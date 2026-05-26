"""
fine-tune GPT-2 small on the per-artist lyric splits from data.py.

the artist tags `<ARTIST: name>` are registered as special tokens so the BPE
doesn't split them. weighted sampling keeps each artist roughly equally
represented since song counts are uneven. AdamW + cosine schedule with warmup.
per-artist val loss + exposure get logged to wandb.

usage:
    python train.py
    python train.py --epochs 5 --lr 3e-5
    python train.py --no-wandb     # smoke test, skips wandb
"""

import argparse
import json
import logging
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

# shut up the "process just got forked" warning from tokenizers when the
# dataloader spawns workers. has to happen before the transformers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)

from data import ARTISTS, PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"

# defaults, can override from CLI
MODEL_NAME = "gpt2"
BLOCK_SIZE = 1024
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 25
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(name):
    path = PROCESSED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python data.py` first.")
    return [json.loads(line) for line in path.open()]


class LyricsDataset(Dataset):
    # each example is {input_ids, attention_mask, labels, artist}
    # pad positions in labels get set to -100 so the LM loss skips them
    def __init__(self, rows, tokenizer, block_size):
        self.examples = []
        truncated = 0
        for r in rows:
            enc = tokenizer(
                r["text"],
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attn = enc["attention_mask"].squeeze(0)
            if attn.sum().item() == block_size:
                truncated += 1
            labels = input_ids.clone()
            labels[attn == 0] = -100
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attn,
                "labels": labels,
                "artist": r["artist"],
            })
        if truncated:
            log.info("  %d/%d examples hit the %d-token cap and got truncated",
                     truncated, len(rows), block_size)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "artists": [b["artist"] for b in batch],
    }


def build_weighted_sampler(train_ds):
    # weight each example by 1/count(its artist) so all 9 artists get
    # roughly equal exposure even though song counts are uneven
    counts = Counter(ex["artist"] for ex in train_ds.examples)
    weights = [1.0 / counts[ex["artist"]] for ex in train_ds.examples]
    return WeightedRandomSampler(
        weights, num_samples=len(train_ds), replacement=True
    )


@torch.no_grad()
def per_artist_val_loss(model, val_loader, device):
    # mean CE loss on val, grouped by artist tag
    model.eval()
    by_artist = {a: [] for a in ARTISTS}
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[..., :-1, :].contiguous()
        target = labels[..., 1:].contiguous()
        ce = F.cross_entropy(
            logits.transpose(1, 2), target,
            reduction="none", ignore_index=-100,
        )
        # average over non-padding positions for each example separately
        mask = (target != -100).float()
        per_example = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for art, l in zip(batch["artists"], per_example.tolist()):
            by_artist[art].append(l)
    return {
        a: (sum(v) / len(v) if v else float("nan"))
        for a, v in by_artist.items()
    }


def train(args):
    set_seed(args.seed)
    device = pick_device()
    log.info("Device: %s", device)

    # tokenizer + the artist control tokens
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    artist_tokens = [f"<ARTIST: {a}>" for a in ARTISTS]
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": artist_tokens,
        "pad_token": "<|pad|>",
    })
    log.info("Added %d special tokens. Vocab size: %d", added, len(tokenizer))

    log.info("Tokenizing splits...")
    train_ds = LyricsDataset(load_split("train"), tokenizer, args.block_size)
    val_ds = LyricsDataset(load_split("val"), tokenizer, args.block_size)
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate,
    )

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %.1fM", n_params / 1e6)

    # standard AdamW: no weight decay on biases or LayerNorm weights
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_frac),
        num_training_steps=total_steps,
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="female-pop-music-lyrics",
            config={
                "model": MODEL_NAME,
                "block_size": args.block_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_frac": args.warmup_frac,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "device": device,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "artists": ARTISTS,
            },
        )

    CKPT_DIR.mkdir(exist_ok=True)
    best_val = float("inf")
    exposure = Counter()
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            for a in batch["artists"]:
                exposure[a] += 1
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % LOG_EVERY == 0:
                lr_now = scheduler.get_last_lr()[0]
                log.info("ep %d step %d  loss %.3f  lr %.2e",
                         epoch, global_step, loss.item(), lr_now)
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_now,
                        "step": global_step,
                    })

        by_artist = per_artist_val_loss(model, val_loader, device)
        valid = [v for v in by_artist.values() if not math.isnan(v)]
        agg = sum(valid) / len(valid) if valid else float("nan")
        log.info("epoch %d  agg val loss %.3f", epoch, agg)
        for a, v in by_artist.items():
            log.info("  %-22s %.3f", a, v)
        if use_wandb:
            wandb.log({
                "val/agg_loss": agg,
                **{f"val/loss/{a}": v for a, v in by_artist.items()},
                **{f"exposure/{a}": exposure[a] for a in ARTISTS},
                "epoch": epoch,
            })

        if agg < best_val:
            best_val = agg
            best_dir = CKPT_DIR / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            log.info("Saved best to %s (val %.3f)", best_dir, best_val)

    final_dir = CKPT_DIR / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("Done in %.1fs. Best val %.3f.", time.time() - t0, best_val)
    if use_wandb:
        wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
