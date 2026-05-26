"""
evaluate the fine-tuned model on the held-out test split.

two things happen here:

1) per-artist held-out PPL with a swapped-tag control. for each test example
   we compute the lyric-token PPL when we condition on each of the 9 artist
   tags. that gives a 9x9 grid. if the diagonal (true_artist == cond_artist)
   is meaningfully lower than the off-diagonal, the artist token is actually
   doing work.

2) stylometry. TTR for lexical diversity, mean line length, a (crude) rhyme
   density measure, VADER sentiment, and TF-IDF top terms per artist. real
   lyrics vs generated samples per artist.

everything writes to eval_results/ (CSVs, PNG plots, and the raw generations).

needs: pip install scikit-learn pandas matplotlib vaderSentiment

usage:
    python eval.py
    python eval.py --ckpt checkpoints/final
    python eval.py --n-generated 30
    python eval.py --skip-generate     # PPL + stylometry of reals only
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data import ARTISTS, PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DEFAULT = PROJECT_ROOT / "checkpoints" / "best"
RESULTS_DIR = PROJECT_ROOT / "eval_results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_dir, device):
    tokenizer = GPT2TokenizerFast.from_pretrained(ckpt_dir)
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir).to(device).eval()
    return model, tokenizer


def load_test_split():
    path = PROCESSED_DIR / "test.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python data.py` first.")
    return [json.loads(line) for line in path.open()]


def _strip_artist_tag(text):
    # drop the `<ARTIST: name>\n` prefix that data.py added
    m = re.match(r"<ARTIST:[^>]+>\n", text)
    return text[m.end():] if m else text


# === perplexity ===

@torch.no_grad()
def lyric_ppl(model, tokenizer, lyrics, conditioning_artist, device,
              block_size=1024):
    # PPL of `lyrics` given a particular artist tag. only the lyric-token
    # positions count - the prefix gets masked with -100 so we're measuring
    # surprise on the lyrics, not on the tag.
    prefix = f"<ARTIST: {conditioning_artist}>\n"
    full = prefix + lyrics
    enc = tokenizer(full, truncation=True, max_length=block_size,
                    return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100
    out = model(input_ids=input_ids, labels=labels)
    return float(torch.exp(out.loss).item())


def ppl_confusion_matrix(model, tokenizer, test_rows, device):
    # rows = true artist, cols = the artist tag we condition on. cells are
    # the mean PPL over the test examples in that (true, cond) cell.
    sums = {t: {c: 0.0 for c in ARTISTS} for t in ARTISTS}
    counts = {t: {c: 0 for c in ARTISTS} for t in ARTISTS}
    t0 = time.time()
    for i, row in enumerate(test_rows):
        true_artist = row["artist"]
        lyrics = _strip_artist_tag(row["text"])
        for cond in ARTISTS:
            p = lyric_ppl(model, tokenizer, lyrics, cond, device)
            sums[true_artist][cond] += p
            counts[true_artist][cond] += 1
        if (i + 1) % 10 == 0:
            log.info("  PPL grid: %d/%d  (%.1fs)", i + 1, len(test_rows), time.time() - t0)
    mat = pd.DataFrame(
        [[sums[t][c] / max(1, counts[t][c]) for c in ARTISTS] for t in ARTISTS],
        index=ARTISTS, columns=ARTISTS,
    )
    mat.index.name = "true_artist"
    mat.columns.name = "conditioning_artist"
    return mat


def ppl_summary(matrix):
    # per artist: correct-tag PPL, mean swapped-tag PPL, and the gap
    rows = []
    for a in ARTISTS:
        correct = matrix.loc[a, a]
        others = matrix.loc[a, [c for c in ARTISTS if c != a]]
        rows.append({
            "artist": a,
            "ppl_correct_tag": correct,
            "ppl_swapped_mean": others.mean(),
            "ppl_swapped_min": others.min(),
            "delta": others.mean() - correct,
            "delta_pct": (others.mean() - correct) / correct * 100,
        })
    return pd.DataFrame(rows)


def plot_confusion(matrix, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ARTISTS)))
    ax.set_xticklabels(ARTISTS, rotation=45, ha="right")
    ax.set_yticks(range(len(ARTISTS)))
    ax.set_yticklabels(ARTISTS)
    ax.set_xlabel("Conditioning artist tag")
    ax.set_ylabel("True artist (test lyrics)")
    ax.set_title("PPL by (true, conditioning) artist (lower is better)")
    mean = matrix.values.mean()
    for i in range(len(ARTISTS)):
        for j in range(len(ARTISTS)):
            v = matrix.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > mean else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary(summary, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, summary["ppl_correct_tag"], w, label="Correct tag")
    ax.bar(x + w / 2, summary["ppl_swapped_mean"], w, label="Swapped tag (mean)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["artist"], rotation=45, ha="right")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Correct- vs swapped-tag PPL per artist")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# === generation ===

@torch.no_grad()
def generate_samples(model, tokenizer, artist, n, device,
                     max_new_tokens=200, temperature=0.9,
                     top_p=0.95, batch_size=5):
    # generate n samples by feeding just the artist control token as prompt
    prompt = f"<ARTIST: {artist}>\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out_texts = []
    while len(out_texts) < n:
        k = min(batch_size, n - len(out_texts))
        out = model.generate(
            input_ids,
            do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
        for seq in out:
            text = tokenizer.decode(seq, skip_special_tokens=False)
            out_texts.append(_strip_artist_tag(text))
    return out_texts


# === stylometry ===

WORD_RE = re.compile(r"[A-Za-z']+")


def word_tokens(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def type_token_ratio(text):
    words = word_tokens(text)
    return len(set(words)) / max(1, len(words))


def avg_line_length(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return float(np.mean([len(word_tokens(l)) for l in lines]))


def rhyme_density(text, window=3, suffix_len=3):
    # fraction of lines whose last word's tail (last N chars) shows up in
    # any of the previous `window` lines' tails. it's a crude, phoneme-free
    # proxy but it's consistent across artists so the comparison still works.
    lines = [l for l in text.splitlines() if l.strip()]
    tails = []
    rhymes = 0
    for l in lines:
        words = word_tokens(l)
        if not words:
            tails.append(None)
            continue
        tail = words[-1][-suffix_len:]
        recent = [t for t in tails[-window:] if t]
        if tail in recent:
            rhymes += 1
        tails.append(tail)
    return rhymes / max(1, len(lines))


_SIA = None


def sentiment_compound(text):
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    return _SIA.polarity_scores(text)["compound"]


def stylometry(texts):
    sentiments = [sentiment_compound(t) for t in texts]
    return {
        "n": len(texts),
        "ttr_mean": float(np.mean([type_token_ratio(t) for t in texts])),
        "line_len_mean": float(np.mean([avg_line_length(t) for t in texts])),
        "rhyme_density_mean": float(np.mean([rhyme_density(t) for t in texts])),
        "sentiment_mean": float(np.mean(sentiments)),
        "sentiment_std": float(np.std(sentiments)),
    }


def stylometry_table(real_by_artist, gen_by_artist):
    rows = []
    for a in ARTISTS:
        rows.append({"artist": a, "source": "real", **stylometry(real_by_artist[a])})
        if a in gen_by_artist:
            rows.append({"artist": a, "source": "generated", **stylometry(gen_by_artist[a])})
    return pd.DataFrame(rows)


def tfidf_top_terms(corpus_by_artist, top_k=10):
    # treat each artist's lyrics as one document; pull top-k distinctive terms.
    artists = list(corpus_by_artist.keys())
    docs = [" ".join(corpus_by_artist[a]).lower() for a in artists]
    vec = TfidfVectorizer(min_df=2, stop_words="english", max_features=5000)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for i, a in enumerate(artists):
        scores = X[i].toarray().ravel()
        top = np.argsort(-scores)[:top_k]
        rows.append({"artist": a, "top_terms": ", ".join(vocab[top])})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT)
    p.add_argument("--n-generated", type=int, default=20,
                   help="samples per artist used for the stylometry comparison")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    device = pick_device()
    log.info("Loading %s on %s", args.ckpt, device)
    model, tokenizer = load_model(args.ckpt, device)

    test_rows = load_test_split()
    log.info("Test split: %d examples", len(test_rows))

    # 1) PPL grid + summary
    log.info("PPL grid (%d examples x %d conditioning artists)...",
             len(test_rows), len(ARTISTS))
    matrix = ppl_confusion_matrix(model, tokenizer, test_rows, device)
    matrix.to_csv(RESULTS_DIR / "ppl_matrix.csv")
    summary = ppl_summary(matrix)
    summary.to_csv(RESULTS_DIR / "ppl_summary.csv", index=False)
    plot_confusion(matrix, RESULTS_DIR / "ppl_matrix.png")
    plot_summary(summary, RESULTS_DIR / "ppl_summary.png")
    log.info("PPL summary:\n%s", summary.round(3).to_string(index=False))

    # 2) stylometry: real vs generated per artist
    real_by_artist = defaultdict(list)
    for r in test_rows:
        real_by_artist[r["artist"]].append(_strip_artist_tag(r["text"]))

    gen_by_artist = {}
    if not args.skip_generate:
        samples_dir = RESULTS_DIR / "samples"
        samples_dir.mkdir(exist_ok=True)
        for a in ARTISTS:
            log.info("Generating %d samples for %s...", args.n_generated, a)
            samples = generate_samples(
                model, tokenizer, a, args.n_generated, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            gen_by_artist[a] = samples
            slug = a.lower().replace(" ", "_")
            (samples_dir / f"{slug}.txt").write_text(
                "\n\n---\n\n".join(samples), encoding="utf-8")

    style = stylometry_table(real_by_artist, gen_by_artist)
    style.to_csv(RESULTS_DIR / "stylometry.csv", index=False)
    log.info("Stylometry:\n%s", style.round(3).to_string(index=False))

    tfidf = tfidf_top_terms(real_by_artist)
    tfidf.to_csv(RESULTS_DIR / "tfidf_top_terms.csv", index=False)
    log.info("TF-IDF top terms (real lyrics):\n%s", tfidf.to_string(index=False))

    log.info("All outputs in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
