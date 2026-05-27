"""
gradio GUI for the fine-tuned per-artist GPT-2.

two tabs:
  - single artist: pick an artist, optional prompt, set the sampling knobs,
    generate one chunk of lyrics.
  - compare: same prompt + same sampling settings, 2 or 3 artist tags,
    results rendered in parallel columns. this is the side-by-side mode
    CLAUDE.md asks for.

the model + tokenizer load once at import time. each click reuses them.

usage:
    python app.py
    python app.py --ckpt checkpoints/final
    python app.py --share          # public gradio link
    python app.py --port 7861
"""

import argparse
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from data import ARTISTS

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DEFAULT = PROJECT_ROOT / "checkpoints" / "best"

NONE_CHOICE = "(none)"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# parse args first so --ckpt actually takes effect when we load the model
_parser = argparse.ArgumentParser()
_parser.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT)
_parser.add_argument("--share", action="store_true")
_parser.add_argument("--port", type=int, default=7860)
_args, _ = _parser.parse_known_args()

DEVICE = pick_device()
print(f"Loading {_args.ckpt} on {DEVICE}...")
TOKENIZER = GPT2TokenizerFast.from_pretrained(_args.ckpt)
MODEL = GPT2LMHeadModel.from_pretrained(_args.ckpt).to(DEVICE).eval()
print(f"Model ready ({sum(p.numel() for p in MODEL.parameters()) / 1e6:.1f}M params).")


def _strip_artist_tag(text):
    # drop the `<ARTIST: name>\n` prefix the model echoes back from the prompt
    m = re.match(r"<ARTIST:[^>]+>\n", text)
    return text[m.end():] if m else text


@torch.no_grad()
def generate(artist, prompt, max_new_tokens, temperature, top_k, top_p,
             rep_penalty, no_repeat_ngram, seed):
    if artist == NONE_CHOICE:
        return ""
    # negative seed means "don't bother setting one"
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
    full = f"<ARTIST: {artist}>\n{(prompt or '').strip()}"
    ids = TOKENIZER(full, return_tensors="pt").input_ids.to(DEVICE)
    out = MODEL.generate(
        ids,
        do_sample=True,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k and int(top_k) > 0 else 0,
        top_p=float(top_p),
        repetition_penalty=float(rep_penalty),
        no_repeat_ngram_size=int(no_repeat_ngram),
        pad_token_id=TOKENIZER.pad_token_id,
    )
    decoded = TOKENIZER.decode(out[0], skip_special_tokens=False)
    return _strip_artist_tag(decoded).strip()


def build_single_tab():
    gr.Markdown("Pick one artist and generate lyrics in their style.")
    artist = gr.Dropdown(ARTISTS, value=ARTISTS[0], label="Artist")
    prompt = gr.Textbox(
        label="Prompt (optional)",
        placeholder="e.g. 'walking down the street tonight'",
        lines=2,
    )
    with gr.Row():
        max_new = gr.Slider(20, 400, value=200, step=10, label="Max new tokens")
        temp = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
    with gr.Row():
        top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k (0 = off)")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
    with gr.Row():
        rep_pen = gr.Slider(1.0, 2.0, value=1.2, step=0.05,
                            label="Repetition penalty (1.0 = off)")
        no_rep = gr.Slider(0, 6, value=3, step=1,
                           label="No-repeat n-gram (0 = off)")
        seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
    btn = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Generated lyrics", lines=18)
    btn.click(
        generate,
        inputs=[artist, prompt, max_new, temp, top_k, top_p, rep_pen, no_rep, seed],
        outputs=output,
    )


def build_compare_tab():
    gr.Markdown(
        "Same prompt + same sampling settings, but each column conditions on a "
        "different artist tag. Set a fixed seed to make the comparison apples-to-apples."
    )
    prompt = gr.Textbox(
        label="Prompt (optional)",
        placeholder="leave blank to let each artist start from scratch",
        lines=2,
    )
    with gr.Row():
        max_new = gr.Slider(20, 400, value=200, step=10, label="Max new tokens")
        temp = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
    with gr.Row():
        top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k (0 = off)")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
    with gr.Row():
        rep_pen = gr.Slider(1.0, 2.0, value=1.2, step=0.05,
                            label="Repetition penalty (1.0 = off)")
        no_rep = gr.Slider(0, 6, value=3, step=1,
                           label="No-repeat n-gram (0 = off)")
        seed = gr.Number(value=42, label="Seed (-1 = random)", precision=0)
    with gr.Row():
        a = gr.Dropdown(ARTISTS, value=ARTISTS[0], label="Artist A")
        b = gr.Dropdown(ARTISTS, value=ARTISTS[1], label="Artist B")
        c = gr.Dropdown([NONE_CHOICE] + ARTISTS, value=NONE_CHOICE,
                       label="Artist C (optional)")
    btn = gr.Button("Generate all", variant="primary")
    with gr.Row():
        out_a = gr.Textbox(label="Artist A", lines=20)
        out_b = gr.Textbox(label="Artist B", lines=20)
        out_c = gr.Textbox(label="Artist C", lines=20)

    def gen_all(prompt, max_new, temp, top_k, top_p, rep_pen, no_rep, seed, a, b, c):
        return (
            generate(a, prompt, max_new, temp, top_k, top_p, rep_pen, no_rep, seed),
            generate(b, prompt, max_new, temp, top_k, top_p, rep_pen, no_rep, seed),
            generate(c, prompt, max_new, temp, top_k, top_p, rep_pen, no_rep, seed),
        )

    btn.click(
        gen_all,
        inputs=[prompt, max_new, temp, top_k, top_p, rep_pen, no_rep, seed, a, b, c],
        outputs=[out_a, out_b, out_c],
    )


with gr.Blocks(title="Female Pop Lyric Generator") as demo:
    gr.Markdown("# Female Pop Lyric Generator")
    gr.Markdown(
        "GPT-2 small fine-tuned on lyrics from 9 contemporary female "
        "pop singer-songwriters. The artist tag is a learned control token, "
        "so different tags should produce different styles."
    )
    with gr.Tabs():
        with gr.Tab("Single artist"):
            build_single_tab()
        with gr.Tab("Compare"):
            build_compare_tab()


if __name__ == "__main__":
    demo.launch(server_port=_args.port, share=_args.share, theme=gr.themes.Soft())
