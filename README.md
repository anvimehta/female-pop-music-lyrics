# female-pop-music-lyrics

Fine-tunes **GPT-2 Small** on lyrics from nine contemporary female pop
singer-songwriters, using `<ARTIST: name>` as a learned control token, then
exposes the model through an interactive Gradio GUI for single-artist and
side-by-side generation.

**Research question:** can a transformer learn *distinct* writing styles for
artists with heavily overlapping genres and themes, or does it collapse them
into a generic "pop cluster" voice?

## Artists

Taylor Swift, Lana Del Rey, Sabrina Carpenter, Kelsea Ballerini, Maisie Peters,
Lorde, Gracie Abrams, Phoebe Bridgers, Olivia Rodrigo.

## Setup

```bash
pip install -r requirements.txt
export GENIUS_ACCESS_TOKEN=<your token from https://genius.com/api-clients>
```

## 1. Data

Scrapes the Genius API, cleans/dedupes, then writes per-artist train/val/test
splits.

```bash
python data.py                     # both phases
python data.py --scrape            # just get raw data
python data.py --process           # just re-clean/split from cached raw
python data.py --max-per-artist 0  # no cap on songs per artist
```

Outputs:
- `data/raw/<artist>.json` — cached Genius pulls
- `data/processed/{train,val,test}.jsonl` — examples of the form
  `{"artist": "...", "title": "...", "text": "<ARTIST: name>\n<lyrics>"}`

The `<ARTIST: name>` prefix is the control token. **Don't strip it** at
training or eval time.

## 2. Train

Fine-tunes `gpt2` (124M) with the artist tags registered as **special tokens**
(so BPE doesn't split them), weighted sampling to equalize per-artist exposure,
AdamW + cosine schedule with warmup, and per-artist val loss logged to W&B.

```bash
python train.py                              # default: 3 epochs, lr 5e-5
python train.py --epochs 5 --lr 3e-5
python train.py --no-wandb                   # smoke test, skip W&B
```

Checkpoints land in `checkpoints/best/` (lowest val loss) and
`checkpoints/final/`.

## 3. Eval
```bash
python eval.py                          # PPL grid + stylometry + generation
python eval.py --skip-generate          # PPL + stylometry of reals only
python eval.py --ckpt checkpoints/final
```

Outputs:
- `ppl_matrix.{csv,png}` — 9×9 PPL grid: rows = true artist, cols =
  conditioning tag. **Swapped-tag control**: if the diagonal is lower than the
  off-diagonal, the artist token is actually doing work.
- `ppl_summary.{csv,png}` — per-artist correct-tag vs swapped-tag PPL gap.
- `stylometry.csv` — TTR, mean line length, rhyme density, sentiment for real
  vs generated samples per artist.
- `tfidf_top_terms.csv` — top distinctive terms per artist (real lyrics).
- `samples/<artist>.txt` — raw generations used for the stylometry comparison.

## 4. GUI

```bash
python app.py                              # http://localhost:7860
python app.py --ckpt checkpoints/final
python app.py --share                      # public gradio link
python app.py --port 7861
```

Two tabs:
- **Single artist** — pick one of the nine, optional prompt, full sampling
  controls (temperature, top-k, top-p, repetition penalty, no-repeat n-gram,
  max new tokens, seed).
- **Compare** — same prompt + same sampling settings, 2–3 artist tags rendered
  in parallel columns. Set a fixed seed for apples-to-apples comparisons.

The model loads once at startup; each click reuses it.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── data.py              # scrape, clean, format, split
├── train.py             # fine-tuning loop with W&B logging
├── eval.py              # per-artist perplexity + stylometry
├── app.py               # Gradio GUI
├── checkpoints/         # not committed
├── data/                # not committed
├── eval_results/        # CSVs + plots from eval.py
└── wandb/               # local W&B run state
```
