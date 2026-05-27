"""Scrape, clean, and split lyrics for nine female pop artists.

Two phases:
- scrape: hit the Genius API, cache raw JSON per artist under data/raw/.
- process: load cached JSON, clean, dedupe, split per artist, write JSONL splits.

Run both with: python data.py
Run only one with: python data.py --scrape   (or --process)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

ARTISTS = [
    "Taylor Swift",
    "Lana Del Rey",
    "Sabrina Carpenter",
    "Kelsea Ballerini",
    "Maisie Peters",
    "Lorde",
    "Gracie Abrams",
    "Phoebe Bridgers",
    "Olivia Rodrigo",
]

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _format_example(artist: str, lyrics: str) -> str:
    """Wrap lyrics with the artist control token used during training."""
    return f"<ARTIST: {artist}>\n{lyrics}"


_GENIUS_TRAILING = re.compile(r"\d*\s*Embed\s*$", re.IGNORECASE)
_YOU_MIGHT_ALSO_LIKE = re.compile(r"You might also like", re.IGNORECASE)
_LEADING_HEADER = re.compile(r"^[^\n]*Lyrics\s*\n", re.IGNORECASE)
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_lyrics(raw: str) -> str:
    """Strip Genius artifacts: the page header, inline tags, trailing 'Embed'."""
    text = raw.strip()
    text = _LEADING_HEADER.sub("", text, count=1)
    text = _YOU_MIGHT_ALSO_LIKE.sub("", text)
    text = _GENIUS_TRAILING.sub("", text).strip()
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def scrape_artist(genius, artist: str, max_songs):
    """Pull up to max_songs from Genius (None or 0 = no cap, all songs).

    Returns [{title, lyrics}, ...]. Songs with empty lyrics are dropped with
    an explicit warning so failures aren't silent.
    """
    cap = None if not max_songs or max_songs <= 0 else max_songs
    log.info("Scraping %s (cap=%s)...", artist, "all" if cap is None else cap)
    result = genius.search_artist(
        artist,
        max_songs=cap,
        sort="popularity",
        include_features=False,
    )
    if result is None:
        log.warning("No artist result for %s", artist)
        return []
    songs = []
    for song in result.songs:
        if not song.lyrics or not song.lyrics.strip():
            log.warning("Empty lyrics — %s / %s", artist, song.title)
            continue
        songs.append({"title": song.title, "lyrics": song.lyrics})
    log.info("  kept %d songs for %s", len(songs), artist)
    return songs


def scrape_all(max_per_artist: int) -> None:
    """Scrape every artist and cache raw JSON to data/raw/{slug}.json."""
    try:
        import lyricsgenius
    except ImportError:
        log.error("lyricsgenius not installed. Run: pip install lyricsgenius")
        sys.exit(1)

    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        log.error(
            "GENIUS_ACCESS_TOKEN env var not set. Create a client at "
            "https://genius.com/api-clients and `export GENIUS_ACCESS_TOKEN=...`"
        )
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    genius = lyricsgenius.Genius(
        token,
        skip_non_songs=True,
        excluded_terms=[
            "(Remix)", "(Live)", "(Demo)", "(Acoustic)",
            "(Taylor's Version)", "Voice Memo", "Commentary",
        ],
        remove_section_headers=True,
        retries=3,
        timeout=30,
    )
    genius.verbose = False

    for artist in ARTISTS:
        out_path = RAW_DIR / f"{_slug(artist)}.json"
        if out_path.exists():
            log.info("Cached, skipping: %s", out_path.name)
            continue
        songs = scrape_artist(genius, artist, max_per_artist)
        out_path.write_text(json.dumps(songs, indent=2, ensure_ascii=False))
        log.info("Wrote %s (%d songs)", out_path.name, len(songs))


def _dedupe(songs: list[dict]) -> list[dict]:
    """Drop near-duplicates by normalized title (re-records, lives, demos)."""
    seen: set[str] = set()
    out = []
    for s in songs:
        key = re.sub(r"[^a-z0-9]", "", s["title"].lower())
        key = re.sub(
            r"(taylorsversion|remix|acoustic|demo|live|extended|deluxe|fromthevault)",
            "", key,
        )
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def split_indices(
    n: int, val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list[int], list[int], list[int]]:
    """Shuffle 0..n-1 and slice into train/val/test index lists."""
    idx = list(range(n))
    rng.shuffle(idx)
    if n < 3:
        return idx, [], []
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]


def process_all(
    val_frac: float, test_frac: float, seed: int, min_chars: int
) -> None:
    """Clean, dedupe, split per artist, write JSONL splits to data/processed/."""
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        log.error("No raw lyrics under %s. Run with --scrape first.", RAW_DIR)
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for artist in ARTISTS:
        path = RAW_DIR / f"{_slug(artist)}.json"
        if not path.exists():
            log.warning("Missing raw file for %s — skipping", artist)
            continue
        raw_songs = json.loads(path.read_text())

        cleaned = []
        for s in raw_songs:
            text = clean_lyrics(s.get("lyrics", ""))
            if len(text) < min_chars:
                continue
            cleaned.append({"title": s["title"], "lyrics": text})
        cleaned = _dedupe(cleaned)

        train_i, val_i, test_i = split_indices(
            len(cleaned), val_frac, test_frac, rng
        )
        for split_name, idxs in (("train", train_i), ("val", val_i), ("test", test_i)):
            for i in idxs:
                song = cleaned[i]
                splits[split_name].append({
                    "artist": artist,
                    "title": song["title"],
                    "text": _format_example(artist, song["lyrics"]),
                })

        log.info(
            "%s: train=%d val=%d test=%d  (raw=%d, cleaned=%d)",
            artist, len(train_i), len(val_i), len(test_i),
            len(raw_songs), len(cleaned),
        )

    for split_name, rows in splits.items():
        rng.shuffle(rows)
        out_path = PROCESSED_DIR / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Wrote %s (%d examples)", out_path, len(rows))

    total = sum(len(v) for v in splits.values())
    log.info("Done. %d total examples.", total)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scrape", action="store_true", help="Run the Genius scrape phase.")
    p.add_argument("--process", action="store_true", help="Run the clean/split phase.")
    p.add_argument(
        "--max-per-artist", type=int, default=100,
        help="Cap on songs pulled per artist (default 100). Pass 0 for no cap.",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument(
        "--min-chars", type=int, default=200,
        help="Drop lyrics shorter than this many characters after cleaning.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.scrape and not args.process:
        args.scrape = args.process = True

    if args.scrape:
        scrape_all(args.max_per_artist)
    if args.process:
        process_all(args.val_frac, args.test_frac, args.seed, args.min_chars)


if __name__ == "__main__":
    main()
