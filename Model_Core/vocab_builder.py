"""Build per-category candidate vocabularies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .data_loader import load_fluency_csv, unique_words_by_category


def build_vocab_from_human(
    fluency_path: Path,
    category: str,
    extra_words: Optional[Iterable[str]] = None,
    min_length: int = 2,
) -> list[str]:
    df = load_fluency_csv(fluency_path, category=category)
    words = set(unique_words_by_category(df).get(category, []))
    if extra_words:
        words.update(w.strip().lower() for w in extra_words if w.strip())
    vocab = sorted(w for w in words if w.isalpha() and len(w) >= min_length)
    return vocab


def build_vocab_from_llm_scores(
    scores_path: Path,
    top_n: int = 500,
) -> list[str]:
    df = pd.read_parquet(scores_path)
    df = df.sort_values("logprob", ascending=False).head(top_n)
    return sorted(df["word"].astype(str).str.lower().unique().tolist())


def merge_vocab(
    human_words: Iterable[str],
    llm_words: Iterable[str],
) -> list[str]:
    merged = set(w.strip().lower() for w in human_words if w.strip())
    merged.update(w.strip().lower() for w in llm_words if w.strip())
    return sorted(w for w in merged if w.isalpha() and len(w) > 1)


def save_vocab(vocab: list[str], path: Path, category: str, meta: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"category": category, "n_words": len(vocab), "words": vocab}
    if meta:
        payload["meta"] = meta
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_vocab(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data["words"]
