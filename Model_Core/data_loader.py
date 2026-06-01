"""Load and validate human semantic fluency data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

FluencyCohort = Literal["oaf", "zhu", "zhu_cleaned", "combined", "demo"]

COHORT_LABELS = {
    "oaf": "BASE animal fluency (primary; file tag oaf)",
    "zhu": "Zhu choice-task appendix",
    "zhu_cleaned": "Zhu animal naming appendix (1024 cap; consecutive repeats removed)",
    "combined": "BASE + Zhu combined",
    "demo": "demo fluency",
}

REQUIRED_COLUMNS = {"subject", "trial", "category", "position", "word"}
OPTIONAL_COLUMNS = {"rt", "iri", "raw_word", "clean_word"}


def normalize_word(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def resolve_fluency_path(
    processed_dir: Path,
    raw_dir: Path,
    category: str,
    cohort: FluencyCohort = "oaf",
) -> Path:
    """Resolve human fluency CSV; primary analysis defaults to BASE (cohort id oaf)."""
    candidates = {
        "oaf": processed_dir / f"{category}_fluency_oaf.csv",
        "zhu": processed_dir / f"{category}_fluency_zhu.csv",
        "zhu_cleaned": processed_dir / f"{category}_fluency_zhu_cleaned.csv",
        "combined": processed_dir / f"{category}_fluency.csv",
        "demo": raw_dir / "demo_fluency.csv",
    }
    path = candidates[cohort]
    if path.exists():
        return path
    if cohort == "oaf" and candidates["combined"].exists():
        return candidates["combined"]
    if cohort != "demo":
        raise FileNotFoundError(
            f"Fluency file not found for cohort={cohort!r}: {path}\n"
            "Run: python scripts/09_prepare_human_data.py"
        )
    return candidates["demo"]


def load_fluency_csv(
    path: Path,
    category: Optional[str] = None,
) -> pd.DataFrame:
    """Load long-format fluency data."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    if "clean_word" in df.columns:
        df["word"] = normalize_word(df["clean_word"])
    else:
        df["word"] = normalize_word(df["word"])

    if category is not None:
        df = df[df["category"] == category].copy()

    df = df.sort_values(["subject", "trial", "position"]).reset_index(drop=True)
    return df


def lists_from_fluency(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, trial) with word list."""
    rows = []
    for (subject, trial, cat), g in df.groupby(["subject", "trial", "category"]):
        g = g.sort_values("position")
        row = {
            "subject": subject,
            "trial": trial,
            "category": cat,
            "words": g["word"].tolist(),
            "positions": g["position"].tolist(),
            "rt": g["rt"].tolist() if "rt" in g.columns else None,
            "iri": g["iri"].tolist() if "iri" in g.columns else None,
        }
        if "condition" in g.columns:
            row["condition"] = g["condition"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def unique_words_by_category(df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cat, g in df.groupby("category"):
        words = sorted(g["word"].unique().tolist())
        out[str(cat)] = words
    return out


def word_frequencies(df: pd.DataFrame, category: str) -> dict[str, float]:
    g = df[df["category"] == category]
    counts = g["word"].value_counts()
    total = counts.sum()
    return {w: c / total for w, c in counts.items()}


def train_test_split_subjects(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = df["subject"].unique()
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(subjects) * test_fraction))
    test_subj = set(rng.choice(subjects, size=n_test, replace=False))
    train = df[~df["subject"].isin(test_subj)].copy()
    test = df[df["subject"].isin(test_subj)].copy()
    return train, test


def save_demo_fluency(path: Path, n_subjects: int = 5) -> Path:
    """Write a small demo CSV for pipeline smoke tests."""
    animals = [
        "dog", "cat", "horse", "cow", "sheep", "pig", "goat", "chicken",
        "eagle", "shark", "lion", "tiger", "bear", "wolf", "fox", "deer",
        "rabbit", "mouse", "elephant", "giraffe", "zebra", "monkey",
        "dolphin", "whale", "penguin", "snake", "frog", "turtle",
    ]
    rows = []
    rng = np.random.default_rng(0)
    for s in range(1, n_subjects + 1):
        trial_words = list(rng.choice(animals, size=18, replace=False))
        for pos, w in enumerate(trial_words, start=1):
            rows.append(
                {
                    "subject": f"S{s:02d}",
                    "trial": 1,
                    "category": "animals",
                    "position": pos,
                    "word": w,
                    "rt": float(rng.uniform(0.8, 3.0)),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
