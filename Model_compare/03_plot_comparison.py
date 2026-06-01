#!/usr/bin/env python3
"""ABC-RF model-comparison figure: posterior model probabilities + feature importance."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_LABELS = {
    "M0_frequency": "M0 frequency",
    "M1_crw": "M1 CRW",
    "M2_crw_jump": "M2 CRW+jump",
    "M3_fixed_T": "M3 fixed T",
    "M4_adaptive_T": "M4 adaptive T",
    "M4b_adaptive_T_global": "M4b adaptive T + global",
}

FEATURE_LABELS = {
    "mean_adjacent_distance": "Mean distance",
    "sd_adjacent_distance": "SD distance",
    "switch_rate_distance": "Switch rate",
    "n_relevant_bigrams_norm": "Relevant bigrams",
}


def plot_abcrf(posterior: pd.DataFrame, importance: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    post = posterior.sort_values("posterior_vote_probability", ascending=True)
    colors = ["#d65222" if m == "M4b_adaptive_T_global" else "#9cb6c9" for m in post["model_id"]]
    ax = axes[0]
    ax.barh([MODEL_LABELS.get(m, m) for m in post["model_id"]], post["posterior_vote_probability"], color=colors)
    for y, v in enumerate(post["posterior_vote_probability"]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ABC-RF posterior probability")
    ax.set_title("A. Model selection", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    imp = importance.sort_values("importance", ascending=True)
    ax = axes[1]
    ax.barh([FEATURE_LABELS.get(f, f) for f in imp["feature"]], imp["importance"], color="#4f7fb5")
    ax.set_xlabel("Random-forest importance")
    ax.set_title("B. Feature importance", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="ABC-RF model-comparison figure.")
    p.add_argument(
        "--abcrf-prefix",
        default="Model_compare/data/animals_oaf_lopez_style_reference_wideprior_200x50_cap150_abcrf_primary_4feat",
    )
    p.add_argument("--out", default="Model_compare/figures/animals_oaf_abcrf_primary4_comparison")
    args = p.parse_args()

    posterior = pd.read_csv(ROOT / (args.abcrf_prefix + "_posterior.csv"))
    importance = pd.read_csv(ROOT / (args.abcrf_prefix + "_feature_importance.csv"))
    plot_abcrf(posterior, importance, ROOT / args.out)
    print(f"Wrote {args.out}.png")


if __name__ == "__main__":
    main()
