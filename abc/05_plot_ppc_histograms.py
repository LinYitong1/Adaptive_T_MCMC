#!/usr/bin/env python3
"""Per-stat PPC histograms for M4b: one panel per ABC target statistic.

Each panel shows the distribution of M4b PPC draws, with
- the human group-level value as a red dashed line,
- the PPC median (solid blue) and 90% interval (light blue band).
"""

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
import numpy as np
import pandas as pd
import seaborn as sns

from Model_Core.lopez_features import LOPEZ_FEATURES_PRIMARY

PANEL_LABELS = {
    "mean_adjacent_distance": "Mean adjacent distance",
    "sd_adjacent_distance": "SD adjacent distance",
    "switch_rate_distance": "Distance switch rate",
    "n_relevant_bigrams_norm": "Relevant bigrams (CN)",
}

PANELS = [(s, PANEL_LABELS.get(s, s.replace("_", " ").title())) for s in LOPEZ_FEATURES_PRIMARY]


def _panel(ax, x: np.ndarray, human: float, label: str, n_bins: int) -> None:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        ax.set_title(f"{label}\n(no finite PPC draws)", loc="left", fontsize=10)
        ax.axis("off")
        return

    q05, q50, q95 = np.quantile(finite, [0.05, 0.50, 0.95])
    in_band = bool(q05 <= human <= q95)

    ax.hist(
        finite,
        bins=n_bins,
        color="#9cb6c9",
        edgecolor="#2d5f8b",
        linewidth=0.6,
        alpha=0.85,
    )

    ax.axvspan(q05, q95, color="#cdd9e3", alpha=0.45, zorder=0)
    ax.axvline(q50, color="#2d5f8b", linewidth=1.4, linestyle="-", label=f"PPC median = {q50:.3g}")
    ax.axvline(human, color="#d62728", linewidth=1.6, linestyle="--", label=f"Human mean = {human:.3g}")

    ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

    text = (
        f"q05 = {q05:.3g}\n"
        f"q95 = {q95:.3g}\n"
        f"human in 90% : {'yes' if in_band else 'no'}"
    )
    color = "#2d8b53" if in_band else "#a83232"
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85),
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-stat PPC histograms with human mean.")
    parser.add_argument(
        "--ppc-draws",
        default="abc/data/animals_oaf_m4b_abc_ppc_wideprior_1000ref_tol001_100x50_ppc_draws.csv",
        help="Path to *_ppc_draws.csv (one row per PPC draw).",
    )
    parser.add_argument(
        "--human-stats",
        default="abc/data/animals_oaf_lopez_style_reference_M4b_wideprior_1000x50_cap150_human_group_stats.csv",
        help="Path to *_human_group_stats.csv (one row, group-level human means).",
    )
    parser.add_argument(
        "--out",
        default="abc/figures/animals_oaf_m4b_ppc_wideprior_1000ref_tol001_histograms",
        help="Output prefix (writes .png and .pdf).",
    )
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--ncols", type=int, default=2)
    args = parser.parse_args()

    ppc = pd.read_csv(ROOT / args.ppc_draws)
    human_row = pd.read_csv(ROOT / args.human_stats).iloc[0]

    panels = [(stat, label) for stat, label in PANELS if stat in ppc.columns and stat in human_row.index]
    if not panels:
        raise RuntimeError("None of the expected statistics are present in the PPC table.")

    ncols = max(1, args.ncols)
    nrows = int(np.ceil(len(panels) / ncols))

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 2.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    n_draws = len(ppc)

    for ax, (stat, label) in zip(axes_flat, panels):
        x = pd.to_numeric(ppc[stat], errors="coerce").to_numpy(float)
        human = float(human_row[stat])
        _panel(ax, x, human, label, args.n_bins)

    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    fig.suptitle(
        f"M4b posterior predictive distributions  (n_draws = {n_draws})",
        fontweight="bold",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")

    summary_rows = []
    for stat, label in panels:
        x = pd.to_numeric(ppc[stat], errors="coerce").to_numpy(float)
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            continue
        q05, q50, q95 = np.quantile(finite, [0.05, 0.50, 0.95])
        human = float(human_row[stat])
        summary_rows.append(
            {
                "stat": stat,
                "label": label,
                "n_draws": int(len(finite)),
                "ppc_q05": float(q05),
                "ppc_q50": float(q50),
                "ppc_q95": float(q95),
                "ppc_mean": float(np.mean(finite)),
                "ppc_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "human": human,
                "human_in_90pct_interval": bool(q05 <= human <= q95),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out.with_name(out.name + "_summary.csv"), index=False)

    print(f"Wrote {out.with_suffix('.png')}")
    print(f"Wrote {out.with_suffix('.pdf')}")
    print(f"Wrote {out.with_name(out.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
