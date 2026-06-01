#!/usr/bin/env python3
"""Nature-style standardized PPC forest plot for M4b."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model_Core.config import load_config
from Model_Core.data_loader import load_fluency_csv, lists_from_fluency, resolve_fluency_path
from Model_Core.levy_flight import fit_mu_lbn
from Model_Core.lopez_features import (
    LOPEZ_FEATURES_PRIMARY,
    MU_LATENCY_MEDIAN_NORM,
    MU_LATENCY_RAW,
)

# PPC fit figure shows ONLY the 4 ABC target features. p90, similarity, and the
# latency exponents are not ABC criteria; the latency posterior is shown separately
# (scripts/36).
PLOT_ROWS = [
    ("Semantic trajectory", "mean_adjacent_distance", "Mean adjacent distance"),
    ("Semantic trajectory", "sd_adjacent_distance", "SD adjacent distance"),
    ("Semantic trajectory", "switch_rate_distance", "Distance switch rate"),
    ("Associative structure", "n_relevant_bigrams_norm", "Relevant bigrams (CN)"),
]

ALL_PLOT_ROWS = PLOT_ROWS


def bootstrap_latency_mu_sd(
    cohort: str,
    n_boot: int,
    seed: int,
) -> tuple[float, int, int]:
    """Block-bootstrap SD of pooled human latency mu (lists resampled with replacement).

    Returns (sd, n_skipped_boots, n_successful_boots).
    """
    paths, _ = load_config()
    df = load_fluency_csv(
        resolve_fluency_path(paths.processed, paths.raw, "animals", cohort=cohort),  # type: ignore[arg-type]
        category="animals",
    )
    lists = lists_from_fluency(df)
    lat_by_list = []
    for iris in lists["iri"]:
        if not isinstance(iris, list):
            lat_by_list.append(np.array([], dtype=float))
            continue
        vals = pd.to_numeric(pd.Series(iris), errors="coerce")
        vals = vals[np.isfinite(vals) & (vals > 0)].to_numpy(float)
        lat_by_list.append(vals)
    rng = np.random.default_rng(seed)
    mus = []
    n_skipped = 0
    n = len(lat_by_list)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pooled = np.concatenate([lat_by_list[i] for i in idx])
        if len(pooled) < 10:
            n_skipped += 1
            continue
        fit = fit_mu_lbn(pooled, series_type="latency")
        if np.isfinite(fit.mu_hat):
            mus.append(float(fit.mu_hat))
        else:
            n_skipped += 1
    sd = float(np.std(mus, ddof=1)) if len(mus) > 1 else np.nan
    return sd, n_skipped, len(mus)


def _per_list_median_norm_mu_sd(human_lists: pd.DataFrame) -> float:
    """SD across lists of per-list median-normalized LBN mu (for forest scaling)."""
    mus = []
    for _, row in human_lists.iterrows():
        iris = row.get("iri")
        if not isinstance(iris, list):
            continue
        vals = pd.to_numeric(pd.Series(iris), errors="coerce").dropna().to_numpy(float)
        base = vals[np.isfinite(vals) & (vals >= 1.0)]
        if len(base) < 2:
            continue
        med = float(np.median(base))
        if med <= 0:
            continue
        fit = fit_mu_lbn(base / med, series_type="latency")
        if np.isfinite(fit.mu_hat):
            mus.append(float(fit.mu_hat))
    return float(np.std(mus, ddof=1)) if len(mus) > 1 else np.nan


def build_scale_table(
    human_per_list: pd.DataFrame,
    human_lists: pd.DataFrame,
    *,
    latency_mu_sd: float,
) -> pd.DataFrame:
    rows = []
    median_norm_list_sd = _per_list_median_norm_mu_sd(human_lists)
    for group, stat, label in ALL_PLOT_ROWS:
        if stat in human_per_list.columns:
            vals = pd.to_numeric(human_per_list[stat], errors="coerce").dropna()
            scale = float(vals.std(ddof=1))
            source = "human list-level SD"
        elif stat == MU_LATENCY_MEDIAN_NORM:
            scale = median_norm_list_sd if np.isfinite(median_norm_list_sd) else latency_mu_sd
            source = (
                "human per-list median-norm mu SD"
                if np.isfinite(median_norm_list_sd)
                else "human list-bootstrap SD (fallback)"
            )
        elif stat == MU_LATENCY_RAW:
            scale = latency_mu_sd
            source = "human list-bootstrap SD (raw IRI pooled)"
        else:
            scale = np.nan
            source = "missing"
        rows.append({"group": group, "stat": stat, "label": label, "scale": scale, "scale_source": source})
    return pd.DataFrame(rows)


def make_plot(plot_df: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    plot_df = plot_df.copy()
    plot_df["y"] = np.arange(len(plot_df))[::-1]
    groups = plot_df.groupby("group", sort=False)
    for _, g in groups:
        y0, y1 = g["y"].min() - 0.45, g["y"].max() + 0.45
        ax.axhspan(y0, y1, color="#f3f4f3", zorder=0)

    ax.axvspan(-1, 1, color="#d8dde1", alpha=0.55, zorder=0)
    ax.axvline(0, color="#d65222", linewidth=1.3, zorder=1)
    ax.hlines(
        plot_df["y"],
        plot_df["q05_delta"],
        plot_df["q95_delta"],
        color="#9cb6c9",
        linewidth=6,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(plot_df["median_delta"], plot_df["y"], color="#2d5f8b", s=44, zorder=3)

    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Standardized model-human discrepancy")
    ax.set_title(
        "M4b PPC: 4-feature ABC target",
        loc="left",
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)

    # Group labels on the left margin.
    for group, g in groups:
        ax.text(
            ax.get_xlim()[0],
            g["y"].max() + 0.62,
            group,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#404040",
        )

    max_abs = np.nanmax(np.abs(plot_df[["q05_delta", "q95_delta", "median_delta"]].to_numpy(float)))
    lim = max(2.5, min(12.0, np.ceil(max_abs + 0.5)))
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standardized M4b PPC forest plot.")
    parser.add_argument(
        "--ppc-summary",
        default="abc/data/animals_oaf_m4b_abc_ppc_wideprior_1000ref_tol001_100x50_ppc_summary.csv",
    )
    parser.add_argument("--human-per-list", default="abc/data/animals_oaf_lopez_style_reference_M4b_wideprior_1000x50_cap150_human_per_list_stats.csv")
    parser.add_argument("--cohort", default="oaf")
    parser.add_argument("--latency-bootstrap", type=int, default=0,
                        help="Latency-mu bootstrap is only needed for diagnostic mu rows, "
                             "which are not in the 4-feature forest; 0 skips it.")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--out", default="abc/figures/animals_oaf_m4b_ppc_wideprior_1000ref_tol001_forest")
    args = parser.parse_args()

    ppc = pd.read_csv(ROOT / args.ppc_summary)
    human_per_list = pd.read_csv(ROOT / args.human_per_list)
    paths, _ = load_config()
    fluency_path = resolve_fluency_path(
        paths.processed, paths.raw, "animals", cohort=args.cohort  # type: ignore[arg-type]
    )
    human_lists = lists_from_fluency(load_fluency_csv(fluency_path, category="animals"))
    latency_mu_sd, n_boot_skipped, n_boot_ok = bootstrap_latency_mu_sd(
        args.cohort, args.latency_bootstrap, args.seed
    )
    print(
        f"Latency mu bootstrap: n_boot={args.latency_bootstrap}, "
        f"n_ok={n_boot_ok}, n_skipped={n_boot_skipped}, sd={latency_mu_sd:.4f}"
    )
    scales = build_scale_table(
        human_per_list, human_lists, latency_mu_sd=latency_mu_sd
    )
    df = ppc.merge(scales, on="stat", how="inner")
    for col in ["ppc_q05", "ppc_q50", "ppc_q95"]:
        df[col.replace("ppc_", "") + "_delta"] = (df[col] - df["human"]) / df["scale"]
    df = df.rename(columns={"q50_delta": "median_delta"})
    df["abs_median_delta"] = df["median_delta"].abs()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    analysis_path = out.with_name(out.name + "_analysis.csv")
    df.to_csv(analysis_path, index=False)
    boot_meta_path = out.with_name(out.name + "_bootstrap_meta.json")
    boot_meta = {
        "cohort": args.cohort,
        "n_boot_requested": args.latency_bootstrap,
        "n_boot_successful": n_boot_ok,
        "n_boot_skipped": n_boot_skipped,
        "latency_mu_bootstrap_sd": latency_mu_sd,
        "scale_source": "human list-bootstrap SD",
    }
    boot_meta_path.write_text(json.dumps(boot_meta, indent=2), encoding="utf-8")
    make_plot(df, out)

    print(df[[
        "group",
        "stat",
        "human",
        "ppc_q50",
        "ppc_q05",
        "ppc_q95",
        "scale",
        "scale_source",
        "median_delta",
        "q05_delta",
        "q95_delta",
        "human_in_90pct_interval",
    ]].round(4).to_string(index=False))
    print(f"Wrote {analysis_path}")
    print(f"Wrote {boot_meta_path}")
    print(f"Wrote {out.with_suffix('.png')}")
    print(f"Wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
