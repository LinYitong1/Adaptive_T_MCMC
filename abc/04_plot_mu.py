#!/usr/bin/env python3
"""Standalone M4b latency heavy-tail figure (panel-E style, model latency).

The latency exponent is NOT an ABC fitting target (human latencies are seconds,
simulated latencies are retrieval turns). It is reported here as a separate timing
diagnostic. This mirrors panel E of the human-signature figure (log-binned density +
power-law slope) but on the *model* latency (retrieval turns), pooled across the
accepted M4b posterior draws. Only the raw exponent is shown and the human value is
not drawn. A short targeted simulation is run from the fitted posterior; the main
ABC/PPC pipeline is not re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model_Core.baselines import generate_for_model
from Model_Core.config import load_config
from Model_Core.levy_flight import fit_mu_lbn, latency_series_from_trace
from Model_Core.sampler import build_assets
from Model_Core.simulate import load_category_bundle

PARAM_COLS = ["eta", "T_min", "T_max", "a", "b", "sigma_q", "gamma0", "gamma1"]


def pool_model_turns(
    posterior: pd.DataFrame,
    *,
    n_lists_per_draw: int,
    max_steps: int,
    latency_cap: int,
    seed: int,
) -> np.ndarray:
    """Pool per-turn model latencies (retrieval turns) across accepted posterior draws."""
    paths, base = load_config()
    base.model_id = "M4b_adaptive_T_global"  # type: ignore[assignment]
    base.use_latency_cap = True
    base.latency_cap = int(latency_cap)
    base.max_steps = int(max_steps)
    vocab, llm_df, emb = load_category_bundle(paths, "animals", params=base)
    assets = build_assets(vocab, llm_df, emb, k_neighbors=base.k_neighbors)

    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for _, row in posterior.iterrows():
        params = deepcopy(base)
        for col in PARAM_COLS:
            setattr(params, col, float(row[col]))
        params.max_steps = int(row.get("max_steps", max_steps))
        params.latency_cap = int(row.get("latency_cap", latency_cap))
        for i in range(n_lists_per_draw):
            _, trace = generate_for_model(
                assets,
                params,
                list_id=f"mu_fig_{i}",
                rng=np.random.default_rng(int(rng.integers(0, 1_000_000_000))),
            )
            lat = latency_series_from_trace(trace)
            if len(lat):
                parts.append(lat)
    return np.concatenate(parts) if parts else np.array([])


def log_binned_pdf(values: np.ndarray, n_bins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values) & (values > 0)]
    edges = np.geomspace(values.min(), values.max(), n_bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = counts / (counts.sum() * widths)
    mask = (counts > 0) & np.isfinite(pdf) & (pdf > 0)
    return centers[mask], pdf[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot M4b model-latency heavy-tail (panel-E style).")
    parser.add_argument(
        "--posterior-draws",
        default="abc/data/animals_oaf_m4b_abc_ppc_wideprior_1000ref_tol001_100x50_posterior_draws.csv",
    )
    parser.add_argument("--n-lists-per-draw", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--latency-cap", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--out", default="abc/figures/animals_oaf_m4b_mu_posterior")
    args = parser.parse_args()

    posterior = pd.read_csv(ROOT / args.posterior_draws)
    turns = pool_model_turns(
        posterior,
        n_lists_per_draw=args.n_lists_per_draw,
        max_steps=args.max_steps,
        latency_cap=args.latency_cap,
        seed=args.seed,
    )
    if len(turns) < 2:
        raise SystemExit("No model latencies pooled; check posterior-draws input.")

    fit = fit_mu_lbn(turns, series_type="latency")
    x, y = log_binned_pdf(turns, n_bins=18)

    import seaborn as sns

    sns.set_theme(style="white", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(x, y, s=22, color="#2d5f8b", alpha=0.85, zorder=3)
    if np.isfinite(fit.mu_hat):
        tail = x >= np.quantile(x, 0.35)
        if tail.sum() >= 2:
            x0, y0 = x[tail][0], y[tail][0]
            yfit = y0 * (x[tail] / x0) ** (-fit.mu_hat)
            ax.plot(x[tail], yfit, color="#d65222", linewidth=1.8, zorder=4)
        ax.text(
            0.06,
            0.10,
            rf"$\mu_{{latency}}={fit.mu_hat:.2f}$",
            transform=ax.transAxes,
            fontsize=12,
            color="#1b1b1b",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("M4b heavy-tailed retrieval latencies", loc="left", fontweight="bold")
    ax.set_xlabel("Latency (retrieval turns)")
    ax.set_ylabel("Log-binned density")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Pooled model latencies: n={len(turns)}, median={np.median(turns):.2f} turns, "
          f"mu_raw={fit.mu_hat:.3f} (R2={fit.mu_r2:.3f})")
    print(f"Wrote {out.with_suffix('.png')}")
    print(f"Wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
