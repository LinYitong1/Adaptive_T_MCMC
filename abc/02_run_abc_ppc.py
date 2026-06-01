#!/usr/bin/env python3
"""Rejection ABC for M4b parameters followed by posterior predictive checks."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model_Core.config import load_config
from Model_Core.data_loader import load_fluency_csv, lists_from_fluency, resolve_fluency_path
from Model_Core.sampler import build_assets
from Model_Core.simulate import load_category_bundle
from Model_Core.lopez_features import (
    LOPEZ_DIAGNOSTIC_SEMANTIC,
    LOPEZ_FEATURES_DIAGNOSTIC_TIMING,
    LOPEZ_FEATURES_PRIMARY,
)
from Model_Core import lopez_reference as lopez

ABC_FEATURES = list(LOPEZ_FEATURES_PRIMARY)
# fLL (lexical) and list_length are not reported as PPC content; the PPC focuses on
# process signatures (trajectory + nRB) and the timing diagnostic mu.
PPC_STATS = (
    ABC_FEATURES
    + list(LOPEZ_DIAGNOSTIC_SEMANTIC)
    + list(LOPEZ_FEATURES_DIAGNOSTIC_TIMING)
)

PARAM_COLS = [
    "eta",
    "T_min",
    "T_max",
    "a",
    "b",
    "sigma_q",
    "gamma0",
    "gamma1",
]


def compute_distances(ref: pd.DataFrame, human: pd.Series, features: list[str]) -> pd.DataFrame:
    """Weighted Euclidean ABC distance; scales by M4b prior-predictive SD (ddof=1)."""
    m4b = ref[ref["model_id"] == "M4b_adaptive_T_global"].dropna(subset=features).copy()
    x = m4b[features].astype(float)
    h = human[features].astype(float)
    # Sample SD (ddof=1): standard for ABC tolerance on finite reference draws.
    scale = x.std(ddof=1).replace(0, np.nan)
    z = (x - h) / scale
    m4b["abc_distance"] = np.sqrt((z.to_numpy(float) ** 2).sum(axis=1))
    return m4b.sort_values("abc_distance").reset_index(drop=True)


def params_from_row(base, row: pd.Series):
    """Restore M4b parameters from a reference/posterior row (seed set by caller)."""
    p = deepcopy(base)
    p.model_id = "M4b_adaptive_T_global"  # type: ignore[assignment]
    for col in PARAM_COLS:
        setattr(p, col, float(row[col]))
    p.max_steps = int(row.get("max_steps", base.max_steps))
    p.use_latency_cap = True
    p.latency_cap = int(row.get("latency_cap", base.latency_cap))
    return p


def ppc_simulate(
    posterior: pd.DataFrame,
    assets,
    asset_dict: dict,
    base_params,
    lopez,
    freq_ref: dict,
    bigram_ref: dict,
    *,
    n_draws: int,
    n_lists: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    choices = posterior.index.to_numpy()
    for ppc_draw in range(n_draws):
        row = posterior.loc[int(rng.choice(choices))]
        params = params_from_row(base_params, row)
        params.random_seed = int(rng.integers(0, 1_000_000_000))
        stats = lopez.simulate_reference_row(
            assets,
            asset_dict,
            params,
            n_lists,
            freq_ref,
            bigram_ref,
        )
        rows.append(
            {
                "ppc_draw": ppc_draw,
                "source_draw": int(row["draw"]),
                "abc_distance": float(row["abc_distance"]),
                **{c: float(row[c]) for c in PARAM_COLS},
                **stats,
            }
        )
    return pd.DataFrame(rows)


def summarize_posterior(posterior: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Accepted posterior vs full M4b reference (prior-predictive) quantiles."""
    rows = []
    for col in PARAM_COLS + ["abc_distance"]:
        post = posterior[col].dropna().astype(float)
        prior_x = prior[col].dropna().astype(float)
        rows.append(
            {
                "parameter": col,
                "mean": float(post.mean()),
                "sd": float(post.std(ddof=1)) if len(post) > 1 else 0.0,
                "q05": float(post.quantile(0.05)),
                "q50": float(post.quantile(0.50)),
                "q95": float(post.quantile(0.95)),
                "prior_mean": float(prior_x.mean()) if len(prior_x) else np.nan,
                "prior_sd": float(prior_x.std(ddof=1)) if len(prior_x) > 1 else 0.0,
                "prior_q05": float(prior_x.quantile(0.05)) if len(prior_x) else np.nan,
                "prior_q50": float(prior_x.quantile(0.50)) if len(prior_x) else np.nan,
                "prior_q95": float(prior_x.quantile(0.95)) if len(prior_x) else np.nan,
                "n_prior_draws": int(len(prior_x)),
                "n_accepted_draws": int(len(post)),
            }
        )
    return pd.DataFrame(rows)


def ppc_interval(
    ppc: pd.DataFrame,
    human: pd.Series,
    features: list[str],
    *,
    n_accepted: int,
    accept_frac: float,
    n_unique_posterior_used: int,
    ppc_lists_per_draw: int,
) -> pd.DataFrame:
    rows = []
    for f in features + ["list_length"]:
        if f not in ppc.columns or f not in human:
            continue
        x = ppc[f].dropna().astype(float)
        rows.append(
            {
                "stat": f,
                "human": float(human[f]),
                "ppc_mean": float(x.mean()),
                "ppc_sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "ppc_q05": float(x.quantile(0.05)),
                "ppc_q50": float(x.quantile(0.50)),
                "ppc_q95": float(x.quantile(0.95)),
                "human_in_90pct_interval": bool(x.quantile(0.05) <= float(human[f]) <= x.quantile(0.95)),
                "n_ppc_draws": int(len(x)),
                "n_accepted": int(n_accepted),
                "accept_frac": float(accept_frac),
                "n_unique_posterior_used": int(n_unique_posterior_used),
                "ppc_lists_per_draw": int(ppc_lists_per_draw),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="M4b rejection ABC + PPC.")
    parser.add_argument("--prefix", default="abc/data/animals_oaf_lopez_style_reference_M4b_wideprior_1000x50_cap150")
    parser.add_argument("--accept-frac", type=float, default=0.01)
    parser.add_argument("--ppc-draws", type=int, default=100)
    parser.add_argument(
        "--ppc-lists",
        "--ppc-lists-per-draw",
        dest="ppc_lists",
        type=int,
        default=50,
        help="Simulated lists per PPC draw (sensitivity: e.g. 50 vs 749 human lists).",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument(
        "--include-length",
        action="store_true",
        help="Add list_length to the ABC distance target (productivity-aware inference).",
    )
    parser.add_argument("--out-prefix", default="abc/data/animals_oaf_m4b_abc_ppc_wideprior_1000ref_tol001_100x50")
    args = parser.parse_args()

    paths, base = load_config()
    prefix = paths.root / args.prefix
    ref = pd.read_csv(prefix.with_name(prefix.name + "_model_reference.csv"))
    human = pd.read_csv(prefix.with_name(prefix.name + "_human_group_stats.csv")).iloc[0]
    meta = json.loads(prefix.with_name(prefix.name + "_meta.json").read_text())

    abc_features = list(ABC_FEATURES) + (["list_length"] if args.include_length else [])
    ranked = compute_distances(ref, human, abc_features)
    n_accept = max(1, int(np.ceil(len(ranked) * args.accept_frac)))
    posterior = ranked.head(n_accept).copy()

    vocab, llm_df, emb = load_category_bundle(paths, "animals", params=base)
    assets = build_assets(vocab, llm_df, emb, k_neighbors=base.k_neighbors)
    asset_dict = {
        "word_index": assets.word_index,
        "distance_matrix": np.sqrt(assets.distance_sq),
        "embeddings": emb["embeddings"],
    }
    fluency_path = resolve_fluency_path(paths.processed, paths.raw, "animals", cohort=meta.get("cohort", "oaf"))  # type: ignore[arg-type]
    human_df = load_fluency_csv(fluency_path, category="animals")
    human_lists = lists_from_fluency(human_df)
    freq_ref = lopez.build_frequency_reference(
        human_lists,
        list(vocab),
        alpha=float(meta.get("freq_alpha", 0.5)),
    )
    bigram_ref = lopez.build_bigram_reference(
        human_lists,
        min_count=int(meta.get("bigram_min_count", 3)),
        window=int(meta.get("bigram_window", 2)),
        sig_z=float(meta.get("bigram_sig_z", 1.96)),
    )

    ppc = ppc_simulate(
        posterior,
        assets,
        asset_dict,
        base,
        lopez,
        freq_ref,
        bigram_ref,
        n_draws=args.ppc_draws,
        n_lists=args.ppc_lists,
        seed=args.seed,
    )
    n_unique_posterior_used = int(ppc["source_draw"].nunique()) if "source_draw" in ppc.columns else int(len(posterior))

    post_summary = summarize_posterior(posterior, ranked)
    ppc_summary = ppc_interval(
        ppc,
        human,
        PPC_STATS,
        n_accepted=len(posterior),
        accept_frac=args.accept_frac,
        n_unique_posterior_used=n_unique_posterior_used,
        ppc_lists_per_draw=args.ppc_lists,
    )

    out_prefix = paths.root / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    posterior_path = out_prefix.with_name(out_prefix.name + "_posterior_draws.csv")
    post_summary_path = out_prefix.with_name(out_prefix.name + "_posterior_summary.csv")
    ppc_path = out_prefix.with_name(out_prefix.name + "_ppc_draws.csv")
    ppc_summary_path = out_prefix.with_name(out_prefix.name + "_ppc_summary.csv")
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")

    posterior.to_csv(posterior_path, index=False)
    post_summary.to_csv(post_summary_path, index=False)
    ppc.to_csv(ppc_path, index=False)
    ppc_summary.to_csv(ppc_summary_path, index=False)
    run_meta = {
        "abc_features": abc_features,
        "include_length_in_abc": bool(args.include_length),
        "ppc_stats": PPC_STATS,
        "features": abc_features,
        "reference_prefix": str(prefix),
        "accept_frac": float(args.accept_frac),
        "n_m4b_reference_rows": int(len(ranked)),
        "n_accepted": int(len(posterior)),
        "n_unique_posterior_used": n_unique_posterior_used,
        "ppc_draws": int(args.ppc_draws),
        "ppc_lists_per_draw": int(args.ppc_lists),
        "seed": int(args.seed),
        "abc_distance_scaling": "M4b prior-predictive SD per feature, ddof=1 (sample SD)",
        "display_scaling_note": (
            "ABC rejection uses model prior-predictive SD (this script). "
            "The PPC forest (03_plot_ppc_forest.py) standardizes by human list-level SD."
        ),
    }
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("Accepted posterior parameter summary")
    print(post_summary.round(4).to_string(index=False))
    print("\nPPC summary")
    print(ppc_summary.round(4).to_string(index=False))
    print(f"\nWrote {posterior_path}")
    print(f"Wrote {post_summary_path}")
    print(f"Wrote {ppc_path}")
    print(f"Wrote {ppc_summary_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
