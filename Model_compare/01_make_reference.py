#!/usr/bin/env python3
"""Build the six-model ABC-RF reference table (prior-predictive simulated data).

Simulates `n_draws` prior draws per model (M0-M4b), each summarized over `n_lists`
lists, and writes the reference + human group/per-list summaries used by
`02_run_abcrf.py`. Defaults reproduce the wide-prior, cap=150 primary reference.

    python Model_compare/01_make_reference.py            # full (slow)
    python Model_compare/01_make_reference.py --n-draws 4 --n-lists 5   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model_Core.abcrf_priors import MODELS
from Model_Core.config import load_config
from Model_Core.lopez_reference import build_reference


def main() -> None:
    p = argparse.ArgumentParser(description="Six-model ABC-RF reference table.")
    p.add_argument("--category", default="animals")
    p.add_argument("--cohort", default="oaf")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--n-draws", type=int, default=200, help="Prior draws per model")
    p.add_argument("--n-lists", type=int, default=50, help="Simulated lists per draw")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--latency-cap", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--freq-alpha", type=float, default=0.5)
    p.add_argument("--bigram-min-count", type=int, default=3)
    p.add_argument("--bigram-window", type=int, default=2)
    p.add_argument("--bigram-sig-z", type=float, default=1.96)
    p.add_argument("--wide-prior", action="store_true", default=True,
                   help="Wide adaptive-model priors (primary setting; on by default).")
    p.add_argument("--default-prior", dest="wide_prior", action="store_false",
                   help="Disable the wide prior.")
    p.add_argument(
        "--out-prefix",
        default="Model_compare/data/animals_oaf_lopez_style_reference_wideprior_200x50_cap150",
    )
    args = p.parse_args()

    paths, base = load_config()
    build_reference(
        paths,
        base,
        out_prefix=args.out_prefix,
        category=args.category,
        cohort=args.cohort,
        models=args.models,
        n_draws=args.n_draws,
        n_lists=args.n_lists,
        max_steps=args.max_steps,
        latency_cap=args.latency_cap,
        seed=args.seed,
        freq_alpha=args.freq_alpha,
        bigram_min_count=args.bigram_min_count,
        bigram_window=args.bigram_window,
        bigram_sig_z=args.bigram_sig_z,
        wide_prior=args.wide_prior,
    )


if __name__ == "__main__":
    main()
