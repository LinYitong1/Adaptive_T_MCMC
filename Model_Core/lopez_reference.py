"""Lopez-style per-list summary statistics and ABC reference-table builder.

This module is the shared core for both pipelines:
  * Model_compare/  (six-model ABC-RF reference)
  * abc/            (M4b posterior reference + PPC)

Statistics (per list):
  - mean/sd/p90 of adjacent embedding distance, distance-based switch rate
  - mean adjacent similarity (diagnostic, ~collinear with mean distance)
  - frequency likelihood fLL (diagnostic; human empirical document frequency)
  - n_relevant_bigrams_norm nRB (CN relevant word-couples, length-detrended)
  - pooled heavy-tail latency exponent mu (diagnostic; logarithmic binning)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from .abcrf_priors import MODELS, sample_prior
from .baselines import generate_for_model
from .config import ModelParams, ProjectPaths
from .data_loader import load_fluency_csv, lists_from_fluency, resolve_fluency_path
from .levy_flight import levy_metrics_pooled
from .lopez_features import (
    LOPEZ_FEATURES_DIAGNOSTIC_TIMING,
    LOPEZ_FEATURES_PRIMARY,
    LOPEZ_FEATURES_RAW_MU,
    LOPEZ_FEATURES_WITH_MEDIAN_NORM_MU,
    MU_LATENCY_MEDIAN_NORM,
    MU_LATENCY_RAW,
)
from .metrics import (
    adjacent_semantic_distances,
    bigram_relevance,
    detect_switches_distance,
    switch_rate,
)
from .sampler import build_assets
from .simulate import load_category_bundle

# Primary ABC target (4): 3 trajectory stats + nRB. See lopez_features for rationale.
FEATURES = list(LOPEZ_FEATURES_PRIMARY)

PARAM_KEYS = [
    "eta", "T_min", "T_max", "a", "b", "sigma_q", "p_global", "p_jump",
    "fixed_T", "gamma0", "gamma1", "latency_cap", "max_steps", "random_seed",
]


# ---------------------------------------------------------------------------
# Reference distributions (built once from human lists)
# ---------------------------------------------------------------------------

def _pair_key(w1: str, w2: str) -> tuple[str, str]:
    """Unordered word-pair key."""
    return tuple(sorted((str(w1), str(w2))))


def build_frequency_reference(human_lists: pd.DataFrame, vocab: list[str], alpha: float) -> dict:
    """Document-frequency reference for fLL (Guimera et al. 2026, Eq. 10).

    f_w = number of lists containing word w; p_appear = (f_w + alpha) / (N + 2 alpha).
    Keyed over the empirical human corpus UNION the model vocab so human words outside
    the candidate vocabulary keep their true human document frequency.
    """
    N = int(len(human_lists))
    doc_counts: Counter[str] = Counter()
    lengths = []
    for words in human_lists["words"]:
        lengths.append(len(words))
        doc_counts.update(set(words))
    lambda_bar = float(np.mean(lengths)) if lengths else float("nan")
    denom = float(N + 2.0 * alpha)
    support = set(doc_counts) | set(vocab)
    p_appear = {w: (doc_counts.get(w, 0) + alpha) / denom for w in support}
    default = alpha / denom
    return {
        "p_appear": p_appear,
        "default": default,
        "N": N,
        "lambda_bar": lambda_bar,
        "alpha": float(alpha),
    }


def build_bigram_reference(
    human_lists: pd.DataFrame,
    min_count: int,
    *,
    window: int = 2,
    sig_z: float = 1.96,
) -> dict:
    """Conceptual-network (CN) relevant word-couples, after Goni et al. (2011).

    A couple co-occurring within a moving window of ``window`` positions in more lists
    than a Binomial(N, p0) chance baseline (p0 = (f_i/N)(f_j/N)) and at least
    ``min_count`` lists is 'relevant'. The per-list nRB count (adjacent transitions
    whose couple is relevant) is detrended for list length using a linear slope.
    """
    N = int(len(human_lists))
    doc_freq: Counter[str] = Counter()
    cooc: Counter[tuple[str, str]] = Counter()
    rows = []
    for _, row in human_lists.iterrows():
        words = row["words"]
        doc_freq.update(set(words))
        pairs_in_list: set[tuple[str, str]] = set()
        n = len(words)
        for a in range(n):
            for b in range(a + 1, min(a + 1 + window, n)):
                if words[a] != words[b]:
                    pairs_in_list.add(_pair_key(words[a], words[b]))
        cooc.update(pairs_in_list)
        rows.append({"length": n, "words": words})

    relevant: set[tuple[str, str]] = set()
    for (wi, wj), obs in cooc.items():
        if obs < min_count:
            continue
        p0 = (doc_freq[wi] / N) * (doc_freq[wj] / N) if N else 0.0
        if p0 <= 0.0 or p0 >= 1.0:
            continue
        mu = N * p0
        sigma = math.sqrt(N * p0 * (1.0 - p0))
        if obs > mu + sig_z * sigma:
            relevant.add((wi, wj))

    lengths = np.array([r["length"] for r in rows], dtype=float)
    counts = np.array(
        [
            sum(1 for w1, w2 in pairwise(r["words"]) if _pair_key(w1, w2) in relevant)
            for r in rows
        ],
        dtype=float,
    )
    length_mean = float(np.mean(lengths)) if len(lengths) else np.nan
    if len(lengths) > 1 and float(np.var(lengths)) > 0:
        slope = float(np.cov(lengths, counts, ddof=0)[0, 1] / np.var(lengths))
    else:
        slope = 0.0

    return {
        "relevant_pairs": relevant,
        "cooc_counts": cooc,
        "doc_freq": dict(doc_freq),
        "n_relevant_pairs": len(relevant),
        "length_mean": length_mean,
        "slope": slope,
        "min_count": min_count,
        "window": int(window),
        "sig_z": float(sig_z),
    }


# ---------------------------------------------------------------------------
# Per-list statistics
# ---------------------------------------------------------------------------

def _frequency_likelihood(words: list[str], freq_ref: dict) -> tuple[float, float]:
    """fLL (Guimera et al. 2026, Eq. 10): geometric mean of per-word appearance probs."""
    if not words:
        return np.nan, np.nan
    p_appear = freq_ref["p_appear"]
    default = freq_ref["default"]
    lambda_bar = freq_ref["lambda_bar"]
    exponent = (len(words) / lambda_bar) if lambda_bar and lambda_bar > 0 else 1.0
    logs = []
    for w in words:
        p = p_appear.get(w, default)
        term = 1.0 - (1.0 - p) ** exponent
        term = min(max(term, 1e-12), 1.0)
        logs.append(math.log(term))
    log_fll = float(np.mean(logs))
    return log_fll, float(math.exp(log_fll))


def _relevant_bigram_stats(words: list[str], ref: dict) -> dict[str, float]:
    pairs = [_pair_key(w1, w2) for w1, w2 in pairwise(words)]
    count = float(sum(1 for pair in pairs if pair in ref["relevant_pairs"]))
    length = float(len(words))
    norm = count - ref["slope"] * (length - ref["length_mean"])
    rate = count / max(1.0, length - 1.0)
    return {
        "n_relevant_bigrams": count,
        "n_relevant_bigrams_rate": float(rate),
        "n_relevant_bigrams_norm": float(norm),
    }


def _distance_stats(words: list[str], asset_dict: dict, switch_quantile: float) -> dict[str, float]:
    dists = adjacent_semantic_distances(
        words, asset_dict["word_index"], asset_dict["distance_matrix"]
    )
    switches = detect_switches_distance(dists, switch_quantile)
    return {
        "list_length": float(len(words)),
        "mean_adjacent_distance": float(np.mean(dists)) if dists else np.nan,
        "sd_adjacent_distance": float(np.std(dists, ddof=1)) if len(dists) > 1 else np.nan,
        "p90_adjacent_distance": float(np.quantile(dists, 0.90)) if dists else np.nan,
        "switch_rate_distance": switch_rate(switches),
        "mean_adjacent_similarity": bigram_relevance(
            words, asset_dict["distance_matrix"], asset_dict["word_index"]
        ),
    }


def summarize_words(
    words: list[str],
    asset_dict: dict,
    freq_ref: dict,
    bigram_ref: dict,
    switch_quantile: float,
) -> dict[str, float]:
    log_fll, fll = _frequency_likelihood(words, freq_ref)
    return {
        **_distance_stats(words, asset_dict, switch_quantile),
        "log_frequency_likelihood": log_fll,
        "frequency_likelihood": fll,
        **_relevant_bigram_stats(words, bigram_ref),
    }


def human_trace_from_lists(human_lists: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for list_idx, row in human_lists.reset_index(drop=True).iterrows():
        words = row["words"]
        iris = row.get("iri")
        if not isinstance(iris, list):
            iris = [np.nan] * len(words)
        for t, word in enumerate(words):
            lat = iris[t] if t < len(iris) else np.nan
            rows.append({"list_idx": int(list_idx), "t": int(t), "word": word, "latency": lat})
    return pd.DataFrame(rows)


def aggregate_reference(
    per_list: pd.DataFrame,
    trace_df: pd.DataFrame,
    asset_dict: dict,
    switch_quantile: float,
    *,
    include_tail_latency_robustness: bool = False,
) -> dict[str, float]:
    out = per_list.mean(numeric_only=True).to_dict()
    out["n_lists"] = float(len(per_list))
    if not trace_df.empty:
        out.update(
            levy_metrics_pooled(
                trace_df,
                asset_dict["word_index"],
                asset_dict["distance_matrix"],
                tail_quantile=switch_quantile,
                include_tail_latency_robustness=include_tail_latency_robustness,
            )
        )
    return out


def parameter_row(params: ModelParams) -> dict:
    return {k: getattr(params, k) for k in PARAM_KEYS if hasattr(params, k)}


def simulate_reference_row(
    assets,
    asset_dict: dict,
    params: ModelParams,
    n_lists: int,
    freq_ref: dict,
    bigram_ref: dict,
    *,
    include_tail_latency_robustness: bool = False,
) -> dict:
    per_list_rows = []
    traces = []
    seed = int(params.random_seed or 0)
    for list_idx in range(n_lists):
        rng = np.random.default_rng(seed + list_idx)
        words, trace = generate_for_model(
            assets,
            params,
            list_id=f"lopez_ref_{params.model_id}_{seed}_{list_idx}",
            rng=rng,
        )
        per_list_rows.append(
            summarize_words(words, asset_dict, freq_ref, bigram_ref, params.switch_quantile)
        )
        if trace is not None and not trace.empty:
            tr = trace.copy()
            tr["list_idx"] = list_idx
            traces.append(tr)

    per_list = pd.DataFrame(per_list_rows)
    trace_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    return aggregate_reference(
        per_list,
        trace_df,
        asset_dict,
        params.switch_quantile,
        include_tail_latency_robustness=include_tail_latency_robustness,
    )


# ---------------------------------------------------------------------------
# Reference-table orchestration (used by Model_compare and abc CLIs)
# ---------------------------------------------------------------------------

def build_reference(
    paths: ProjectPaths,
    base: ModelParams,
    *,
    out_prefix: str,
    category: str = "animals",
    cohort: str = "oaf",
    models: list[str] | None = None,
    n_draws: int = 200,
    n_lists: int = 50,
    max_steps: int = 500,
    latency_cap: int = 150,
    seed: int = 20260601,
    freq_alpha: float = 0.5,
    bigram_min_count: int = 3,
    bigram_window: int = 2,
    bigram_sig_z: float = 1.96,
    wide_prior: bool = False,
) -> dict:
    """Build the human-summary + model prior-predictive reference table.

    Writes ``{out_prefix}_{human_group_stats,human_per_list_stats,model_reference,
    model_summary_by_model,meta}``. Returns the human summary dict.
    """
    base.category = category
    base.max_steps = max_steps
    base.use_latency_cap = True
    base.latency_cap = latency_cap

    vocab, llm_df, emb = load_category_bundle(paths, category, params=base)
    assets = build_assets(vocab, llm_df, emb, k_neighbors=base.k_neighbors)
    asset_dict = {
        "word_index": assets.word_index,
        "distance_matrix": np.sqrt(assets.distance_sq),
        "embeddings": emb["embeddings"],
    }

    fluency_path = resolve_fluency_path(paths.processed, paths.raw, category, cohort=cohort)
    human_lists = lists_from_fluency(load_fluency_csv(fluency_path, category=category))
    freq_ref = build_frequency_reference(human_lists, list(vocab), alpha=freq_alpha)
    bigram_ref = build_bigram_reference(
        human_lists, min_count=bigram_min_count, window=bigram_window, sig_z=bigram_sig_z
    )

    human_per_list = pd.DataFrame(
        [
            {
                "source": "human",
                "subject": row["subject"],
                "trial": row["trial"],
                **summarize_words(row["words"], asset_dict, freq_ref, bigram_ref, base.switch_quantile),
            }
            for _, row in human_lists.iterrows()
        ]
    )
    human_trace = human_trace_from_lists(human_lists)
    human_summary = aggregate_reference(human_per_list, human_trace, asset_dict, base.switch_quantile)
    human_summary.update({"source": "human", "cohort": cohort, "human_fluency_path": str(fluency_path)})

    models = models or list(MODELS)
    rng = np.random.default_rng(seed)
    rows = []
    total = len(models) * n_draws
    done = 0
    for model_id in models:
        for draw in range(n_draws):
            params = sample_prior(model_id, rng, base, wide_prior=wide_prior)
            params.max_steps = max_steps
            params.use_latency_cap = True
            params.latency_cap = latency_cap
            row = {"source": "model", "model_id": model_id, "draw": draw, **parameter_row(params)}
            try:
                params.validate()
                row.update(
                    simulate_reference_row(assets, asset_dict, params, n_lists, freq_ref, bigram_ref)
                )
            except Exception as exc:  # keep long runs alive on a single bad draw
                row["error"] = repr(exc)
            rows.append(row)
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"[{done}/{total}] completed", flush=True)

    out = paths.root / out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    human_group_path = out.with_name(out.name + "_human_group_stats.csv")
    human_per_list_path = out.with_name(out.name + "_human_per_list_stats.csv")
    reference_path = out.with_name(out.name + "_model_reference.csv")
    model_summary_path = out.with_name(out.name + "_model_summary_by_model.csv")
    meta_path = out.with_name(out.name + "_meta.json")

    pd.DataFrame([human_summary]).to_csv(human_group_path, index=False)
    human_per_list.to_csv(human_per_list_path, index=False)
    reference = pd.DataFrame(rows)
    reference.to_csv(reference_path, index=False)
    if not reference.empty and "model_id" in reference.columns:
        model_summary = reference.groupby("model_id").agg(
            n_rows=("model_id", "size"),
            n_valid=("model_id", lambda s: len(reference.loc[s.index].dropna(subset=FEATURES))),
            list_length_mean=("list_length", "mean"),
            list_length_sd=("list_length", "std"),
            **{feature: (feature, "mean") for feature in FEATURES},
        )
        model_summary.reset_index().to_csv(model_summary_path, index=False)

    meta = {
        "abcrf_primary_features": FEATURES,
        "prior_variant": "wide" if wide_prior else "default",
        "abcrf_robustness_feature_sets": {
            "with_median_norm_mu": LOPEZ_FEATURES_WITH_MEDIAN_NORM_MU,
            "with_raw_mu": LOPEZ_FEATURES_RAW_MU,
        },
        "diagnostic_columns_not_used_as_primary_features": [
            "p90_adjacent_distance",
            "list_length",
            "log_frequency_likelihood",
            "frequency_likelihood",
            "mean_adjacent_similarity",
            *LOPEZ_FEATURES_DIAGNOSTIC_TIMING,
        ],
        "models": models,
        "n_draws_per_model": n_draws,
        "n_lists_per_draw": n_lists,
        "max_steps": max_steps,
        "latency_cap": latency_cap,
        "seed": seed,
        "freq_alpha": freq_alpha,
        "frequency_likelihood_method": "doc_freq_length_exponent_geomean_eq10",
        "bigram_method": "cn_window_cooccurrence_significance",
        "bigram_min_count": bigram_min_count,
        "bigram_window": bigram_window,
        "bigram_sig_z": bigram_sig_z,
        "n_relevant_bigram_pairs": len(bigram_ref["relevant_pairs"]),
        "relevant_bigram_length_slope": bigram_ref["slope"],
        "human_mean_length_for_bigram_norm": bigram_ref["length_mean"],
        "cohort": cohort,
        "human_fluency_path": str(fluency_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {human_group_path}")
    print(f"Wrote {human_per_list_path}")
    print(f"Wrote {reference_path}")
    print(f"Wrote {model_summary_path}")
    print(f"Wrote {meta_path}")
    return human_summary
