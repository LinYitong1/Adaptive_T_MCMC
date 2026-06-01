"""Behavioral and mechanism metrics for model evaluation."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None  # type: ignore


def list_length(words: Sequence[str]) -> int:
    return len(words)


def unique_count(words: Sequence[str]) -> int:
    return len(set(words))


def repetition_rate(words: Sequence[str]) -> float:
    if not words:
        return 0.0
    return 1.0 - unique_count(words) / len(words)


def adjacent_semantic_distances(
    words: Sequence[str],
    word_index: dict[str, int],
    distance_matrix: np.ndarray,
) -> list[float]:
    dists = []
    for w1, w2 in pairwise(words):
        i, j = word_index.get(w1), word_index.get(w2)
        if i is None or j is None:
            continue
        dists.append(float(distance_matrix[i, j]))
    return dists


def mean_semantic_distance(dists: list[float]) -> float:
    return float(np.mean(dists)) if dists else np.nan


def detect_switches_distance(
    dists: list[float],
    quantile: float = 0.80,
) -> list[int]:
    if not dists:
        return []
    thr = float(np.quantile(dists, quantile))
    return [int(d > thr) for d in dists]


def detect_switches_cluster(
    words: Sequence[str],
    embeddings: np.ndarray,
    word_index: dict[str, int],
    n_clusters: int = 8,
) -> list[int]:
    if KMeans is None or len(words) < 2:
        return []
    idx = [word_index[w] for w in words if w in word_index]
    if len(idx) < 2:
        return []
    X = embeddings[idx]
    k = min(n_clusters, len(idx))
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(X)
    return [int(labels[i] != labels[i + 1]) for i in range(len(labels) - 1)]


def switch_rate(switches: list[int]) -> float:
    return float(np.mean(switches)) if switches else 0.0


def bigram_overlap(words_a: Sequence[str], words_b: Sequence[str]) -> float:
    bg_a = set(zip(words_a, words_a[1:]))
    bg_b = set(zip(words_b, words_b[1:]))
    if not bg_a and not bg_b:
        return 1.0
    return len(bg_a & bg_b) / max(1, len(bg_a | bg_b))


def bigram_relevance(
    words: Sequence[str],
    distance_matrix: np.ndarray,
    word_index: dict[str, int],
) -> float:
    """Mean 1 / (1 + distance) for adjacent pairs."""
    scores = []
    for w1, w2 in pairwise(words):
        i, j = word_index.get(w1), word_index.get(w2)
        if i is None or j is None:
            continue
        d = distance_matrix[i, j]
        scores.append(1.0 / (1.0 + d))
    return float(np.mean(scores)) if scores else np.nan


def kl_frequency(human_freq: dict[str, float], model_words: Sequence[str]) -> float:
    model_counts = Counter(model_words)
    total = sum(model_counts.values()) or 1
    model_freq = {w: c / total for w, c in model_counts.items()}
    keys = set(human_freq) | set(model_freq)
    kl = 0.0
    for k in keys:
        p = human_freq.get(k, 1e-12)
        q = model_freq.get(k, 1e-12)
        kl += p * np.log(p / q)
    return float(kl)


def delta_E_before_switch(trace: pd.DataFrame, switches: list[int]) -> float:
    if trace.empty or not switches:
        return np.nan
    vals = []
    de = trace["delta_E"].values if "delta_E" in trace.columns else []
    for i, sw in enumerate(switches):
        if sw and i < len(de):
            vals.append(de[i])
    return float(np.mean(vals)) if vals else np.nan


def slope_delta_E_before_switch(trace: pd.DataFrame, switches: list[int]) -> float:
    if "delta_E" not in trace.columns or len(trace) < 2:
        return np.nan
    de = trace["delta_E"].values
    dde = np.diff(de)
    vals = [dde[i] for i, sw in enumerate(switches) if sw and i < len(dde)]
    return float(np.mean(vals)) if vals else np.nan


def latency_summary(trace: pd.DataFrame) -> dict[str, float]:
    """Summarize per-word latency and cumulative latency_sum (turns)."""
    if trace.empty:
        return {"mean_latency": np.nan, "latency_sum": np.nan}

    if "latency" in trace.columns:
        lat = pd.to_numeric(trace["latency"], errors="coerce")
    elif "latency_attempts" in trace.columns:
        lat = pd.to_numeric(trace["latency_attempts"], errors="coerce")
    elif "n_pass_before" in trace.columns:
        lat = pd.to_numeric(trace["n_pass_before"], errors="coerce") + 1.0
    else:
        return {"mean_latency": np.nan, "latency_sum": np.nan}

    lat = lat.dropna()
    if lat.empty:
        return {"mean_latency": np.nan, "latency_sum": np.nan}

    if "latency_sum" in trace.columns:
        total = float(pd.to_numeric(trace["latency_sum"], errors="coerce").iloc[-1])
    else:
        total = float(lat.sum())

    return {
        "mean_latency": float(lat.mean()),
        "latency_sum": total,
    }


def summarize_list(
    words: list[str],
    trace: pd.DataFrame,
    assets: dict,
    human_freq: Optional[dict[str, float]] = None,
    switch_quantile: float = 0.80,
) -> dict:
    word_index = assets["word_index"]
    dist = assets["distance_matrix"]
    emb = assets.get("embeddings")
    adj = adjacent_semantic_distances(words, word_index, dist)
    sw_d = detect_switches_distance(adj, switch_quantile)
    sw_c = (
        detect_switches_cluster(words, emb, word_index)
        if emb is not None
        else []
    )
    out = {
        "list_length": list_length(words),
        "unique_count": unique_count(words),
        "repetition_rate": repetition_rate(words),
        "mean_adjacent_distance": mean_semantic_distance(adj),
        "switch_rate_distance": switch_rate(sw_d),
        "switch_rate_cluster": switch_rate(sw_c) if sw_c else np.nan,
        "bigram_relevance": bigram_relevance(words, dist, word_index),
        "delta_E_before_switch": delta_E_before_switch(trace, sw_d),
        "slope_delta_E_before_switch": slope_delta_E_before_switch(trace, sw_d),
        "mean_T": float(trace["T"].mean()) if "T" in trace.columns and len(trace) else np.nan,
        "mean_delta_E": float(trace["delta_E"].mean()) if "delta_E" in trace.columns and len(trace) else np.nan,
        **latency_summary(trace),
    }
    if trace.attrs.get("latency_sum") is not None:
        out["latency_sum"] = float(trace.attrs["latency_sum"])
    if human_freq:
        out["frequency_KL"] = kl_frequency(human_freq, words)
    from .levy_flight import levy_metrics_for_list

    out.update(
        levy_metrics_for_list(words, trace, word_index, dist)
    )
    return out


def human_per_list_metrics(
    lists_df: pd.DataFrame,
    assets: dict,
    human_freq: Optional[dict[str, float]] = None,
    switch_quantile: float = 0.80,
) -> pd.DataFrame:
    """One row per human list with full summarize_list metrics."""
    per_list = []
    for _, row in lists_df.iterrows():
        trace = pd.DataFrame()
        iris = row.get("iri")
        if iris is not None and isinstance(iris, list):
            vals = pd.to_numeric(pd.Series(iris), errors="coerce").dropna()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                trace = pd.DataFrame({"latency": vals.values})
        per_list.append(
            summarize_list(
                row["words"], trace, assets, human_freq, switch_quantile
            )
        )
    return pd.DataFrame(per_list)


def human_summary_statistics(
    lists_df: pd.DataFrame,
    assets: dict,
    human_freq: dict[str, float],
    switch_quantile: float = 0.80,
) -> dict:
    """Aggregate summary stats across human lists."""
    per_df = human_per_list_metrics(
        lists_df, assets, human_freq, switch_quantile
    )
    agg = per_df.mean(numeric_only=True).to_dict()
    agg["n_lists"] = len(per_df)

    word_index = assets["word_index"]
    dist = assets["distance_matrix"]
    lat_parts: list[float] = []
    sem_parts: list[float] = []
    for _, row in lists_df.iterrows():
        words = row["words"]
        sem_parts.extend(
            adjacent_semantic_distances(words, word_index, dist)
        )
        iris = row.get("iri")
        if iris is not None and isinstance(iris, list):
            vals = pd.to_numeric(pd.Series(iris), errors="coerce").dropna()
            lat_parts.extend(vals[np.isfinite(vals) & (vals > 0)].tolist())

    from .levy_flight import fit_mu_lbn, _empty_mu, _pack_mu

    if len(lat_parts) >= 2:
        fit = fit_mu_lbn(np.array(lat_parts), series_type="latency")
        agg.update(_pack_mu(fit, prefix="mu_latency_pooled"))
    else:
        agg.update(_empty_mu("mu_latency_pooled"))

    from .levy_flight import semantic_tail_subset

    sem = np.array([s for s in sem_parts if np.isfinite(s) and s > 0])
    if len(sem) >= 2:
        fit_s = fit_mu_lbn(sem, min_value=1e-6, series_type="semantic")
        agg.update(_pack_mu(fit_s, prefix="mu_semantic_pooled"))
        tail = semantic_tail_subset(sem, quantile=switch_quantile)
        agg["mu_semantic_tail_quantile"] = float(switch_quantile)
        agg["mu_semantic_tail_n_total"] = float(len(sem))
        if len(tail) >= 6:
            agg["mu_semantic_tail_frac"] = float(len(tail) / len(sem))
            fit_t = fit_mu_lbn(tail, min_value=1e-6, series_type="semantic_tail")
            agg.update(_pack_mu(fit_t, prefix="mu_semantic_tail_pooled"))
        else:
            agg["mu_semantic_tail_frac"] = (
                float(len(tail) / len(sem)) if len(sem) else np.nan
            )
            agg.update(_empty_mu("mu_semantic_tail_pooled"))
    else:
        agg.update(_empty_mu("mu_semantic_pooled"))
        agg.update(_empty_mu("mu_semantic_tail_pooled"))

    return agg


def distance_to_human_summary(
    model_stats: dict,
    human_stats: dict,
    weights: dict[str, float],
    human_sd: Optional[dict[str, float]] = None,
) -> float:
    total = 0.0
    for key, w in weights.items():
        if key not in human_stats or key not in model_stats:
            continue
        mv = model_stats[key]
        hv = human_stats[key]
        if not np.isfinite(mv) or not np.isfinite(hv):
            continue
        sd = (human_sd or {}).get(key, 1.0) or 1.0
        diff = (mv - hv) / sd
        total += w * diff**2
    return float(total)


# --- Distributional comparison (human heterogeneity) ---

DISTRIBUTION_METRICS = [
    "list_length",
    "switch_rate_distance",
    "mean_adjacent_distance",
    "bigram_relevance",
    "mu_latency_pooled",
]

# Local semantic coherence / trajectory naturalness (06 / calibration macro fit).
LOCAL_TRANSITION_EVAL_WEIGHTS: dict[str, float] = {
    "mean_adjacent_distance": 1.0,
    "switch_rate_distance": 1.5,
    "bigram_relevance": 1.0,
    "mu_latency_pooled": 1.0,
}

# Depletion-timing targets (replay / trace summaries). Often NaN in list-level
# human_stats — use timing_distance + stepwise K1 for primary process tests.
DEPLETION_TIMING_EVAL_WEIGHTS: dict[str, float] = {
    "delta_E_before_switch": 1.5,
    "slope_delta_E_before_switch": 1.0,
    "switch_rate_distance": 1.0,
}

# Legacy union (backward compatible with 06_evaluate_models / 14_calibrate).
MECHANISM_EVAL_WEIGHTS: dict[str, float] = {
    **LOCAL_TRANSITION_EVAL_WEIGHTS,
    **DEPLETION_TIMING_EVAL_WEIGHTS,
}

PRODUCTIVITY_EVAL_WEIGHTS: dict[str, float] = {
    "list_length": 1.0,
}


def distribution_stats(series: pd.Series) -> dict[str, float]:
    """Mean, SD, and quantiles for one per-list metric."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n": 0.0,
        }
    return {
        "mean": float(x.mean()),
        "sd": float(x.std()) if len(x) > 1 else 0.0,
        "p10": float(x.quantile(0.10)),
        "p50": float(x.quantile(0.50)),
        "p90": float(x.quantile(0.90)),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": float(len(x)),
    }


def human_distribution_bands(
    per_list_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Per-metric human distribution (for p10–p90 coverage checks)."""
    metrics = metrics or DISTRIBUTION_METRICS
    bands: dict[str, dict[str, float]] = {}
    for m in metrics:
        if m not in per_list_df.columns:
            continue
        bands[m] = distribution_stats(per_list_df[m])
    return bands


def augment_stats_with_distribution(
    mean_stats: dict,
    per_list_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> dict:
    """Add {metric}_sd, {metric}_p10, … to aggregate stats (mean already in mean_stats)."""
    metrics = metrics or DISTRIBUTION_METRICS
    out = dict(mean_stats)
    for m in metrics:
        if m not in per_list_df.columns:
            continue
        d = distribution_stats(per_list_df[m])
        for suffix, val in d.items():
            if suffix == "mean":
                continue
            out[f"{m}_{suffix}"] = val
    return out


def coverage_in_human_band(
    per_list_df: pd.DataFrame,
    bands: dict[str, dict[str, float]],
    metrics: Optional[list[str]] = None,
) -> dict[str, float]:
    """Fraction of lists falling within human [p10, p90] for each metric."""
    metrics = metrics or DISTRIBUTION_METRICS
    out: dict[str, float] = {}
    for m in metrics:
        if m not in per_list_df.columns or m not in bands:
            continue
        lo, hi = bands[m].get("p10"), bands[m].get("p90")
        if not np.isfinite(lo) or not np.isfinite(hi):
            out[f"{m}_in_human_p10_p90"] = np.nan
            continue
        x = pd.to_numeric(per_list_df[m], errors="coerce")
        out[f"{m}_in_human_p10_p90"] = float(((x >= lo) & (x <= hi)).mean())
    return out


def human_distribution_reference_table(
    bands: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Long table: one row per metric with human distribution stats."""
    rows = []
    for metric, d in bands.items():
        rows.append({"metric": metric, **d})
    return pd.DataFrame(rows)
