"""Lévy-flight / heavy-tail metrics (Zhu, Sanborn, & Chater, NeurIPS 2018 style).

Primary metric: power-law exponent mu from latency_turns via logarithmic
binning (LBN) in log-log space on the latency distribution.

Also supports mu on semantic step lengths (embedding flight distances).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class LevyFitResult:
    mu_hat: float
    mu_slope: float
    mu_intercept: float
    mu_r2: float
    mu_n_bins_used: int
    n_points: int
    series_type: str  # latency | semantic_distance


def lbn_histogram(
    x: np.ndarray,
    n_bins: int = 12,
    min_value: float = 1.0,
) -> pd.DataFrame:
    """Logarithmic binning (geometric midpoints), Zhu / study1c style."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= min_value)]
    if len(x) < 2:
        return pd.DataFrame()

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return pd.DataFrame()

    breaks = np.unique(np.exp(np.linspace(np.log(xmin), np.log(xmax), n_bins + 1)))
    if len(breaks) < 3:
        return pd.DataFrame()

    counts, edges = np.histogram(x, bins=breaks)
    widths = np.diff(edges)
    mids = np.sqrt(edges[:-1] * edges[1:])
    total = counts.sum()
    if total == 0:
        return pd.DataFrame()

    density = counts / (total * widths)
    df = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "width": widths,
            "midpoint": mids,
            "count": counts,
            "density": density,
        }
    )
    return df[
        (df["count"] > 0)
        & np.isfinite(df["midpoint"])
        & (df["midpoint"] > 0)
        & np.isfinite(df["density"])
        & (df["density"] > 0)
    ]


def fit_mu_lbn(
    x: Sequence[float],
    n_bins: int = 12,
    min_points_for_fit: int = 6,
    min_value: float = 1.0,
    series_type: str = "latency",
) -> LevyFitResult:
    """
    Estimate mu from log-log slope of binned density:
      log10(density) ~ log10(midpoint)  =>  mu_hat = -slope
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= min_value)]
    n = len(x)
    empty = LevyFitResult(
        np.nan, np.nan, np.nan, np.nan, 0, n, series_type
    )
    if n < 2:
        return empty

    df = lbn_histogram(x, n_bins=n_bins, min_value=min_value)
    if len(df) < min_points_for_fit:
        return LevyFitResult(
            np.nan, np.nan, np.nan, np.nan, len(df), n, series_type
        )

    lx = np.log10(df["midpoint"].values)
    ly = np.log10(df["density"].values)
    slope, intercept = np.polyfit(lx, ly, 1)
    y_hat = intercept + slope * lx
    ss_res = np.sum((ly - y_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return LevyFitResult(
        mu_hat=float(-slope),
        mu_slope=float(slope),
        mu_intercept=float(intercept),
        mu_r2=float(r2),
        mu_n_bins_used=len(df),
        n_points=n,
        series_type=series_type,
    )


def _clean_latency_array(x: Sequence[float], min_value: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x >= min_value)]


def fit_mu_lbn_variants(
    x: Sequence[float],
    *,
    series_type: str = "latency",
    min_value_raw: float = 1.0,
    n_bins: int = 12,
) -> list[dict]:
    """LBN mu under raw, median-normalized, and upper-quantile tail cuts.

    Intended for human (seconds) vs model (turns) sensitivity: normalization
    and quantile tails put both sides on a more comparable scale-free footing.
    """
    base = _clean_latency_array(x, min_value_raw)
    rows: list[dict] = []

    def _row(variant: str, arr: np.ndarray, note: str) -> None:
        fit = fit_mu_lbn(arr, n_bins=n_bins, min_value=1.0, series_type=series_type)
        rows.append(
            {
                "variant": variant,
                "mu_hat": fit.mu_hat,
                "mu_r2": fit.mu_r2,
                "n_points": fit.n_points,
                "mu_n_bins_used": fit.mu_n_bins_used,
                "note": note,
            }
        )

    _row(
        "A_raw_lbn",
        base,
        f"min_value={min_value_raw} on raw units (current default pipeline)",
    )
    if len(base) < 2:
        return rows

    med = float(np.median(base))
    if med > 0:
        norm = base / med
        _row(
            "B_median_norm_lbn",
            norm,
            f"x / median(x) with median={med:.4g} on pre-cut data; LBN min_value=1",
        )

    for q, label in ((0.50, "C_tail50_lbn"), (0.75, "C_tail75_lbn")):
        thr = float(np.quantile(base, q))
        tail = base[base >= thr - 1e-12]
        _row(
            label,
            tail,
            f"upper {(1 - q) * 100:.0f}% of values after min_value_raw cut; thr={thr:.4g}",
        )

    # Human-oriented: include fast IRI (<1 s) when raw units are seconds
    if min_value_raw >= 1.0:
        loose = _clean_latency_array(x, min_value=0.0)
        loose = loose[loose > 0]
        if len(loose) >= 2:
            _row(
                "A_raw_lbn_min0",
                loose,
                "no lower cutoff except x>0 (sensitivity for human seconds)",
            )

    return rows


def pool_human_iri_seconds(human_lists: pd.DataFrame) -> np.ndarray:
    """Pool per-word IRI (seconds) across all fluency lists."""
    parts: list[np.ndarray] = []
    for _, row in human_lists.iterrows():
        iris = row.get("iri")
        if not isinstance(iris, list):
            continue
        vals = pd.to_numeric(pd.Series(iris), errors="coerce").dropna().to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            parts.append(vals)
    return np.concatenate(parts) if parts else np.array([])


def fit_mu_ccdf(
    x: Sequence[float],
    min_value: float = 1e-12,
    series_type: str = "latency",
) -> LevyFitResult:
    """CCDF power-law fit: mu = -1 / slope(log CCDF vs log x)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > min_value)]
    n = len(x)
    empty = LevyFitResult(np.nan, np.nan, np.nan, np.nan, 0, n, series_type)
    if n < 2:
        return empty

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    ccdf = 1.0 - ranks / (n + 1)
    mask = ccdf < 1.0
    if mask.sum() < 2:
        return empty

    lx = np.log10(ccdf[mask])
    ly = np.log10(x[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    if slope == 0:
        return LevyFitResult(np.nan, np.nan, np.nan, np.nan, 0, n, series_type)

    mu = -1.0 / slope
    y_hat = intercept + slope * lx
    ss_res = np.sum((ly - y_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return LevyFitResult(
        mu_hat=float(mu),
        mu_slope=float(slope),
        mu_intercept=float(intercept),
        mu_r2=float(r2),
        mu_n_bins_used=int(mask.sum()),
        n_points=n,
        series_type=series_type,
    )


def latency_series_from_trace(trace: pd.DataFrame) -> np.ndarray:
    """Per-emission latency (turns), aligned with social LLM latency_turns."""
    if trace.empty:
        return np.array([])
    if "latency" in trace.columns:
        return pd.to_numeric(trace["latency"], errors="coerce").dropna().values
    if "latency_attempts" in trace.columns:
        return pd.to_numeric(trace["latency_attempts"], errors="coerce").dropna().values
    if "n_pass_before" in trace.columns:
        return (pd.to_numeric(trace["n_pass_before"], errors="coerce").dropna() + 1).values
    return np.array([])


def semantic_step_lengths(
    words: Sequence[str],
    word_index: dict[str, int],
    distance_matrix: np.ndarray,
) -> np.ndarray:
    dists = []
    for w1, w2 in zip(words[:-1], words[1:]):
        i, j = word_index.get(w1), word_index.get(w2)
        if i is None or j is None:
            continue
        dists.append(float(distance_matrix[i, j]))
    return np.array(dists, dtype=float)


def collect_pooled_series(
    traces: pd.DataFrame,
    word_index: dict[str, int],
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool latency and semantic steps across all lists in a batch."""
    lat_parts: list[np.ndarray] = []
    sem_parts: list[np.ndarray] = []
    if traces.empty:
        return np.array([]), np.array([])

    if "list_idx" in traces.columns:
        groups = list(traces.groupby("list_idx"))
    else:
        groups = [(0, traces)]
    for _, g in groups:
        lat = latency_series_from_trace(g)
        if len(lat):
            lat_parts.append(lat)
        if "word" in g.columns:
            words = g["word"].tolist()
            if len(words) >= 2:
                steps = semantic_step_lengths(words, word_index, distance_matrix)
                steps = steps[np.isfinite(steps) & (steps > 0)]
                if len(steps):
                    sem_parts.append(steps)

    lat_all = np.concatenate(lat_parts) if lat_parts else np.array([])
    sem_all = np.concatenate(sem_parts) if sem_parts else np.array([])
    return lat_all, sem_all


def semantic_tail_subset(
    steps: np.ndarray,
    quantile: float = 0.80,
    min_points: int = 6,
) -> np.ndarray:
    """Keep large semantic hops (>= global quantile), for heavy-tail mu fits."""
    steps = np.asarray(steps, dtype=float)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if len(steps) < 2:
        return np.array([])
    thr = float(np.quantile(steps, quantile))
    tail = steps[steps >= thr - 1e-12]
    return tail if len(tail) >= min_points else np.array([])


def pooled_latency_mu_metrics(
    lat_all: np.ndarray,
    *,
    n_bins: int = 12,
    raw_min_value: float = 1.0,
    include_tail_robustness: bool = False,
    use_ccdf_fallback: bool = False,
) -> dict[str, float]:
    """Pooled latency exponents: raw (descriptive) + median-normalized (model comparison)."""
    out: dict[str, float] = {}
    base = _clean_latency_array(lat_all, raw_min_value)

    def _fit_pack(arr: np.ndarray, prefix: str) -> None:
        if len(arr) < 2:
            out.update(_empty_mu(prefix))
            if use_ccdf_fallback:
                out.update(_empty_mu(f"{prefix}_ccdf"))
            return
        fit = fit_mu_lbn(arr, n_bins=n_bins, min_value=1.0, series_type="latency")
        out.update(_pack_mu(fit, prefix=prefix))
        if use_ccdf_fallback and not np.isfinite(fit.mu_hat):
            fit_c = fit_mu_ccdf(arr, series_type="latency")
            out.update(_pack_mu(fit_c, prefix=f"{prefix}_ccdf"))

    _fit_pack(base, "mu_latency_raw_pooled")

    if len(base) >= 2 and float(np.median(base)) > 0:
        _fit_pack(base / float(np.median(base)), "mu_latency_median_norm")
    else:
        out.update(_empty_mu("mu_latency_median_norm"))

    if include_tail_robustness and len(base) >= 2:
        for q, prefix in ((0.50, "mu_latency_tail50"), (0.75, "mu_latency_tail75")):
            thr = float(np.quantile(base, q))
            _fit_pack(base[base >= thr - 1e-12], prefix)

    return out


def levy_metrics_pooled(
    traces: pd.DataFrame,
    word_index: dict[str, int],
    distance_matrix: np.ndarray,
    n_bins: int = 12,
    use_ccdf_fallback: bool = True,
    tail_quantile: float = 0.80,
    min_tail_points: int = 6,
    latency_min_value: float = 1.0,
    include_tail_latency_robustness: bool = False,
) -> dict[str, float]:
    """Fit mu once on all lists pooled; semantic tail uses large hops only."""
    lat_all, sem_all = collect_pooled_series(traces, word_index, distance_matrix)
    out: dict[str, float] = {}

    out.update(
        pooled_latency_mu_metrics(
            lat_all,
            n_bins=n_bins,
            raw_min_value=latency_min_value,
            include_tail_robustness=include_tail_latency_robustness,
            use_ccdf_fallback=use_ccdf_fallback,
        )
    )

    if len(sem_all) >= 2:
        fit_s = fit_mu_lbn(
            sem_all, n_bins=n_bins, min_value=1e-6, series_type="semantic"
        )
        out.update(_pack_mu(fit_s, prefix="mu_semantic_pooled"))
        tail = semantic_tail_subset(
            sem_all, quantile=tail_quantile, min_points=min_tail_points
        )
        out["mu_semantic_tail_quantile"] = float(tail_quantile)
        out["mu_semantic_tail_n_total"] = float(len(sem_all))
        if len(tail) >= min_tail_points:
            out["mu_semantic_tail_frac"] = float(len(tail) / len(sem_all))
            fit_t = fit_mu_lbn(
                tail, n_bins=n_bins, min_value=1e-6, series_type="semantic_tail"
            )
            out.update(_pack_mu(fit_t, prefix="mu_semantic_tail_pooled"))
        else:
            out["mu_semantic_tail_frac"] = (
                float(len(tail) / len(sem_all)) if len(sem_all) else np.nan
            )
            out.update(_empty_mu("mu_semantic_tail_pooled"))
    else:
        out.update(_empty_mu("mu_semantic_pooled"))
        out.update(_empty_mu("mu_semantic_tail_pooled"))
        out["mu_semantic_tail_quantile"] = float(tail_quantile)
        out["mu_semantic_tail_n_total"] = 0.0
        out["mu_semantic_tail_frac"] = np.nan
    return out


def levy_metrics_for_list(
    words: Sequence[str],
    trace: pd.DataFrame,
    word_index: Optional[dict[str, int]] = None,
    distance_matrix: Optional[np.ndarray] = None,
    n_bins: int = 12,
    use_ccdf_fallback: bool = True,
) -> dict[str, float]:
    """Sequence-level mu (latency + optional semantic flight distances)."""
    out: dict[str, float] = {}
    lat = latency_series_from_trace(trace)
    if len(lat) >= 2:
        fit = fit_mu_lbn(lat, n_bins=n_bins, series_type="latency")
        out.update(_pack_mu(fit, prefix="mu_latency"))
        if use_ccdf_fallback and not np.isfinite(fit.mu_hat):
            fit_c = fit_mu_ccdf(lat, series_type="latency")
            out.update(_pack_mu(fit_c, prefix="mu_latency_ccdf"))
    else:
        out.update(_empty_mu("mu_latency"))
        out.update(_empty_mu("mu_latency_ccdf"))

    if word_index is not None and distance_matrix is not None and len(words) >= 2:
        steps = semantic_step_lengths(words, word_index, distance_matrix)
        steps = steps[np.isfinite(steps) & (steps > 0)]
        if len(steps) >= 2:
            fit_s = fit_mu_lbn(steps, n_bins=n_bins, min_value=1e-6, series_type="semantic")
            out.update(_pack_mu(fit_s, prefix="mu_semantic"))
        else:
            out.update(_empty_mu("mu_semantic"))
    return out


def _pack_mu(fit: LevyFitResult, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}": fit.mu_hat,
        f"{prefix}_r2": fit.mu_r2,
        f"{prefix}_n": float(fit.n_points),
        f"{prefix}_n_bins": float(fit.mu_n_bins_used),
    }


def _empty_mu(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}": np.nan,
        f"{prefix}_r2": np.nan,
        f"{prefix}_n": 0.0,
        f"{prefix}_n_bins": 0.0,
    }
