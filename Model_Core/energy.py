"""Standardized energy, patch depletion gap, and local entropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)


def standardize_scores(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return (values - mean) / (sd + 1e-12)


def compute_energy(
    neg_logprob: float | np.ndarray,
    distance_sq: float | np.ndarray,
    mu_p: float,
    sd_p: float,
    mu_d: float,
    sd_d: float,
    eta: float,
) -> float | np.ndarray:
    z_p = standardize_scores(np.asarray(neg_logprob, dtype=float), mu_p, sd_p)
    z_d = standardize_scores(np.asarray(distance_sq, dtype=float), mu_d, sd_d)
    return z_p + eta * z_d


@dataclass
class EnergyGapResult:
    delta_E: float
    E_min_all: float
    E_min_avail: float


def compute_energy_gap(
    E_t: np.ndarray,
    local_indices: Sequence[int],
    generated_indices: Iterable[int],
) -> EnergyGapResult:
    all_local = np.asarray(local_indices, dtype=int)
    gen_set = set(generated_indices)
    available = [i for i in all_local if i not in gen_set]

    E_min_all = float(np.min(E_t[all_local])) if len(all_local) else float("inf")

    if not available:
        E_min_avail = float("inf")
    else:
        E_min_avail = float(np.min(E_t[available]))

    delta_E = E_min_avail - E_min_all
    if not np.isfinite(delta_E):
        delta_E = 0.0
    delta_E = max(0.0, delta_E)
    return EnergyGapResult(delta_E=delta_E, E_min_all=E_min_all, E_min_avail=E_min_avail)


def compute_local_entropy(
    E_t: np.ndarray,
    available_indices: Sequence[int],
) -> tuple[float, float]:
    if not available_indices:
        return 0.0, 0.0
    logits = -E_t[np.asarray(available_indices, dtype=int)]
    q = softmax(logits)
    H = float(-np.sum(q * np.log(q + 1e-12)))
    N_eff = float(np.exp(H))
    return H, N_eff


def compute_energy_for_candidates(
    current_idx: int,
    neg_logprobs: np.ndarray,
    distance_sq_matrix: np.ndarray,
    mu_p: float,
    sd_p: float,
    mu_d: float,
    sd_d: float,
    eta: float,
) -> np.ndarray:
    dist_sq = distance_sq_matrix[current_idx]
    return compute_energy(neg_logprobs, dist_sq, mu_p, sd_p, mu_d, sd_d, eta)


def temperature_from_energy_gap(
    delta_E: float,
    T_min: float,
    T_max: float,
    a: float,
    b: float,
) -> float:
    s = a + b * delta_E
    return float(T_min + (T_max - T_min) * sigmoid(s))


def temperature_full(
    delta_E: float,
    H: float,
    R: float,
    T_min: float,
    T_max: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    s = a + b * delta_E - c * H + d * R
    return float(T_min + (T_max - T_min) * sigmoid(s))


def discretize_temperature(T_cont: float, ladder: np.ndarray) -> float:
    return float(ladder[np.argmin(np.abs(ladder - T_cont))])


def temperature_level_probs(s: float, ladder: np.ndarray, lam: float = 1.0) -> np.ndarray:
    logits = lam * s * np.arange(1, len(ladder) + 1)
    return softmax(logits)
