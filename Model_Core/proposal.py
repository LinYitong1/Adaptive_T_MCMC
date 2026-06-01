"""Local, global, and mixture proposal distributions."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .energy import softmax, sigmoid


def local_proposal_probs(
    current_idx: int,
    knn_indices: np.ndarray,
    distance_sq_matrix: np.ndarray,
    sigma_q: float,
    exclude: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    neigh = knn_indices[current_idx]
    dist_sq = distance_sq_matrix[current_idx, neigh]
    logits = -dist_sq / (2.0 * sigma_q**2)
    if exclude:
        mask = np.array([i not in exclude for i in neigh], dtype=bool)
        neigh = neigh[mask]
        logits = logits[mask]
    if len(neigh) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    probs = softmax(logits)
    return neigh, probs


def global_proposal_probs(
    log_scores: np.ndarray,
    exclude: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(log_scores))
    if exclude:
        mask = np.array([i not in exclude for i in indices], dtype=bool)
        indices = indices[mask]
        logits = log_scores[mask]
    else:
        logits = log_scores
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    probs = softmax(logits.astype(float))
    return indices, probs


def p_global_from_temperature(T: float, gamma0: float, gamma1: float) -> float:
    return float(sigmoid(gamma0 + gamma1 * T))


def sample_proposal(
    rng: np.random.Generator,
    current_idx: int,
    p_global: float,
    knn_indices: np.ndarray,
    distance_sq_matrix: np.ndarray,
    log_scores: np.ndarray,
    sigma_q: float,
    generated_indices: set[int],
) -> tuple[int, str, float]:
    """Return (candidate_idx, proposal_type, q_forward)."""
    if rng.random() < p_global:
        indices, probs = global_proposal_probs(log_scores, exclude=generated_indices)
        ptype = "global"
    else:
        indices, probs = local_proposal_probs(
            current_idx,
            knn_indices,
            distance_sq_matrix,
            sigma_q,
            exclude=generated_indices,
        )
        ptype = "local"

    if len(indices) == 0:
        indices, probs = global_proposal_probs(log_scores, exclude=generated_indices)
        ptype = "global_fallback"

    if len(indices) == 0:
        raise RuntimeError("No available proposal candidates")

    j = int(rng.choice(len(indices), p=probs))
    return int(indices[j]), ptype, float(probs[j])


def proposal_reverse_prob(
    candidate_idx: int,
    current_idx: int,
    p_global: float,
    knn_indices: np.ndarray,
    distance_sq_matrix: np.ndarray,
    log_scores: np.ndarray,
    sigma_q: float,
    proposal_type: str,
) -> float:
    """Approximate reverse proposal probability (for asymmetric correction)."""
    if proposal_type.startswith("global"):
        _, probs = global_proposal_probs(log_scores)
        idx = np.where(probs > 0)[0]
        # find candidate in global set
        all_idx, all_probs = global_proposal_probs(log_scores)
        match = np.where(all_idx == candidate_idx)[0]
        if len(match) == 0:
            return 1e-12
        q_g = all_probs[match[0]]
        _, local_probs = local_proposal_probs(
            candidate_idx, knn_indices, distance_sq_matrix, sigma_q
        )
        all_l, lp = local_proposal_probs(
            candidate_idx, knn_indices, distance_sq_matrix, sigma_q
        )
        q_l = 0.0
        m2 = np.where(all_l == current_idx)[0]
        if len(m2):
            q_l = lp[m2[0]]
        return (1 - p_global) * q_l + p_global * q_g
    else:
        all_l, lp = local_proposal_probs(
            candidate_idx, knn_indices, distance_sq_matrix, sigma_q
        )
        m = np.where(all_l == current_idx)[0]
        if len(m) == 0:
            return 1e-12
        q_l = lp[m[0]]
        all_g, gp = global_proposal_probs(log_scores)
        m2 = np.where(all_g == current_idx)[0]
        q_g = gp[m2[0]] if len(m2) else 1e-12
        return (1 - p_global) * q_l + p_global * q_g
