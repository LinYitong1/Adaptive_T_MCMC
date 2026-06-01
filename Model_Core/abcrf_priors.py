"""Prior distributions over model-family parameters for ABC / ABC-RF.

Defines the six retrieval mechanisms (M0-M4b) and how their free parameters are
drawn. Budget fields (max_steps, latency_cap) are set by the caller after sampling.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .config import ModelParams

MODELS = [
    "M0_frequency",
    "M1_crw",
    "M2_crw_jump",
    "M3_fixed_T",
    "M4_adaptive_T",
    "M4b_adaptive_T_global",
]


def _tmax_from_tmin(rng: np.random.Generator, t_min: float) -> float:
    return float(rng.uniform(t_min + 0.3, 4.0))


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_prior(
    model_id: str,
    rng: np.random.Generator,
    base: ModelParams,
    *,
    wide_prior: bool = False,
) -> ModelParams:
    """Draw model-specific parameters; budget fields are set outside this function.

    wide_prior: for adaptive models, lower eta's effective floor (log-uniform on
    [0.01, 1.0]) and bias the restart slope gamma1 toward 0, increasing the prior
    density of long, low-restart lists (matches human productivity better).
    """
    p = deepcopy(base)
    p.model_id = model_id  # type: ignore[assignment]
    p.category = "animals"
    p.random_seed = int(rng.integers(0, 1_000_000_000))

    if model_id == "M0_frequency":
        return p

    if model_id == "M1_crw":
        p.sigma_q = float(rng.uniform(0.15, 0.60))
        return p

    if model_id == "M2_crw_jump":
        p.p_jump = float(rng.uniform(0.02, 0.40))
        p.sigma_q = float(rng.uniform(0.15, 0.60))
        return p

    if model_id == "M3_fixed_T":
        p.eta = _loguniform(rng, 0.01, 1.00) if wide_prior else float(rng.uniform(0.05, 1.00))
        p.fixed_T = float(rng.uniform(0.5, 3.5))
        p.sigma_q = float(rng.uniform(0.15, 0.60))
        p.p_global = float(rng.uniform(0.02, 0.50))
        return p

    if model_id in ("M4_adaptive_T", "M4b_adaptive_T_global"):
        p.eta = _loguniform(rng, 0.01, 1.00) if wide_prior else float(rng.uniform(0.05, 1.00))
        p.T_min = float(rng.uniform(0.3, 1.5))
        p.T_max = _tmax_from_tmin(rng, p.T_min)
        p.a = float(rng.uniform(-2.0, 2.0))
        p.b = float(rng.uniform(0.2, 6.0))
        p.sigma_q = float(rng.uniform(0.15, 0.60))
        if model_id == "M4_adaptive_T":
            p.p_global = float(rng.uniform(0.02, 0.50))
        else:
            p.gamma0 = float(rng.uniform(-3.0, 1.0))
            # Bias toward low restart slope (more mass near 0 -> longer lists).
            p.gamma1 = float(3.0 * rng.random() ** 2) if wide_prior else float(rng.uniform(0.0, 3.0))
        return p

    raise ValueError(f"Unsupported model_id: {model_id}")
