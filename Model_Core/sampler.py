"""Main list-generation sampler (M3–M4b) and acceptance.

Design notes
------------
Target distribution
    pi(w) ∝ exp(-E_t(w) / T)
    where E_t(w) = z[-log P_LLM(w|h_t)] + eta * z[d(e(w), e(w_t))^2]
    is the *full* energy function (LLM availability + semantic distance cost).

Acceptance criterion
    Standard Metropolis–Hastings on the full energy:
        log alpha = -(E(w') - E(w)) / T + log q(w|w') - log q(w'|w)
    Energy enters both the temperature controller (via delta_E → T_t)
    AND the acceptance step (via E_t directly).  This ensures eta is
    identifiable from behaviour data and the model is internally
    consistent with the design document.

Temperature role
    T_t is resolved ONCE per step from (delta_E, H, R_fail) before the
    inner proposal loop.  This implements step-level temperature control:
    T is fixed during the retry loop for a given emission step, and
    only updated between emitted words.  R_fail accumulates across all
    failed attempts within the current step and is reset to 0 after
    each successful emission.

Proposal asymmetry
    The full MH ratio includes log q_reverse - log q_forward.
    proposal_reverse_prob() must use identical parameters (sigma_q,
    knn_indices, log_scores, p_g, ptype) to sample_proposal() to avoid
    systematic bias.

Censoring
    Previously generated words are excluded from emission but remain in
    the LLM context (h_t) and embedding space, shaping the energy
    landscape and proposal distribution.  Two failure types are tracked
    separately:
        pass_repeat    – candidate is in generated set (censoring)
        pass_mh_reject – candidate was proposed but MH rejected it
    pass_forbidden_category is also tracked for diagnostic purposes.

Category filtering
    allowed_categories is passed via ModelParams (or CategoryAssets) to
    avoid hardcoding task-specific labels in this module.  Falls back to
    allowed_words if provided, or no filtering if neither is set.

Edge cases
    - Empty local neighbourhood (avail = []):
        delta_E is capped at DELTA_E_MAX for numerical stability.
        H and N_eff are set to 0.0 explicitly.
    - T = 0 is guarded against; T is clipped to T_FLOOR before use.
    - t=0 trace row includes NaN sentinels for fields that are undefined
      at initialisation, so downstream column operations do not silently
      drop the row.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import ModelParams
from .energy import (
    compute_energy_for_candidates,
    compute_energy_gap,
    compute_local_entropy,
    temperature_from_energy_gap,
    temperature_full,
)
from .latency_budget import LatencyTracker, inner_search_limit, tracker_from_params
from .proposal import (
    p_global_from_temperature,
    proposal_reverse_prob,
    sample_proposal,
)

# ---------------------------------------------------------------------------
# Numerical safety constants
# ---------------------------------------------------------------------------

T_FLOOR: float = 1e-8
"""Minimum temperature to prevent division by zero in acceptance."""

DELTA_E_MAX: float = 50.0
"""Cap on delta_E fed to the temperature controller (avoids sigmoid saturation
from inf when the local neighbourhood is fully exhausted)."""


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def accept_candidate(
    E_current: float,
    E_candidate: float,
    T: float,
    q_forward: float,
    q_reverse: float,
    rng: np.random.Generator,
) -> bool:
    """Metropolis–Hastings acceptance on the full energy function.

    alpha = min(1, exp(-(E(w') - E(w)) / T) * q(w|w') / q(w'|w))

    Both the LLM availability term and the semantic distance term of E_t
    are included, so eta is identifiable and the acceptance step is
    consistent with the energy landscape used for depletion detection.

    Parameters
    ----------
    E_current   : E_t(w_t)  – energy of the current word
    E_candidate : E_t(w')   – energy of the proposed word
    T           : temperature (must be > 0; caller should clip to T_FLOOR)
    q_forward   : proposal probability q(w' | w_t)
    q_reverse   : proposal probability q(w_t | w')
    rng         : numpy random generator
    """
    T = max(T, T_FLOOR)
    log_alpha = -(E_candidate - E_current) / T
    log_alpha += math.log(q_reverse + 1e-300) - math.log(q_forward + 1e-300)
    u = math.log(rng.random() + 1e-300)
    return u < min(0.0, log_alpha)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_first_word(
    rng: np.random.Generator,
    log_scores: np.ndarray,
    generated: set[int],
    allowed_indices: Optional[set[int]] = None,
) -> int:
    """Sample the first word proportional to the LLM global prior.

    Uses the context-free log_scores (prompt-only LLM distribution) so
    the first word reflects baseline lexical availability, not position
    in semantic space (no current_word yet → distance term undefined).
    """
    avail = [
        i for i in range(len(log_scores))
        if i not in generated and (allowed_indices is None or i in allowed_indices)
    ]
    if not avail:
        raise RuntimeError("No available initial word candidates")
    logits = log_scores[avail]
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(avail, p=probs))


# ---------------------------------------------------------------------------
# Temperature and p_global resolution
# ---------------------------------------------------------------------------

def resolve_temperature(
    params: ModelParams,
    delta_E: float,
    H: float,
    R: float,
) -> float:
    """Map depletion signals to temperature for the current step.

    Temperature is resolved once per emission step (step-level control).
    It does not update during the inner retry loop; changes in R_fail
    within a step are only reflected at the *next* step.

    delta_E is capped at DELTA_E_MAX before being passed to the sigmoid
    controller to prevent numerical saturation when the local
    neighbourhood is fully exhausted (avail = []).
    """
    delta_E = min(delta_E, DELTA_E_MAX)
    mid = params.model_id
    if mid == "M3_fixed_T":
        T = params.fixed_T
    elif mid in ("M4_adaptive_T", "M4b_adaptive_T_global"):
        T = temperature_from_energy_gap(
            delta_E, params.T_min, params.T_max, params.a, params.b
        )
    else:
        # Default: energy-gap adaptive (same as M4)
        T = temperature_from_energy_gap(
            delta_E, params.T_min, params.T_max, params.a, params.b
        )
    return max(T, T_FLOOR)


def resolve_p_global(params: ModelParams, T: float) -> float:
    """Return the global-proposal mixing weight for this step."""
    if params.model_id == "M4b_adaptive_T_global":
        return p_global_from_temperature(T, params.gamma0, params.gamma1)
    return params.p_global


# ---------------------------------------------------------------------------
# Category assets
# ---------------------------------------------------------------------------

class CategoryAssets:
    """Preloaded tensors for one task category.

    Parameters
    ----------
    words          : vocabulary list (index order defines all arrays)
    word_index     : word → integer index mapping
    neg_logprobs   : -log P_LLM(w | prompt) for each word  [shape: (n,)]
    log_scores     : log P_LLM(w | prompt)  for each word  [shape: (n,)]
                     NOTE: context-free (V1).  Upgrade to history-dependent
                     scores in V2; mu_p / sd_p must be re-estimated then.
    distance_sq    : pairwise squared embedding distances   [shape: (n, n)]
    knn_indices    : k-nearest neighbours for each word     [shape: (n, k)]
    mu_p, sd_p     : standardisation stats for neg_logprob  (from vocab distribution)
    mu_d, sd_d     : standardisation stats for distance_sq  (from vocab pair distribution)
    categories     : optional semantic category label per word [shape: (n,)]
                     Used for allowed_idx filtering; pass via ModelParams.allowed_categories
                     to avoid hardcoding labels in the sampler.
    """

    def __init__(
        self,
        words: list[str],
        word_index: dict[str, int],
        neg_logprobs: np.ndarray,
        log_scores: np.ndarray,
        distance_sq: np.ndarray,
        knn_indices: np.ndarray,
        mu_p: float,
        sd_p: float,
        mu_d: float,
        sd_d: float,
        categories: Optional[np.ndarray] = None,
    ):
        self.words = words
        self.word_index = word_index
        self.neg_logprobs = neg_logprobs
        self.log_scores = log_scores
        self.distance_sq = distance_sq
        self.knn_indices = knn_indices
        self.mu_p = mu_p
        self.sd_p = sd_p
        self.mu_d = mu_d
        self.sd_d = sd_d
        self.categories = categories


def build_assets(
    words: list[str],
    llm_df,
    emb_cache: dict,
    k_neighbors: int = 20,
) -> CategoryAssets:
    """Construct CategoryAssets from raw LLM scores and an embedding cache."""
    from .embedding import build_knn, squared_distance_matrix

    word_index = {w: i for i, w in enumerate(words)}
    nlp = np.zeros(len(words))
    ls = np.zeros(len(words))
    llm_map = {r["word"]: r for _, r in llm_df.iterrows()}
    for w, i in word_index.items():
        row = llm_map.get(w)
        if row is None:
            nlp[i] = 10.0
            ls[i] = -10.0
        else:
            nlp[i] = float(row["neg_logprob"])
            ls[i] = float(row["logprob"])

    old_index = emb_cache["word_index"]
    n = len(words)
    embeddings = np.zeros((n, emb_cache["embeddings"].shape[1]))
    categories = np.array(["unknown"] * n, dtype=object)
    cache_categories = emb_cache.get("categories")
    for i, w in enumerate(words):
        oi = old_index.get(w)
        if oi is not None:
            embeddings[i] = emb_cache["embeddings"][oi]
            if cache_categories is not None:
                categories[i] = str(cache_categories[oi])

    dist_sq = squared_distance_matrix(embeddings)
    dist = np.sqrt(dist_sq)
    knn_indices, _ = build_knn(dist, k_neighbors)

    mu_p = float(llm_df["neg_logprob"].mean())
    sd_p = float(llm_df["neg_logprob"].std() + 1e-12)
    stats = emb_cache.get("stats", {})
    iu = np.triu_indices(n, k=1)
    pair_vals = dist_sq[iu] if n > 1 else np.array([0.0])
    mu_d = float(stats.get("mu_d", np.mean(pair_vals)))
    sd_d = float(stats.get("sd_d", np.std(pair_vals) + 1e-12))

    return CategoryAssets(
        words=words,
        word_index=word_index,
        neg_logprobs=nlp,
        log_scores=ls,
        distance_sq=dist_sq,
        knn_indices=knn_indices,
        mu_p=mu_p,
        sd_p=sd_p,
        mu_d=mu_d,
        sd_d=sd_d,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# Allowed-index resolution  (replaces hardcoded animal_categories)
# ---------------------------------------------------------------------------

def _resolve_allowed_idx(
    assets: CategoryAssets,
    params: ModelParams,
    allowed_words: Optional[set[str]],
) -> Optional[set[int]]:
    """Build the set of permitted candidate indices.

    Priority:
      1. params.allowed_categories  – set of category label strings from ModelParams
      2. allowed_words              – explicit word-level allowlist
      3. None                       – no filtering

    Returning None means all words are permitted.  An empty set after
    filtering is treated as None (no filtering) to avoid blocking all
    proposals, with a warning logged.
    """
    allowed_idx: Optional[set[int]] = None

    if getattr(params, "allowed_categories", None) and assets.categories is not None:
        allowed_idx = {
            i for i, cat in enumerate(assets.categories)
            if str(cat) in params.allowed_categories
        }
    elif allowed_words is not None:
        allowed_idx = {
            assets.word_index[w] for w in allowed_words if w in assets.word_index
        }

    if allowed_idx is not None and len(allowed_idx) == 0:
        import warnings
        warnings.warn(
            "allowed_idx resolved to empty set; disabling category filtering "
            "to avoid blocking all proposals.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return allowed_idx


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

_NAN = float("nan")

_INIT_TRACE_SENTINELS: dict[str, Any] = {
    "delta_E": _NAN,
    "H": _NAN,
    "N_eff": _NAN,
    "T": _NAN,
    "R": 0,
    "n_pass_before": 0,
    "pass_repeat": 0,
    "pass_mh_reject": 0,
    "pass_forbidden_category": 0,
    "proposal_type": "init",
    "E": _NAN,       # filled below with actual log_score of first word
    "E_current": _NAN,
}
"""NaN sentinels for the t=0 row so downstream column operations do not
silently skip the initialisation step."""


def _finalize_trace(
    trace_rows: list[dict],
    stop_reason: str,
    time: LatencyTracker,
) -> pd.DataFrame:
    """Assign stop_reason to the final row and build the DataFrame.

    stop_reason is written only to the last row and to trace.attrs.
    Earlier rows retain their in-loop value of 'running' to avoid
    misleadingly attributing the terminal condition to earlier steps.
    """
    if trace_rows:
        trace_rows[-1]["stop_reason"] = stop_reason
    trace = pd.DataFrame(trace_rows)
    if not trace.empty:
        trace.attrs["stop_reason"] = stop_reason
        trace.attrs["latency_sum"] = time.latency_sum
    return trace


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def generate_list(
    assets: CategoryAssets,
    params: ModelParams,
    list_id: str = "sim_001",
    subject_sim: str = "SIM",
    rng: Optional[np.random.Generator] = None,
    allowed_words: Optional[set[str]] = None,
) -> tuple[list[str], pd.DataFrame]:
    """Generate one semantic fluency list with full trace logging.

    Returns
    -------
    generated_words : list of emitted word strings (including the first word)
    trace           : DataFrame with one row per emitted word (t=0 … T)

    Trace columns
    -------------
    subject_sim, list_id, t, word, current_word
    delta_E, H, N_eff, T, R
    n_pass_before      – number of failed proposals before acceptance
    pass_repeat        – how many candidates were censored (in generated set)
    pass_mh_reject     – how many candidates failed MH acceptance
    pass_forbidden_category – how many candidates were outside allowed_idx
    proposal_type      – 'local' | 'global' | 'init'
    E, E_current       – full energy of accepted and current word
    accepted           – always True (only accepted steps are logged)
    model_id, stop_reason
    latency_* fields from LatencyTracker
    """
    if rng is None:
        rng = np.random.default_rng(params.random_seed)

    words = assets.words
    allowed_idx = _resolve_allowed_idx(assets, params, allowed_words)

    generated_words: list[str] = []
    generated_idx: set[int] = set()
    trace_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ t=0
    current_idx = initialize_first_word(
        rng, assets.log_scores, generated_idx, allowed_idx
    )
    current_word = words[current_idx]
    generated_words.append(current_word)
    generated_idx.add(current_idx)

    time = tracker_from_params(params)
    stop_reason = "max_steps"

    first_lat = 1
    if time.would_exceed(first_lat):
        return generated_words, _finalize_trace(trace_rows, "latency_cap", time)

    trace_rows.append(
        {
            "subject_sim": subject_sim,
            "list_id": list_id,
            "t": 0,
            "word": current_word,
            "current_word": None,
            "model_id": params.model_id,
            "accepted": True,
            "stop_reason": "running",
            **_INIT_TRACE_SENTINELS,
            # Override E sentinel with the actual first-word log_score
            # (energy is undefined at t=0 because there is no current_word,
            #  but LLM log_score is available as a partial record)
            "E": float(assets.log_scores[current_idx]),
            **time.commit(first_lat),
        }
    )

    R_fail = 0
    search_limit = inner_search_limit(params)

    # ------------------------------------------------------------------ t≥1
    for t in range(1, params.max_steps):
        if time.exhausted():
            stop_reason = "latency_cap"
            break

        # ---- compute full energy landscape for current position ----------
        E_t = compute_energy_for_candidates(
            current_idx,
            assets.neg_logprobs,
            assets.distance_sq,
            assets.mu_p,
            assets.sd_p,
            assets.mu_d,
            assets.sd_d,
            params.eta,
        )

        # ---- depletion signals ------------------------------------------
        local_indices = assets.knn_indices[current_idx].tolist()
        gap = compute_energy_gap(E_t, local_indices, generated_idx)

        avail = [i for i in local_indices if i not in generated_idx]
        if avail:
            H, N_eff = compute_local_entropy(E_t, avail)
        else:
            # Local neighbourhood fully exhausted: depletion is maximal.
            # delta_E will be capped at DELTA_E_MAX inside resolve_temperature.
            H, N_eff = 0.0, 0.0

        # ---- step-level temperature (fixed for this emission step) ------
        T = resolve_temperature(params, gap.delta_E, H, R_fail)
        p_g = resolve_p_global(params, T)

        # ---- inner proposal / acceptance loop ---------------------------
        accepted = False
        pass_count = 0          # total attempts this step
        pass_repeat = 0         # censored (already generated)
        pass_mh_reject = 0      # MH rejected
        pass_forbidden = 0      # outside allowed_idx

        while pass_count < search_limit:
            if time.would_exceed(pass_count + 1):
                stop_reason = "latency_cap"
                break
            # -- propose --------------------------------------------------
            try:
                cand_idx, ptype, q_fwd = sample_proposal(
                    rng,
                    current_idx,
                    p_g,
                    assets.knn_indices,
                    assets.distance_sq,
                    assets.log_scores,
                    params.sigma_q,
                    set(),           # exclusion set handled explicitly below
                )
            except RuntimeError:
                # No valid proposal could be drawn (e.g. all neighbours
                # exhausted inside sample_proposal); give up this step.
                R_fail += 1
                stop_reason = "no_proposal"
                break

            # -- category filter ------------------------------------------
            if allowed_idx is not None and cand_idx not in allowed_idx:
                pass_forbidden += 1
                pass_count += 1
                R_fail += 1
                continue

            # -- censoring ------------------------------------------------
            if cand_idx in generated_idx:
                pass_repeat += 1
                pass_count += 1
                R_fail += 1
                continue

            # -- MH acceptance on full energy E_t -------------------------
            E_cur = float(E_t[current_idx])
            E_cand = float(E_t[cand_idx])

            # q_reverse must use identical parameters to sample_proposal
            # (same sigma_q, knn_indices, log_scores, p_g, ptype) to ensure
            # the MH ratio is correct.
            q_rev = proposal_reverse_prob(
                cand_idx,
                current_idx,
                p_g,
                assets.knn_indices,
                assets.distance_sq,
                assets.log_scores,
                params.sigma_q,
                ptype,
            )

            if accept_candidate(E_cur, E_cand, T, q_fwd, q_rev, rng):
                word_latency = pass_count + 1
                if time.would_exceed(word_latency):
                    stop_reason = "latency_cap"
                    break

                cand_word = words[cand_idx]
                trace_rows.append(
                    {
                        "subject_sim": subject_sim,
                        "list_id": list_id,
                        "t": t,
                        "word": cand_word,
                        "current_word": current_word,
                        "delta_E": gap.delta_E,
                        "H": H,
                        "N_eff": N_eff,
                        "T": T,
                        "R": R_fail,
                        "n_pass_before": pass_count,
                        "pass_repeat": pass_repeat,
                        "pass_mh_reject": pass_mh_reject,
                        "pass_forbidden_category": pass_forbidden,
                        "proposal_type": ptype,
                        "E": E_cand,
                        "E_current": E_cur,
                        "accepted": True,
                        "model_id": params.model_id,
                        "stop_reason": "running",
                        **time.commit(word_latency),
                    }
                )
                current_word = cand_word
                current_idx = cand_idx
                generated_words.append(cand_word)
                generated_idx.add(cand_idx)
                R_fail = 0
                accepted = True
                break

            # -- MH rejected ----------------------------------------------
            pass_mh_reject += 1
            pass_count += 1
            R_fail += 1

        # ---- end of inner loop ------------------------------------------
        if not accepted:
            if stop_reason not in ("latency_cap", "no_proposal"):
                stop_reason = "search_limit"
            break

    return generated_words, _finalize_trace(trace_rows, stop_reason, time)
