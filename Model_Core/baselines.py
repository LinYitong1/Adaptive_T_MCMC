"""Baseline samplers M0–M2 (non-MH or simplified)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import ModelParams
from .proposal import global_proposal_probs, local_proposal_probs
from .latency_budget import inner_search_limit, tracker_from_params
from .sampler import CategoryAssets, initialize_first_word


def _finalize_trace(trace_rows: list, stop_reason: str, time) -> pd.DataFrame:
    from .sampler import _finalize_trace as fin

    return fin(trace_rows, stop_reason, time)


def generate_frequency_baseline(
    assets: CategoryAssets,
    params: ModelParams,
    list_id: str = "M0_001",
    subject_sim: str = "SIM",
    rng: Optional[np.random.Generator] = None,
    allowed_words: Optional[set[str]] = None,
) -> tuple[list[str], pd.DataFrame]:
    """M0: LLM-prior frequency baseline with post-sample PASS checks."""
    rng = rng or np.random.default_rng(params.random_seed)
    n = len(assets.words)
    animal_categories = {"mammal", "bird", "water", "insect", "reptile"}
    allowed_idx = None
    if assets.categories is not None:
        allowed_idx = {
            i for i, cat in enumerate(assets.categories)
            if str(cat) in animal_categories
        }
    elif allowed_words is not None:
        allowed_idx = {assets.word_index[w] for w in allowed_words if w in assets.word_index}
    if allowed_idx is not None and len(allowed_idx) == 0:
        allowed_idx = None
    probs = np.exp(assets.log_scores - np.max(assets.log_scores))
    probs /= probs.sum()

    words: list[str] = []
    generated_idx: set[int] = set()
    trace_rows = []
    time = tracker_from_params(params)
    stop_reason = "max_steps"
    search_limit = inner_search_limit(params)

    current_idx = initialize_first_word(
        rng, assets.log_scores, generated_idx, allowed_idx
    )
    first_lat = 1
    if time.would_exceed(first_lat):
        return words, _finalize_trace(trace_rows, "latency_cap", time)
    words.append(assets.words[current_idx])
    generated_idx.add(current_idx)
    trace_rows.append(
        {
            "subject_sim": subject_sim,
            "list_id": list_id,
            "t": 0,
            "word": assets.words[current_idx],
            "current_word": None,
            "proposal_type": "frequency",
            "accepted": True,
            "model_id": "M0_frequency",
            **time.commit(first_lat),
        }
    )

    for t in range(1, params.max_steps):
        if time.exhausted():
            stop_reason = "latency_cap"
            break
        accepted = False
        current_word = words[-1]
        pass_count = 0
        while pass_count < search_limit:
            if time.would_exceed(pass_count + 1):
                stop_reason = "latency_cap"
                break
            cand = int(rng.choice(n, p=probs))
            if allowed_idx is not None and cand not in allowed_idx:
                pass_count += 1
                continue
            if cand in generated_idx:
                pass_count += 1
                continue
            word_latency = pass_count + 1
            if time.would_exceed(word_latency):
                stop_reason = "latency_cap"
                break
            words.append(assets.words[cand])
            generated_idx.add(cand)
            trace_rows.append(
                {
                    "subject_sim": subject_sim,
                    "list_id": list_id,
                    "t": t,
                    "word": assets.words[cand],
                    "current_word": current_word,
                    "n_pass_before": pass_count,
                    "proposal_type": "frequency",
                    "accepted": True,
                    "model_id": "M0_frequency",
                    **time.commit(word_latency),
                }
            )
            accepted = True
            break
        if not accepted:
            if stop_reason != "latency_cap":
                stop_reason = "search_limit" if pass_count >= search_limit else "no_proposal"
            break

    return words, _finalize_trace(trace_rows, stop_reason, time)


def generate_crw_baseline(
    assets: CategoryAssets,
    params: ModelParams,
    use_jump: bool = False,
    list_id: str = "M1_001",
    subject_sim: str = "SIM",
    rng: Optional[np.random.Generator] = None,
    allowed_words: Optional[set[str]] = None,
) -> tuple[list[str], pd.DataFrame]:
    """M1/M2: random walk retrieval with post-sample PASS checks."""
    rng = rng or np.random.default_rng(params.random_seed)
    animal_categories = {"mammal", "bird", "water", "insect", "reptile"}
    allowed_idx = None
    if assets.categories is not None:
        allowed_idx = {
            i for i, cat in enumerate(assets.categories)
            if str(cat) in animal_categories
        }
    elif allowed_words is not None:
        allowed_idx = {assets.word_index[w] for w in allowed_words if w in assets.word_index}
    if allowed_idx is not None and len(allowed_idx) == 0:
        allowed_idx = None
    generated_idx: set[int] = set()
    current_idx = initialize_first_word(
        rng, assets.log_scores, generated_idx, allowed_idx
    )
    words = [assets.words[current_idx]]
    generated_idx.add(current_idx)
    trace_rows = []

    p_jump = params.p_jump if use_jump else 0.0
    time = tracker_from_params(params)
    stop_reason = "max_steps"
    search_limit = inner_search_limit(params)
    model_id = "M2_crw_jump" if use_jump else "M1_crw"

    first_lat = 1
    if time.would_exceed(first_lat):
        return words, _finalize_trace(trace_rows, "latency_cap", time)
    trace_rows.append(
        {
            "subject_sim": subject_sim,
            "list_id": list_id,
            "t": 0,
            "word": words[0],
            "current_word": None,
            "proposal_type": "init",
            "accepted": True,
            "model_id": model_id,
            **time.commit(first_lat),
        }
    )

    for t in range(1, params.max_steps):
        if time.exhausted():
            stop_reason = "latency_cap"
            break
        accepted = False
        pass_count = 0
        while pass_count < search_limit:
            if time.would_exceed(pass_count + 1):
                stop_reason = "latency_cap"
                break
            if use_jump and rng.random() < p_jump:
                indices, probs = global_proposal_probs(
                    assets.log_scores, exclude=None
                )
                ptype = "jump"
            else:
                indices, probs = local_proposal_probs(
                    current_idx,
                    assets.knn_indices,
                    assets.distance_sq,
                    params.sigma_q,
                    exclude=None,
                )
                ptype = "local"
            if len(indices) == 0:
                break
            j = int(rng.choice(len(indices), p=probs))
            cand = int(indices[j])
            if allowed_idx is not None and cand not in allowed_idx:
                pass_count += 1
                continue
            if cand in generated_idx:
                pass_count += 1
                continue
            word_latency = pass_count + 1
            if time.would_exceed(word_latency):
                stop_reason = "latency_cap"
                break
            trace_rows.append(
                {
                    "subject_sim": subject_sim,
                    "list_id": list_id,
                    "t": t,
                    "word": assets.words[cand],
                    "current_word": assets.words[current_idx],
                    "n_pass_before": pass_count,
                    "proposal_type": ptype,
                    "accepted": True,
                    "model_id": model_id,
                    **time.commit(word_latency),
                }
            )
            current_idx = cand
            words.append(assets.words[cand])
            generated_idx.add(cand)
            accepted = True
            break
        if not accepted:
            if stop_reason != "latency_cap":
                stop_reason = "search_limit" if pass_count >= search_limit else "no_proposal"
            break

    return words, _finalize_trace(trace_rows, stop_reason, time)


def generate_for_model(
    assets: CategoryAssets,
    params: ModelParams,
    list_id: str = "sim",
    subject_sim: str = "SIM",
    rng: Optional[np.random.Generator] = None,
    allowed_words: Optional[set[str]] = None,
) -> tuple[list[str], pd.DataFrame]:
    """Dispatch to the appropriate generator by model_id."""
    from .sampler import generate_list

    mid = params.model_id
    if mid == "M0_frequency":
        return generate_frequency_baseline(
            assets, params, list_id, subject_sim, rng, allowed_words
        )
    if mid == "M1_crw":
        return generate_crw_baseline(
            assets, params, use_jump=False, list_id=list_id,
            subject_sim=subject_sim, rng=rng, allowed_words=allowed_words
        )
    if mid == "M2_crw_jump":
        return generate_crw_baseline(
            assets, params, use_jump=True, list_id=list_id,
            subject_sim=subject_sim, rng=rng, allowed_words=allowed_words
        )
    return generate_list(
        assets, params, list_id=list_id, subject_sim=subject_sim,
        rng=rng, allowed_words=allowed_words
    )
