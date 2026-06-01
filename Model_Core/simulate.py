"""Batch simulation across models and parameter sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .baselines import generate_for_model
from .config import ModelParams, ProjectPaths
from .metrics import summarize_list
from .sampler import build_assets
from .vocab_builder import load_vocab


def load_category_bundle(
    paths: ProjectPaths,
    category: str,
    params: Optional[ModelParams] = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Load from exported data/ or directly from social probe assets."""
    if params and params.use_probe_assets and params.probe_json and params.embedding_cache:
        from .probe_assets import load_probe_bundle

        bundle = load_probe_bundle(
            Path(params.probe_json),
            Path(params.embedding_cache),
            category=category,
            max_candidates=params.max_candidates,
            k_neighbors=params.k_neighbors,
            category_map_path=Path(params.category_map) if params.category_map else None,
        )
        emb = {
            "words": bundle["words"],
            "word_index": bundle["word_index"],
            "embeddings": bundle["embeddings"],
            "categories": bundle.get("categories"),
            "distance_matrix": bundle["distance_matrix"],
            "distance_sq": bundle["distance_sq"],
            "knn_indices": bundle["knn_indices"],
            "knn_distances": bundle["knn_distances"],
            "stats": bundle["stats"],
        }
        return bundle["words"], bundle["llm_df"], emb

    from .config import require_standalone_data

    require_standalone_data(paths, category)
    vocab = load_vocab(paths.vocab / f"{category}_vocab.json")
    llm_df = pd.read_parquet(paths.llm_scores / f"{category}_prompt_scores.parquet")
    from .embedding import load_embedding_cache

    emb = load_embedding_cache(category, paths.embeddings)
    emb["word_index"] = {w: i for i, w in enumerate(vocab)}
    emb["words"] = vocab
    # Prefer the precomputed full distance matrix when present; otherwise derive it
    # from the squared-distance cache (distance = sqrt(distance_sq)). The minimal
    # release ships only distance_sq.npy to stay under platform file-size limits.
    dm_path = paths.embeddings / f"{category}_distance_matrix.npy"
    if dm_path.exists():
        emb["distance_matrix"] = np.load(dm_path)
    else:
        emb["distance_matrix"] = np.sqrt(emb["distance_sq"])
    categories_path = paths.vocab / f"{category}_categories.json"
    if categories_path.exists():
        with open(categories_path) as f:
            cat_map = json.load(f)
        emb["categories"] = np.array([cat_map.get(w, "unknown") for w in vocab], dtype=object)
    return vocab, llm_df, emb


def run_simulations(
    paths: ProjectPaths,
    params: ModelParams,
    n_lists: int = 100,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    out_dir = out_dir or paths.simulations
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab, llm_df, emb = load_category_bundle(paths, params.category, params=params)
    assets = build_assets(vocab, llm_df, emb, k_neighbors=params.k_neighbors)
    asset_dict = {
        "word_index": assets.word_index,
        "distance_matrix": np.sqrt(assets.distance_sq),
        "embeddings": emb["embeddings"],
    }

    all_traces = []
    summaries = []
    base_seed = params.random_seed or 0

    for i in range(n_lists):
        rng = np.random.default_rng(base_seed + i)
        p = ModelParams(**{**params.__dict__})
        words, trace = generate_for_model(
            assets,
            p,
            list_id=f"{params.model_id}_{i:05d}",
            subject_sim="SIM",
            rng=rng,
        )
        if not trace.empty:
            trace["list_idx"] = i
            all_traces.append(trace)
        summaries.append(
            {
                "list_idx": i,
                "model_id": params.model_id,
                **summarize_list(
                    words,
                    trace,
                    asset_dict,
                    human_freq=None,
                    switch_quantile=params.switch_quantile,
                ),
            }
        )

    trace_df = pd.concat(all_traces, ignore_index=True) if all_traces else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    if not trace_df.empty:
        from .levy_flight import levy_metrics_pooled

        pooled = levy_metrics_pooled(
            trace_df,
            asset_dict["word_index"],
            asset_dict["distance_matrix"],
            tail_quantile=params.switch_quantile,
        )
        for key, val in pooled.items():
            summary_df[key] = val

    prefix = f"{params.category}_{params.model_id}"
    trace_path = out_dir / f"{prefix}_trace.parquet"
    summary_path = out_dir / f"{prefix}_summary.parquet"
    trace_df.to_parquet(trace_path, index=False)
    summary_df.to_parquet(summary_path, index=False)

    meta = {"n_lists": n_lists, "params": params.__dict__}
    with open(out_dir / f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return summary_df
