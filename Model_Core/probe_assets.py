"""Load social-LLM probe_summary.json + probe_embeddings_5k.npz (shared assets)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .embedding import build_knn, cosine_distance_matrix, save_embedding_cache, standardization_stats_from_matrix
from .vocab_builder import save_vocab


def default_probe_paths(repo_root: Optional[Path] = None) -> dict[str, Path]:
    """Default paths into 20260519_github_minimal probe bundle."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2] / "20260519_github_minimal"
    assets = repo_root / "01_probe_assets"
    return {
        "probe_json": assets / "probe_summary.json",
        "embedding_cache": assets / "probe_embeddings_5k.npz",
        "category_map": assets / "category_map.json",
    }


def build_candidates_from_probe_json(
    data: dict,
    max_candidates: int = 5000,
) -> tuple[list[str], np.ndarray, str]:
    """Same aggregation as social_mh_base.build_candidates_from_json."""
    llm_tokens = data.get("all_token_logits")
    source = "all_token_logits"
    if not llm_tokens:
        llm_tokens = data.get("topk_tokens")
        source = "topk_tokens"
    if not llm_tokens:
        raise ValueError("probe JSON missing all_token_logits and topk_tokens")

    aggregated: dict[str, float] = defaultdict(float)
    total_mass = 0.0

    for td in llm_tokens:
        raw = td.get("clean_token", td.get("canon", td.get("clean", td.get("token", ""))))
        if raw is None:
            continue
        clean = str(raw).replace("▁", "").lower().strip()
        prob = td.get("probability", td.get("prob"))
        if prob is None:
            continue
        p = float(prob)
        total_mass += p
        if clean and clean.isalpha() and len(clean) > 1:
            aggregated[clean] += p

    if not aggregated:
        raise ValueError("no valid alphabetic tokens after filtering")

    items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    limited = items[:max_candidates]
    mass_trunc = sum(v for _, v in limited)
    if mass_trunc <= 0:
        raise ValueError("truncated mass is non-positive")

    words = [k for k, _ in limited]
    probs = np.array([v / mass_trunc for _, v in limited], dtype=np.float64)
    return words, probs, f"{source} (aggregated/cleaned)"


def load_probe_embedding_npz(
    cache_npz: Path,
    words: list[str],
    probs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align vectors from social cache to probe-derived word order."""
    npz = np.load(cache_npz, allow_pickle=True)
    cache_tokens = [str(t) for t in npz["tokens"]]
    cache_vectors = np.asarray(npz["vectors"], dtype=np.float64)
    cache_probs = np.asarray(npz["probabilities"], dtype=np.float64)
    cache_categories = (
        [str(c) for c in npz["categories"]]
        if "categories" in npz.files
        else ["unknown"] * len(cache_tokens)
    )

    vec_by_token = {t: cache_vectors[i] for i, t in enumerate(cache_tokens)}
    prob_by_token = {t: float(cache_probs[i]) for i, t in enumerate(cache_tokens)}
    cat_by_token = {t: cache_categories[i] for i, t in enumerate(cache_tokens)}

    embeddings = np.zeros((len(words), cache_vectors.shape[1]), dtype=np.float64)
    aligned_probs = probs.copy()
    aligned_categories = np.array(["unknown"] * len(words), dtype=object)
    missing = []
    for i, w in enumerate(words):
        v = vec_by_token.get(w)
        if v is None:
            missing.append(w)
            embeddings[i] = 0.0
        else:
            embeddings[i] = v
            aligned_categories[i] = cat_by_token.get(w, "unknown")
        if w in prob_by_token:
            aligned_probs[i] = prob_by_token[w]

    if missing:
        raise KeyError(
            f"{len(missing)} tokens missing from embedding cache (first: {missing[:3]}). "
            "Rebuild probe_embeddings_5k.npz or lower max_candidates."
        )

    # Renormalize after any prob realignment
    s = aligned_probs.sum()
    if s > 0:
        aligned_probs = aligned_probs / s

    return embeddings, {
        "words": words,
        "prob_by_token": prob_by_token,
        "categories": aligned_categories,
    }


def probe_to_llm_scores_df(
    words: list[str],
    probs: np.ndarray,
    category: str = "animals",
) -> pd.DataFrame:
    rows = []
    for w, p in zip(words, probs):
        p = max(float(p), 1e-300)
        rows.append(
            {
                "category": category,
                "word": w,
                "prob": p,
                "logprob": math.log(p),
                "neg_logprob": -math.log(p),
            }
        )
    return pd.DataFrame(rows)


def load_probe_bundle(
    probe_json: Path,
    embedding_cache: Path,
    category: str = "animals",
    max_candidates: int = 5000,
    k_neighbors: int = 20,
    extra_words: Optional[list[str]] = None,
    category_map_path: Optional[Path] = None,
    filter_category: Optional[str] = None,
) -> dict[str, Any]:
    """
    Load social-LLM probe lexicon + precomputed MiniLM embeddings.

    Returns dict compatible with build_assets() / simulate.load_category_bundle().
    """
    with open(probe_json, encoding="utf-8") as f:
        data = json.load(f)

    words, probs, source = build_candidates_from_probe_json(data, max_candidates=max_candidates)

    if category_map_path and filter_category:
        with open(category_map_path, encoding="utf-8") as f:
            cat_map = {str(k).lower(): str(v) for k, v in json.load(f).items()}
        keep = [w for w in words if cat_map.get(w, "other") == filter_category]
        if keep:
            idx = [words.index(w) for w in keep]
            words = keep
            probs = probs[idx]
            probs = probs / probs.sum()

    if extra_words:
        existing = set(words)
        for w in extra_words:
            wl = w.strip().lower()
            if wl and wl.isalpha() and wl not in existing:
                words.append(wl)
                existing.add(wl)
                probs = np.append(probs, 1e-8)
        probs = probs / probs.sum()

    embeddings, emb_meta = load_probe_embedding_npz(embedding_cache, words, probs)
    categories = emb_meta["categories"]

    dist = cosine_distance_matrix(embeddings)
    dist_sq = dist ** 2
    mu_d, sd_d = standardization_stats_from_matrix(dist_sq)
    knn_idx, knn_dist = build_knn(dist, k_neighbors)

    llm_df = probe_to_llm_scores_df(words, probs, category=category)
    word_index = {w: i for i, w in enumerate(words)}

    return {
        "words": words,
        "word_index": word_index,
        "llm_df": llm_df,
        "embeddings": embeddings,
        "categories": categories,
        "distance_matrix": dist,
        "distance_sq": dist_sq,
        "knn_indices": knn_idx,
        "knn_distances": knn_dist,
        "stats": {"mu_d": mu_d, "sd_d": sd_d, "candidate_source": source},
        "probe_json": str(probe_json.resolve()),
        "embedding_cache": str(embedding_cache.resolve()),
    }


def export_probe_to_project_data(
    paths,  # ProjectPaths from config
    probe_json: Path,
    embedding_cache: Path,
    category: str = "animals",
    max_candidates: int = 5000,
    k_neighbors: int = 20,
    extra_words: Optional[list[str]] = None,
) -> dict:
    """Write vocab / parquet / embedding cache under semantic_tempering_sft/data/."""
    bundle = load_probe_bundle(
        probe_json,
        embedding_cache,
        category=category,
        max_candidates=max_candidates,
        k_neighbors=k_neighbors,
        extra_words=extra_words,
    )
    words = bundle["words"]

    save_vocab(
        words,
        paths.vocab / f"{category}_vocab.json",
        category,
        meta={
            "source": "probe_assets",
            "probe_json": bundle["probe_json"],
            "embedding_cache": bundle["embedding_cache"],
            "candidate_source": bundle["stats"]["candidate_source"],
            "n_words": len(words),
        },
    )

    llm_path = paths.llm_scores / f"{category}_prompt_scores.parquet"
    bundle["llm_df"].to_parquet(llm_path, index=False)

    save_embedding_cache(
        category,
        words,
        bundle["embeddings"],
        bundle["distance_matrix"],
        bundle["knn_indices"],
        bundle["knn_distances"],
        paths.embeddings,
        bundle["stats"]["mu_d"],
        bundle["stats"]["sd_d"],
    )

    categories_path = paths.vocab / f"{category}_categories.json"
    with open(categories_path, "w") as f:
        json.dump(
            {w: str(c) for w, c in zip(words, bundle["categories"])},
            f,
            indent=2,
        )

    meta_path = paths.data / f"{category}_probe_source.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "probe_json": bundle["probe_json"],
                "embedding_cache": bundle["embedding_cache"],
                "n_words": len(words),
                "categories": str(categories_path.resolve()),
            },
            f,
            indent=2,
        )

    return bundle
