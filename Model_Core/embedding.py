"""Embedding computation, distance matrix, and kNN indices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore
    torch = None  # type: ignore


def encode_words(
    words: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers required: pip install sentence-transformers")
    if device is None:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        words,
        convert_to_numpy=True,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(words) > 100,
    )
    return np.asarray(emb, dtype=np.float64)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance on unit-normalized embeddings (= 2 - 2*cos)."""
    # For normalized vectors: ||a-b||^2 = 2 - 2*dot(a,b)
    sim = embeddings @ embeddings.T
    dist_sq = np.clip(2.0 - 2.0 * sim, 0.0, None)
    return np.sqrt(dist_sq)


def squared_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    d = cosine_distance_matrix(embeddings)
    return d ** 2


def build_knn(
    distance_matrix: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = distance_matrix.shape[0]
    k = min(k, n - 1)
    knn_indices = np.zeros((n, k), dtype=np.int64)
    knn_distances = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        order = np.argsort(distance_matrix[i])
        neigh = order[order != i][:k]
        knn_indices[i] = neigh
        knn_distances[i] = distance_matrix[i, neigh]
    return knn_indices, knn_distances


def standardization_stats_from_matrix(dist_sq: np.ndarray) -> tuple[float, float]:
    iu = np.triu_indices(dist_sq.shape[0], k=1)
    vals = dist_sq[iu]
    return float(np.mean(vals)), float(np.std(vals) + 1e-12)


def save_embedding_cache(
    category: str,
    words: list[str],
    embeddings: np.ndarray,
    distance_matrix: np.ndarray,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    out_dir: Path,
    mu_d: float,
    sd_d: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = category
    np.save(out_dir / f"{prefix}_embeddings.npy", embeddings)
    np.save(out_dir / f"{prefix}_distance_matrix.npy", distance_matrix)
    np.save(out_dir / f"{prefix}_distance_sq.npy", distance_matrix ** 2)
    np.save(out_dir / f"{prefix}_knn_indices.npy", knn_indices)
    np.save(out_dir / f"{prefix}_knn_distances.npy", knn_distances)
    word_index = {w: i for i, w in enumerate(words)}
    with open(out_dir / f"{prefix}_word_index.json", "w") as f:
        json.dump(word_index, f, indent=2)
    stats = {"mu_d": mu_d, "sd_d": sd_d, "n_words": len(words)}
    with open(out_dir / f"{prefix}_embedding_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_embedding_cache(category: str, emb_dir: Path) -> dict:
    prefix = category
    words_idx_path = emb_dir / f"{prefix}_word_index.json"
    with open(words_idx_path) as f:
        word_index = json.load(f)
    words = [None] * len(word_index)
    for w, i in word_index.items():
        words[i] = w
    distance_sq = np.load(emb_dir / f"{prefix}_distance_sq.npy")
    # The full distance matrix is derived as sqrt(distance_sq) when the (large) cached
    # matrix is absent; the minimal release ships only distance_sq.npy.
    dm_path = emb_dir / f"{prefix}_distance_matrix.npy"
    distance_matrix = np.load(dm_path) if dm_path.exists() else np.sqrt(distance_sq)
    return {
        "words": words,
        "word_index": word_index,
        "embeddings": np.load(emb_dir / f"{prefix}_embeddings.npy"),
        "distance_matrix": distance_matrix,
        "distance_sq": distance_sq,
        "knn_indices": np.load(emb_dir / f"{prefix}_knn_indices.npy"),
        "knn_distances": np.load(emb_dir / f"{prefix}_knn_distances.npy"),
        "stats": json.loads((emb_dir / f"{prefix}_embedding_stats.json").read_text()),
    }
