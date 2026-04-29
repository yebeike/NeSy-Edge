"""NPZ-backed embedding-matrix storage for perception bundles."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.retrieval.artifacts import embedding_bundle_paths


def save_embedding_matrix(
    manifest_id: str,
    bundle_id: str,
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    paths: ProjectPaths | None = None,
) -> None:
    """Save one dense embedding matrix to the canonical NPZ bundle path."""
    bundle = embedding_bundle_paths(manifest_id, bundle_id, paths=paths)
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    np.savez_compressed(bundle.embeddings_path, embeddings=array)


def load_embedding_matrix(
    manifest_id: str,
    bundle_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> np.ndarray:
    """Load one dense embedding matrix from the canonical NPZ bundle path."""
    bundle = embedding_bundle_paths(manifest_id, bundle_id, paths=paths)
    with np.load(bundle.embeddings_path) as payload:
        if "embeddings" not in payload:
            raise KeyError("Embedding bundle is missing the 'embeddings' array")
        matrix = payload["embeddings"]
    if matrix.ndim != 2:
        raise ValueError("Loaded embedding matrix must be 2-dimensional")
    return matrix
