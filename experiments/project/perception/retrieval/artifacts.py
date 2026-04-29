"""Project-owned artifact helpers for the perception layer."""

from __future__ import annotations

from typing import Iterable

from experiments.project.perception.core.models import EmbeddingBundleMetadata, ReferenceRow
from experiments.project.perception.paths import EmbeddingBundlePaths, ProjectPaths, project_paths
from experiments.project.shared.jsonio import read_json, write_json


def embedding_bundle_paths(
    manifest_id: str,
    bundle_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> EmbeddingBundlePaths:
    """Resolve one perception embedding bundle under the canonical artifact tree."""
    resolved_paths = paths or project_paths()
    return resolved_paths.perception_embedding_bundle(manifest_id, bundle_id)


def load_embedding_bundle_metadata(
    manifest_id: str,
    bundle_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> EmbeddingBundleMetadata:
    """Load one perception embedding-bundle metadata document."""
    bundle = embedding_bundle_paths(manifest_id, bundle_id, paths=paths)
    return EmbeddingBundleMetadata.from_dict(read_json(bundle.metadata_path))


def write_embedding_bundle_metadata(
    metadata: EmbeddingBundleMetadata,
    *,
    paths: ProjectPaths | None = None,
) -> EmbeddingBundlePaths:
    """Write metadata for one perception embedding bundle."""
    bundle = embedding_bundle_paths(
        metadata.manifest_id,
        metadata.bundle_id,
        paths=paths,
    )
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    write_json(bundle.metadata_path, metadata.to_dict())
    return bundle


def load_reference_index(
    manifest_id: str,
    bundle_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> list[dict[str, str]]:
    """Load the stable reference-index file for one embedding bundle."""
    bundle = embedding_bundle_paths(manifest_id, bundle_id, paths=paths)
    payload = read_json(bundle.reference_index_path)
    if not isinstance(payload, list):
        raise TypeError("reference_index.json must contain a list payload")
    rows: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("reference_index.json rows must be objects")
        rows.append({str(key): str(value) for key, value in item.items()})
    return rows


def write_reference_index(
    manifest_id: str,
    bundle_id: str,
    reference_rows: Iterable[ReferenceRow],
    *,
    paths: ProjectPaths | None = None,
) -> EmbeddingBundlePaths:
    """Write a stable reference index aligned to embedding row order."""
    bundle = embedding_bundle_paths(manifest_id, bundle_id, paths=paths)
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "reference_id": row.reference_id,
            "case_id": row.case_id,
            "dataset_id": row.dataset_id,
        }
        for row in reference_rows
    ]
    write_json(bundle.reference_index_path, payload)
    return bundle
