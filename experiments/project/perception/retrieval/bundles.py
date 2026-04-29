"""Builders for project-owned perception embedding bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from experiments.project.perception.core.models import EmbeddingBundleMetadata
from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.retrieval.artifacts import (
    write_embedding_bundle_metadata,
    write_reference_index,
)
from experiments.project.perception.retrieval.embedding_store import save_embedding_matrix
from experiments.project.perception.retrieval.probes import PerceptionProbeHarness


class ReferenceEmbeddingBackend(Protocol):
    """Minimal backend protocol required for bundle building."""

    def embed_references(self, references: list[str]) -> list[list[float]]:
        """Encode reference texts into dense numeric vectors."""


@dataclass(frozen=True)
class BuiltEmbeddingBundle:
    """Result summary for one built perception embedding bundle."""

    manifest_id: str
    bundle_id: str
    dataset_ids: list[str]
    reference_count: int
    embedding_dimension: int
    embedding_model_id: str


def build_embedding_bundle(
    manifest_id: str,
    bundle_id: str,
    *,
    backend: ReferenceEmbeddingBackend,
    embedding_model_id: str,
    preprocessing_version: str = "bounded_embedding_v1",
    paths: ProjectPaths | None = None,
) -> BuiltEmbeddingBundle:
    """Build and persist one perception reference-embedding bundle."""
    harness = PerceptionProbeHarness.from_manifest(manifest_id, paths=paths)
    probe_rows = harness.probe_reference_texts()
    matrix = backend.embed_references([item.text for item in probe_rows])
    if not matrix:
        raise ValueError("Embedding backend returned an empty reference matrix")

    dataset_ids = sorted({row.dataset_id for row in harness.reference_rows})
    embedding_dimension = len(matrix[0])
    metadata = EmbeddingBundleMetadata(
        schema_version="perception.embedding_bundle.v1",
        bundle_id=bundle_id,
        manifest_id=manifest_id,
        dataset_id=",".join(dataset_ids),
        reference_count=len(harness.reference_rows),
        preprocessing_version=preprocessing_version,
        embedding_model_id=embedding_model_id,
        embedding_dimension=embedding_dimension,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    write_embedding_bundle_metadata(metadata, paths=paths)
    write_reference_index(
        manifest_id,
        bundle_id,
        harness.reference_rows,
        paths=paths,
    )
    save_embedding_matrix(
        manifest_id,
        bundle_id,
        matrix,
        paths=paths,
    )
    return BuiltEmbeddingBundle(
        manifest_id=manifest_id,
        bundle_id=bundle_id,
        dataset_ids=dataset_ids,
        reference_count=len(harness.reference_rows),
        embedding_dimension=embedding_dimension,
        embedding_model_id=embedding_model_id,
    )
