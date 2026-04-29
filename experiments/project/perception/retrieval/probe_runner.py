"""Reusable runners for perception lexical and embedding probes."""

from __future__ import annotations

from typing import Protocol

from experiments.project.perception.core.models import RetrievalCandidate
from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.retrieval.artifacts import load_reference_index
from experiments.project.perception.retrieval.embedding_store import load_embedding_matrix
from experiments.project.perception.retrieval.probes import PerceptionProbeHarness


class QueryEmbeddingBackend(Protocol):
    """Minimal backend protocol required for query-side embedding probes."""

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Encode query texts into dense numeric vectors."""


def load_reference_embedding_map(
    manifest_id: str,
    bundle_id: str,
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, list[float]]:
    """Load one embedding map aligned by reference id."""
    index_rows = load_reference_index(manifest_id, bundle_id, paths=paths)
    matrix = load_embedding_matrix(manifest_id, bundle_id, paths=paths)
    return {
        row["reference_id"]: matrix[index].tolist()
        for index, row in enumerate(index_rows)
    }


def run_lexical_probe(
    manifest_id: str,
    *,
    query_text: str,
    dataset_id: str,
    top_k: int = 5,
    paths: ProjectPaths | None = None,
) -> list[RetrievalCandidate]:
    """Run the project-owned lexical perception probe."""
    harness = PerceptionProbeHarness.from_manifest(manifest_id, paths=paths)
    return harness.lexical_candidates(query_text, dataset_id, top_k=top_k)


def run_embedding_probe(
    manifest_id: str,
    bundle_id: str,
    *,
    query_text: str,
    dataset_id: str,
    query_backend: QueryEmbeddingBackend,
    top_k: int = 5,
    paths: ProjectPaths | None = None,
) -> list[RetrievalCandidate]:
    """Run the project-owned embedding perception probe."""
    harness = PerceptionProbeHarness.from_manifest(manifest_id, paths=paths)
    reference_embeddings = load_reference_embedding_map(
        manifest_id,
        bundle_id,
        paths=paths,
    )
    return harness.embedding_candidates(
        query_text,
        dataset_id,
        query_embedder=lambda text: query_backend.embed_queries([text])[0],
        reference_embeddings=reference_embeddings,
        top_k=top_k,
    )
