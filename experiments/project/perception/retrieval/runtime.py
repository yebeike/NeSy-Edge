"""Runtime retrievers that connect project-owned bundles to the parser."""

from __future__ import annotations

from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.retrieval.probe_runner import (
    QueryEmbeddingBackend,
    load_reference_embedding_map,
)
from experiments.project.perception.retrieval.probes import PerceptionProbeHarness
from experiments.project.perception.retrieval.retrieval import (
    EMBEDDING_QWEN_METHOD_ID,
    LEGACY_LEXICAL_METHOD_ID,
)


class LexicalManifestRetriever:
    """Project-owned lexical retriever bound to one manifest bundle."""

    def __init__(
        self,
        manifest_id: str,
        *,
        paths: ProjectPaths | None = None,
    ) -> None:
        self.manifest_id = manifest_id
        self.method_id = LEGACY_LEXICAL_METHOD_ID
        self.harness = PerceptionProbeHarness.from_manifest(manifest_id, paths=paths)

    def __call__(self, query_text: str, dataset_id: str, top_k: int) -> list:
        return self.harness.lexical_candidates(query_text, dataset_id, top_k=top_k)

    def warmup(self, _dataset_id: str) -> None:
        return None


class EmbeddingManifestRetriever:
    """Project-owned embedding retriever bound to one manifest and bundle."""

    def __init__(
        self,
        manifest_id: str,
        bundle_id: str,
        *,
        query_backend: QueryEmbeddingBackend,
        method_id: str = EMBEDDING_QWEN_METHOD_ID,
        paths: ProjectPaths | None = None,
    ) -> None:
        self.manifest_id = manifest_id
        self.bundle_id = bundle_id
        self.method_id = method_id
        self.query_backend = query_backend
        self.harness = PerceptionProbeHarness.from_manifest(manifest_id, paths=paths)
        self.reference_embeddings = load_reference_embedding_map(
            manifest_id,
            bundle_id,
            paths=paths,
        )

    def __call__(self, query_text: str, dataset_id: str, top_k: int) -> list:
        return self.harness.embedding_candidates(
            query_text,
            dataset_id,
            query_embedder=lambda text: self.query_backend.embed_queries([text])[0],
            reference_embeddings=self.reference_embeddings,
            top_k=top_k,
        )

    def warmup(self, _dataset_id: str) -> None:
        if hasattr(self.query_backend, "warmup"):
            self.query_backend.warmup()
