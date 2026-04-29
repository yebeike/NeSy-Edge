"""Runnable probe harnesses for perception-layer retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from experiments.project.perception.core.models import ReferenceRow, RetrievalCandidate
from experiments.project.perception.core.preprocessing import bounded_embedding_text
from experiments.project.perception.paths import ProjectPaths
from experiments.project.perception.retrieval.embeddings import embedding_probe_candidates
from experiments.project.perception.retrieval.references import load_reference_rows
from experiments.project.perception.retrieval.retrieval import lexical_probe_candidates


QueryEmbedder = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class ProbeReferenceText:
    """One bounded reference text prepared for a perception probe."""

    reference_id: str
    dataset_id: str
    text: str


class PerceptionProbeHarness:
    """Lightweight harness for lexical and embedding retrieval probes."""

    def __init__(self, reference_rows: list[ReferenceRow]) -> None:
        self.reference_rows = list(reference_rows)
        self.reference_texts = {
            row.reference_id: bounded_embedding_text(row.clean_alert, row.dataset_id)
            for row in self.reference_rows
        }

    @classmethod
    def from_manifest(
        cls,
        manifest_id: str,
        *,
        paths: ProjectPaths | None = None,
    ) -> "PerceptionProbeHarness":
        """Build one probe harness from a project-owned manifest."""
        return cls(load_reference_rows(manifest_id, paths=paths))

    def probe_reference_texts(self) -> list[ProbeReferenceText]:
        """Return bounded reference texts in stable row order."""
        return [
            ProbeReferenceText(
                reference_id=row.reference_id,
                dataset_id=row.dataset_id,
                text=self.reference_texts[row.reference_id],
            )
            for row in self.reference_rows
        ]

    def lexical_candidates(
        self,
        query_text: str,
        dataset_id: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:
        """Run the lexical probe against bounded reference texts."""
        bounded_query = bounded_embedding_text(query_text, dataset_id)
        normalized_rows = [
            ReferenceRow(
                reference_id=row.reference_id,
                case_id=row.case_id,
                dataset_id=row.dataset_id,
                raw_log=row.raw_log,
                clean_alert=self.reference_texts[row.reference_id],
                gt_template=row.gt_template,
                metadata=row.metadata,
            )
            for row in self.reference_rows
        ]
        return lexical_probe_candidates(bounded_query, normalized_rows, top_k=top_k)

    def embedding_candidates(
        self,
        query_text: str,
        dataset_id: str,
        *,
        query_embedder: QueryEmbedder,
        reference_embeddings: dict[str, Sequence[float]],
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:
        """Run the embedding probe against bounded reference texts."""
        bounded_query = bounded_embedding_text(query_text, dataset_id)
        query_embedding = query_embedder(bounded_query)
        return embedding_probe_candidates(
            query_embedding,
            self.reference_rows,
            reference_embeddings,
            top_k=top_k,
        )
