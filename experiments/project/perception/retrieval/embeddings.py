"""Embedding-oriented helpers for perception retrieval probes."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from experiments.project.perception.core.models import ReferenceRow, RetrievalCandidate


Vector = Sequence[float]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Compute cosine similarity for two equal-length numeric vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")
    if not left:
        raise ValueError("Embedding vectors must not be empty")

    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def embedding_probe_candidates(
    query_embedding: Vector,
    reference_rows: Iterable[ReferenceRow],
    reference_embeddings: dict[str, Vector],
    *,
    top_k: int = 5,
) -> list[RetrievalCandidate]:
    """Rank reference rows with precomputed embeddings for probe evaluation."""
    candidates: list[RetrievalCandidate] = []
    for row in reference_rows:
        if row.reference_id not in reference_embeddings:
            continue
        score = cosine_similarity(query_embedding, reference_embeddings[row.reference_id])
        candidates.append(
            RetrievalCandidate(
                reference_id=row.reference_id,
                case_id=row.case_id,
                template=row.gt_template,
                score=score,
                matched_text=row.clean_alert,
            )
        )
    candidates.sort(
        key=lambda item: (item.score, item.template, item.reference_id),
        reverse=True,
    )
    return candidates[:top_k]
