"""Fallback parser helpers for project-owned perception protocol runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from experiments.project.perception.core.models import RetrievalCandidate


FallbackParser = Callable[
    [str, str, list[RetrievalCandidate]],
    tuple[str, dict[str, Any]],
]


def top_reference_template_fallback(
    _query_text: str,
    _dataset_id: str,
    candidates: list[RetrievalCandidate],
) -> tuple[str, dict[str, Any]]:
    """Use the top retrieved reference template as a lightweight fallback."""
    if not candidates:
        return "UNKNOWN_TEMPLATE", {"fallback_mode": "unknown_template"}
    best = candidates[0]
    return best.template, {
        "fallback_mode": "top_reference_template",
        "fallback_reference_id": best.reference_id,
        "fallback_reference_score": round(best.score, 6),
        "ref_count": len(candidates),
    }
