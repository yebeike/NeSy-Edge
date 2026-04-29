"""Probe-ready retrieval helpers for the perception layer."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

from experiments.project.perception.core.models import ReferenceRow, RetrievalCandidate


LEGACY_LEXICAL_METHOD_ID = "lexical_l2_legacy"
EMBEDDING_MINI_METHOD_ID = "embedding_l2_mini"
EMBEDDING_QWEN_METHOD_ID = "embedding_l2_qwen"


def fingerprint_text(text: str) -> str:
    """Apply a lightweight dynamic-value collapse used by the lexical probe."""
    return re.sub(r"\d+", "N", text or "")


def token_set(text: str) -> set[str]:
    """Extract a rough token set for the lexical probe."""
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_./:-]*", (text or "").lower()))


def lexical_similarity(query_text: str, reference_text: str) -> float:
    """Legacy lexical probe score used as a debug comparison baseline."""
    query_fp = fingerprint_text(query_text)
    reference_fp = fingerprint_text(reference_text)
    sequence_ratio = SequenceMatcher(None, query_fp, reference_fp).ratio()
    query_tokens = token_set(query_text)
    reference_tokens = token_set(reference_text)
    overlap = len(query_tokens & reference_tokens) / max(
        1, len(query_tokens | reference_tokens)
    )
    return 0.7 * sequence_ratio + 0.3 * overlap


def lexical_probe_candidates(
    query_text: str,
    reference_rows: Iterable[ReferenceRow],
    *,
    top_k: int = 5,
) -> list[RetrievalCandidate]:
    """Rank reference rows with the legacy lexical probe for comparison only."""
    candidates = [
        RetrievalCandidate(
            reference_id=row.reference_id,
            case_id=row.case_id,
            template=row.gt_template,
            score=lexical_similarity(query_text, row.clean_alert),
            matched_text=row.clean_alert,
        )
        for row in reference_rows
    ]
    candidates.sort(
        key=lambda item: (item.score, item.template, item.reference_id),
        reverse=True,
    )
    return candidates[:top_k]
