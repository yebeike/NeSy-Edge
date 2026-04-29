"""Evaluation helpers for project-owned perception protocol runs."""

from __future__ import annotations

import re


def normalize_template(text: str) -> str:
    """Normalize template strings before exact-match parsing accuracy checks."""
    if not isinstance(text, str):
        return ""
    value = text.lower().strip()
    value = re.sub(r"<\*?>", " __VAR__ ", value)
    value = re.sub(r"<[^>]+>", " __VAR__ ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def exact_match_hit(predicted_template: str, ground_truth_template: str) -> float:
    """Return `1.0` when normalized predicted and ground-truth templates match."""
    return float(
        normalize_template(predicted_template)
        == normalize_template(ground_truth_template)
    )
