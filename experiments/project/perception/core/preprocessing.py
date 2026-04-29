"""Bounded preprocessing helpers for the rebuilt perception layer."""

from __future__ import annotations

import re


_DEFAULT_QUERY_CHAR_BUDGET = {
    "hdfs": 170,
    "openstack": 180,
    "hadoop": 220,
}


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace in a text snippet."""
    return re.sub(r"\s+", " ", (text or "").strip())


def prepare_runtime_alert(text: str, dataset_id: str) -> str:
    """Apply dataset-specific header stripping without semantic remapping."""
    value = normalize_whitespace(text)
    dataset_key = dataset_id.lower()

    if dataset_key == "hdfs":
        match = re.match(
            r"^\d{6}\s+\d{6}\s+\d+\s+(?:INFO|WARN|ERROR)\s+[^:]+:\s*(.*)$",
            value,
        )
        if match:
            return normalize_whitespace(match.group(1))
        return value

    if dataset_key == "hadoop":
        match = re.match(
            r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s+"
            r"(?:INFO|WARN|ERROR)\s+[^:]+:\s*(.*)$",
            value,
        )
        if match:
            return normalize_whitespace(match.group(1))
        match = re.match(r"^[^\]]+\]\s+org\.apache\.[^:]+:\s*(.*)$", value)
        if match:
            return normalize_whitespace(match.group(1))
        return value

    if dataset_key == "openstack":
        match = re.match(
            r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s+\d+\s+"
            r"(?:DEBUG|INFO|WARN|ERROR)\s+[^:]+:\s*(.*)$",
            value,
        )
        if match:
            value = normalize_whitespace(match.group(1))
        match = re.search(r"(\[instance:\s+[^\]]+\].*)$", value)
        if match:
            return normalize_whitespace(match.group(1))
        return value

    return value


def bounded_embedding_text(
    text: str,
    dataset_id: str,
    *,
    char_budget: int | None = None,
) -> str:
    """Prepare bounded embedding input without template-family rewrites."""
    value = prepare_runtime_alert(text, dataset_id)
    dataset_key = dataset_id.lower()

    if dataset_key == "hdfs":
        value = re.sub(r"\bblk_-?\d+\b", "blk_<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"\bblock-id:-?\d+\b", "block-id:<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b", "<*>:<*>", value)
        value = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<*>", value)
        value = re.sub(r"\b\d+\b", "<*>", value)
    elif dataset_key == "openstack":
        value = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            "<*>",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\b[a-f0-9]{24,}\b", "<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"/var/lib/nova/instances(?:/_base)?/[A-Za-z0-9._-]+", "<*>", value)
        value = re.sub(r"\((?:/[^)]+|<\*>)\)", "(<*>)", value)
        value = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "<*>", value)
        value = re.sub(r"\b\d+\.\d+\b", "<*>.<*>", value)
        value = re.sub(r"\b\d+\b", "<*>", value)
    elif dataset_key == "hadoop":
        value = re.sub(r"\battempt_[a-z0-9_:-]+\b", "attempt_<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"\batt_[a-z0-9_:-]+\b", "att_<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"\btask_\d+_\d+_[mr]_\d+\b", "task_<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"\btk_\d+_\d+_[mr]_\d+\b", "tk_<*>", value, flags=re.IGNORECASE)
        value = re.sub(r"\bDFSClient_NONMAPREDUCE_\d+_\d+\b", "DFSClient_NONMAPREDUCE_<*>_<*>", value)
        value = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "<*>:<*>", value)
        value = re.sub(r"\b\d+\.\d+\b", "<*>.<*>", value)
        value = re.sub(r"\b\d+\b", "<*>", value)

    value = normalize_whitespace(value)
    limit = char_budget or _DEFAULT_QUERY_CHAR_BUDGET.get(dataset_key, 220)
    if len(value) > limit:
        clipped = value[:limit]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        value = clipped
    return value
