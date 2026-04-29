"""Text normalization and sparse similarity helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_STOPWORDS = {
    "",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "with",
    "by",
    "from",
    "this",
    "that",
    "http",
    "https",
    "status",
    "len",
    "time",
}


def normalize_text(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", " <id> ", value)
    value = re.sub(r"\breq-[0-9a-f-]{8,}\b", " <req> ", value)
    value = re.sub(r"\bblk_-?\d+\b", " <blk> ", value)
    value = re.sub(r"\b(?:application|appattempt|container)_[0-9_]+\b", " <job> ", value)
    value = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " <ip> ", value)
    value = re.sub(r"hdfs://\S+", " hdfs_path ", value)
    value = re.sub(r"/var/lib/nova/instances/_base/[a-z0-9]+", " base_path ", value)
    value = re.sub(r"/[a-z0-9._:/-]+", " path ", value)
    value = re.sub(r"\b\d+\b", " <num> ", value)
    value = re.sub(r"[^a-z0-9_<>\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_set(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if token not in _STOPWORDS and len(token) > 1
    }


def token_counter(texts: Iterable[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(token_set(text))
    return counter


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for token, value in left.items():
        dot += float(value) * float(right.get(token, 0.0))
    if dot <= 0.0:
        return 0.0
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left.values()))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
