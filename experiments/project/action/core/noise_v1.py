"""Deterministic semantic noise variant v1."""

from __future__ import annotations

import random
import re
from collections import Counter


_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>:-]+")

_OPENSTACK_MAP = {
    "pending_task": "queued_work",
    "instance_sync": "instance_refresh",
    "active_base_files": "live_base_files",
    "detail_low": "detail_small",
    "detail_high": "detail_heavy",
    "warning": "notice",
    "error": "fault",
    "claim": "lifecycle",
    "successful": "lifecycle",
    "creating": "artifact",
    "image": "artifact",
    "vm": "guest",
    "stopped": "state",
    "paused": "state",
    "resumed": "state",
    "active": "state",
    "unknown": "storage",
    "removable": "storage",
    "base": "storage",
    "swap": "storage",
    "file": "storage",
    "files": "storage",
    "servers": "request",
    "server": "request",
    "detail": "request",
    "instance": "guest",
    "compute": "service",
    "manager": "service",
    "cache": "service",
    "pending": "control",
    "task": "control",
    "sync": "control",
    "power": "control",
    "state": "control",
}

_HADOOP_MAP = {
    "machine": "infra",
    "network": "infra",
    "disk": "infra",
    "container": "infra",
    "manager": "infra",
    "failed": "issue",
    "error": "issue",
    "exception": "issue",
    "shuffle": "transfer",
    "fetcher": "transfer",
    "port": "transfer",
    "reader": "transfer",
    "connection": "link",
    "connect": "link",
    "closed": "link",
    "remote": "link",
    "timeout": "delay",
    "timed": "delay",
    "block": "storage",
    "outputstream": "storage",
    "pagerank": "batchjob",
    "wordcount": "batchjob",
}

_GLOBAL_MAP = {
    "trace": "sequence",
    "event": "signal",
    "tail": "ending",
    "component": "module",
    "repair": "recovery",
    "root": "source",
}

_HDFS_THRESHOLD_MAP = {
    "e21": 0.18,
    "e23": 0.18,
    "e5": 0.22,
    "e9": 0.24,
    "e11": 0.24,
    "e22": 0.28,
    "e2": 0.52,
    "e3": 0.56,
    "e4": 0.6,
    "e26": 0.72,
    "e20": 0.74,
    "e6": 0.84,
    "e16": 0.84,
    "e27": 0.9,
    "e28": 0.92,
}


def _rename_token(token: str, dataset: str) -> str:
    lowered = token.lower()
    if dataset == "HDFS" and lowered.startswith("e") and lowered[1:].isdigit():
        return f"step_{lowered[1:]}"
    if dataset == "OpenStack" and lowered in _OPENSTACK_MAP:
        return _OPENSTACK_MAP[lowered]
    if dataset == "Hadoop" and lowered in _HADOOP_MAP:
        return _HADOOP_MAP[lowered]
    return _GLOBAL_MAP.get(lowered, token)


def _effective_noise_level(dataset: str, *, channel: str, noise_level: float) -> float:
    multipliers = {
        ("HDFS", "text"): 0.85,
        ("HDFS", "counter"): 0.65,
        ("Hadoop", "text"): 1.45,
        ("Hadoop", "counter"): 1.25,
        ("OpenStack", "text"): 1.7,
        ("OpenStack", "counter"): 1.4,
    }
    scaled = noise_level * multipliers.get((dataset, channel), 1.0)
    return max(0.0, min(1.0, scaled))


def _hdfs_threshold(token: str) -> float:
    lowered = token.lower()
    if lowered in _HDFS_THRESHOLD_MAP:
        return _HDFS_THRESHOLD_MAP[lowered]
    if lowered.startswith("e") and lowered[1:].isdigit():
        return 0.68
    return 0.5


def _hdfs_partial_ratio(token: str, *, effective_noise: float) -> float:
    threshold = _hdfs_threshold(token)
    if effective_noise <= threshold:
        return 0.0
    span = max(1.0 - threshold, 1e-6)
    ratio = (effective_noise - threshold) / span
    return max(0.0, min(1.0, ratio))


def inject_text_noise(text: str, *, dataset: str, noise_level: float, seed: int) -> str:
    effective_noise = _effective_noise_level(dataset, channel="text", noise_level=noise_level)
    if effective_noise <= 0.0:
        return text
    if dataset == "HDFS":
        def replace_hdfs(match: re.Match[str]) -> str:
            token = match.group(0)
            if _hdfs_partial_ratio(token, effective_noise=effective_noise) <= 0.0:
                return token
            return _rename_token(token, dataset)

        return _TOKEN_RE.sub(replace_hdfs, text)
    rng = random.Random(f"{dataset}:text:{seed}:{noise_level}:{len(text)}")

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if rng.random() > effective_noise:
            return token
        return _rename_token(token, dataset)

    return _TOKEN_RE.sub(replace, text)


def inject_counter_noise(counter: Counter[str], *, dataset: str, noise_level: float, seed: int) -> Counter[str]:
    effective_noise = _effective_noise_level(dataset, channel="counter", noise_level=noise_level)
    if effective_noise <= 0.0:
        return Counter(counter)
    if dataset == "HDFS":
        noised: Counter[str] = Counter()
        for token, value in counter.items():
            replacement = _rename_token(token, dataset)
            ratio = _hdfs_partial_ratio(token, effective_noise=effective_noise)
            moved = int(round(float(value) * ratio))
            kept = int(value) - moved
            if kept > 0:
                noised[token] += kept
            if moved > 0:
                noised[replacement] += moved
        return noised
    rng = random.Random(f"{dataset}:counter:{seed}:{noise_level}:{len(counter)}")
    noised: Counter[str] = Counter()
    for token, value in counter.items():
        replacement = token
        if rng.random() <= effective_noise:
            replacement = _rename_token(token, dataset)
        noised[replacement] += value
    return noised
