"""Deterministic semantic noise variant v2."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import replace

from experiments.project.action.core.schema import ActionCase, ActionQuery


_TOKEN_RE = re.compile(r"[A-Za-z0-9_<>:/\\.-]+")

_OPENSTACK_SYNONYM_MAP = {
    "get": "fetch",
    "post": "submit",
    "pending_task": "queued_work",
    "instance_sync": "instance_refresh",
    "active_base_files": "live_base_files",
    "unknown_base_file": "unresolved_base_file",
    "removable_base_files": "candidate_base_files",
    "remove_swap_file": "cleanup_swap_file",
    "warning": "notice",
    "error": "fault",
    "imagecache": "image_store",
    "nova-api": "api_service",
    "nova-compute": "compute_service",
}

_HADOOP_SYNONYM_MAP = {
    "machine": "worker",
    "network": "link",
    "disk": "storage",
    "failed": "broken",
    "exception": "fault",
    "shuffle": "transfer",
    "pagerank": "rank_job",
    "wordcount": "count_job",
    "finalmerge": "final_combine",
    "fetcher": "fetch_unit",
}

_GLOBAL_SYNONYM_MAP = {
    "trace": "sequence",
    "event": "signal",
    "tail": "ending",
    "component": "module",
    "repair": "recovery",
    "root": "source",
}

_OPENSTACK_GENERIC_MAP = {
    "get": "api_call",
    "post": "api_call",
    "fetch": "api_call",
    "submit": "api_call",
    "pending_task": "state_marker",
    "queued_work": "state_marker",
    "instance_sync": "state_marker",
    "instance_refresh": "state_marker",
    "active_base_files": "storage_marker",
    "live_base_files": "storage_marker",
    "unknown_base_file": "storage_marker",
    "unresolved_base_file": "storage_marker",
    "removable_base_files": "storage_marker",
    "candidate_base_files": "storage_marker",
    "remove_swap_file": "storage_marker",
    "cleanup_swap_file": "storage_marker",
    "imagecache": "storage_component",
    "image_store": "storage_component",
    "nova-api": "service_component",
    "nova-compute": "service_component",
    "api_service": "service_component",
    "compute_service": "service_component",
    "server": "service_component",
    "compute_manager": "service_component",
    "detail": "detail_flag",
    "detail_low": "detail_flag",
    "detail_high": "detail_flag",
}

_HADOOP_GENERIC_MAP = {
    "machine": "infra_issue",
    "worker": "infra_issue",
    "network": "infra_issue",
    "link": "infra_issue",
    "disk": "infra_issue",
    "storage": "infra_issue",
    "failed": "status_issue",
    "broken": "status_issue",
    "exception": "status_issue",
    "fault": "status_issue",
    "shuffle": "phase_marker",
    "transfer": "phase_marker",
    "finalmerge": "phase_marker",
    "final_combine": "phase_marker",
    "pagerank": "batch_job",
    "rank_job": "batch_job",
    "wordcount": "batch_job",
    "count_job": "batch_job",
    "fetcher": "phase_marker",
    "fetch_unit": "phase_marker",
}

_GLOBAL_GENERIC_MAP = {
    "trace": "signal_block",
    "sequence": "signal_block",
    "event": "signal_marker",
    "signal": "signal_marker",
    "component": "module_marker",
    "module": "module_marker",
}

_OPENSTACK_DISTRACTORS = [
    "control-plane poll: api_call /resource/detail returned status 200",
    "scheduler heartbeat: service_component state update observed",
]

_HADOOP_DISTRACTORS = [
    "auxiliary alert: phase_marker checkpoint advanced without decisive status",
    "auxiliary alert: worker progress updated after prior task retry",
]

_HDFS_DISTRACTORS = [
    "Auxiliary trace note: prefix path remained structurally consistent before termination.",
    "Auxiliary trace note: replicated block handling repeated a prior transition pattern.",
]


def _stable_score(*parts: object) -> float:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64 - 1)


def _line_kind(dataset: str, line: str) -> str:
    lowered = line.lower()
    if dataset == "HDFS":
        if lowered.startswith("tail events"):
            return "tail"
        if lowered.startswith("full event trace"):
            return "full_trace"
        if lowered.startswith("prefix events"):
            return "prefix"
        return "meta"
    if dataset == "Hadoop":
        if lowered.startswith("workload context"):
            return "workload"
        if lowered.startswith("- "):
            return "alert"
        if "late_tail" in lowered or "support" in lowered:
            return "late"
        return "meta"
    if dataset == "OpenStack":
        if "servers/detail" in lowered or "post /v2" in lowered:
            return "api"
        if "instance:" in lowered:
            return "instance"
        if "pending_task" in lowered or "instance_sync" in lowered:
            return "anchor"
        return "control"
    return "default"


def _event_code_generic(line_kind: str) -> str:
    if line_kind == "tail":
        return "tail_event"
    if line_kind == "prefix":
        return "prefix_event"
    return "phase_event"


def _dataset_synonym(token: str, dataset: str) -> str:
    lowered = token.lower()
    if dataset == "HDFS" and lowered.startswith("e") and lowered[1:].isdigit():
        return f"step_{lowered[1:]}"
    if dataset == "OpenStack":
        return _OPENSTACK_SYNONYM_MAP.get(lowered, _GLOBAL_SYNONYM_MAP.get(lowered, token))
    if dataset == "Hadoop":
        return _HADOOP_SYNONYM_MAP.get(lowered, _GLOBAL_SYNONYM_MAP.get(lowered, token))
    return _GLOBAL_SYNONYM_MAP.get(lowered, token)


def _dataset_generic(token: str, dataset: str, *, line_kind: str) -> str:
    lowered = token.lower()
    if dataset == "HDFS" and lowered.startswith("e") and lowered[1:].isdigit():
        return _event_code_generic(line_kind)
    if dataset == "OpenStack":
        if re.fullmatch(r"[0-9a-f]{8,}", lowered):
            return "id_token"
        return _OPENSTACK_GENERIC_MAP.get(lowered, _GLOBAL_GENERIC_MAP.get(lowered, token))
    if dataset == "Hadoop":
        return _HADOOP_GENERIC_MAP.get(lowered, _GLOBAL_GENERIC_MAP.get(lowered, token))
    return _GLOBAL_GENERIC_MAP.get(lowered, token)


def _importance(token: str, dataset: str, *, line_kind: str) -> str:
    lowered = token.lower()
    if dataset == "HDFS":
        if lowered.startswith("e") and lowered[1:].isdigit():
            if line_kind in {"tail", "full_trace"}:
                return "high"
            if line_kind == "prefix":
                return "medium"
        return "low"
    if dataset == "Hadoop":
        if lowered in {"pagerank", "wordcount"}:
            return "workload"
        if lowered in {"machine", "network", "disk", "worker", "link", "storage", "exception", "fault", "failed", "broken"}:
            return "high"
        if lowered in {"shuffle", "transfer", "finalmerge", "final_combine", "fetcher", "fetch_unit"}:
            return "medium"
        if line_kind in {"alert", "late"}:
            return "medium"
        return "low"
    if dataset == "OpenStack":
        if lowered in {
            "get",
            "post",
            "fetch",
            "submit",
            "pending_task",
            "queued_work",
            "instance_sync",
            "instance_refresh",
            "active_base_files",
            "live_base_files",
            "unknown_base_file",
            "unresolved_base_file",
            "removable_base_files",
            "candidate_base_files",
            "remove_swap_file",
            "cleanup_swap_file",
            "imagecache",
            "image_store",
            "compute_manager",
            "server",
        }:
            return "high"
        if "nova-" in lowered or "/v2/" in lowered or "servers/detail" in lowered:
            return "medium"
        return "low"
    return "low"


def _thresholds(noise_level: float, importance: str, *, graph: bool) -> tuple[float, float, float]:
    if graph:
        shifted = max(0.0, noise_level - 0.3) / 0.7
        if importance == "high":
            return 0.15 * shifted, 0.65 * shifted, 0.85 * shifted
        if importance == "workload":
            return 0.10 * shifted, 0.25 * shifted, 0.35 * shifted
        if importance == "medium":
            return 0.25 * shifted, 0.55 * shifted, 0.70 * shifted
        return 0.30 * shifted, 0.35 * shifted, 0.45 * shifted

    if importance == "workload":
        rename = 0.20 * noise_level
        generic = 0.25 * max(0.0, noise_level - 0.6) / 0.4
        mask = 0.05 * max(0.0, noise_level - 0.9) / 0.1
        return rename, generic, mask
    if importance == "high":
        rename = 0.10 * noise_level
        generic = max(0.0, noise_level - 0.2)
        mask = max(0.0, noise_level - 0.6) / 0.4
        return rename, generic, mask
    if importance == "medium":
        rename = 0.35 * noise_level
        generic = 0.75 * max(0.0, noise_level - 0.3) / 0.7
        mask = 0.40 * max(0.0, noise_level - 0.8) / 0.2
        return rename, generic, mask
    rename = 0.50 * noise_level
    generic = 0.20 * max(0.0, noise_level - 0.6) / 0.4
    return rename, generic, 0.0


def _apply_token_transform(
    token: str,
    *,
    dataset: str,
    noise_level: float,
    seed: int,
    namespace: str,
    line_kind: str,
    graph: bool,
) -> str:
    if noise_level <= 0.0:
        return token
    score = _stable_score(dataset, seed, namespace, line_kind, token)
    importance = _importance(token, dataset, line_kind=line_kind)
    rename_threshold, generic_threshold, mask_threshold = _thresholds(
        noise_level,
        importance,
        graph=graph,
    )
    if score < mask_threshold:
        return _dataset_generic(token, dataset, line_kind=line_kind)
    if score < generic_threshold:
        return _dataset_generic(token, dataset, line_kind=line_kind)
    if score < rename_threshold:
        return _dataset_synonym(token, dataset)
    return token


def _transform_text(text: str, *, dataset: str, noise_level: float, seed: int, graph: bool) -> str:
    if noise_level <= 0.0:
        return text
    lines = text.splitlines()
    transformed: list[str] = []
    for line_index, line in enumerate(lines):
        kind = _line_kind(dataset, line)

        def replace_token(match: re.Match[str]) -> str:
            token = match.group(0)
            namespace = f"line:{line_index}:{match.start()}"
            return _apply_token_transform(
                token,
                dataset=dataset,
                noise_level=noise_level,
                seed=seed,
                namespace=namespace,
                line_kind=kind,
                graph=graph,
            )

        transformed.append(_TOKEN_RE.sub(replace_token, line))

    if not graph:
        transformed.extend(_distractor_lines(dataset, noise_level=noise_level, seed=seed))
    return "\n".join(transformed)


def _distractor_lines(dataset: str, *, noise_level: float, seed: int) -> list[str]:
    if noise_level < 0.6:
        return []
    pool = {
        "HDFS": _HDFS_DISTRACTORS,
        "Hadoop": _HADOOP_DISTRACTORS,
        "OpenStack": _OPENSTACK_DISTRACTORS,
    }[dataset]
    lines = []
    for index, line in enumerate(pool):
        threshold = min(1.0, (noise_level - 0.5) / 0.5)
        if _stable_score(dataset, seed, "distractor", index) < threshold:
            lines.append(line)
    return lines


def inject_text_noise_v2(text: str, *, dataset: str, noise_level: float, seed: int) -> str:
    return _transform_text(text, dataset=dataset, noise_level=noise_level, seed=seed, graph=False)


def inject_counter_noise_v2(
    counter: Counter[str],
    *,
    dataset: str,
    noise_level: float,
    seed: int,
    namespace: str,
    graph: bool,
    line_kind: str = "feature",
) -> Counter[str]:
    if noise_level <= 0.0:
        return Counter(counter)
    noised: Counter[str] = Counter()
    drop_threshold = 0.0
    if graph:
        drop_threshold = 0.45 * max(0.0, noise_level - 0.6) / 0.4
    else:
        drop_threshold = 0.20 * max(0.0, noise_level - 0.8) / 0.2
    for token, value in counter.items():
        token_score = _stable_score(dataset, seed, namespace, token, "drop")
        if token_score < drop_threshold:
            continue
        replacement = _apply_token_transform(
            token,
            dataset=dataset,
            noise_level=noise_level,
            seed=seed,
            namespace=f"{namespace}:{token}",
            line_kind=line_kind,
            graph=graph,
        )
        noised[replacement] += value
    return noised


def _noise_graph_facts(
    facts: dict[str, list[str]],
    *,
    dataset: str,
    noise_level: float,
    seed: int,
) -> dict[str, list[str]]:
    if noise_level <= 0.0:
        return {key: list(values) for key, values in facts.items()}
    drop_threshold = 0.55 * max(0.0, noise_level - 0.6) / 0.4
    payload: dict[str, list[str]] = {}
    for fact_key, values in facts.items():
        transformed_values: list[str] = []
        for index, value in enumerate(values):
            if _stable_score(dataset, seed, fact_key, index, "fact_drop") < drop_threshold:
                continue
            transformed_values.append(
                _transform_text(
                    value,
                    dataset=dataset,
                    noise_level=noise_level,
                    seed=seed,
                    graph=True,
                )
            )
        payload[fact_key] = transformed_values or ["Graph evidence weakened under semantic noise."]
    return payload


def build_query_v2(case: ActionCase, *, noise_level: float, seed: int) -> ActionQuery:
    noisy_graph_features = {
        name: inject_counter_noise_v2(
            counter,
            dataset=case.dataset,
            noise_level=noise_level,
            seed=seed,
            namespace=f"graph_feature:{name}",
            graph=True,
            line_kind="graph_feature",
        )
        for name, counter in case.graph_features.items()
    }
    noisy_case = replace(
        case,
        graph_features=noisy_graph_features,
        graph_facts=_noise_graph_facts(
            case.graph_facts,
            dataset=case.dataset,
            noise_level=noise_level,
            seed=seed,
        ),
        metadata={
            **case.metadata,
            "noise_variant": "v2",
            "noise_level": noise_level,
        },
    )
    return ActionQuery(
        base_case=noisy_case,
        noise_level=noise_level,
        seed=seed,
        incident_text=inject_text_noise_v2(
            case.incident_text,
            dataset=case.dataset,
            noise_level=noise_level,
            seed=seed,
        ),
        raw_features={
            name: inject_counter_noise_v2(
                counter,
                dataset=case.dataset,
                noise_level=noise_level,
                seed=seed,
                namespace=f"raw_feature:{name}",
                graph=False,
                line_kind="raw_feature",
            )
            for name, counter in case.raw_features.items()
        },
        metadata={
            "noise_level": noise_level,
            "seed": seed,
            "noise_variant": "v2",
        },
    )
