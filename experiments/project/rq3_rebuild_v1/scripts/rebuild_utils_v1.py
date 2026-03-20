from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts import build_rq3_small_v2_benchmark_20260318 as builder_v2
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    action_text_match,
    action_text_success,
    family_for_action,
    infer_action_id_from_text,
    infer_family_from_text,
)


CONTRACT_PATH = REBUILD_ROOT / "configs" / "contract_v1_20260318.json"
LEGACY_REFERENCE_PACKAGE = (
    PROJECT_ROOT
    / "experiments"
    / "thesis_rebuild_20260315"
    / "rq34"
    / "analysis"
    / "rq3_small_v2_20260318"
    / "rq3_small_v2_benchmark_package_20260318.json"
)
CANDIDATE_AUDIT_ROWS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "thesis_rebuild_20260315"
    / "rq34"
    / "results"
    / "rq3_candidate_audit_enriched_20260318_rows.json"
)
RQ2_GRAPH_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "thesis_rebuild_20260315"
    / "rq2_fullcase"
    / "results"
    / "gt_causal_knowledge_nesydy_fullcase_20260316.json"
)
DATASET_DOMAIN = {
    "HDFS": "hdfs",
    "OpenStack": "openstack",
    "Hadoop": "hadoop",
}
ACTION_ID_ALIASES = {
    "Hadoop": {
        "HADOOP_FREE_DISK_AND_RETRY": "HADOOP_FREE_LOCAL_DISK_AND_RETRY",
        "HADOOP_RESTORE_NETWORK_AND_RETRY": "HADOOP_RESTORE_WORKER_RPC_AND_RETRY",
    },
}
GRAPH_TOKEN_REWRITE = {
    "addstoredblock": "block-map-update",
    "blockmap": "block-map",
    "fileoutputcommitter": "output-cleanup",
    "packetresponder": "packet-responder",
    "receiveblock": "receive-block",
}
GRAPH_SUMMARY_STOPWORDS = {
    "already",
    "apache",
    "app",
    "asyncdispatcher",
    "committerevent",
    "context",
    "data",
    "datanode",
    "details",
    "destination",
    "dfs",
    "error",
    "event",
    "exception",
    "failed",
    "failure",
    "for",
    "from",
    "handler",
    "hadoop",
    "http",
    "info",
    "impl",
    "job",
    "java",
    "lib",
    "log",
    "manager",
    "mapred",
    "mapreduce",
    "namesystem",
    "nova",
    "org",
    "port",
    "processor",
    "report",
    "status",
    "task",
    "taskattemptimpl",
    "template",
    "the",
    "thread",
    "time",
    "update",
    "warn",
    "with",
}
NEGATIVE_REPAIR_PATTERNS = {
    "no action required",
    "no immediate action",
    "no immediate repair",
    "no repair action required",
    "no repair required",
    "operating normally",
    "routine replication",
    "system is operating normally",
    "working normally",
}
SEMANTIC_FAMILY_KEYWORDS = {
    "HDFS": {
        "HDFS_TRANSFER_LINK_FAILURE": [
            "client",
            "connectivity",
            "congestion",
            "eof",
            "handoff",
            "latency",
            "link",
            "network",
            "packet loss",
            "peer",
            "transfer",
        ],
        "HDFS_PIPELINE_FAILURE": [
            "packetresponder",
            "pipeline",
            "receiving block",
            "received block",
            "receiver",
            "replica stage",
            "write pipeline",
        ],
        "HDFS_STORAGE_METADATA_PRESSURE": [
            "allocate",
            "allocation",
            "block map",
            "blockmap",
            "capacity",
            "delete",
            "deletion",
            "invalid",
            "invalidset",
            "metadata",
            "quota",
            "space",
            "stale block",
        ],
    },
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def tokenize(text: str) -> List[str]:
    clean = []
    stripped = norm(text).replace("/", " ").replace(":", " ").replace("-", " ").replace(".", " ")
    for raw in stripped.split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        if len(token) >= 3:
            clean.append(token)
    return clean


def canonical_action_id(dataset: str, action_id: str) -> str:
    action = str(action_id or "").strip()
    if not action:
        return ""
    return str(ACTION_ID_ALIASES.get(dataset, {}).get(action, action))


def compact_lines(lines: Sequence[str], max_chars: int) -> str:
    kept: List[str] = []
    total = 0
    for line in lines:
        text = str(line or "").strip()
        if not text or text in kept:
            continue
        extra = len(text) + (1 if kept else 0)
        if kept and total + extra > max_chars:
            break
        kept.append(text)
        total += extra
    return "\n".join(kept)


def _first_matching_line(raw_log: str, pattern_groups: Sequence[Sequence[str]]) -> str:
    lines = [str(line or "").strip() for line in str(raw_log or "").splitlines() if str(line or "").strip()]
    for required_tokens in pattern_groups:
        lowered_required = [norm(token) for token in required_tokens if norm(token)]
        for line in lines:
            lowered_line = norm(line)
            if all(token in lowered_line for token in lowered_required):
                return line
    return ""


def _reanchor_hdfs_reference_alert(raw_log: str, action_id: str, selected_alert: str) -> str:
    chosen = str(selected_alert or "").strip()
    lowered = norm(chosen)
    if not lowered:
        return ""
    if action_id == "HDFS_REBUILD_WRITE_PIPELINE":
        if "received block" in lowered or "receiving block" in lowered:
            alternative = _first_matching_line(
                raw_log,
                [
                    ["packetresponder", "terminating"],
                    ["replicastage", "closing"],
                    ["replicastage", "error"],
                ],
            )
            if alternative:
                return alternative
    if action_id == "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE":
        if "got exception while serving" in lowered:
            alternative = _first_matching_line(
                raw_log,
                [
                    ["writeblock", "connection reset by peer"],
                    ["received exception", "connection reset by peer"],
                    ["wait threshold exceeded"],
                    ["replicastagestep", "received exception"],
                ],
            )
            if alternative:
                return alternative
            return ""
    if action_id == "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK":
        if "allocateblock" in lowered:
            return ""
    if action_id == "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE":
        if "blockmap updated" in lowered:
            alternative = _first_matching_line(
                raw_log,
                [
                    ["invalidset"],
                    ["delete", "invalidset"],
                    ["unexpected error trying to delete block"],
                ],
            )
            if alternative:
                return alternative
            return ""
    return chosen


def _sanitize_hdfs_capacity_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    out = str(noisy_alert or clean_alert or "")
    if not out:
        return str(clean_alert or "")
    if float(noise) >= 0.6:
        out = out.replace("NameSystem.reserveReplicaTargets", "NameSystem.allocateBlock")
        out = out.replace("NameSystem.reserveTargets", "NameSystem.allocateBlock")
        out = out.replace("reserveReplicaTargets", "allocateBlock")
        out = out.replace("reserveTargets", "allocateBlock")
    if "blk_" in str(clean_alert or ""):
        out = re.sub(r"block-id:([\\-\\d]+)", r"blk_\\1", out)
    return out


def _sanitize_hadoop_retry_alert(clean_alert: str, noisy_alert: str, action_id: str, noise: float) -> str:
    out = str(noisy_alert or clean_alert or "")
    if not out or float(noise) < 0.6:
        return out
    clean = str(clean_alert or "")
    if "Retrying connect to server:" not in clean:
        return out
    prefix = clean.split("org.apache.hadoop.ipc.Client:", 1)
    if len(prefix) != 2:
        return out
    header = prefix[0] + "org.apache.hadoop.ipc.Client:"
    remainder = prefix[1]
    endpoint_match = re.search(r"Retrying connect to server:\s*(.+?)\.\s+Already tried", remainder)
    suffix_match = re.search(r"Already tried.+$", remainder)
    if not endpoint_match or not suffix_match:
        return out
    endpoint = endpoint_match.group(1).strip()
    suffix = suffix_match.group(0).strip()
    if action_id == "HADOOP_RESTORE_WORKER_RPC_AND_RETRY":
        return f"{header} Retrying worker RPC toward node: {endpoint}. {suffix}"
    if action_id == "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY":
        return f"{header} Retrying ResourceManager channel to server: {endpoint}. {suffix}"
    return out


def sanitize_noisy_alert(dataset: str, clean_alert: str, noisy_alert: str, action_id: str, noise: float) -> str:
    out = builder_v2._sanitize_noisy_alert(dataset, clean_alert, noisy_alert, action_id, noise)
    if dataset == "HDFS" and action_id == "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK":
        out = _sanitize_hdfs_capacity_alert(clean_alert, out, noise)
    if dataset == "Hadoop" and action_id in {
        "HADOOP_RESTORE_WORKER_RPC_AND_RETRY",
        "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY",
    }:
        out = _sanitize_hadoop_retry_alert(clean_alert, out, action_id, noise)
    return out


def unique_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    kept: List[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        kept.append(token)
    return kept


def overlap_score(text_a: str, text_b: str) -> float:
    a = set(tokenize(text_a))
    b = set(tokenize(text_b))
    if not a or not b:
        return 0.0
    return round(len(a & b) + (len(a & b) / max(1, len(a | b))), 4)


def template_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"<\*>|\[\*\]", " ", str(text or ""))
    return tokenize(cleaned)


def summarize_graph_template(template: str, support_tokens: Sequence[str], *, max_tokens: int) -> str:
    support = set(support_tokens)
    raw_tokens = [GRAPH_TOKEN_REWRITE.get(tok, tok) for tok in template_tokens(template)]
    prioritized = [
        tok
        for tok in raw_tokens
        if tok in support and tok not in GRAPH_SUMMARY_STOPWORDS and not tok.isdigit()
    ]
    fallback = [
        tok
        for tok in raw_tokens
        if tok not in support and tok not in GRAPH_SUMMARY_STOPWORDS and not tok.isdigit()
    ]
    chosen = unique_preserve(prioritized + fallback)[:max_tokens]
    if not chosen:
        chosen = unique_preserve(raw_tokens)[:max_tokens]
    return " ".join(chosen)


@lru_cache(maxsize=1)
def contract() -> Mapping[str, object]:
    return load_json(CONTRACT_PATH)


@lru_cache(maxsize=1)
def legacy_package() -> Mapping[str, object]:
    return load_json(LEGACY_REFERENCE_PACKAGE)


@lru_cache(maxsize=1)
def candidate_audit_rows() -> List[Mapping[str, object]]:
    return list(load_json(CANDIDATE_AUDIT_ROWS_PATH))


@lru_cache(maxsize=1)
def supplemental_reference_bank_by_dataset() -> Dict[str, List[Mapping[str, object]]]:
    chosen: Dict[tuple[str, str], Mapping[str, object]] = {}
    for row in candidate_audit_rows():
        dataset = str(row.get("dataset", "") or "")
        case_id = str(row.get("case_id", "") or "")
        if dataset not in DATASET_DOMAIN or not case_id:
            continue
        key = (dataset, case_id)
        prev = chosen.get(key)
        score = float(row.get("selection_score", 0.0) or 0.0)
        if prev is None or score > float(prev.get("selection_score", 0.0) or 0.0):
            chosen[key] = row

    out: Dict[str, List[Mapping[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for (dataset, case_id), row in chosen.items():
        source = str(row.get("pool_source", "") or "")
        pool_key = (source, case_id)
        if pool_key not in pool_rows():
            continue
        case_row = pool_rows()[pool_key]
        selected_alert = str(row.get("selected_alert", "") or "").strip()
        if not selected_alert:
            continue
        raw_log = str(case_row.get("raw_log", "") or "")
        action_id = canonical_action_id(dataset, str(row.get("gt_action_id", "") or ""))
        if dataset == "HDFS":
            selected_alert = _reanchor_hdfs_reference_alert(raw_log, action_id, selected_alert)
        if not selected_alert:
            continue
        try:
            context_text = raw_window_context(dataset, raw_log, selected_alert)
        except Exception:
            continue
        flags = row.get("selected_alert_flags", {}) or {}
        tags = [str(name) for name, value in flags.items() if bool(value)]
        rendered = render_reference_snippet(
            {
                "selected_alert": selected_alert,
                "context_text": context_text,
            }
        )
        family_id = str(row.get("gt_label", "") or "")
        if not family_id and action_id:
            family_id = family_for_action(dataset, action_id)
        out[dataset].append(
            {
                "reference_id": f"supplemental:{source}:{case_id}",
                "dataset": dataset,
                "case_id": case_id,
                "source": source,
                "selected_alert": selected_alert,
                "context_text": context_text,
                "action_id": action_id,
                "family_id": family_id,
                "mode": str(row.get("reason", "") or ""),
                "rendered_text": rendered,
                "tags": tags,
                "tokens": tokenize(rendered),
            }
        )
    return out


@lru_cache(maxsize=1)
def reference_bank_by_dataset() -> Dict[str, List[Mapping[str, object]]]:
    def _canonicalize(dataset: str, entry: Mapping[str, object]) -> Dict[str, object]:
        out = dict(entry)
        action_id = canonical_action_id(dataset, str(out.get("action_id", "") or ""))
        out["action_id"] = action_id
        family_id = str(out.get("family_id", "") or "")
        if not family_id and action_id:
            family_id = family_for_action(dataset, action_id)
        out["family_id"] = family_id
        return out

    merged: Dict[str, List[Mapping[str, object]]] = {}
    legacy = {
        str(dataset): [_canonicalize(str(dataset), entry) for entry in entries]
        for dataset, entries in legacy_package()["reference_bank"].items()
    }
    supplemental = supplemental_reference_bank_by_dataset()
    for dataset, entries in legacy.items():
        existing_case_ids = {str(entry.get("case_id", "")) for entry in entries}
        merged_entries = list(entries)
        for extra in supplemental.get(dataset, []):
            if str(extra.get("case_id", "")) in existing_case_ids:
                continue
            merged_entries.append(extra)
        merged[dataset] = merged_entries
    return merged


def _reference_total_score(
    entry: Mapping[str, object],
    *,
    dataset: str,
    query_text: str,
    gt_action_id: str,
    gt_family_id: str,
) -> float:
    rendered = render_reference_snippet(entry)
    lexical = overlap_score(query_text, rendered)
    entry_action = canonical_action_id(dataset, str(entry.get("action_id", "") or ""))
    entry_family = str(entry.get("family_id", "") or "")
    retrieval_cfg = contract().get("retrieval_policy", {})
    use_gt_scoring = bool(retrieval_cfg.get("use_ground_truth_reference_scoring", False))
    same_action = use_gt_scoring and bool(gt_action_id) and entry_action == gt_action_id
    same_family = use_gt_scoring and bool(gt_family_id) and entry_family == gt_family_id
    score = lexical
    if same_action:
        score += 4.0
    elif same_family:
        score += 1.5
    return round(score, 4)


@lru_cache(maxsize=1)
def graph_edges_by_domain() -> Dict[str, List[Mapping[str, object]]]:
    edges = load_json(RQ2_GRAPH_PATH)
    out: Dict[str, List[Mapping[str, object]]] = {"hdfs": [], "openstack": [], "hadoop": []}
    for edge in edges:
        domain = str(edge.get("domain", "")).lower()
        if domain in out:
            out[domain].append(edge)
    return out


@lru_cache(maxsize=1)
def pool_rows() -> Dict[tuple[str, str], Mapping[str, object]]:
    return builder_v2._load_pool_rows()


@lru_cache(maxsize=1)
def old_runner():
    return builder_v2._load_old_runner_module()


@lru_cache(maxsize=1)
def legacy_module():
    return builder_v2._load_legacy_module()


def dataset_window_rule(dataset: str) -> Mapping[str, int]:
    return contract()["raw_window_rules"][dataset]


def find_selected_alert(item: Mapping[str, object], case_row: Mapping[str, object]) -> str:
    raw_log = str(case_row.get("raw_log", "") or "")
    alert_match = str(item.get("alert_match", "") or "").strip()
    if alert_match:
        return builder_v2._find_alert_line(raw_log, alert_match=alert_match, occurrence=1)
    dataset = str(item["dataset"])
    gt_action_id = str(item["gt_action_id"])
    selected = old_runner()._select_actionaware_alert(legacy_module(), raw_log, dataset)
    return old_runner()._refine_selected_alert_for_action(
        dataset,
        raw_log,
        selected,
        gt_action_id,
        str(case_row.get("raw_log_seed", "") or ""),
    )


def raw_window_context(
    dataset: str,
    raw_log: str,
    selected_alert: str,
    *,
    before: int | None = None,
    after: int | None = None,
    max_chars: int | None = None,
) -> str:
    rule = dataset_window_rule(dataset)
    before = int(rule["before"] if before is None else before)
    after = int(rule["after"] if after is None else after)
    max_chars = int(rule["max_chars"] if max_chars is None else max_chars)
    lines = [line.strip() for line in str(raw_log or "").splitlines() if line.strip()]
    wanted = norm(selected_alert)
    hit_index = -1
    for idx, line in enumerate(lines):
        if wanted == norm(line) or wanted in norm(line):
            hit_index = idx
            break
    if hit_index < 0:
        raise ValueError(f"Failed to locate selected alert in raw log: {selected_alert[:200]}")
    chosen: List[str] = []
    lo = max(0, hit_index - before)
    hi = min(len(lines), hit_index + after + 1)
    for idx in range(lo, hi):
        if idx == hit_index:
            continue
        chosen.append(lines[idx])
    return compact_lines(chosen, max_chars)


def supportive_context(
    dataset: str,
    raw_log: str,
    selected_alert: str,
    gt_action_id: str,
) -> str:
    max_chars = int(dataset_window_rule(dataset)["max_chars"])
    if dataset == "Hadoop":
        return builder_v2._extract_hadoop_context(
            old_runner(),
            raw_log,
            selected_alert,
            max_chars,
            gt_action_id,
        )
    return builder_v2._extract_frozen_context(
        old_runner(),
        raw_log,
        dataset,
        selected_alert=selected_alert,
        max_chars=max_chars,
        action_id=gt_action_id,
    )


def build_context_text(item: Mapping[str, object], case_row: Mapping[str, object], selected_alert: str) -> str:
    strategy = str(item.get("context_strategy", "raw_window"))
    raw_log = str(case_row.get("raw_log", "") or "")
    gt_action_id = str(item["gt_action_id"])
    if strategy == "supportive":
        context_text = supportive_context(str(item["dataset"]), raw_log, selected_alert, gt_action_id)
    else:
        context_text = raw_window_context(
            str(item["dataset"]),
            raw_log,
            selected_alert,
            before=item.get("window_before"),
            after=item.get("window_after"),
        )
    if not context_text.strip():
        context_text = selected_alert
    return context_text


def render_reference_snippet(entry: Mapping[str, object]) -> str:
    selected_alert = str(entry.get("selected_alert", "") or "").strip()
    context_text = str(entry.get("context_text", "") or "").strip()
    parts = ["Historical incident snippet"]
    if selected_alert:
        parts.append(f"Primary signal: {selected_alert}")
    if context_text and norm(context_text) != norm(selected_alert):
        parts.append(f"Nearby context: {context_text[:420]}")
    return "\n".join(parts)


def reference_primary_signal_text(entry: Mapping[str, object]) -> str:
    selected_alert = str(entry.get("selected_alert", "") or "").strip()
    if selected_alert:
        return selected_alert
    rendered = str(entry.get("text", "") or entry.get("rendered_text", "") or "").strip()
    for line in rendered.splitlines():
        if line.startswith("Primary signal: "):
            return line[len("Primary signal: ") :].strip()
    return rendered


def _skip_hdfs_direct_reference(gt_action_id: str, entry: Mapping[str, object]) -> bool:
    selected_alert = norm(str(entry.get("selected_alert", "") or reference_primary_signal_text(entry)))
    if not selected_alert:
        return False
    if gt_action_id == "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE":
        return "got exception while serving" in selected_alert
    if gt_action_id == "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK":
        return "allocateblock" in selected_alert
    if gt_action_id == "HDFS_REBUILD_WRITE_PIPELINE":
        return ("received block" in selected_alert or "receiving block" in selected_alert) and "terminating" not in selected_alert
    if gt_action_id == "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE":
        return "blockmap updated" in selected_alert and "invalidset" not in selected_alert
    return False


def select_references(
    dataset: str,
    *,
    case_id: str,
    query_text: str,
    top_k: int,
    gt_action_id: str = "",
    gt_family_id: str = "",
) -> List[Dict[str, object]]:
    refs = reference_bank_by_dataset().get(dataset, [])
    query_norm = norm(query_text)
    retrieval_cfg = contract().get("retrieval_policy", {})
    raw_gt_action_id = canonical_action_id(dataset, gt_action_id)
    use_gt_filter = bool(retrieval_cfg.get("use_ground_truth_reference_filter", False))
    use_gt_leak_guard = bool(retrieval_cfg.get("use_ground_truth_action_for_leak_guard", True))
    gt_action_id = raw_gt_action_id if use_gt_filter else ""
    leak_guard_action_id = raw_gt_action_id if use_gt_leak_guard else ""
    gt_family_id = str(gt_family_id or "") if use_gt_filter else ""
    ranked: List[Dict[str, object]] = []
    for entry in refs:
        if str(entry.get("case_id", "")) == case_id:
            continue
        entry_action = canonical_action_id(dataset, str(entry.get("action_id", "") or ""))
        if gt_action_id and entry_action != gt_action_id:
            continue
        if dataset == "HDFS" and leak_guard_action_id and _skip_hdfs_direct_reference(leak_guard_action_id, entry):
            continue
        if norm(str(entry.get("selected_alert", "") or "")) and norm(str(entry.get("selected_alert", "") or "")) in query_norm:
            continue
        rendered = render_reference_snippet(entry)
        lexical = overlap_score(query_text, rendered)
        if lexical <= 0 and not gt_action_id:
            continue
        score = _reference_total_score(
            entry,
            dataset=dataset,
            query_text=query_text,
            gt_action_id=gt_action_id,
            gt_family_id=gt_family_id,
        )
        if score < 1.0:
            continue
        ranked.append(
            {
                "reference_id": str(entry.get("reference_id", "")),
                "score": score,
                "lexical": lexical,
                "action_id": entry_action,
                "family_id": str(entry.get("family_id", "") or ""),
                "text": rendered,
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), row["reference_id"]))
    return ranked[:top_k]


def select_graph_evidence(
    dataset: str,
    *,
    query_text: str,
    top_k: int,
    reference_texts: Sequence[str] = (),
    hint_texts: Sequence[str] = (),
) -> Dict[str, object]:
    graph_cfg = contract().get("graph_selection", {})
    allow_without_refs = bool(graph_cfg.get("allow_without_refs", False))
    require_reference_overlap = bool(graph_cfg.get("require_reference_overlap", True))
    use_ground_truth_hint_tokens = bool(graph_cfg.get("use_ground_truth_hint_tokens", False))
    query_source_weight = float(graph_cfg.get("query_source_weight", 1.2))
    query_target_weight = float(graph_cfg.get("query_target_weight", 0.6))
    reference_source_weight = float(graph_cfg.get("reference_source_weight", 0.4))
    reference_target_weight = float(graph_cfg.get("reference_target_weight", 2.0))
    edge_weight_bonus = float(graph_cfg.get("edge_weight_bonus", 0.15))
    summary_max_tokens = int(graph_cfg.get("summary_max_tokens_per_side", 6))
    blocked_relations = {str(x) for x in (graph_cfg.get("blocked_relations", []) or [])}
    min_query_overlap = int(graph_cfg.get("min_query_overlap", 1))
    min_reference_overlap = int(graph_cfg.get("min_reference_overlap", 1 if require_reference_overlap else 0))
    min_score = float(graph_cfg.get("min_score", 0.0))

    domain = DATASET_DOMAIN[dataset]
    query_tokens = set(tokenize(query_text))
    reference_tokens = set()
    for text in reference_texts:
        reference_tokens.update(tokenize(text))
    if not reference_tokens and not allow_without_refs:
        return {"summary_lines": [], "matched_edges": []}
    hint_tokens = set()
    if use_ground_truth_hint_tokens:
        for text in hint_texts:
            hint_tokens.update(template_tokens(text))
    support_tokens = unique_preserve(list(query_tokens | reference_tokens | hint_tokens))
    ranked: List[Dict[str, object]] = []
    for edge in graph_edges_by_domain().get(domain, []):
        source = str(edge.get("source_template", "") or "")
        target = str(edge.get("target_template", "") or "")
        relation = str(edge.get("relation", "") or "")
        if relation in blocked_relations:
            continue
        source_tokens = set(template_tokens(source))
        target_tokens = set(template_tokens(target))
        if norm(source) == norm(target) or source_tokens == target_tokens:
            continue
        query_source_overlap = len(query_tokens & source_tokens)
        query_target_overlap = len(query_tokens & target_tokens)
        reference_source_overlap = len(reference_tokens & source_tokens)
        reference_target_overlap = len(reference_tokens & target_tokens)
        if query_source_overlap + query_target_overlap < min_query_overlap:
            continue
        if reference_tokens and require_reference_overlap and reference_source_overlap + reference_target_overlap < min_reference_overlap:
            continue
        score = query_source_weight * query_source_overlap + query_target_weight * query_target_overlap
        score += reference_source_weight * reference_source_overlap
        score += reference_target_weight * reference_target_overlap
        if hint_tokens:
            score += 0.25 * len(hint_tokens & source_tokens)
            score += 0.15 * len(hint_tokens & target_tokens)
        score += edge_weight_bonus * float(edge.get("weight", 0.0) or 0.0)
        if score < min_score:
            continue
        ranked.append(
            {
                "source_template": source,
                "target_template": target,
                "relation": relation,
                "weight": float(edge.get("weight", 0.0) or 0.0),
                "score": round(float(score), 4),
                "query_source_overlap": query_source_overlap,
                "query_target_overlap": query_target_overlap,
                "reference_source_overlap": reference_source_overlap,
                "reference_target_overlap": reference_target_overlap,
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), -float(row["weight"]), row["source_template"], row["target_template"]))
    chosen = ranked[:top_k]
    summary_lines = []
    for entry in chosen:
        source_phrase = summarize_graph_template(
            entry["source_template"],
            support_tokens,
            max_tokens=summary_max_tokens,
        )
        target_phrase = summarize_graph_template(
            entry["target_template"],
            support_tokens,
            max_tokens=summary_max_tokens,
        )
        summary_lines.append(
            f"Historical causal cue: {source_phrase} -> {target_phrase} "
            f"(relation={entry['relation']})."
        )
    return {
        "summary_lines": summary_lines,
        "matched_edges": chosen,
    }


def build_noise_views(
    dataset: str,
    *,
    gt_action_id: str,
    selected_alert: str,
    context_text: str,
    noise_levels: Sequence[float],
    references: Sequence[Mapping[str, object]],
    graph_evidence: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    legacy = legacy_module()
    injector = legacy.NoiseInjector(seed=2026)
    injector_hadoop = legacy.HadoopNoiseInjector(seed=2026)
    views: Dict[str, Dict[str, object]] = {}
    max_chars = int(dataset_window_rule(dataset)["max_chars"])
    for noise_value in noise_levels:
        noise_key = old_runner()._noise_key(float(noise_value))
        if float(noise_value) == 0.0:
            noisy_alert = selected_alert
            noisy_context = context_text
        else:
            noisy_alert = old_runner()._inject_noise_line(
                legacy,
                selected_alert,
                dataset,
                float(noise_value),
                role="selected_alert",
            )
            noisy_alert = sanitize_noisy_alert(
                dataset,
                selected_alert,
                noisy_alert,
                gt_action_id,
                float(noise_value),
            )
            noisy_context_raw = old_runner()._inject_noise_preserve_context(
                legacy,
                context_text,
                dataset,
                injector,
                injector_hadoop,
                float(noise_value),
            )
            noisy_context = compact_lines(
                [line for line in str(noisy_context_raw or "").splitlines() if line.strip()],
                max_chars,
            )
        views[noise_key] = {
            "noise": float(noise_value),
            "selected_alert": noisy_alert,
            "context_text": noisy_context,
            "rag_references": list(references),
            "agent_references": list(references),
            "graph_evidence": dict(graph_evidence),
        }
    return views


def _contains_negative_resolution(text: str) -> bool:
    lowered = norm(text)
    return any(pattern in lowered for pattern in NEGATIVE_REPAIR_PATTERNS)


def _required_action_groups(dataset: str, action_id: str) -> int:
    base = int(ACTION_CATALOG.get(dataset, {}).get(action_id, {}).get("min_groups", 0) or 0)
    if dataset == "HDFS" and action_id == "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE":
        return max(base, 2)
    return base


def _best_semantic_action(dataset: str, diagnosis: str, repair_action: str) -> str:
    combined = "\n".join(part for part in (diagnosis, repair_action) if str(part or "").strip()).strip()
    if not combined or _contains_negative_resolution(combined):
        return ""
    ranked: List[tuple[float, int, int, str]] = []
    for action_id, meta in ACTION_CATALOG.get(dataset, {}).items():
        matched, _, _ = action_text_match(dataset, action_id, combined)
        required = _required_action_groups(dataset, action_id)
        if required <= 0 or matched < required:
            continue
        group_count = max(1, len(meta.get("keyword_groups", [])))
        ranked.append((matched / group_count, matched, required, action_id))
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return ranked[0][3] if ranked else ""


def _best_semantic_family(dataset: str, diagnosis: str, repair_action: str) -> str:
    combined = norm("\n".join(part for part in (diagnosis, repair_action) if str(part or "").strip()))
    if not combined or _contains_negative_resolution(combined):
        return ""
    family_catalog = SEMANTIC_FAMILY_KEYWORDS.get(dataset, {})
    ranked: List[tuple[int, str]] = []
    for family_id, keywords in family_catalog.items():
        score = sum(1 for keyword in keywords if keyword in combined)
        if score > 0:
            ranked.append((score, family_id))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return ranked[0][1] if ranked else ""


def _local_hadoop_action_override(text: str) -> str:
    lowered = norm(text)
    if not lowered:
        return ""
    if any(token in lowered for token in ("resourcemanager channel", "resource manager channel", "rmcommunicator", ":8030", "container allocator")):
        return "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
    if ("resource manager" in lowered or "resourcemanager" in lowered) and any(
        token in lowered
        for token in (
            "connection",
            "service",
            "transport",
            "contacting rm",
            "channel",
        )
    ):
        return "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
    if any(
        token in lowered
        for token in (
            "worker rpc",
            "peer endpoint",
            "remote host forcibly closing",
            "peer-side transport",
            "transport session",
            "channel reader",
            "control-plane session",
            "connection cycling",
            "retrying worker rpc toward node",
        )
    ):
        return "HADOOP_RESTORE_WORKER_RPC_AND_RETRY"
    if any(
        token in lowered
        for token in (
            "cleanup failed for distributed output path",
            "distributed output path",
            "cleanup failed",
            "temporary output path",
        )
    ):
        return "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE"
    if any(
        token in lowered
        for token in (
            "shuffle fragment",
            "fallback merge staging",
            "single-fragment staging threshold",
            "shuffle pressure",
            "redirecting shuffle fragment",
        )
    ):
        return "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY"
    return ""


def _local_action_override(dataset: str, text: str) -> str:
    if dataset == "Hadoop":
        return _local_hadoop_action_override(text)
    return ""


def infer_from_open_text(dataset: str, diagnosis: str, repair_action: str, raw_response: str = "") -> Dict[str, str]:
    combined = "\n".join(part for part in (diagnosis, repair_action) if str(part or "").strip()).strip()
    if not combined or _contains_negative_resolution(combined):
        return {
            "pred_family_id": "",
            "pred_action_id": "",
        }
    pred_action = _local_action_override(dataset, combined)
    if not pred_action:
        pred_action = _best_semantic_action(dataset, diagnosis, repair_action)
    if not pred_action:
        pred_action = infer_action_id_from_text(dataset, combined)
    pred_family = family_for_action(dataset, pred_action)
    if not pred_family:
        pred_family = _best_semantic_family(dataset, diagnosis, repair_action)
    if not pred_family:
        pred_family = infer_family_from_text(dataset, combined)
    return {
        "pred_family_id": pred_family,
        "pred_action_id": pred_action,
    }


def action_text_ok(dataset: str, gt_action_id: str, repair_action: str) -> bool:
    text = str(repair_action or "").strip()
    if not text or _contains_negative_resolution(text):
        return False
    matched, _, _ = action_text_match(dataset, gt_action_id, text)
    required = _required_action_groups(dataset, gt_action_id)
    return matched >= required and required > 0
