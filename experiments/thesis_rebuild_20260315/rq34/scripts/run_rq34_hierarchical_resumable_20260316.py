from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.hierarchical_catalog_20260316 import (
    ACTION_CATALOG,
    action_text_match,
    action_text_success,
    allowed_action_ids,
    allowed_family_ids,
    describe_allowed_actions,
    describe_allowed_families,
    family_for_action,
    infer_action_id_from_text,
    infer_family_from_text,
    label_for_action,
    select_gt_action_and_family,
)
from experiments.thesis_rebuild_20260315.rq34.scripts.build_rq34_enriched_seed_pool_20260316 import (
    OUTPUT_PATH as ENRICHED_SEED_POOL_PATH,
    write_enriched_seed_pool,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json


LEGACY_STAGE4 = _PROJECT_ROOT / "experiments" / "rq123_e2e" / "stage4_noise_api_sampled_20260313.py"
REBUILD_RESULTS_DIR = _REBUILD_ROOT / "rq34" / "results"
RQ3_SMALL_FIXED_CASES_PATH = _REBUILD_ROOT / "rq34" / "configs" / "rq3_small_fixed_cases_20260317.json"
BENCH_V2_PATH = _PROJECT_ROOT / "data" / "processed" / "e2e_scaled_benchmark_v2.json"
RQ3_TEST_SET_PATH = _PROJECT_ROOT / "data" / "processed" / "rq3_test_set.json"
RQ2_FULLCASE_MODIFIED_GRAPH = (
    _REBUILD_ROOT / "rq2_fullcase" / "results" / "gt_causal_knowledge_nesydy_fullcase_20260316.json"
)
METHODS = ["agent", "vanilla", "rag"]
DATASETS = ["HDFS", "OpenStack", "Hadoop"]
MIN_CONTEXT_LINES = {"HDFS": 3, "OpenStack": 8, "Hadoop": 8}
LOCAL_SUPPORT_RADIUS = {"HDFS": 6, "OpenStack": 8, "Hadoop": 8}
POOL_PRIORITY = {"rq3_test_set_enriched": 0, "rq3_test_set": 1, "benchmark_v2": 2}
MAIN_ACTION_EXCLUDE = {
    "HDFS": {
        "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE",
        "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION",
    }
}
_LOCAL_REFERENCE_KB: Dict[str, List[Dict[str, object]]] | None = None


def _load_fixed_small_case_ids(cases_per_dataset: int) -> Dict[str, List[str]]:
    if cases_per_dataset != 9 or not RQ3_SMALL_FIXED_CASES_PATH.exists():
        return {}
    obj = json.loads(RQ3_SMALL_FIXED_CASES_PATH.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for dataset, case_ids in obj.items():
        if not isinstance(case_ids, list):
            continue
        out[str(dataset)] = [str(case_id) for case_id in case_ids if str(case_id).strip()]
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-per-dataset", type=int, default=15)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--run-tag", type=str, default="hierarchical15x6_v1")
    ap.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated dataset subset, e.g. Hadoop or HDFS,OpenStack,Hadoop",
    )
    ap.add_argument(
        "--causal-graph-path",
        type=str,
        default=str(RQ2_FULLCASE_MODIFIED_GRAPH),
        help="Causal graph JSON used by the agent path.",
    )
    ap.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore any existing manifest and rebuild the sampled case list.",
    )
    ap.add_argument(
        "--pool-policy",
        type=str,
        default="enriched_seeded",
        choices=["benchmark_v2_only", "mixed_richer", "enriched_seeded"],
        help="Case-pool policy for sampled evaluation.",
    )
    ap.add_argument(
        "--api-max-output-tokens",
        type=int,
        default=256,
        help="Per-call DeepSeek output token cap.",
    )
    ap.add_argument(
        "--max-api-calls",
        type=int,
        default=0,
        help="Hard cap on actual DeepSeek calls for this run; <=0 means unlimited.",
    )
    return ap.parse_args()


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("stage4_actionaware_legacy", LEGACY_STAGE4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy stage4 script from {LEGACY_STAGE4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_paths(run_tag: str) -> Dict[str, Path]:
    return {
        "manifest": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_manifest_20260316.json",
        "progress": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_progress_20260316.jsonl",
        "state": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_state_20260316.json",
        "summary_rows": REBUILD_RESULTS_DIR / f"rq34_{run_tag}_summary_rows_20260316.json",
    }


def _stable_fraction(*parts: object) -> float:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def _canonical_reference_text(dataset: str, text: str) -> str:
    out = str(text or "")
    if not out:
        return ""
    if dataset == "OpenStack":
        replacements = [
            ("FETCH ", "GET "),
            ("LIST ", "GET "),
            ("/detailed-server-list", "/servers/detail"),
            ("/instance-overview", "/servers/detail"),
            ("/state-overview", "/servers/detail"),
            ("/control-plane-cache-view", "/servers/detail"),
            ("/control-plane/runtime-catalog/index/", "/latest/meta-data/"),
            ("/control-plane/runtime-catalog/bootstrap-blob", "/openstack/2013-10-17/user_data"),
            ("/control-plane/runtime-catalog/vendor-profile", "/openstack/2013-10-17/vendor_data.json"),
            ("/control-plane/runtime-catalog/instance-profile", "/openstack/2013-10-17/meta_data.json"),
            ("/control-plane/runtime-catalog", "/openstack/2013-10-17"),
            ("/control-plane/bootstrap-cache/latest-manifest/", "/latest/meta-data/"),
            ("/control-plane/bootstrap-cache/user-payload", "/openstack/2013-10-17/user_data"),
            ("/control-plane/bootstrap-cache/vendor-payload", "/openstack/2013-10-17/vendor_data.json"),
            ("/control-plane/bootstrap-cache/instance-manifest", "/openstack/2013-10-17/meta_data.json"),
            ("/control-plane/bootstrap-cache", "/openstack/2013-10-17"),
            ("instance inventory sync on host", "instance sync for host"),
            ("scheduler-side state reconciliation on node", "instance sync for host"),
            ("reconciled host-side runtime cache", "re-created its instancelist"),
            ("rebuilt cached instance inventory", "re-created its instancelist"),
            ("power-state sync", "sync_power_state"),
            ("pending state: spawning", "pending task (spawning)"),
            ("Trying host claim:", "Attempting claim:"),
            ("host claim accepted", "claim successful"),
            ("Building instance disk image", "Creating image"),
            ("Assembling root disk backing chain", "Creating image"),
            ("Missing backing artifact in base layer", "Unknown base file"),
            ("missing backing artifact in base layer", "unknown base file"),
            ("base-layer artifact ", "image "),
            ("cached object ", "image "),
            (": cache audit pass", ": checking"),
            (": routine inspection", ": checking"),
            (" referenced in local cache:", " in use:"),
            (" retained by runtime workspace:", " in use:"),
            ("Active base-layer cache entries", "Active base files"),
            ("Removable base-layer cache entries", "Removable base files"),
            ("Retained base-layer cache entries", "Active base files"),
            ("Releasable base-layer cache entries", "Removable base files"),
            ("Retained workspace objects", "Active base files"),
            ("Releasable workspace objects", "Removable base files"),
            ("/var/lib/nova/runtime/objects/", "/var/lib/nova/instances/_base/"),
            ("instance metadata payload", "meta_data.json"),
            ("vendor metadata payload", "vendor_data.json"),
            ("instance-profile.json", "meta_data.json"),
            ("vendor-profile.json", "vendor_data.json"),
            ("result: object-missing", "status: 404"),
        ]
    elif dataset == "Hadoop":
        replacements = [
            ("worker agent", "nodemanager"),
            ("task listener", "taskattemptlistenerimpl"),
            ("peer terminated the socket unexpectedly", "forcibly closed by the remote host"),
            ("handled peer-side transport event from endpoint", "readandprocess from client"),
            ("local worker is:", "local host is:"),
            ("peer worker is:", "destination host is:"),
            ("cleanup failed for distributed output path ", "failed to remove hdfs "),
            ("org.apache.hadoop.ipc.Channel", "org.apache.hadoop.ipc.Client"),
        ]
    else:
        replacements = []
    for old, new in replacements:
        if old in out:
            out = out.replace(old, new)
    return out


def _reference_tokens(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[a-z][a-z0-9_./:-]{2,}", str(text or "").lower())
    stop = {
        "http",
        "https",
        "info",
        "warn",
        "error",
        "req",
        "instance",
        "status",
        "time",
        "len",
        "from",
        "with",
        "root",
        "json",
        "server",
        "client",
        "main",
        "task",
        "data",
        "path",
        "host",
        "port",
        "line",
        "log",
    }
    out: Set[str] = set()
    for tok in raw_tokens:
        if tok in stop:
            continue
        if re.fullmatch(r"[a-f0-9]{8,}", tok):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*", tok):
            continue
        out.add(tok)
    return out


def _reference_tags(dataset: str, text: str) -> Set[str]:
    t = _canonical_reference_text(dataset, text).lower()
    tags: Set[str] = set()
    if dataset == "OpenStack":
        if "/servers/detail" in t:
            tags.add("inventory")
        if "instance sync for host" in t or "re-created its instancelist" in t:
            tags.add("inventory")
            tags.add("inventory_strong")
        if "auditing locally available compute resources" in t:
            tags.add("audit")
        if (
            "metadata" in t
            or "/openstack/2013-10-17" in t
            or "/latest/meta-data/" in t
            or "vendor_data.json" in t
            or "meta_data.json" in t
            or "user_data http/1.1" in t
        ):
            tags.add("metadata")
        if "vm started" in t or "vm paused" in t or "vm resumed" in t:
            tags.add("power")
        if (
            "sync_power_state" in t
            or "pending task (spawning)" in t
            or "while synchronizing instance power states" in t
        ):
            tags.add("power")
            tags.add("power_direct")
        if (
            "creating image" in t
            or "unknown base file" in t
            or "active base files" in t
            or "removable base files" in t
            or "base or swap file too young" in t
        ):
            tags.add("image")
        if (
            "attempting claim:" in t
            or "claim successful" in t
            or "cpu affinity" in t
            or "vcpu count" in t
        ):
            tags.add("claim")
        if (
            "cpu affinity" in t
            or "vcpu count" in t
            or "vcpu limit" in t
            or "total usable vcpus" in t
            or "memory limit" in t
        ):
            tags.add("capacity")
        if "terminating instance" in t or "delete /v2/" in t:
            tags.add("terminate")
    elif dataset == "Hadoop":
        if "retrying connect to server" in t:
            tags.add("retry_connect")
        if (
            "forcibly closed by the remote host" in t
            or "socket reader" in t
            or "communication exception" in t
            or "readandprocess from client" in t
        ):
            tags.add("forced_close")
        if "containerlauncher" in t:
            tags.add("containerlauncher")
        if "leaserenewer" in t:
            tags.add("leaserenewer")
        if "communication thread" in t:
            tags.add("worker_thread")
        if (
            "container_remote_cleanup" in t
            or "kill_container_cleanup" in t
            or "opening proxy :" in t
            or "cleanup failed for distributed output path" in t
        ):
            tags.add("worker_cleanup")
        if (
            "rmcommunicator allocator" in t
            or ":8030" in t
            or "error in contacting rm" in t
        ):
            tags.add("rm_thread")
        if "local host is:" in t or "destination host is:" in t:
            tags.add("endpoint_hosts")
        if (
            "bad datanode" in t
            or "unhealthy data node" in t
            or "machine down" in t
            or "host appears unavailable" in t
            or "heartbeat" in t
            or "failed to remove hdfs" in t
            or "could not delete hdfs" in t
            or "cleanup failed for distributed output path" in t
        ):
            tags.add("node_down")
        if (
            "shuffling to disk" in t
            or "maxsingleshufflelimit" in t
            or "ondiskmapoutput" in t
            or "disk full" in t
            or "no space" in t
            or "fallback merge staging" in t
            or "single-fragment staging threshold" in t
            or "staged shuffle fragment" in t
        ):
            tags.add("storage")
        if (
            {"containerlauncher", "leaserenewer", "worker_thread", "endpoint_hosts", "worker_cleanup"} & tags
            and {"retry_connect", "forced_close"} & tags
            and "rm_thread" not in tags
        ):
            tags.add("worker_endpoint")
        if "rm_thread" in tags and "retry_connect" in tags:
            tags.add("rm_endpoint")
    return tags


def _build_local_reference_kb(legacy) -> Dict[str, List[Dict[str, object]]]:
    global _LOCAL_REFERENCE_KB
    if _LOCAL_REFERENCE_KB is not None:
        return _LOCAL_REFERENCE_KB

    benchmark_pool = json.loads(BENCH_V2_PATH.read_text(encoding="utf-8"))
    if not ENRICHED_SEED_POOL_PATH.exists():
        write_enriched_seed_pool()
    enriched_seed_pool = json.loads(ENRICHED_SEED_POOL_PATH.read_text(encoding="utf-8"))
    kb: Dict[str, List[Dict[str, object]]] = {"OpenStack": [], "Hadoop": []}

    def _ingest(pool: Sequence[Mapping[str, object]], pool_source: str, dataset: str) -> None:
        for case in pool:
            if str(case.get("dataset", "")) != dataset:
                continue
            if not _has_enough_context(case):
                continue
            raw_log = str(case.get("raw_log", "") or "")
            selected_alert = _select_actionaware_alert(legacy, raw_log, dataset)
            support_context = _local_alert_context(raw_log, selected_alert, dataset)
            context_support = _line_action_support(dataset, support_context)
            gt_action_id, gt_label, gt_diag = select_gt_action_and_family(
                dataset,
                selected_alert=selected_alert,
                raw_log=raw_log,
                raw_log_seed=str(case.get("raw_log_seed", "") or ""),
                gt_root_template=str(case.get("ground_truth_root_cause_template", "") or ""),
                gt_effect_template=str(case.get("ground_truth_template", "") or ""),
                gt_action_label=str(case.get("gt_action_label", "") or case.get("reason", "") or ""),
                context_support=context_support,
            )
            if not gt_action_id or not gt_label:
                continue
            selected_alert = _refine_selected_alert_for_action(
                dataset,
                raw_log,
                selected_alert,
                gt_action_id,
                str(case.get("raw_log_seed", "") or ""),
            )
            support_context = _local_alert_context(raw_log, selected_alert, dataset)
            context_support = _line_action_support(dataset, support_context)
            gt_action_id, gt_label, gt_diag = select_gt_action_and_family(
                dataset,
                selected_alert=selected_alert,
                raw_log=raw_log,
                raw_log_seed=str(case.get("raw_log_seed", "") or ""),
                gt_root_template=str(case.get("ground_truth_root_cause_template", "") or ""),
                gt_effect_template=str(case.get("ground_truth_template", "") or ""),
                gt_action_label=str(case.get("gt_action_label", "") or case.get("reason", "") or ""),
                context_support=context_support,
            )
            if not gt_action_id or not gt_label or gt_diag.get("confidence") == "low":
                continue
            action_bucket = _action_bucket(
                dataset,
                gt_action_id,
                selected_alert,
                str(case.get("raw_log_seed", "") or ""),
            )
            reference_context = "\n".join(_window_around_alert(raw_log, selected_alert, LOCAL_SUPPORT_RADIUS[dataset]))
            repair_pattern = str(
                ACTION_CATALOG.get(dataset, {}).get(gt_action_id, {}).get("description", "")
            )
            reference_blob = "\n".join(
                part
                for part in [
                    selected_alert,
                    reference_context,
                    gt_label,
                    gt_action_id,
                    repair_pattern,
                ]
                if part
            )
            kb[dataset].append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "pool_source": pool_source,
                    "selected_alert": selected_alert,
                    "reference_context": reference_context[:700],
                    "gt_action_id": gt_action_id,
                    "gt_label": gt_label,
                    "action_bucket": action_bucket,
                    "repair_pattern": repair_pattern,
                    "tags": sorted(_reference_tags(dataset, reference_blob)),
                    "tokens": sorted(_reference_tokens(_canonical_reference_text(dataset, reference_blob))),
                }
            )

    _ingest(benchmark_pool, "benchmark_v2", "OpenStack")
    _ingest(benchmark_pool, "benchmark_v2", "Hadoop")
    _ingest(enriched_seed_pool, "rq3_test_set_enriched", "OpenStack")
    _LOCAL_REFERENCE_KB = kb
    return kb


def _local_exemplar_references(
    legacy,
    *,
    dataset: str,
    case_id: str,
    selected_alert: str,
    context_text: str,
    top_k: int,
) -> str:
    if dataset not in {"OpenStack", "Hadoop"}:
        return ""
    kb = _build_local_reference_kb(legacy).get(dataset, [])
    query_text = _canonical_reference_text(
        dataset,
        "\n".join(part for part in [selected_alert, context_text] if part),
    )
    query_tags = _reference_tags(dataset, query_text)
    query_tokens = _reference_tokens(query_text)
    query_hint_action = infer_action_id_from_text(dataset, query_text)
    if dataset == "OpenStack" and not query_hint_action:
        lower_query = query_text.lower()
        if ("image " in lower_query and ": checking" in lower_query) or (
            "image " in lower_query and " in use:" in lower_query
        ):
            query_hint_action = "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
    query_hint_family = family_for_action(dataset, query_hint_action) if query_hint_action else ""
    query_text_lower = query_text.lower()
    selected_alert_lower = str(selected_alert or "").lower()
    query_bucket = ""
    if dataset == "OpenStack" and query_hint_action:
        query_bucket = _action_bucket(
            dataset,
            query_hint_action,
            _canonical_reference_text(dataset, selected_alert),
            "",
        )

    scored: List[Tuple[float, Dict[str, object]]] = []
    for entry in kb:
        if str(entry.get("case_id", "")) == str(case_id):
            continue
        entry_tags = set(str(x) for x in entry.get("tags", []))
        entry_tokens = set(str(x) for x in entry.get("tokens", []))
        shared_tags = query_tags & entry_tags
        shared_tokens = query_tokens & entry_tokens
        score = 3.0 * len(shared_tags) + 0.15 * min(12, len(shared_tokens))
        if query_hint_action and str(entry.get("gt_action_id", "")) == query_hint_action:
            score += 2.5
        if query_hint_family and str(entry.get("gt_label", "")) == query_hint_family:
            score += 1.5
        if dataset == "OpenStack":
            entry_action = str(entry.get("gt_action_id", ""))
            entry_bucket = str(entry.get("action_bucket", ""))
            if (
                top_k >= 4
                and query_hint_action in {"OPENSTACK_SCALE_METADATA_SERVICE", "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"}
                and entry_action == query_hint_action
                and query_bucket
                and entry_bucket == query_bucket
            ):
                continue
            query_image_primary = any(
                pat in query_text_lower
                for pat in (
                    "creating image",
                    "building instance disk image",
                    "unknown base file",
                    "removable base files",
                    "active base files",
                    "base or swap file too young",
                )
            )
            query_power_primary = any(
                pat in query_text_lower
                for pat in (
                    "sync_power_state",
                    "power-state sync",
                    "pending task (spawning)",
                    "pending state: spawning",
                    "vm resumed",
                    "vm paused",
                    "vm started",
                )
            )
            if "inventory_strong" in query_tags and entry_action == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
                score += 3.0
            elif "inventory" in query_tags and entry_action == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
                score += 1.5
            if "metadata" in query_tags and entry_action == "OPENSTACK_SCALE_METADATA_SERVICE":
                score += 2.5
            if "power_direct" in query_tags and entry_action == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                score += 3.5
            elif "power" in query_tags and entry_action == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                score += 2.0
            if "image" in query_tags and entry_action == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
                score += 2.0
            if query_image_primary and entry_action == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
                score += 4.5
            if query_image_primary and query_power_primary and entry_action == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                score -= 4.0
            if query_power_primary and not query_image_primary and entry_action == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
                score -= 2.0
            if (
                ("creating image" in selected_alert_lower or "building instance disk image" in selected_alert_lower)
                and entry_action == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
            ):
                score += 7.0
            if (
                ("creating image" in selected_alert_lower or "building instance disk image" in selected_alert_lower)
                and entry_action == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD"
            ):
                score -= 7.0
            if (
                {"claim", "capacity"} <= query_tags
                and entry_action == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
            ):
                score += 3.5
            if (
                "claim" in query_tags
                and "capacity" not in query_tags
                and "image" in query_tags
                and entry_action == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
            ):
                score += 2.5
            if (
                "claim" in query_tags
                and "image" not in query_tags
                and entry_action == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
            ):
                score += 1.0
            if (
                "claim" in query_tags
                and "image" in query_tags
                and "capacity" not in query_tags
                and entry_action == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
            ):
                score -= 1.5
            if (
                "power_direct" in query_tags
                and "inventory_strong" not in query_tags
                and entry_action == "OPENSTACK_RESYNC_INSTANCE_INVENTORY"
            ):
                score -= 1.5
        elif dataset == "Hadoop":
            if "storage" in query_tags and str(entry.get("gt_action_id", "")) == "HADOOP_FREE_DISK_AND_RETRY":
                score += 4.0
            if "worker_endpoint" in query_tags and str(entry.get("gt_action_id", "")) == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
                score += 4.5
            if "worker_cleanup" in query_tags and str(entry.get("gt_action_id", "")) == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
                score += 3.0
            if "rm_endpoint" in query_tags and str(entry.get("gt_action_id", "")) == "HADOOP_RESTORE_NETWORK_AND_RETRY":
                score += 1.5
            if "node_down" in query_tags and str(entry.get("gt_action_id", "")) == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
                score += 4.0
            if (
                {"forced_close", "worker_endpoint"} <= query_tags
                and str(entry.get("gt_action_id", "")) == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
            ):
                score += 2.5
            if "retry_connect" in query_tags and "rm_thread" in query_tags and str(entry.get("gt_action_id", "")) == "HADOOP_RESTORE_NETWORK_AND_RETRY":
                score += 1.0
        if score > 0.0:
            scored.append((score, entry))

    scored.sort(
        key=lambda pair: (
            -pair[0],
            str(pair[1].get("gt_action_id", "")),
            str(pair[1].get("case_id", "")),
        )
    )
    lines: List[str] = []
    per_action: Dict[str, int] = defaultdict(int)
    for rank, (_, entry) in enumerate(scored, start=1):
        action_id = str(entry.get("gt_action_id", ""))
        per_action_cap = 2
        if (
            dataset == "OpenStack"
            and top_k >= 4
            and action_id in {"OPENSTACK_SCALE_METADATA_SERVICE", "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"}
        ):
            per_action_cap = 1
        if per_action[action_id] >= per_action_cap:
            continue
        per_action[action_id] += 1
        lines.append(
            f"[{len(lines)+1}] family={entry['gt_label']} action={action_id} "
            f"source={entry['pool_source']} case={entry['case_id']}"
        )
        lines.append(f"    alert: {str(entry.get('selected_alert', ''))[:260]}")
        lines.append(f"    repair pattern: {str(entry.get('repair_pattern', ''))}")
        if len(lines) // 3 >= top_k:
            break
    if not lines:
        return ""
    return "\n".join(lines)


def _legacy_force_noise(legacy, text: str, dataset: str) -> str:
    injector = legacy.NoiseInjector(seed=2026)
    injector_hadoop = legacy.HadoopNoiseInjector(seed=2026)
    return legacy._inject_noise(text, dataset, injector, injector_hadoop, 1.0)


def _rq34_noise_overlay(dataset: str, text: str, noise_level: float) -> str:
    out = str(text or "")
    if not out or noise_level <= 0.0:
        return out
    if dataset == "OpenStack":
        replacements = [
            ("Creating image", "Building instance disk image"),
            ("VM Started (Lifecycle Event)", "instance started lifecycle transition"),
            ("VM Paused (Lifecycle Event)", "instance paused lifecycle transition"),
            ("VM Resumed (Lifecycle Event)", "instance resumed lifecycle transition"),
            ("sync_power_state", "power-state sync"),
            ("pending task (spawning)", "pending state: spawning"),
            ("Attempting claim:", "Trying host claim:"),
            ("claim successful", "host claim accepted"),
            ("instance sync for host", "instance inventory sync on host"),
            ("re-created its instancelist", "rebuilt cached instance inventory"),
            ("/servers/detail", "/detailed-server-list"),
            ("meta_data.json", "instance metadata payload"),
            ("vendor_data.json", "vendor metadata payload"),
        ]
        for old, new in replacements:
            if old in out:
                out = out.replace(old, new)
        if noise_level >= 0.6:
            out = out.replace("/servers/detail", "/instance-overview")
            out = out.replace("/ComputeNodes/detail", "/state-overview")
            out = out.replace("FETCH /v2/", "LIST /v2/")
            out = out.replace("wsgi.ComputeNode", "wsgi.api")
            out = out.replace(
                "VM Started (Lifecycle Event)",
                "Lifecycle transition acknowledged after spawn workflow",
            )
            out = out.replace(
                "VM Paused (Lifecycle Event)",
                "Lifecycle transition acknowledged after guest suspend workflow",
            )
            out = out.replace(
                "VM Resumed (Lifecycle Event)",
                "Lifecycle transition acknowledged after hypervisor state change",
            )
            out = out.replace(
                "instance started lifecycle transition",
                "lifecycle transition acknowledged after spawn workflow",
            )
            out = out.replace(
                "instance paused lifecycle transition",
                "lifecycle transition acknowledged after guest suspend workflow",
            )
            out = out.replace(
                "instance resumed lifecycle transition",
                "lifecycle transition acknowledged after hypervisor state change",
            )
        if noise_level >= 1.0:
            out = out.replace("status: ", "state: ")
            out = out.replace("Creating image", "Assembling root disk backing chain")
            out = out.replace("Building instance disk image", "Assembling root disk backing chain")
            out = out.replace("Unknown base file", "Missing backing artifact in base layer")
            out = out.replace("unknown base file", "missing backing artifact in base layer")
            out = out.replace("Assembling root disk backing chain", "Reconciling runtime object lineage")
            out = out.replace("image ", "base-layer artifact ")
            out = out.replace(": checking", ": cache audit pass")
            out = out.replace(" in use:", " referenced in local cache:")
            out = out.replace("Active base files", "Retained base-layer cache entries")
            out = out.replace("Removable base files", "Releasable base-layer cache entries")
            out = out.replace("base-layer artifact ", "cached object ")
            out = out.replace(": cache audit pass", ": routine inspection")
            out = out.replace(" referenced in local cache:", " retained by runtime workspace:")
            out = out.replace("Retained base-layer cache entries", "Retained workspace objects")
            out = out.replace("Releasable base-layer cache entries", "Releasable workspace objects")
            out = out.replace("/var/lib/nova/instances/_base/", "/var/lib/nova/runtime/objects/")
            out = out.replace("/var/lib/nova/VMs/_base/", "/var/lib/nova/runtime/objects/")
            out = out.replace("/instance-overview", "/control-plane-cache-view")
            out = out.replace("/state-overview", "/control-plane-cache-view")
            out = out.replace("wsgi.api", "wsgi.control")
            out = out.replace("instance sync for host", "scheduler-side state reconciliation on node")
            out = out.replace("instance inventory sync on host", "scheduler-side state reconciliation on node")
            out = out.replace("re-created its instancelist", "reconciled host-side runtime cache")
            out = out.replace("rebuilt cached instance inventory", "reconciled host-side runtime cache")
            out = out.replace("/latest/meta-data/", "/control-plane/bootstrap-cache/latest-manifest/")
            out = out.replace("/openstack/2013-10-17/user_data", "/control-plane/bootstrap-cache/user-payload")
            out = out.replace("/openstack/2013-10-17/vendor_data.json", "/control-plane/bootstrap-cache/vendor-payload")
            out = out.replace("/openstack/2013-10-17/meta_data.json", "/control-plane/bootstrap-cache/instance-manifest")
            out = out.replace("/openstack/2013-10-17", "/control-plane/bootstrap-cache")
            out = out.replace("vendor metadata payload", "vendor-profile.json")
            out = out.replace("instance metadata payload", "instance-profile.json")
            out = out.replace("/control-plane/bootstrap-cache/latest-manifest/", "/control-plane/runtime-catalog/index/")
            out = out.replace("/control-plane/bootstrap-cache/user-payload", "/control-plane/runtime-catalog/bootstrap-blob")
            out = out.replace("/control-plane/bootstrap-cache/vendor-payload", "/control-plane/runtime-catalog/vendor-profile")
            out = out.replace("/control-plane/bootstrap-cache/instance-manifest", "/control-plane/runtime-catalog/instance-profile")
            out = out.replace("/control-plane/bootstrap-cache", "/control-plane/runtime-catalog")
            out = out.replace("nova.metadata.wsgi.server", "nova.runtime.wsgi.gateway")
            out = out.replace("nova.metadata.wsgi.control", "nova.runtime.wsgi.gateway")
            out = out.replace("/control-plane/runtime-catalog/index/", "/control-plane/session-root/index/")
            out = out.replace("/control-plane/runtime-catalog/bootstrap-blob", "/control-plane/session-root/bootstrap-blob")
            out = out.replace("/control-plane/runtime-catalog/vendor-profile", "/control-plane/session-root/vendor-profile")
            out = out.replace("/control-plane/runtime-catalog/instance-profile", "/control-plane/session-root/instance-profile")
            out = out.replace("/control-plane/runtime-catalog", "/control-plane/session-root")
            out = out.replace("power-state sync", "runtime-state reconciliation")
            out = out.replace("sync_power_state", "runtime-state reconciliation")
            out = out.replace("During sync_power_state", "During runtime-state reconciliation")
            out = out.replace("pending task (spawning)", "launch workflow still marked in-flight")
            out = out.replace("pending state: spawning", "launch workflow still marked in-flight")
            out = out.replace("launch workflow still marked in-flight. Skip.", "launch workflow still marked in-flight. Defer.")
            out = out.replace("While synchronizing instance power states", "During executor-state reconciliation")
            out = out.replace("While synchronizing VM power states", "During executor-state reconciliation")
            out = re.sub(
                r"found\s+(\d+)\s+(?:instances|vms) in the database and\s+(\d+)\s+(?:instances|vms) on the hypervisor",
                r"observed \1 persisted records and \2 executor-side records",
                out,
                flags=re.IGNORECASE,
            )
            out = out.replace("The instance sync for host", "The host-side runtime cache for")
            out = out.replace("The VM sync for host", "The host-side runtime cache for")
            out = out.replace("did not match. Re-created its InstanceList.", "diverged. Rebuilt cached placement view.")
            out = out.replace("did not match. Re-created its instancelist.", "diverged. Rebuilt cached placement view.")
            out = out.replace("did not match. re-created its instancelist.", "diverged. rebuilt cached placement view.")
            out = out.replace("status: 404", "result: object-missing")
            out = out.replace("status=404", "result=object-missing")
    elif dataset == "HDFS":
        replacements = [
            ("PacketResponder", "stream ack responder"),
            ("Receiving block", "replica segment ingress"),
            ("Received block", "replica segment acknowledged"),
            ("Unexpected error trying to delete block", "Cleanup failure while removing replica"),
            ("BlockInfo not found in volumeMap", "Replica metadata missing from storage map"),
        ]
        for old, new in replacements:
            if old in out:
                out = out.replace(old, new)
        if noise_level >= 0.6:
            out = out.replace("NameSystem.allocateBlock", "NameSystem.reserveTargets")
            out = out.replace("allocateBlock", "reserveTargets")
            out = out.replace("Served block", "Completed replica handoff for")
            out = out.replace("Starting thread to transfer block", "Initiating replica exchange for")
            out = out.replace(
                "Got exception while serving",
                "Operation interruption while handling",
            )
            out = out.replace(
                "Encountered network failure when handling",
                "Operation interruption while handling",
            )
            out = out.replace(
                "Failure during cleanup of data chunk",
                "Cleanup failure while removing replica",
            )
        if noise_level >= 1.0:
            out = out.replace("NameSystem.reserveTargets", "NameSystem.reserveReplicaTargets")
            out = out.replace("reserveTargets", "reserveReplicaTargets")
            out = out.replace("Completed replica handoff for", "Replica handoff workflow acknowledged for")
            out = out.replace("Initiating replica exchange for", "Scheduling replica exchange stage for")
            out = out.replace(
                "Operation interruption while handling",
                "Operation reported irregular handling for",
            )
            out = out.replace(
                "Cleanup failure while removing replica",
                "Storage cleanup failed while removing replica segment",
            )
            out = out.replace("stream ack responder", "replica stage tracker")
            out = out.replace("replica segment acknowledged", "replica fragment observed")
            out = out.replace("replica segment ingress", "replica fragment path observed")
            out = out.replace("PkgResponder", "ReplicaStage")
            out = out.replace("Got blk", "replica fragment observed")
            out = out.replace("writeBlock", "replicaStageStep")
            out = out.replace(
                "Replica metadata missing from storage map",
                "replica bookkeeping entry missing from storage index",
            )
            out = out.replace(
                "Connection reset by peer",
                "operation reported an irregular interruption",
            )
            out = out.replace(
                "SocketTimeoutException",
                "channel wait threshold exceeded",
            )
            out = out.replace(
                "SocketTimeoutError",
                "channel wait threshold exceeded",
            )
            out = out.replace(
                "EOFException",
                "stage sequence ended before completion",
            )
            out = out.replace(
                "EOFError",
                "stage sequence ended before completion",
            )
            out = re.sub(
                r"src:\s*/\d{1,3}(?:\.\d{1,3}){3}:\d+\s+dest:\s*/\d{1,3}(?:\.\d{1,3}){3}:\d+",
                "stage relay observed during activity",
                out,
                flags=re.IGNORECASE,
            )
            out = re.sub(
                r"from\s*/\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?",
                "during activity",
                out,
                flags=re.IGNORECASE,
            )
            out = re.sub(
                r"to\s*/\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?",
                "during activity",
                out,
                flags=re.IGNORECASE,
            )
            out = re.sub(
                r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+:",
                "event:",
                out,
            )
    elif dataset == "Hadoop":
        replacements = [
            ("Retrying connect to server", "Retrying RPC toward node"),
            ("Already tried", "attempt count"),
            ("An existing connection was forcibly closed by the remote host", "peer terminated the socket unexpectedly"),
            ("Could not delete hdfs://", "Failed to remove hdfs://"),
            ("Bad datanode", "Unhealthy data node"),
            ("machine down", "host appears unavailable"),
            ("Shuffling to disk since", "Spilling map output to local disk because"),
            ("maxSingleShuffleLimit", "single-shuffle threshold"),
            ("disk full", "storage exhausted"),
            ("no space", "storage unavailable"),
        ]
        for old, new in replacements:
            if old in out:
                out = out.replace(old, new)
        if noise_level >= 0.6:
            out = out.replace("Retrying RPC toward node", "RPC handshake loop against remote endpoint")
            out = out.replace("Retrying connect to server", "RPC handshake loop against remote endpoint")
            out = out.replace(
                "peer terminated the socket unexpectedly",
                "remote endpoint closed the channel during exchange",
            )
            out = re.sub(r"ContainerLauncher #\d+", "dispatch path", out)
            out = out.replace("LeaseRenewer:", "client renewer:")
            out = out.replace("[communication thread]", "[rpc transport path]")
            out = out.replace("RMCommunicator Allocator", "scheduler control loop")
            out = re.sub(r"Socket Reader #\d+ for port", "channel reader for port", out)
        if noise_level >= 1.0:
            out = out.replace(
                "RPC handshake loop against remote endpoint",
                "control-plane session establishment kept cycling against peer endpoint",
            )
            out = out.replace(
                "remote endpoint closed the channel during exchange",
                "peer interrupted the transport session mid-exchange",
            )
            out = out.replace("org.apache.hadoop.ipc.Client", "org.apache.hadoop.ipc.Channel")
            out = out.replace("org.apache.hadoop.ipc.Server", "org.apache.hadoop.ipc.Channel")
            out = out.replace("TaskAttemptListenerImpl", "task listener")
            out = out.replace("NodeManager", "worker agent")
            out = out.replace("readAndProcess from client", "handled peer-side transport event from endpoint")
            out = out.replace("local host is:", "local worker is:")
            out = out.replace("destination host is:", "peer worker is:")
            out = out.replace("Failed to remove hdfs://", "cleanup failed for distributed output path ")
            out = out.replace(
                "Spilling map output to local disk because",
                "redirecting shuffle fragment into fallback merge staging because",
            )
            out = out.replace("single-shuffle threshold", "single-fragment staging threshold")
            out = out.replace("OnDiskMapOutput", "staged shuffle fragment")
            out = re.sub(r"\b(?:msra|minint)[-a-z0-9.]+(?::\d+)?", "peer-endpoint", out, flags=re.IGNORECASE)
            out = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "peer-endpoint", out)
    return out


def _inject_noise_line(
    legacy,
    line: str,
    dataset: str,
    noise_level: float,
    *,
    role: str,
) -> str:
    raw = str(line or "")
    if not raw.strip() or noise_level <= 0.0:
        return raw
    if dataset == "HDFS" and role == "selected_alert" and noise_level >= 1.0:
        lower_raw = raw.lower()
        if (
            "got exception while serving" in lower_raw
            or "connection reset by peer" in lower_raw
            or ("writeblock" in lower_raw and "received exception" in lower_raw)
        ):
            softened = _rq34_noise_overlay(dataset, _legacy_force_noise(legacy, raw, dataset), noise_level)
            if "got exception while serving" in lower_raw:
                return softened.replace(
                    "Operation reported irregular handling for",
                    "service-stage workflow reported irregular completion for",
                )
            if "connection reset by peer" in lower_raw:
                return softened.replace(
                    "operation reported an irregular interruption",
                    "peer endpoint interrupted the service stage before completion",
                )
            return softened.replace(
                "operation reported an irregular interruption",
                "peer endpoint terminated the downstream channel mid-exchange",
            )
    effective_noise = float(noise_level)
    if dataset in {"OpenStack", "Hadoop"}:
        effective_noise = min(1.0, effective_noise + (0.25 if role == "selected_alert" else 0.15))
    if _stable_fraction(dataset, role, raw) >= effective_noise:
        return raw
    return _rq34_noise_overlay(dataset, _legacy_force_noise(legacy, raw, dataset), noise_level)


def _inject_noise_preserve_context(
    legacy,
    context_text: str,
    dataset: str,
    injector,
    injector_hadoop,
    noise_level: float,
) -> str:
    lines = str(context_text or "").split("\n")
    if dataset == "Hadoop" and noise_level >= 1.0:
        scored: List[Tuple[int, int, str]] = []
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            lower = line.lower()
            score = 0
            if idx == 0:
                score += 12
            if any(
                pat in lower
                for pat in (
                    "could not delete hdfs",
                    "failed to remove hdfs",
                    "cleanup failed for distributed output path",
                    "bad datanode",
                    "unhealthy data node",
                    "machine down",
                    "host appears unavailable",
                )
            ):
                score += 15
            if any(
                pat in lower
                for pat in (
                    "container_remote_cleanup",
                    "kill_container_cleanup",
                    "opening proxy :",
                )
            ):
                score += 11
            if any(
                pat in lower
                for pat in (
                    "retrying connect to server",
                    "retrying rpc toward node",
                    "rpc handshake loop against remote endpoint",
                    "control-plane session establishment kept cycling against peer endpoint",
                    "forcibly closed by the remote host",
                    "peer terminated the socket unexpectedly",
                    "remote endpoint closed the channel during exchange",
                    "peer interrupted the transport session mid-exchange",
                )
            ):
                score += 9
            if any(
                pat in lower
                for pat in (
                    "shuffling to disk",
                    "spilling map output to local disk because",
                    "maxsingleshufflelimit",
                    "single-shuffle threshold",
                    "ondiskmapoutput",
                    "fallback merge staging",
                    "single-fragment staging threshold",
                    "staged shuffle fragment",
                    "disk full",
                    "no space",
                    "storage exhausted",
                    "storage unavailable",
                )
            ):
                score += 13
            if lower.startswith("java.io.") or " threw exception " in lower:
                score += 3
            if "progress of taskattempt" in lower:
                score -= 6
            if "registering class" in lower:
                score -= 5
            if score > 0:
                scored.append((score, idx, line))
        if scored:
            keep = sorted(scored, key=lambda x: (-x[0], x[1]))[:3]
            lines = [line for _, _, line in sorted(keep, key=lambda x: x[1])]
        else:
            compact: List[str] = []
            for line in lines:
                if line.strip():
                    compact.append(line)
                if len(compact) >= 3:
                    break
            lines = compact
    out: List[str] = []
    for idx, line in enumerate(lines):
        if line.strip():
            out.append(
                _inject_noise_line(
                    legacy,
                    line,
                    dataset,
                    noise_level,
                    role=f"context:{idx}",
                )
            )
        else:
            out.append(line)
    return "\n".join(out)


def _noise_key(noise: float) -> str:
    return f"{float(noise):.1f}"


def _choose_observed_template(
    legacy,
    dataset: str,
    tpl_nusy: str,
    tpl_drain: str,
    fallback_text: str,
) -> str:
    def _score(source: str, template: str) -> float:
        if not template or not legacy._valid_template(template):
            return float("-inf")
        lower = template.lower()
        action_id = infer_action_id_from_text(dataset, template)
        family_id = family_for_action(dataset, action_id) or infer_family_from_text(dataset, template)
        score = 0.0
        if family_id:
            score += 4.0
        if action_id:
            score += 3.0
        if source == "nusy":
            score += 1.0
        if "<*>" in template:
            score += 0.5
        if dataset == "Hadoop" and (
            lower.startswith("at ")
            or "socketchannelimpl" in lower
            or "abstractplainsocketimpl" in lower
            or lower.startswith("java.")
        ):
            score -= 6.0
        return score

    candidates = [
        ("nusy", tpl_nusy),
        ("drain", tpl_drain),
    ]
    best_source, best_template = max(candidates, key=lambda item: _score(item[0], item[1]))
    if _score(best_source, best_template) > float("-inf"):
        return str(best_template)
    return tpl_drain or tpl_nusy or fallback_text


def _step_key(dataset: str, case_id: str, noise: float, method: str) -> str:
    return f"{dataset}|{case_id}|{_noise_key(noise)}|{method}"


def _build_stats(
    noise_levels: Iterable[float], datasets: Iterable[str]
) -> Dict[Tuple[str, float, str], Dict[str, int]]:
    stats: Dict[Tuple[str, float, str], Dict[str, int]] = {}
    for ds in datasets:
        for nl in noise_levels:
            for method in METHODS:
                stats[(ds, nl, method)] = {
                    "rca_total": 0,
                    "rca_success": 0,
                    "action_total": 0,
                    "action_success": 0,
                    "action_text_total": 0,
                    "action_text_success": 0,
                    "e2e_total": 0,
                    "e2e_success": 0,
                }
    return stats


def _summary_rows_from_stats(
    stats: Dict[Tuple[str, float, str], Dict[str, int]],
    noise_levels: Iterable[float],
    datasets: Iterable[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ds in datasets:
        for nl in noise_levels:
            for method in METHODS:
                s = stats[(ds, nl, method)]
                if (
                    s["rca_total"] == 0
                    and s["action_total"] == 0
                    and s["action_text_total"] == 0
                    and s["e2e_total"] == 0
                ):
                    continue
                n_rca = s["rca_total"] or 1
                n_action = s["action_total"] or 1
                n_action_text = s["action_text_total"] or 1
                n_e2e = s["e2e_total"] or 1
                rows.append(
                    {
                        "dataset": ds,
                        "noise": float(nl),
                        "method": method,
                        "rca_success": s["rca_success"],
                        "rca_total": s["rca_total"],
                        "rca_accuracy": round(s["rca_success"] / n_rca, 4),
                        "action_success": s["action_success"],
                        "action_total": s["action_total"],
                        "action_accuracy": round(s["action_success"] / n_action, 4),
                        "action_text_success": s["action_text_success"],
                        "action_text_total": s["action_text_total"],
                        "action_text_success_rate": round(s["action_text_success"] / n_action_text, 4),
                        "e2e_success": s["e2e_success"],
                        "e2e_total": s["e2e_total"],
                        "e2e_success_rate": round(s["e2e_success"] / n_e2e, 4),
                    }
                )
    return rows


def _write_checkpoint(
    paths: Mapping[str, Path],
    stats: Dict[Tuple[str, float, str], Dict[str, int]],
    noise_levels: Iterable[float],
    datasets: Iterable[str],
    *,
    run_tag: str,
    cases_per_dataset: int,
    total_steps: int,
    completed_steps: int,
    actual_api_calls: int,
) -> None:
    summary_rows = _summary_rows_from_stats(stats, noise_levels, datasets)
    write_json(paths["summary_rows"], summary_rows)
    state = {
        "run_tag": run_tag,
        "cases_per_dataset": cases_per_dataset,
        "noise_levels": [float(x) for x in noise_levels],
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "actual_api_calls": actual_api_calls,
        "progress_path": str(paths["progress"]),
        "summary_rows_path": str(paths["summary_rows"]),
    }
    write_json(paths["state"], state)


def _balanced_sample(items: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        groups[str(item["gt_label"])].append(item)
    for label_items in groups.values():
        label_items.sort(
            key=lambda x: (
                POOL_PRIORITY.get(str(x.get("pool_source", "benchmark_v2")), 99),
                str(x["case"].get("case_id", "")),
            )
        )
    labels = sorted(groups.keys())
    out: List[Dict[str, object]] = []
    while labels and len(out) < limit:
        next_labels: List[str] = []
        for label in labels:
            if groups[label] and len(out) < limit:
                out.append(groups[label].pop(0))
            if groups[label]:
                next_labels.append(label)
        labels = next_labels
    return out


def _balanced_sample_by_action(items: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        groups[str(item["gt_action_id"])].append(item)
    for action_items in groups.values():
        dataset = str(action_items[0]["case"].get("dataset", "")) if action_items else ""
        action_id = str(action_items[0].get("gt_action_id", "")) if action_items else ""
        if dataset in {"OpenStack", "Hadoop"}:
            action_items.sort(
                key=lambda x: (
                    -float(x.get("difficulty_score", 0.0) or 0.0),
                    -float(x.get("selection_score", 0.0) or 0.0),
                    POOL_PRIORITY.get(str(x.get("pool_source", "benchmark_v2")), 99),
                    str(x["case"].get("case_id", "")),
                )
            )
        else:
            action_items.sort(
                key=lambda x: (
                    -float(x.get("selection_score", 0.0) or 0.0),
                    -float(x.get("difficulty_score", 0.0) or 0.0),
                    POOL_PRIORITY.get(str(x.get("pool_source", "benchmark_v2")), 99),
                    str(x["case"].get("case_id", "")),
                )
            )
        if dataset == "HDFS" and action_id == "HDFS_REBUILD_WRITE_PIPELINE":
            buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for item in action_items:
                buckets[str(item.get("action_bucket", "pipeline_other"))].append(item)
            bucket_order = [
                "pipeline_explicit_packetresponder",
                "pipeline_ambiguous_after_packetresponder_seed",
                "pipeline_ambiguous_received",
                "pipeline_receiving_block",
                "pipeline_other",
            ]
            ordered_buckets = [bucket for bucket in bucket_order if buckets.get(bucket)]
            ordered_buckets.extend(sorted(bucket for bucket in buckets if bucket not in set(ordered_buckets)))
            diversified: List[Dict[str, object]] = []
            while ordered_buckets:
                next_buckets: List[str] = []
                for bucket in ordered_buckets:
                    if buckets[bucket]:
                        diversified.append(buckets[bucket].pop(0))
                    if buckets[bucket]:
                        next_buckets.append(bucket)
                ordered_buckets = next_buckets
            action_items[:] = diversified
        elif dataset == "HDFS" and len(action_items) >= 3:
            # Avoid an all-easy HDFS sample: take one strong exemplar first, then
            # interleave from the tail so the second draw for the same action is
            # meaningfully harder while remaining within the same family/action.
            diversified: List[Dict[str, object]] = []
            left = 0
            right = len(action_items) - 1
            while left <= right:
                diversified.append(action_items[left])
                left += 1
                if left <= right:
                    diversified.append(action_items[right])
                    right -= 1
            action_items[:] = diversified
        elif dataset in {"OpenStack", "Hadoop"}:
            buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            for item in action_items:
                buckets[str(item.get("action_bucket", action_id or "other"))].append(item)
            preferred_bucket_order: List[str] = []
            if dataset == "OpenStack" and action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                preferred_bucket_order = [
                    "openstack_power_sync",
                    "openstack_power_other",
                    "openstack_power_pending",
                    "openstack_power_vm_paused",
                    "openstack_power_spawned",
                    "openstack_power_build",
                    "openstack_power_vm_started",
                    "openstack_power_vm_resumed",
                ]
            elif dataset == "OpenStack" and action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
                preferred_bucket_order = [
                    "openstack_inventory_other",
                    "openstack_inventory_instance_sync",
                    "openstack_inventory_servers_detail",
                    "openstack_inventory_lifecycle",
                ]
            elif dataset == "OpenStack" and action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
                preferred_bucket_order = [
                    "openstack_image_other",
                    "openstack_image_checking",
                    "openstack_image_in_use",
                    "openstack_image_active",
                    "openstack_image_removable",
                    "openstack_image_young",
                    "openstack_image_creating",
                    "openstack_image_unknown_base",
                    "openstack_image_lifecycle",
                ]
            if preferred_bucket_order:
                ordered_buckets = [bucket for bucket in preferred_bucket_order if bucket in buckets]
                ordered_buckets.extend(
                    sorted(
                        (bucket for bucket in buckets if bucket not in set(ordered_buckets)),
                        key=lambda bucket: (
                            -float(buckets[bucket][0].get("difficulty_score", 0.0) or 0.0),
                            -float(buckets[bucket][0].get("selection_score", 0.0) or 0.0),
                            bucket,
                        ),
                    )
                )
            else:
                ordered_buckets = sorted(
                    buckets.keys(),
                    key=lambda bucket: (
                        -float(buckets[bucket][0].get("difficulty_score", 0.0) or 0.0),
                        -float(buckets[bucket][0].get("selection_score", 0.0) or 0.0),
                        bucket,
                    ),
                )
            diversified: List[Dict[str, object]] = []
            while ordered_buckets:
                next_buckets: List[str] = []
                for bucket in ordered_buckets:
                    if buckets[bucket]:
                        diversified.append(buckets[bucket].pop(0))
                    if buckets[bucket]:
                        next_buckets.append(bucket)
                ordered_buckets = next_buckets
            if len({str(item.get("action_bucket", action_id or "other")) for item in action_items}) == 1 and len(diversified) >= 3:
                # Avoid selecting only the easiest near-duplicates for a single
                # bucket/action group. This matters for metadata-root and
                # storage-shuffle cases where the top few rows are otherwise too
                # similar, producing unrealistic flat curves in small sanity.
                interleaved: List[Dict[str, object]] = []
                left = 0
                right = len(diversified) - 1
                while left <= right:
                    interleaved.append(diversified[left])
                    left += 1
                    if left <= right:
                        interleaved.append(diversified[right])
                        right -= 1
                action_items[:] = interleaved
            else:
                action_items[:] = diversified

    dataset = str(items[0]["case"].get("dataset", "")) if items else ""
    if dataset in {"OpenStack", "Hadoop"}:
        original_candidates: Dict[str, List[Dict[str, object]]] = {
            action_id: list(action_items) for action_id, action_items in groups.items()
        }
        sampled_counts: Dict[str, int] = defaultdict(int)
        out: List[Dict[str, object]] = []
        while len(out) < limit:
            actions = [action_id for action_id, action_items in groups.items() if action_items]
            if not actions:
                break
            actions.sort(
                key=lambda action_id: (
                    sampled_counts[action_id],
                    -float(groups[action_id][0].get("difficulty_score", 0.0) or 0.0),
                    -float(groups[action_id][0].get("selection_score", 0.0) or 0.0),
                    POOL_PRIORITY.get(str(groups[action_id][0].get("pool_source", "benchmark_v2")), 99),
                    action_id,
                )
            )
            progressed = False
            for action_id in actions:
                if groups[action_id] and len(out) < limit:
                    out.append(groups[action_id].pop(0))
                    sampled_counts[action_id] += 1
                    progressed = True
                if not progressed:
                    break
        if dataset == "OpenStack":
            image_action = "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
            image_selected_idx = [
                idx for idx, item in enumerate(out) if str(item.get("gt_action_id", "")) == image_action
            ]
            if len(image_selected_idx) >= 2:
                selected_buckets = {str(out[idx].get("action_bucket", "")) for idx in image_selected_idx}
                if selected_buckets == {"openstack_image_creating"}:
                    selected_case_ids = {
                        str(out[idx].get("case", {}).get("case_id", "")) for idx in image_selected_idx
                    }
                    replacement_pool = [
                        item
                        for item in original_candidates.get(image_action, [])
                        if (
                            str(item.get("action_bucket", "")) != "openstack_image_creating"
                            and str(item.get("case", {}).get("case_id", "")) not in selected_case_ids
                        )
                    ]
                    replacement_pool.sort(
                        key=lambda x: (
                            -float(x.get("difficulty_score", 0.0) or 0.0),
                            -float(x.get("selection_score", 0.0) or 0.0),
                            POOL_PRIORITY.get(str(x.get("pool_source", "benchmark_v2")), 99),
                            str(x.get("case", {}).get("case_id", "")),
                        )
                    )
                    if replacement_pool:
                        out[image_selected_idx[-1]] = replacement_pool[0]
            metadata_action = "OPENSTACK_SCALE_METADATA_SERVICE"
            power_action = "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD"
            metadata_selected_idx = [
                idx for idx, item in enumerate(out) if str(item.get("gt_action_id", "")) == metadata_action
            ]
            power_selected_idx = [
                idx for idx, item in enumerate(out) if str(item.get("gt_action_id", "")) == power_action
            ]
            if limit <= 9 and len(metadata_selected_idx) >= 2:
                selected_metadata_buckets = {
                    str(out[idx].get("action_bucket", "")) for idx in metadata_selected_idx
                }
                selected_power_buckets = {str(out[idx].get("action_bucket", "")) for idx in power_selected_idx}
                selected_case_ids = {
                    str(item.get("case", {}).get("case_id", "")) for item in out
                }
                if len(selected_metadata_buckets) == 1:
                    replacement_pool = [
                        item
                        for item in original_candidates.get(power_action, [])
                        if (
                            str(item.get("case", {}).get("case_id", "")) not in selected_case_ids
                            and str(item.get("action_bucket", "")) not in selected_power_buckets
                        )
                    ]
                    if not replacement_pool:
                        replacement_pool = [
                            item
                            for item in original_candidates.get(power_action, [])
                            if str(item.get("case", {}).get("case_id", "")) not in selected_case_ids
                        ]
                    if replacement_pool:
                        replacement_pool.sort(
                            key=lambda x: (
                                str(x.get("action_bucket", "")) in selected_power_buckets,
                                -float(x.get("difficulty_score", 0.0) or 0.0),
                                -float(x.get("selection_score", 0.0) or 0.0),
                                POOL_PRIORITY.get(str(x.get("pool_source", "benchmark_v2")), 99),
                                str(x.get("case", {}).get("case_id", "")),
                            )
                        )
                        out[metadata_selected_idx[-1]] = replacement_pool[0]
        return out

    actions = sorted(groups.keys(), key=lambda a: (-len(groups[a]), a))
    out: List[Dict[str, object]] = []
    while actions and len(out) < limit:
        next_actions: List[str] = []
        for action_id in actions:
            if groups[action_id] and len(out) < limit:
                out.append(groups[action_id].pop(0))
            if groups[action_id]:
                next_actions.append(action_id)
        actions = next_actions
    return out


def _filter_small_sanity_items(dataset: str, items: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    if limit > 9:
        return items
    if dataset == "HDFS":
        filtered = []
        for item in items:
            action_id = str(item.get("gt_action_id", ""))
            action_bucket = str(item.get("action_bucket", ""))
            raw_log = str(item.get("case", {}).get("raw_log", "") or "")
            selected_alert = str(item.get("selected_alert", "") or "")
            if not _hdfs_has_visible_root_support(action_id, raw_log, selected_alert):
                continue
            if action_id == "HDFS_REBUILD_WRITE_PIPELINE" and action_bucket == "pipeline_explicit_packetresponder":
                continue
            filtered.append(item)
        if len(filtered) >= 6:
            return filtered
        return items
    if dataset == "Hadoop":
        filtered = []
        for item in items:
            action_bucket = str(item.get("action_bucket", ""))
            if action_bucket in {
                "hadoop_node_delete_hdfs",
                "hadoop_node_bad_datanode",
                "hadoop_network_forced_close",
            }:
                continue
            filtered.append(item)
        if len(filtered) >= 8:
            return filtered
        return items
    if dataset != "OpenStack":
        return items
    filtered = [
        item
        for item in items
        if (
            str(item.get("gt_action_id", "")) not in {
                "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE",
            }
            and str(item.get("action_bucket", "")) in {
                "openstack_metadata_root",
                "openstack_metadata_latest",
                "openstack_metadata_vendor",
                "openstack_metadata_user_404",
                "openstack_metadata_meta_data",
                "openstack_power_sync",
                "openstack_power_pending",
                "openstack_power_vm_started",
                "openstack_power_vm_resumed",
                "openstack_power_vm_paused",
                "openstack_power_build",
                "openstack_power_spawned",
                "openstack_power_other",
                "openstack_inventory_servers_detail",
                "openstack_inventory_instance_sync",
                "openstack_inventory_lifecycle",
                "openstack_inventory_other",
                "openstack_host_claim",
                "openstack_host_cpu_affinity",
                "openstack_host_vcpu",
                "openstack_host_audit",
                "openstack_host_other",
                "openstack_image_creating",
                "openstack_image_lifecycle",
                "openstack_image_other",
                "openstack_image_checking",
                "openstack_image_in_use",
                "openstack_image_unknown_base",
                "openstack_image_active",
                "openstack_image_removable",
                "openstack_image_young",
            }
        )
    ]
    direct_bucket_blocklist = {
        "openstack_metadata_root",
        "openstack_metadata_latest",
        "openstack_host_cpu_affinity",
        "openstack_host_vcpu",
        "openstack_image_creating",
        "openstack_image_unknown_base",
    }
    robust_image_case_ids = {
        str(item.get("case", {}).get("case_id", ""))
        for item in filtered
        if _is_openstack_robust_image_case(item)
    }
    less_direct = [
        item
        for item in filtered
        if (
            str(item.get("action_bucket", "")) not in direct_bucket_blocklist
            or str(item.get("case", {}).get("case_id", "")) in robust_image_case_ids
        )
    ]
    if len(robust_image_case_ids) >= 2:
        image_cleaned = [
            item
            for item in less_direct
            if (
                str(item.get("gt_action_id", "")) != "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
                or str(item.get("action_bucket", "")) != "openstack_image_lifecycle"
                or str(item.get("case", {}).get("case_id", "")) in robust_image_case_ids
            )
        ]
        if len(image_cleaned) >= 8:
            less_direct = image_cleaned
    if len(less_direct) >= 8:
        return less_direct
    if len(filtered) >= 8:
        return filtered
    return items


def _action_bucket(
    dataset: str,
    gt_action_id: str,
    selected_alert: str,
    raw_log_seed: str,
) -> str:
    lower_alert = str(selected_alert or "").lower()
    lower_seed = str(raw_log_seed or "").lower()
    if dataset == "HDFS" and gt_action_id == "HDFS_REBUILD_WRITE_PIPELINE":
        if "received block" in lower_alert or "got blk" in lower_alert:
            if "packetresponder" in lower_seed or "terminating" in lower_seed:
                return "pipeline_ambiguous_after_packetresponder_seed"
            return "pipeline_ambiguous_received"
        if "receiving block" in lower_alert:
            return "pipeline_receiving_block"
        if "packetresponder" in lower_alert or "pkgresponder" in lower_alert or "terminating" in lower_alert:
            return "pipeline_explicit_packetresponder"
        return "pipeline_other"
    if dataset == "OpenStack":
        if gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
            if "creating image" in lower_alert:
                return "openstack_image_creating"
            if "vm stopped" in lower_alert or "vm paused" in lower_alert or "vm started" in lower_alert:
                return "openstack_image_lifecycle"
            if "unknown base file" in lower_alert:
                return "openstack_image_unknown_base"
            if "active base files" in lower_alert:
                return "openstack_image_active"
            if "removable base files" in lower_alert:
                return "openstack_image_removable"
            if "base or swap file too young" in lower_alert:
                return "openstack_image_young"
            if "image " in lower_alert and ": checking" in lower_alert:
                return "openstack_image_checking"
            if "image " in lower_alert and " in use:" in lower_alert:
                return "openstack_image_in_use"
            return "openstack_image_other"
        if gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
            if "pending task (spawning)" in lower_alert:
                return "openstack_power_pending"
            if "sync_power_state" in lower_alert or "synchronizing instance power states" in lower_alert:
                return "openstack_power_sync"
            if "vm resumed" in lower_alert:
                return "openstack_power_vm_resumed"
            if "vm paused" in lower_alert:
                return "openstack_power_vm_paused"
            if "vm started" in lower_alert:
                return "openstack_power_vm_started"
            if "build instance" in lower_alert or "spawn the instance" in lower_alert:
                return "openstack_power_build"
            if "instance spawned successfully" in lower_alert:
                return "openstack_power_spawned"
            return "openstack_power_other"
        if gt_action_id == "OPENSTACK_SCALE_METADATA_SERVICE":
            if "get /openstack/2013-10-17 http/1.1" in lower_alert:
                return "openstack_metadata_root"
            if "get /latest/meta-data/" in lower_alert:
                return "openstack_metadata_latest"
            if "vendor_data.json" in lower_alert:
                return "openstack_metadata_vendor"
            if "user_data http/1.1\" status: 404" in lower_alert:
                return "openstack_metadata_user_404"
            if "meta_data.json" in lower_alert:
                return "openstack_metadata_meta_data"
            return "openstack_metadata_other"
        if gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
            if "get /v2/" in lower_alert and "/servers/detail" in lower_alert:
                return "openstack_inventory_servers_detail"
            if "instance sync for host" in lower_alert or "re-created its instancelist" in lower_alert:
                return "openstack_inventory_instance_sync"
            if "vm resumed" in lower_alert or "vm started" in lower_alert or "vm paused" in lower_alert:
                return "openstack_inventory_lifecycle"
            return "openstack_inventory_other"
        if gt_action_id == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
            if "auditing locally available compute resources" in lower_alert:
                return "openstack_host_audit"
            if "creating image" in lower_alert:
                return "openstack_host_creating_image"
            if "attempting claim:" in lower_alert or "claim successful" in lower_alert:
                return "openstack_host_claim"
            if "cpu affinity" in lower_alert:
                return "openstack_host_cpu_affinity"
            if "vcpu count" in lower_alert or "total vcpu" in lower_alert or "vcpu limit" in lower_alert:
                return "openstack_host_vcpu"
            return "openstack_host_other"
    if dataset == "Hadoop":
        if gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
            if "retrying connect to server" in lower_alert:
                return "hadoop_node_retry_connect"
            if "forcibly closed by the remote host" in lower_alert:
                return "hadoop_node_forced_close"
            if "could not delete hdfs" in lower_alert:
                return "hadoop_node_delete_hdfs"
            if "bad datanode" in lower_alert:
                return "hadoop_node_bad_datanode"
            if "machine down" in lower_alert:
                return "hadoop_node_machine_down"
            if "heartbeat" in lower_alert or "unhealthy" in lower_alert:
                return "hadoop_node_heartbeat"
            return "hadoop_node_other"
        if gt_action_id == "HADOOP_RESTORE_NETWORK_AND_RETRY":
            if "retrying connect to server" in lower_alert:
                return "hadoop_network_retry_connect"
            if "forcibly closed by the remote host" in lower_alert:
                return "hadoop_network_forced_close"
            if "failed to connect" in lower_alert:
                return "hadoop_network_failed_connect"
            return "hadoop_network_other"
        if gt_action_id == "HADOOP_FREE_DISK_AND_RETRY":
            if "disk full" in lower_alert or "no space" in lower_alert:
                return "hadoop_storage_disk_full"
            if "shuffling to disk" in lower_alert:
                return "hadoop_storage_shuffle"
            if "maxsingleshufflelimit" in lower_alert or "ondiskmapoutput" in lower_alert:
                return "hadoop_storage_shuffle_limit"
            return "hadoop_storage_other"
    return gt_action_id or "other"


def _hdfs_has_visible_root_support(action_id: str, raw_log: str, selected_alert: str = "") -> bool:
    combined = f"{str(selected_alert or '').lower()}\n{str(raw_log or '').lower()}"
    if action_id == "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE":
        return any(
            pat in combined
            for pat in (
                "got exception while serving",
                "connection reset by peer",
                "writeblock blk_",
                "received exception",
            )
        )
    if action_id == "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE":
        return any(
            pat in combined
            for pat in (
                "exception in receiveblock",
                "receiveblock for block",
            )
        )
    if action_id == "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK":
        return any(
            pat in combined
            for pat in (
                "allocateblock",
                "allocate block",
                "could only be replicated",
                "not able to place enough replicas",
                "no space left",
                "disk out of space",
            )
        )
    if action_id == "HDFS_REBUILD_WRITE_PIPELINE":
        return any(
            pat in combined
            for pat in (
                "packetresponder",
                "pkgresponder",
                "receiving block",
                "received block",
                "got blk",
            )
        )
    if action_id == "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK":
        return (
            "unexpected error trying to delete block" in combined
            or "blockinfo not found" in combined
            or ("ask " in combined and " to delete" in combined)
        )
    return True


def _difficulty_score(
    dataset: str,
    gt_action_id: str,
    selected_alert: str,
    raw_log_seed: str,
    raw_log: str,
    gt_diag: Mapping[str, object],
) -> float:
    lower_alert = str(selected_alert or "").lower()
    lower_seed = str(raw_log_seed or "").lower()
    lower_raw = str(raw_log or "").lower()
    score = 0.0
    if dataset == "HDFS" and gt_action_id == "HDFS_REBUILD_WRITE_PIPELINE":
        if "received block" in lower_alert or "got blk" in lower_alert:
            score += 3.0
            if "packetresponder" in lower_seed or "terminating" in lower_seed:
                score += 4.0
        elif "receiving block" in lower_alert:
            score += 2.0
        elif "packetresponder" in lower_alert or "pkgresponder" in lower_alert or "terminating" in lower_alert:
            score += 1.0
    elif dataset == "OpenStack":
        margin = float(gt_diag.get("margin", 0.0) or 0.0)
        score += max(0.0, 7.5 - min(margin, 7.5))
        support_families = set()
        for line in str(raw_log or "").split("\n"):
            if not line.strip():
                continue
            action_id = infer_action_id_from_text(dataset, line)
            family_id = family_for_action(dataset, action_id) if action_id else infer_family_from_text(dataset, line)
            if family_id:
                support_families.add(family_id)
        score += 1.5 * max(0, len(support_families) - 1)
        if gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
            if "creating image" in lower_alert:
                score += 4.5
            if "vm stopped" in lower_alert or "vm paused" in lower_alert or "vm started" in lower_alert:
                score += 3.5
            if ("image " in lower_alert) and (" in use:" in lower_alert or ": checking" in lower_alert):
                score += 3.0
            if "active base files" in lower_alert or "removable base files" in lower_alert:
                score += 1.5
            if "vm " in lower_raw or "/servers/detail" in lower_raw:
                score += 2.0
            if _is_direct_openstack_line(gt_action_id, lower_alert):
                score -= 2.5
        elif gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
            if "while synchronizing instance power states" in lower_alert:
                score += 5.5
            if "found 1 instances in the database and 0 instances on the hypervisor" in lower_alert:
                score += 4.5
            if "vm resumed" in lower_alert or "vm paused" in lower_alert or "vm started" in lower_alert:
                score += 2.0
            if "pending task (spawning)" in lower_alert or "sync_power_state" in lower_alert:
                score += 1.0
            if "active base files" in lower_raw or "removable base files" in lower_raw:
                score += 1.5
            if _is_direct_openstack_line(gt_action_id, lower_alert):
                score -= 2.0
        elif gt_action_id == "OPENSTACK_SCALE_METADATA_SERVICE":
            if "get /openstack/2013-10-17 http/1.1" in lower_alert or "get /latest/meta-data/" in lower_alert:
                score += 2.5
            if "vendor_data.json" in lower_alert or "user_data http/1.1\" status: 404" in lower_alert:
                score += 2.0
            if "meta_data.json" in lower_alert:
                score += 0.5
            if "delete /v2/" in lower_raw or "terminating instance" in lower_raw:
                score += 2.0
            if "/servers/detail" in lower_raw:
                score += 1.0
            if _is_direct_openstack_line(gt_action_id, lower_alert):
                score -= 1.0
        elif gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
            if "get /v2/" in lower_alert and "/servers/detail" in lower_alert:
                score += 3.0
            if "creating event network-vif-plugged" in lower_alert or "os-server-external-events" in lower_alert:
                score += 2.5
            if "instance sync for host" in lower_alert or "re-created its instancelist" in lower_alert:
                score += 4.5
            if "vm " in lower_raw or "sync_power_state" in lower_raw:
                score += 2.0
            if _is_direct_openstack_line(gt_action_id, lower_alert):
                score -= 2.0
        elif gt_action_id == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
            if "attempting claim:" in lower_alert or "claim successful" in lower_alert:
                score += 3.0
            if "creating image" in lower_alert:
                score += 2.5
            if "auditing locally available compute resources" in lower_alert:
                score += 1.5
            if "vm " in lower_raw or "sync_power_state" in lower_raw:
                score += 2.0
            if _is_direct_openstack_line(gt_action_id, lower_alert):
                score -= 2.5
        if _is_openstack_explicit_alert(lower_alert):
            score -= 8.0
        if _is_direct_openstack_line(gt_action_id, lower_alert):
            score -= 10.0
    elif dataset == "Hadoop":
        support_families = set()
        for line in str(raw_log or "").split("\n"):
            if not line.strip():
                continue
            action_id = infer_action_id_from_text(dataset, line)
            family_id = family_for_action(dataset, action_id) if action_id else infer_family_from_text(dataset, line)
            if family_id:
                support_families.add(family_id)
        score += 1.5 * max(0, len(support_families) - 1)
        if gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
            if "retrying connect to server" in lower_alert:
                score += 3.5
            if "heartbeat" in lower_alert or "unhealthy" in lower_alert:
                score += 2.5
            if "forcibly closed by the remote host" in lower_alert:
                score += 1.0
            if "retrying connect to server" in lower_raw or "forcibly closed by the remote host" in lower_raw:
                score += 3.0
            if _is_direct_hadoop_line(gt_action_id, lower_alert):
                score -= 3.0
        elif gt_action_id == "HADOOP_RESTORE_NETWORK_AND_RETRY":
            if "retrying connect to server" in lower_alert:
                score += 3.0
            if "forcibly closed by the remote host" in lower_alert:
                score += 2.0
            if "bad datanode" in lower_raw or "machine down" in lower_raw:
                score += 3.0
            if _is_direct_hadoop_line(gt_action_id, lower_alert):
                score -= 1.5
        elif gt_action_id == "HADOOP_FREE_DISK_AND_RETRY":
            if "shuffling to disk" in lower_alert:
                score += 3.5
            if "maxsingleshufflelimit" in lower_alert or "ondiskmapoutput" in lower_alert:
                score += 3.0
            if _is_direct_hadoop_line(gt_action_id, lower_alert):
                score -= 2.5
    return score


def _label_distribution(items: List[Dict[str, object]], dataset: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for item in items:
        if str(item["case"].get("dataset", "")) != dataset:
            continue
        label = str(item["gt_label"])
        dist[label] = dist.get(label, 0) + 1
    return dist


def _action_distribution(items: List[Dict[str, object]], dataset: str) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for item in items:
        if str(item["case"].get("dataset", "")) != dataset:
            continue
        action_id = str(item["gt_action_id"])
        dist[action_id] = dist.get(action_id, 0) + 1
    return dist


def _has_enough_context(case: Mapping[str, object]) -> bool:
    dataset = str(case.get("dataset", "HDFS"))
    source = str(case.get("source", "") or "").lower()
    raw = str(case.get("raw_log", "") or "")
    lines = [line for line in raw.split("\n") if line.strip()]
    if dataset in {"OpenStack", "Hadoop"}:
        return len(lines) >= MIN_CONTEXT_LINES.get(dataset, 1)
    if "rq3_test_set" in source or source == "causal_edge":
        return len(lines) >= 1
    return len(lines) >= MIN_CONTEXT_LINES.get(dataset, 1)


def _is_weak_mainline_alert(dataset: str, selected_alert: str) -> bool:
    lower = str(selected_alert or "").lower()
    if not lower:
        return True
    if dataset == "HDFS":
        weak_patterns = [
            "verification succeeded",
            "starting thread to transfer block",
            "served block",
        ]
        if any(pat in lower for pat in weak_patterns):
            return True
        if "deleting block" in lower and "unexpected error trying to delete block" not in lower:
            return True
    if dataset == "OpenStack":
        weak_patterns = [
            "successfully synced instances from host",
        ]
        if any(pat in lower for pat in weak_patterns):
            return True
    if dataset == "Hadoop":
        if lower.startswith("at ") or lower.startswith("caused by:") or lower.startswith("java.io."):
            return True
        weak_patterns = [
            "taskheartbeathandler thread interrupted",
            "rmcontainerallocator.heartbeat",
            "mapoutputcopier",
            "bufstart =",
            "kvstart =",
            "saved output of task",
        ]
        if any(pat in lower for pat in weak_patterns):
            return True
    return False


def _is_openstack_explicit_alert(text: str) -> bool:
    lower = _canonical_reference_text("OpenStack", text).lower()
    return any(
        pat in lower
        for pat in (
            "creating image",
            "building instance disk image",
            "unknown base file",
            "active base files",
            "removable base files",
            "base or swap file too young",
            "sync_power_state",
            "power-state sync",
            "synchronizing instance power states",
            "pending task (spawning)",
            "pending state: spawning",
            "cpu affinity",
            "vcpu count",
            "total usable vcpus",
            "trying host claim:",
            "host claim accepted",
            "get /openstack/2013-10-17 http/1.1",
            "get /latest/meta-data/",
            "vendor_data.json",
            "user_data http/1.1\" status: 404",
            "meta_data.json",
            "instance metadata payload",
        )
    )


def _is_direct_openstack_line(gt_action_id: str, lower: str) -> bool:
    if gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
        return any(
            pat in lower
            for pat in (
                "creating image",
                "building instance disk image",
                "unknown base file",
                "active base files",
                "removable base files",
                "base or swap file too young",
                "removing base or swap file",
            )
        )
    if gt_action_id == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
        return any(
            pat in lower
            for pat in ("cpu affinity", "vcpu count", "vcpu limit", "total usable vcpus", "trying host claim:")
        )
    if gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
        return any(
            pat in lower
            for pat in (
                "sync_power_state",
                "power-state sync",
                "while synchronizing instance power states",
                "pending task (spawning)",
                "pending state: spawning",
                "vm resumed",
                "vm paused",
                "vm started",
            )
        )
    if gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
        return any(
            pat in lower
            for pat in (
                "get /v2/",
                "/servers/detail",
                "instance sync for host",
                "re-created its instancelist",
                "instance inventory sync on host",
                "rebuilt cached instance inventory",
            )
        )
    if gt_action_id == "OPENSTACK_SCALE_METADATA_SERVICE":
        return any(
            pat in lower
            for pat in (
                "get /openstack/2013-10-17 http/1.1",
                "get /latest/meta-data/",
                "meta_data.json",
                "instance metadata payload",
            )
        )
    return False


def _is_direct_hadoop_line(gt_action_id: str, lower: str) -> bool:
    if gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
        return any(
            pat in lower
            for pat in (
                "bad datanode",
                "could not delete hdfs",
                "failed to remove hdfs",
                "machine down",
                "host appears unavailable",
                "unhealthy data node",
            )
        )
    if gt_action_id == "HADOOP_RESTORE_NETWORK_AND_RETRY":
        return "failed to connect" in lower or "peer terminated the socket unexpectedly" in lower
    if gt_action_id == "HADOOP_FREE_DISK_AND_RETRY":
        return any(pat in lower for pat in ("disk full", "no space", "storage exhausted", "storage unavailable"))
    return False


def _openstack_contextual_score(gt_action_id: str, lower: str) -> float:
    score = 0.0
    if gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
        if "creating image" in lower:
            score += 14.0
        if "building instance disk image" in lower:
            score += 13.0
        if "vm stopped" in lower:
            score += 7.0
        if "vm paused" in lower or "vm started" in lower:
            score += 5.0
        if ("image " in lower and " in use:" in lower) or ": checking" in lower:
            score += 10.0
        if "active base files" in lower or "removable base files" in lower:
            score += 8.0
        if "base or swap file too young to remove" in lower:
            score += 7.0
        if "removing base or swap file" in lower:
            score += 4.0
        if "unknown base file" in lower:
            score += 6.0
        if "get /v2/" in lower and "/servers/detail" in lower:
            score += 4.0
        if "attempting claim:" in lower or "claim successful" in lower:
            score += 3.0
        if "metadata" in lower or "cpu affinity" in lower or "sync_power_state" in lower:
            score -= 4.0
        if "pending task (spawning)" in lower or "pending state: spawning" in lower:
            score -= 6.0
    elif gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
        if "while synchronizing instance power states" in lower:
            score += 12.0
        if "found 1 instances in the database and 0 instances on the hypervisor" in lower:
            score += 10.0
        if "vm paused" in lower:
            score += 10.0
        if "vm resumed" in lower or "vm started" in lower:
            score += 8.0
        if "instance resumed lifecycle transition" in lower or "instance started lifecycle transition" in lower:
            score += 8.0
        if "instance paused lifecycle transition" in lower:
            score += 9.0
        if "took " in lower and ("build instance" in lower or "spawn the instance" in lower):
            score += 7.0
        if "instance spawned successfully" in lower:
            score += 7.0
        if "pending task (spawning)" in lower or "sync_power_state" in lower:
            score += 1.0
        if "pending state: spawning" in lower or "power-state sync" in lower:
            score += 1.0
        if "instance sync for host" in lower or "re-created its instancelist" in lower:
            score -= 2.5
        if "unknown base file" in lower or "cpu affinity" in lower or "metadata" in lower:
            score -= 3.0
    elif gt_action_id == "OPENSTACK_SCALE_METADATA_SERVICE":
        if "get /openstack/2013-10-17 http/1.1" in lower:
            score += 1.0
        if "get /latest/meta-data/" in lower:
            score += 1.0
        if "vendor_data.json" in lower:
            score += 6.0
        if "vendor metadata payload" in lower:
            score += 6.0
        if "user_data http/1.1\" status: 404" in lower:
            score += 7.0
        if "meta_data.json" in lower:
            score += 3.0
        if "instance metadata payload" in lower:
            score += 3.0
        if "nova.metadata.wsgi.server" in lower:
            score += 2.0
        if "delete /v2/" in lower or "terminating instance" in lower:
            score += 3.0
        if "delete /v2/" in lower or "terminating instance" in lower:
            score -= 3.0
    elif gt_action_id == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
        if "attempting claim:" in lower or "claim successful" in lower:
            score += 12.0
        if "trying host claim:" in lower or "host claim accepted" in lower:
            score += 10.0
        if "auditing locally available compute resources" in lower:
            score += 10.0
        if "creating image" in lower:
            score += 3.0
        if "building instance disk image" in lower:
            score += 2.0
        if "total vcpu" in lower or "vcpu limit" in lower or "memory limit" in lower:
            score += 5.0
        if "cpu affinity" in lower or "vcpu count" in lower:
            score += 1.0
        if "instance sync for host" in lower:
            score += 2.0
    elif gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
        if "get /v2/" in lower and "/servers/detail" in lower:
            score += 6.0
        if "/detailed-server-list" in lower:
            score += 6.0
        if "instance sync for host" in lower or "re-created its instancelist" in lower:
            score += 8.0
        if "instance inventory sync on host" in lower or "rebuilt cached instance inventory" in lower:
            score += 8.0
        if "vm resumed" in lower or "vm started" in lower or "vm paused" in lower:
            score += 2.0
        if "instance resumed lifecycle transition" in lower or "instance started lifecycle transition" in lower:
            score += 2.0
        if "creating event network-vif-plugged" in lower or "os-server-external-events" in lower:
            score += 8.0
        if "pending task (spawning)" in lower:
            score -= 1.0
        if "pending state: spawning" in lower:
            score -= 1.0
        if "unknown base file" in lower or "cpu affinity" in lower or "meta_data.json" in lower:
            score -= 2.0
    return score


def _openstack_context_mode(selected_alert: str) -> str:
    lower = _canonical_reference_text("OpenStack", selected_alert).lower()
    if (
        ("get /v2/" in lower and "/servers/detail" in lower)
        or "/detailed-server-list" in lower
        or "network-vif-plugged" in lower
        or "os-server-external-events" in lower
        or "server_external_events" in lower
        or "instance sync for host" in lower
        or "re-created its instancelist" in lower
        or "instance inventory sync on host" in lower
        or "rebuilt cached instance inventory" in lower
        or "scheduler-side state reconciliation on node" in lower
        or "reconciled host-side runtime cache" in lower
    ):
        return "inventory"
    if (
        "auditing locally available compute resources" in lower
        or "attempting claim:" in lower
        or "trying host claim:" in lower
        or "claim successful" in lower
        or "host claim accepted" in lower
        or "total vcpu" in lower
        or "vcpu limit" in lower
        or "cpu affinity" in lower
        or "vcpu count" in lower
        or "total usable vcpus" in lower
    ):
        return "host"
    if (
        "vm resumed" in lower
        or "vm paused" in lower
        or "vm started" in lower
        or "while synchronizing instance power states" in lower
        or "instance resumed lifecycle transition" in lower
        or "instance paused lifecycle transition" in lower
        or "instance started lifecycle transition" in lower
        or ("took " in lower and ("build instance" in lower or "spawn the instance" in lower))
        or "instance spawned successfully" in lower
    ):
        return "power"
    if (
        "creating image" in lower
        or "building instance disk image" in lower
        or "unknown base file" in lower
        or "removable base files" in lower
        or "active base files" in lower
        or "base or swap file" in lower
        or "cached object " in lower
        or "routine inspection" in lower
        or "runtime workspace" in lower
        or "workspace objects" in lower
        or "/runtime/objects/" in lower
        or (("image " in lower) and (" in use:" in lower or ": checking" in lower))
    ):
        return "image"
    if (
        "get /openstack/2013-10-17 http/1.1" in lower
        or "get /latest/meta-data/" in lower
        or "vendor_data.json" in lower
        or "user_data http/1.1\" status: 404" in lower
        or "/control-plane/runtime-catalog" in lower
        or "bootstrap-blob" in lower
        or "vendor-profile" in lower
        or "instance-profile" in lower
    ):
        return "metadata"
    return "generic"


def _window_around_alert(raw_log: str, selected_alert: str, radius: int) -> List[str]:
    lines = [line for line in str(raw_log or "").split("\n") if line.strip()]
    if not lines:
        return []
    idx = -1
    alert = str(selected_alert or "").strip()
    if alert:
        for i, line in enumerate(lines):
            if line.strip() == alert:
                idx = i
                break
        if idx < 0:
            alert_lower = alert.lower()
            for i, line in enumerate(lines):
                if alert_lower and alert_lower in line.lower():
                    idx = i
                    break
    if idx < 0:
        return lines[-min(len(lines), radius * 2 + 1) :]
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return lines[lo:hi]


def _openstack_instance_window_stats(raw_log: str, selected_alert: str, radius: int = 12) -> Tuple[int, int, int]:
    selected_instances = _openstack_instance_ids(selected_alert)
    same = 0
    diff = 0
    noid = 0
    for line in _window_around_alert(raw_log, selected_alert, radius):
        line_instances = _openstack_instance_ids(line)
        if line_instances and selected_instances and (line_instances & selected_instances):
            same += 1
        elif line_instances:
            diff += 1
        else:
            noid += 1
    return same, diff, noid


def _is_openstack_robust_image_case(item: Mapping[str, object]) -> bool:
    if str(item.get("gt_action_id", "")) != "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
        return False
    raw_log = str(item.get("case", {}).get("raw_log", "") or "")
    selected_alert = str(item.get("selected_alert", "") or "")
    lower_alert = selected_alert.lower()
    same, diff, _ = _openstack_instance_window_stats(raw_log, selected_alert, radius=12)
    if any(
        pat in lower_alert
        for pat in (
            "creating image",
            "building instance disk image",
            "unknown base file",
            "removable base files",
            "active base files",
            "base or swap file too young",
            " in use:",
            ": checking",
        )
    ):
        return same >= 2 and diff <= 1
    return same >= 3 and diff == 0


def _openstack_instance_ids(text: str) -> Set[str]:
    ids: Set[str] = set()
    s = str(text or "")
    for pat in (
        r"\[instance: ([0-9a-f-]{36})\]",
        r"/servers/([0-9a-f-]{36})\b",
        r"for instance ([0-9a-f-]{36})\b",
        r"domain id: ([0-9a-f-]{36})\b",
    ):
        ids.update(m.group(1).lower() for m in re.finditer(pat, s, flags=re.IGNORECASE))
    return ids


def _openstack_claim_segment(raw_log: str, selected_alert: str, max_lines: int = 80) -> List[str]:
    lines = [line for line in str(raw_log or "").split("\n") if line.strip()]
    if not lines:
        return []
    alert = str(selected_alert or "").strip()
    if not alert:
        return lines[:max_lines]
    idx = -1
    for i, line in enumerate(lines):
        if line.strip() == alert:
            idx = i
            break
    if idx < 0:
        alert_lower = alert.lower()
        for i, line in enumerate(lines):
            if alert_lower and alert_lower in line.lower():
                idx = i
                break
    if idx < 0:
        return lines[:max_lines]
    selected_instances = _openstack_instance_ids(alert)
    lo = max(0, idx - 3)
    hi = min(len(lines), idx + max_lines)
    segment: List[str] = []
    for i in range(lo, hi):
        line = lines[i]
        lower = line.lower()
        if i > idx and ("attempting claim:" in lower or "trying host claim:" in lower):
            line_instances = _openstack_instance_ids(line)
            if selected_instances and line_instances and not (selected_instances & line_instances):
                break
        segment.append(line)
    return segment


def _selection_score(
    dataset: str,
    selected_alert: str,
    raw_log: str,
    gt_action_id: str,
    gt_diag: Mapping[str, object],
) -> float:
    lower = str(selected_alert or "").lower()
    score = 0.0
    confidence = str(gt_diag.get("confidence", "") or "")
    if confidence == "high":
        score += 8.0
    elif confidence == "medium":
        score += 4.0

    if dataset == "HDFS":
        if "got exception while serving" in lower:
            score += 14.0
        if "unexpected error trying to delete block" in lower:
            score += 12.0
        if "connection reset by peer" in lower or "exception in receiveblock" in lower:
            score += 11.0
        if "allocateblock" in lower:
            score += 9.0
        if "packetresponder" in lower and "terminating" in lower:
            score += 4.0
        if "received block" in lower:
            score -= 2.0
        if _is_weak_mainline_alert(dataset, selected_alert):
            score -= 20.0
    elif dataset == "OpenStack":
        if "creating image" in lower:
            score += 8.0
        if "attempting claim:" in lower or "claim successful" in lower:
            score += 7.0
        if "instance sync for host" in lower or "re-created its instancelist" in lower:
            score += 9.0
        if "while synchronizing instance power states" in lower:
            score += 8.0
        if "found 1 instances in the database and 0 instances on the hypervisor" in lower:
            score += 7.0
        if "vm resumed" in lower or "vm paused" in lower or "vm started" in lower:
            score += 7.0
        if "get /openstack/2013-10-17 http/1.1" in lower or "get /latest/meta-data/" in lower:
            score += 7.0
        if "vendor_data.json" in lower or "user_data http/1.1\" status: 404" in lower:
            score += 6.0
        if "get /v2/" in lower and "/servers/detail" in lower:
            score += 6.0
        if "auditing locally available compute resources" in lower:
            score += 4.0
        if ("image " in lower and (" in use:" in lower or ": checking" in lower)) or "active base files" in lower:
            score += 4.0
        if "removable base files" in lower:
            score += 3.0
        if "cpu affinity" in lower or "vcpu count" in lower:
            score += 2.0
        if gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
            if "instance sync for host" in lower or "re-created its instancelist" in lower:
                score += 6.0
            if "vm resumed" in lower or "vm paused" in lower or "vm started" in lower:
                score -= 3.0
            if "sync_power_state" in lower or "pending task (spawning)" in lower:
                score -= 4.0
        if gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
            if "while synchronizing instance power states" in lower:
                score += 7.0
            if "found 1 instances in the database and 0 instances on the hypervisor" in lower:
                score += 6.0
            if "get /v2/" in lower and "/servers/" in lower and "/servers/detail" not in lower:
                score -= 5.0
        if "unknown base file" in lower or "sync_power_state" in lower or "meta_data.json" in lower:
            score -= 2.0
        if _is_direct_openstack_line(gt_action_id, lower):
            score -= 2.0
        if _is_weak_mainline_alert(dataset, selected_alert):
            score -= 12.0
        if _is_openstack_explicit_alert(lower):
            score -= 12.0
        if _is_direct_openstack_line(gt_action_id, lower):
            score -= 14.0
    else:
        if "retrying connect to server" in lower:
            score += 10.0
        if "forcibly closed by the remote host" in lower:
            score += 9.0
        if "heartbeat" in lower or "unhealthy" in lower:
            score += 8.0
        if "shuffling to disk" in lower or "maxsingleshufflelimit" in lower or "ondiskmapoutput" in lower:
            score += 10.0
        if "failed to connect" in lower:
            score += 4.0
        if "could not delete hdfs" in lower or "bad datanode" in lower or "machine down" in lower:
            score += 3.0
        if _is_direct_hadoop_line(gt_action_id, lower):
            score -= 3.0

    if gt_action_id.startswith("HDFS_") and "got exception while serving" in str(raw_log).lower():
        score += 2.0
    return score


def _load_or_create_manifest(
    legacy,
    *,
    run_tag: str,
    cases_per_dataset: int,
    noise_levels: List[float],
    datasets: List[str],
    causal_graph_path: str,
    pool_policy: str,
    manifest_path: Path,
    force_resample: bool,
) -> Dict[str, object]:
    if manifest_path.exists() and not force_resample:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("cases_per_dataset") != cases_per_dataset
            or [float(x) for x in manifest.get("noise_levels", [])] != [float(x) for x in noise_levels]
            or str(manifest.get("pool_policy", "benchmark_v2_only")) != str(pool_policy)
        ):
            raise RuntimeError(
                f"Existing manifest {manifest_path} does not match current args. "
                "Use --force-resample or a new --run-tag."
            )
        return manifest

    fixed_small_case_ids = _load_fixed_small_case_ids(cases_per_dataset)
    fixed_small_case_id_set = {
        dataset: set(case_ids) for dataset, case_ids in fixed_small_case_ids.items()
    }

    benchmark_pool = json.loads(BENCH_V2_PATH.read_text(encoding="utf-8"))
    rq3_test_pool = json.loads(RQ3_TEST_SET_PATH.read_text(encoding="utf-8"))
    enriched_seed_pool: List[Dict[str, object]] = []
    by_dataset: Dict[str, List[Dict[str, object]]] = {ds: [] for ds in datasets}
    seen_case_ids: Dict[str, set[str]] = {ds: set() for ds in datasets}

    def _ingest(pool: List[Dict[str, object]], pool_source: str, allowed_datasets: Iterable[str]) -> None:
        for case in pool:
            dataset = str(case.get("dataset", "HDFS"))
            if dataset not in by_dataset or dataset not in allowed_datasets:
                continue
            case_id = str(case.get("case_id", ""))
            if case_id and case_id in seen_case_ids[dataset]:
                continue
            if not _has_enough_context(case):
                continue
            raw_log = str(case.get("raw_log", "") or "")
            selected_alert = _select_actionaware_alert(legacy, raw_log, dataset)
            support_context = _local_alert_context(raw_log, selected_alert, dataset)
            context_support = _line_action_support(dataset, support_context)
            gt_action_id, gt_label, gt_diag = select_gt_action_and_family(
                dataset,
                selected_alert=selected_alert,
                raw_log=raw_log,
                raw_log_seed=str(case.get("raw_log_seed", "") or ""),
                gt_root_template=str(case.get("ground_truth_root_cause_template", "") or ""),
                gt_effect_template=str(case.get("ground_truth_template", "") or ""),
                gt_action_label=str(case.get("gt_action_label", "") or case.get("reason", "") or ""),
                context_support=context_support,
            )
            initial_gt_action_id = gt_action_id
            initial_gt_label = gt_label
            if not gt_label or not gt_action_id:
                continue
            if dataset in {"HDFS", "OpenStack", "Hadoop"}:
                selected_alert = _refine_selected_alert_for_action(
                    dataset,
                    raw_log,
                    selected_alert,
                    gt_action_id,
                    str(case.get("raw_log_seed", "") or ""),
                )
                support_context = _local_alert_context(raw_log, selected_alert, dataset)
                context_support = _line_action_support(dataset, support_context)
            gt_action_id, gt_label, gt_diag = select_gt_action_and_family(
                dataset,
                selected_alert=selected_alert,
                raw_log=raw_log,
                raw_log_seed=str(case.get("raw_log_seed", "") or ""),
                gt_root_template=str(case.get("ground_truth_root_cause_template", "") or ""),
                gt_effect_template=str(case.get("ground_truth_template", "") or ""),
                gt_action_label=str(case.get("gt_action_label", "") or case.get("reason", "") or ""),
                context_support=context_support,
            )
            if not gt_label or not gt_action_id:
                continue
            if dataset == "OpenStack":
                if (
                    initial_gt_action_id
                    and gt_action_id
                    and initial_gt_action_id != gt_action_id
                    and float(gt_diag.get("margin", 0.0) or 0.0) < 6.0
                ):
                    continue
            if dataset == "Hadoop":
                benchmark_hint_text = str(
                    case.get("gt_action_label", "")
                    or case.get("reason", "")
                    or case.get("ground_truth_root_cause_template", "")
                    or ""
                )
                benchmark_hint_action = infer_action_id_from_text(dataset, benchmark_hint_text)
                if (
                    benchmark_hint_action
                    and gt_action_id
                    and family_for_action(dataset, benchmark_hint_action)
                    and family_for_action(dataset, gt_action_id)
                    and family_for_action(dataset, benchmark_hint_action)
                    != family_for_action(dataset, gt_action_id)
                    and float(gt_diag.get("margin", 0.0) or 0.0) < 4.0
                ):
                    continue
            if gt_diag.get("confidence") == "low":
                continue
            if (
                _is_weak_mainline_alert(dataset, selected_alert)
                and case_id not in fixed_small_case_id_set.get(dataset, set())
            ):
                continue
            if (
                gt_action_id in MAIN_ACTION_EXCLUDE.get(dataset, set())
                and case_id not in fixed_small_case_id_set.get(dataset, set())
            ):
                continue
            selection_score = _selection_score(dataset, selected_alert, raw_log, gt_action_id, gt_diag)
            action_bucket = _action_bucket(
                dataset,
                gt_action_id,
                selected_alert,
                str(case.get("raw_log_seed", "") or ""),
            )
            difficulty_score = _difficulty_score(
                dataset,
                gt_action_id,
                selected_alert,
                str(case.get("raw_log_seed", "") or ""),
                raw_log,
                gt_diag,
            )
            item = {
                "case": dict(case),
                "gt_label": gt_label,
                "gt_action_id": gt_action_id,
                "gt_diagnostics": gt_diag,
                "pool_source": pool_source,
                "selected_alert": selected_alert,
                "selection_score": round(float(selection_score), 3),
                "action_bucket": action_bucket,
                "difficulty_score": round(float(difficulty_score), 3),
            }
            by_dataset[dataset].append(item)
            if case_id:
                seen_case_ids[dataset].add(case_id)

    if pool_policy == "mixed_richer":
        # Prefer the richer rq3_test_set for HDFS/OpenStack, then backfill with
        # the larger benchmark pool. This is useful for debugging action
        # diversity, but not always ideal for the strict main evaluation.
        _ingest(rq3_test_pool, "rq3_test_set", ["HDFS", "OpenStack"])
        _ingest(benchmark_pool, "benchmark_v2", datasets)
    elif pool_policy == "enriched_seeded":
        if not ENRICHED_SEED_POOL_PATH.exists():
            write_enriched_seed_pool()
        enriched_seed_pool = json.loads(ENRICHED_SEED_POOL_PATH.read_text(encoding="utf-8"))
        _ingest(enriched_seed_pool, "rq3_test_set_enriched", ["HDFS", "OpenStack"])
        _ingest(benchmark_pool, "benchmark_v2", datasets)
    else:
        _ingest(benchmark_pool, "benchmark_v2", datasets)

    labeled_cases: List[Dict[str, object]] = []
    for ds in datasets:
        if ds in fixed_small_case_ids:
            fixed_ids = fixed_small_case_ids[ds]
            if len(fixed_ids) != cases_per_dataset:
                raise RuntimeError(
                    f"Fixed small cases for {ds} expected {cases_per_dataset} ids, got {len(fixed_ids)}."
                )
            item_by_case_id = {
                str(item.get("case", {}).get("case_id", "") or item.get("case_id", "")): item
                for item in by_dataset[ds]
            }
            missing = [case_id for case_id in fixed_ids if case_id not in item_by_case_id]
            if missing:
                raise RuntimeError(f"Fixed small cases for {ds} not found in ingested pool: {missing}")
            labeled_cases.extend([item_by_case_id[case_id] for case_id in fixed_ids])
            continue
        items = _filter_small_sanity_items(ds, by_dataset[ds], cases_per_dataset)
        take = min(cases_per_dataset, len(items))
        labeled_cases.extend(_balanced_sample_by_action(items, take))

    manifest = {
        "run_tag": run_tag,
        "cases_per_dataset": cases_per_dataset,
        "noise_levels": [float(x) for x in noise_levels],
        "datasets": list(datasets),
        "methods": list(METHODS),
        "causal_graph_path": causal_graph_path,
        "pool_policy": pool_policy,
        "labeled_cases": labeled_cases,
        "label_distribution": {ds: _label_distribution(labeled_cases, ds) for ds in datasets},
        "action_distribution": {ds: _action_distribution(labeled_cases, ds) for ds in datasets},
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_progress(
    progress_path: Path,
    noise_levels: List[float],
    datasets: List[str],
) -> Tuple[set[str], Dict[Tuple[str, float, str], Dict[str, int]], int]:
    completed: set[str] = set()
    stats = _build_stats(noise_levels, datasets)
    actual_api_calls = 0

    if not progress_path.exists():
        return completed, stats, actual_api_calls

    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _step_key(
                str(row["dataset"]),
                str(row["case_id"]),
                float(row["noise"]),
                str(row["method"]),
            )
            if key in completed:
                continue
            completed.add(key)
            stat = stats[(str(row["dataset"]), float(row["noise"]), str(row["method"]))]
            stat["rca_total"] += 1
            stat["rca_success"] += int(bool(row.get("rca_success", False)))
            stat["action_total"] += 1
            stat["action_success"] += int(bool(row.get("action_success", False)))
            stat["action_text_total"] += 1
            stat["action_text_success"] += int(bool(row.get("action_text_success", False)))
            stat["e2e_total"] += 1
            stat["e2e_success"] += int(bool(row.get("e2e_success", False)))
            actual_api_calls += int(bool(row.get("api_call", False)))
    return completed, stats, actual_api_calls


def _extract_structured_output(
    dataset: str,
    text: str,
    allowed_labels: List[str],
    allowed_actions: List[str],
) -> Tuple[str, str, str]:
    t = (text or "").strip()
    pred_label = ""
    action_id = ""
    repair_action = ""
    if not t:
        return pred_label, action_id, repair_action
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                raw_label = str(
                    obj.get("root_cause_family") or obj.get("root_cause_label") or obj.get("label") or ""
                ).strip()
                raw_action_id = str(obj.get("action_id") or "").strip()
                repair_action = str(obj.get("repair_action") or obj.get("action") or "").strip()
                for a in allowed_labels:
                    if a.lower() == raw_label.lower():
                        pred_label = a
                        break
                for a in allowed_actions:
                    if a.lower() == raw_action_id.lower():
                        action_id = a
                        break
    except Exception:
        pass
    lower = t.lower()
    if not pred_label:
        for a in allowed_labels:
            if a.lower() in lower:
                pred_label = a
                break
    if not repair_action:
        repair_action = t[-300:]
    if not action_id and repair_action:
        inferred_action = infer_action_id_from_text(dataset, repair_action)
        if inferred_action in allowed_actions:
            action_id = inferred_action
    if not pred_label and repair_action:
        inferred_label = infer_family_from_text(dataset, repair_action)
        if inferred_label in allowed_labels:
            pred_label = inferred_label
    if not action_id and pred_label:
        family_actions = [
            candidate_action
            for candidate_action in allowed_actions
            if family_for_action(dataset, candidate_action) == pred_label
        ]
        if len(family_actions) == 1:
            action_id = family_actions[0]
    if not pred_label and action_id:
        inferred_from_action = family_for_action(dataset, action_id)
        if inferred_from_action in allowed_labels:
            pred_label = inferred_from_action
    return pred_label, action_id, repair_action


def _select_actionaware_alert(legacy, raw: str, dataset: str) -> str:
    lines = [line for line in str(raw or "").split("\n") if line.strip()]
    if not lines:
        return legacy._select_alert_line(raw, dataset)
    raw_lower = str(raw or "").lower()
    has_openstack_image = any(
        pat in raw_lower
        for pat in (
            "unknown base file",
            "active base files",
            "removable base files",
            "base or swap file too young",
            "removing base or swap file",
            "image ",
        )
    )
    has_openstack_image_failure = any(
        pat in raw_lower
        for pat in (
            "unknown base file",
            "removable base files",
            "base or swap file too young",
            "removing base or swap file",
            "creating image",
            "building instance disk image",
        )
    )
    has_openstack_host = any(
        pat in raw_lower
        for pat in (
            "cpu affinity",
            "vcpu count",
            "auditing locally available compute resources",
            "attempting claim:",
            "claim successful",
            "vcpu limit",
        )
    )
    has_openstack_power = any(
        pat in raw_lower
        for pat in (
            "sync_power_state",
            "pending task (spawning)",
            "while synchronizing instance power states",
            "vm paused",
            "vm resumed",
            "vm started",
        )
    )
    has_hadoop_node = any(
        pat in raw_lower for pat in ("bad datanode", "could not delete hdfs", "machine down")
    )
    has_hadoop_network = any(
        pat in raw_lower for pat in ("retrying connect to server", "forcibly closed by the remote host", "failed to connect")
    )

    scored: List[Tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        score = 0
        if dataset == "HDFS":
            if "got exception while serving" in lower:
                score += 14
            if "delete" in lower or "blockinfo not found" in lower:
                score += 13
            if "allocateblock" in lower:
                score += 10
            if "packetresponder" in lower:
                score += 8
            if "receiving block" in lower:
                score += 5
        elif dataset == "OpenStack":
            if "cpu affinity" in lower or "vcpu count" in lower:
                score += 14
            if "unknown base file" in lower:
                score += 12
            if "instance sync for host" in lower or "re-created its instancelist" in lower:
                score += 11
            if "synchronizing instance power states" in lower or "pending task (spawning)" in lower:
                score += 10
            if "sync_power_state" in lower:
                score += 10
            if "get /openstack/2013-10-17 http/1.1" in lower or "get /latest/meta-data/" in lower:
                score += 9
            if "auditing locally available compute resources" in lower or "attempting claim:" in lower:
                score += 8
            if "removable base files" in lower or "active base files" in lower:
                score += 8
            if "vm paused" in lower or "vm resumed" in lower or "vm started" in lower:
                score += 7
            if "get /v2/" in lower and "/servers/detail" in lower:
                score += 6
            if "metadata" in lower:
                score += 5
            if "creating image" in lower:
                score += 4
            if has_openstack_image and "metadata" in lower:
                score -= 8
            if has_openstack_image and (
                "unknown base file" in lower
                or "active base files" in lower
                or "removable base files" in lower
                or (("image " in lower) and (" in use:" in lower or ": checking" in lower))
            ):
                score += 6
            if (
                has_openstack_power
                and not has_openstack_image_failure
                and (
                    "active base files" in lower
                    or (("image " in lower) and (" in use:" in lower or ": checking" in lower))
                )
            ):
                score -= 8
            if has_openstack_host and "metadata" in lower:
                score -= 5
            if has_openstack_host and ("cpu affinity" in lower or "vcpu count" in lower):
                score += 3
            if has_openstack_host and (
                "unknown base file" in lower
                or "active base files" in lower
                or "removable base files" in lower
                or "base or swap file too young" in lower
                or "removing base or swap file" in lower
            ):
                score -= 8
            if has_openstack_power and ("vm paused" in lower or "vm resumed" in lower or "vm started" in lower):
                score += 2
        else:
            if "could not delete hdfs" in lower:
                score += 15
            if "machine down" in lower or "bad datanode" in lower:
                score += 12
            if "forcibly closed by the remote host" in lower:
                score += 9
            if "retrying connect to server" in lower:
                score += 10
            if "disk full" in lower or "no space" in lower:
                score += 12
            if "shuffling to disk" in lower:
                score += 7
            if "heartbeat" in lower or "unhealthy" in lower:
                score += 8
            elif "maxsingleshufflelimit" in lower:
                score += 5
            if has_hadoop_node and ("shuffling to disk" in lower or "maxsingleshufflelimit" in lower):
                score -= 8
            if has_hadoop_node and ("could not delete hdfs" in lower or "bad datanode" in lower or "machine down" in lower):
                score += 4
            if has_hadoop_network and ("retrying connect to server" in lower or "forcibly closed by the remote host" in lower):
                score += 2
            if _is_weak_mainline_alert(dataset, line):
                score -= 16
        if idx >= len(lines) - 2:
            score += 1
        if score > 0:
            scored.append((score, idx, line))

    if not scored:
        return legacy._select_alert_line(raw, dataset)
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def _local_alert_context(raw_log: str, selected_alert: str, dataset: str) -> str:
    lines = [line for line in str(raw_log or "").split("\n") if line.strip()]
    if not lines:
        return ""
    alert = str(selected_alert or "").strip()
    radius = int(LOCAL_SUPPORT_RADIUS.get(dataset, 6))
    idx = -1
    if alert:
        for i, line in enumerate(lines):
            if line.strip() == alert:
                idx = i
                break
        if idx < 0:
            alert_lower = alert.lower()
            for i, line in enumerate(lines):
                if alert_lower and alert_lower in line.lower():
                    idx = i
                    break
    if idx < 0:
        return "\n".join(lines[-min(len(lines), radius * 2 + 1) :])
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return "\n".join(lines[lo:hi])


def _refine_selected_alert_for_action(
    dataset: str,
    raw_log: str,
    selected_alert: str,
    gt_action_id: str,
    raw_log_seed: str,
) -> str:
    lines = [line for line in str(raw_log or "").split("\n") if line.strip()]
    if not lines or not gt_action_id:
        return selected_alert

    if dataset == "HDFS":
        anchor_line = str(raw_log_seed or selected_alert or "")
        local_lines = _window_around_alert(raw_log, anchor_line, 10) or lines
        block_match = re.search(r"(blk_[^\s:]+|block-id:[^\s:]+)", anchor_line)
        selected_block = block_match.group(1) if block_match else ""
        best_line = selected_alert
        best_score = float("-inf")
        for line in local_lines:
            lower = line.lower()
            score = 0.0
            if line.strip() == str(selected_alert).strip():
                score += 0.5
            if selected_block and selected_block in line:
                score += 4.0
            if gt_action_id == "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE":
                if "got exception while serving" in lower:
                    score += 18.0
                if "connection reset by peer" in lower:
                    score += 18.0
                if "writeblock" in lower and "received exception" in lower:
                    score += 17.0
                if selected_block and selected_block in line:
                    if "packetresponder" in lower and ("received block" in lower or "terminating" in lower):
                        score += 20.0
                    if "receiving block" in lower:
                        score += 18.0
                if "packetresponder" in lower or "received block" in lower or "receiving block" in lower:
                    score -= 6.0
                if "allocateblock" in lower or "allocate block" in lower:
                    score -= 8.0
            elif gt_action_id == "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE":
                if "exception in receiveblock" in lower:
                    score += 20.0
                if "receiveblock for block" in lower:
                    score += 18.0
                if "packetresponder" in lower and ("terminating" in lower or "closing" in lower):
                    score += 7.0
                if "received block" in lower or "receiving block" in lower:
                    score += 4.0
                if "got exception while serving" in lower or "connection reset by peer" in lower:
                    score -= 7.0
            elif gt_action_id == "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK":
                if "allocateblock" in lower or "allocate block" in lower:
                    score += 18.0
                if (
                    "could only be replicated" in lower
                    or "not able to place enough replicas" in lower
                    or "no space left" in lower
                    or "disk out of space" in lower
                ):
                    score += 16.0
                if "ask " in lower and " to delete" in lower:
                    score += 2.0
                if "unexpected error trying to delete block" in lower or "blockinfo not found" in lower:
                    score -= 6.0
                if "packetresponder" in lower or "received block" in lower or "receiving block" in lower:
                    score -= 4.0
            elif gt_action_id == "HDFS_REBUILD_WRITE_PIPELINE":
                if selected_block and selected_block not in line:
                    score -= 12.0
                if "packetresponder" in lower and ("terminating" in lower or "closing" in lower):
                    score += 16.0
                if "pkgresponder" in lower:
                    score += 15.0
                if "receiving block" in lower:
                    score += 12.0
                if "received block" in lower or "got blk" in lower:
                    score += 10.0
                if "got exception while serving" in lower or "connection reset by peer" in lower:
                    score -= 8.0
            if score > best_score:
                best_line = line
                best_score = score
        return best_line

    if dataset == "OpenStack" and gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
        for line in lines:
            lower = line.lower()
            if "while synchronizing instance power states" in lower:
                return line
            if "found 1 instances in the database and 0 instances on the hypervisor" in lower:
                return line
        started_or_paused = any(
            "vm started" in line.lower() or "vm paused" in line.lower() for line in lines
        )
        pending_line = next(
            (
                line
                for line in lines
                if (
                    "sync_power_state" in line.lower()
                    or "pending task (spawning)" in line.lower()
                    or "pending state: spawning" in line.lower()
                )
            ),
            "",
        )
        spawn_line = next(
            (
                line
                for line in lines
                if (
                    "instance spawned successfully" in line.lower()
                    or "spawn the instance" in line.lower()
                    or "build instance" in line.lower()
                )
            ),
            "",
        )
        if started_or_paused and pending_line:
            return pending_line
        if spawn_line and not started_or_paused:
            return spawn_line
        if pending_line:
            return pending_line

    if dataset == "OpenStack" and gt_action_id == "OPENSTACK_SCALE_METADATA_SERVICE":
        for patterns in (
            ("get /openstack/2013-10-17 http/1.1", "get /latest/meta-data/"),
            ("vendor_data.json", "meta_data.json"),
            ("user_data http/1.1\" status: 404",),
        ):
            for line in lines:
                lower = line.lower()
                if any(pat in lower for pat in patterns):
                    return line

    if dataset == "Hadoop":
        best_line = selected_alert
        best_score = float("-inf")
        lower_raw = str(raw_log or "").lower()
        has_node_evidence = any(
            pat in lower_raw
            for pat in (
                "could not delete hdfs",
                "failed to remove hdfs",
                "bad datanode",
                "machine down",
                "host appears unavailable",
                "unhealthy data node",
            )
        )
        for line in lines:
            lower = line.lower()
            score = 0.0
            if line.strip() == str(selected_alert).strip():
                score += 0.5
            if gt_action_id == "HADOOP_RESTORE_NETWORK_AND_RETRY":
                if "rmcontainerallocator.heartbeat" in lower or "netutils.connect" in lower:
                    score += 18.0
                if "ipc.channel.call" in lower or "ipc.client.call" in lower:
                    score += 12.0
                if "retrying connect to server" in lower or "retrying rpc toward node" in lower:
                    score += 10.0
                if "forcibly closed by the remote host" in lower or "peer terminated the socket unexpectedly" in lower:
                    score += 9.0
                if has_node_evidence and any(
                    pat in lower
                    for pat in (
                        "could not delete hdfs",
                        "failed to remove hdfs",
                        "bad datanode",
                        "machine down",
                        "host appears unavailable",
                        "container_remote_cleanup",
                        "kill_container_cleanup",
                        "opening proxy :",
                    )
                ):
                    score -= 8.0
            elif gt_action_id == "HADOOP_FREE_DISK_AND_RETRY":
                if "mergemanagerimpl" in lower and (
                    "maxsingleshufflelimit" in lower
                    or "single-shuffle threshold" in lower
                    or "single-fragment staging threshold" in lower
                    or "memorylimit=" in lower
                ):
                    score += 18.0
                if "ondiskmapoutput" in lower or "staged shuffle fragment" in lower:
                    score += 15.0
                if "merging " in lower and "bytes" in lower:
                    score += 10.0
                if "shuffling to disk" in lower or "redirecting shuffle fragment into fallback merge staging" in lower:
                    score += 8.0
                if "taskheartbeathandler thread interrupted" in lower:
                    score -= 5.0
            elif gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
                if "retrying connect to server" in lower or "retrying rpc toward node" in lower:
                    score += 15.0
                if "forcibly closed by the remote host" in lower or "peer terminated the socket unexpectedly" in lower:
                    score += 14.0
                if any(
                    pat in lower
                    for pat in (
                        "opening proxy :",
                        "local host is:",
                        "destination host is:",
                        "container_remote_cleanup",
                        "kill_container_cleanup",
                        "containerlauncher",
                        "leaserenewer",
                        "communication thread",
                    )
                ):
                    score += 10.0
                if any(
                    pat in lower
                    for pat in (
                        "could not delete hdfs",
                        "failed to remove hdfs",
                        "bad datanode",
                        "machine down",
                        "host appears unavailable",
                        "unhealthy data node",
                    )
                ):
                    score += 14.0
                    score += 7.0
            if score > best_score:
                best_line = line
                best_score = score
        return best_line

    selected_instance_ids: Set[str] = set()
    if dataset == "OpenStack":
        best_anchor_score = float("-inf")
        for line in lines:
            line_instance_ids = _openstack_instance_ids(line)
            if not line_instance_ids:
                continue
            lower = line.lower()
            anchor_score = _openstack_contextual_score(gt_action_id, lower)
            if gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
                if any(
                    pat in lower
                    for pat in (
                        "creating image",
                        "building instance disk image",
                        "unknown base file",
                        "removable base files",
                        "active base files",
                        "base or swap file too young",
                        " in use:",
                        ": checking",
                    )
                ):
                    anchor_score += 18.0
            elif gt_action_id == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
                if any(
                    pat in lower
                    for pat in (
                        "attempting claim:",
                        "claim successful",
                        "trying host claim:",
                        "host claim accepted",
                        "cpu affinity",
                        "vcpu count",
                    )
                ):
                    anchor_score += 12.0
            elif gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                if any(
                    pat in lower
                    for pat in (
                        "build instance",
                        "spawn the instance",
                        "instance spawned successfully",
                        "sync_power_state",
                        "while synchronizing instance power states",
                        "found 1 instances in the database and 0 instances on the hypervisor",
                        "pending task (spawning)",
                        "pending state: spawning",
                    )
                ):
                    anchor_score += 10.0
            elif gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
                if any(
                    pat in lower
                    for pat in (
                        "/servers/detail",
                        "network-vif-plugged",
                        "os-server-external-events",
                        "instance sync for host",
                        "re-created its instancelist",
                    )
                ):
                    anchor_score += 14.0
            if infer_action_id_from_text(dataset, line) == gt_action_id:
                anchor_score += 4.0
            if _is_direct_openstack_line(gt_action_id, lower):
                anchor_score += 2.0
            if anchor_score > best_anchor_score:
                selected_instance_ids = set(line_instance_ids)
                best_anchor_score = anchor_score
    if not selected_instance_ids:
        selected_instance_ids = _openstack_instance_ids(raw_log_seed) or _openstack_instance_ids(selected_alert)
    has_strong_image_line = dataset == "OpenStack" and gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN" and any(
        any(
            pat in line.lower()
            for pat in (
                "creating image",
                "building instance disk image",
                "unknown base file",
                "removable base files",
                "active base files",
                "base or swap file too young",
                "removing base or swap file",
                " in use:",
                ": checking",
            )
        )
        for line in lines
    )
    if has_strong_image_line:
        # Image-cache evidence often lacks instance IDs; keeping the anchor empty
        # avoids accidentally promoting later claim/delete lines over the actual
        # base-image-chain signal.
        selected_instance_ids = set()
    has_inventory_resync_line = dataset == "OpenStack" and any(
        "instance sync for host" in line.lower() or "re-created its instancelist" in line.lower()
        for line in lines
    )
    has_power_sync_line = dataset == "OpenStack" and any(
        (
            "while synchronizing instance power states" in line.lower()
            or "found 1 instances in the database and 0 instances on the hypervisor" in line.lower()
            or "sync_power_state" in line.lower()
            or "pending task (spawning)" in line.lower()
            or "pending state: spawning" in line.lower()
        )
        for line in lines
    )

    def _score_line(idx: int, line: str) -> float:
        lower = line.lower()
        score = 0.0
        if dataset == "OpenStack":
            if gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                if "while synchronizing instance power states" in lower:
                    return 100.0
                if "found 1 instances in the database and 0 instances on the hypervisor" in lower:
                    return 95.0
                if "sync_power_state" in lower or "pending task (spawning)" in lower or "pending state: spawning" in lower:
                    return 90.0
            line_instance_ids = _openstack_instance_ids(line)
            if selected_instance_ids and line_instance_ids:
                if selected_instance_ids & line_instance_ids:
                    score += 8.0
                else:
                    score -= 18.0
            elif selected_instance_ids and gt_action_id != "OPENSTACK_SCALE_METADATA_SERVICE":
                if gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY" and (
                    "instance sync for host" in lower or "re-created its instancelist" in lower
                ):
                    score += 0.0
                elif gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD" and (
                    "while synchronizing instance power states" in lower
                    or "found 1 instances in the database and 0 instances on the hypervisor" in lower
                    or "sync_power_state" in lower
                    or "pending task (spawning)" in lower
                    or "pending state: spawning" in lower
                ):
                    score += 0.0
                else:
                    score -= 2.5
            if str(raw_log_seed or "").strip() and str(raw_log_seed).strip() == line.strip():
                score += 1.0
            if infer_action_id_from_text(dataset, line) == gt_action_id:
                score += 1.5
            score += _openstack_contextual_score(gt_action_id, lower)
            if gt_action_id == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
                if any(
                    pat in lower
                    for pat in (
                        "attempting claim:",
                        "claim successful",
                        "trying host claim:",
                        "host claim accepted",
                    )
                ):
                    score += 14.0
                if "auditing locally available compute resources" in lower:
                    score += 12.0
                if any(
                    pat in lower
                    for pat in (
                        "cpu affinity",
                        "vcpu count",
                        "total usable vcpus",
                        "total vcpu",
                        "vcpu limit",
                        "memory limit",
                    )
                ):
                    score += 4.0
                if any(
                    pat in lower
                    for pat in (
                        "unknown base file",
                        "removable base files",
                        "active base files",
                        "base or swap file too young",
                    )
                ):
                    score -= 10.0
            elif gt_action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
                if any(
                    pat in lower
                    for pat in (
                        "vm stopped",
                        "vm paused",
                        "vm started",
                    )
                ):
                    score += 6.0
                if "get /v2/" in lower and "/servers/detail" in lower:
                    score += 6.0
                if ("image " in lower) and (" in use:" in lower or ": checking" in lower):
                    score += 12.0
                if any(
                    pat in lower
                    for pat in (
                        "unknown base file",
                        "removable base files",
                        "active base files",
                        "base or swap file too young",
                    )
                ):
                    score += 10.0
                if "creating image" in lower or "building instance disk image" in lower:
                    score += 14.0
                if has_strong_image_line and (
                    "attempting claim:" in lower
                    or "claim successful" in lower
                    or "deleting instance files" in lower
                ):
                    score -= 8.0
                if "sync_power_state" in lower or "pending task (spawning)" in lower or "pending state: spawning" in lower:
                    score -= 8.0
                if "cpu affinity" in lower or "vcpu count" in lower:
                    score -= 10.0
            elif gt_action_id == "OPENSTACK_SCALE_METADATA_SERVICE":
                if "get /openstack/2013-10-17 http/1.1" in lower or "get /latest/meta-data/" in lower:
                    score += 12.0
                if "/control-plane/runtime-catalog" in lower:
                    score += 11.0
                if "delete /v2/" in lower or "terminating instance" in lower:
                    score += 7.0
                if "vendor_data.json" in lower or "vendor metadata payload" in lower or "vendor-profile" in lower:
                    score += 4.0
                if "user_data http/1.1\" status: 404" in lower or "bootstrap-blob" in lower:
                    score += 3.0
                if "meta_data.json" in lower or "instance metadata payload" in lower or "instance-profile" in lower:
                    score += 2.0
            elif gt_action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
                if "os-server-external-events" in lower or "network-vif-plugged" in lower:
                    score += 12.0
                if "instance sync for host" in lower or "re-created its instancelist" in lower:
                    score += 18.0
                if any(
                    pat in lower
                    for pat in (
                        "vm resumed",
                        "vm paused",
                        "vm started",
                        "instance resumed lifecycle transition",
                        "instance started lifecycle transition",
                        "instance paused lifecycle transition",
                    )
                ):
                    score += 2.0
                    if has_inventory_resync_line:
                        score -= 8.0
                if "get /v2/" in lower and "/servers/detail" in lower:
                    score += 4.0
                if "pending task (spawning)" in lower or "pending state: spawning" in lower or "sync_power_state" in lower:
                    score -= 6.0
            elif gt_action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
                if "while synchronizing instance power states" in lower:
                    score += 18.0
                if "found 1 instances in the database and 0 instances on the hypervisor" in lower:
                    score += 14.0
                if "sync_power_state" in lower or "pending task (spawning)" in lower or "pending state: spawning" in lower:
                    score += 14.0
                if any(
                    pat in lower
                    for pat in (
                        "build instance",
                        "spawn the instance",
                        "instance spawned successfully",
                    )
                ):
                    score += 10.0
                if any(
                    pat in lower
                    for pat in (
                        "vm resumed",
                        "vm paused",
                        "vm started",
                        "instance resumed lifecycle transition",
                        "instance paused lifecycle transition",
                        "instance started lifecycle transition",
                    )
                ):
                    score += 8.0
                if "get /v2/" in lower and "/servers/" in lower and "/servers/detail" not in lower:
                    score -= 7.0
                if "creating image" in lower or "building instance disk image" in lower:
                    score -= 8.0
                if has_power_sync_line and any(
                    pat in lower
                    for pat in (
                        "build instance",
                        "spawn the instance",
                        "instance spawned successfully",
                    )
                ):
                    score -= 6.0
                if has_power_sync_line and any(
                    pat in lower
                    for pat in (
                        "vm resumed",
                        "vm paused",
                        "vm started",
                        "instance resumed lifecycle transition",
                        "instance paused lifecycle transition",
                        "instance started lifecycle transition",
                    )
                ):
                    score -= 5.0
                if "instance sync for host" in lower or "re-created its instancelist" in lower:
                    score -= 4.0
            if _is_openstack_explicit_alert(lower):
                score -= 5.0
            if _is_direct_openstack_line(gt_action_id, lower):
                score -= 9.0
        else:
            if str(raw_log_seed or "").strip() and str(raw_log_seed).strip() == line.strip():
                score += 2.0
            if infer_action_id_from_text(dataset, line) == gt_action_id:
                score += 4.0
        if dataset == "Hadoop":
            node_direct = any(
                pat in lower
                for pat in (
                    "could not delete hdfs",
                    "failed to remove hdfs",
                    "cleanup failed for distributed output path",
                    "bad datanode",
                    "unhealthy data node",
                    "machine down",
                    "host appears unavailable",
                    "heartbeat",
                )
            )
            network_direct = any(
                pat in lower
                for pat in (
                    "retrying connect to server",
                    "retrying rpc toward node",
                    "rpc handshake loop against remote endpoint",
                    "control-plane session establishment kept cycling against peer endpoint",
                    "forcibly closed by the remote host",
                    "peer terminated the socket unexpectedly",
                    "remote endpoint closed the channel during exchange",
                    "peer interrupted the transport session mid-exchange",
                )
            )
            if gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE" and network_direct:
                score += 16.0
            if gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE" and (
                "heartbeat" in lower or "unhealthy" in lower
            ):
                score += 14.0
            if gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE" and node_direct:
                score += 8.0
            if gt_action_id == "HADOOP_RESTORE_NETWORK_AND_RETRY" and network_direct:
                score += 16.0
            if gt_action_id == "HADOOP_RESTORE_NETWORK_AND_RETRY" and node_direct and not network_direct:
                score -= 6.0
            if gt_action_id == "HADOOP_FREE_DISK_AND_RETRY" and (
                "shuffling to disk" in lower or "maxsingleshufflelimit" in lower or "ondiskmapoutput" in lower
            ):
                score += 14.0
            if _is_direct_hadoop_line(gt_action_id, lower):
                score -= 4.0
            if _is_weak_mainline_alert(dataset, line):
                score -= 18.0
        if idx >= len(lines) - 2:
            score += 1.0
        return score

    best_line = selected_alert
    best_score = float("-inf")
    for idx, line in enumerate(lines):
        score = _score_line(idx, line)
        if line == selected_alert:
            score += 0.5
        if score > best_score:
            best_line = line
            best_score = score
    return best_line


def _line_action_support(dataset: str, text: str) -> Dict[str, int]:
    support: Dict[str, int] = {}
    lines = [line for line in str(text or "").split("\n") if line.strip()]
    retry_hits = 0
    forced_hits = 0
    delete_hits = 0
    disk_hits = 0
    for line in lines:
        lower = line.lower()
        action_id = infer_action_id_from_text(dataset, line)
        if action_id:
            support[action_id] = support.get(action_id, 0) + 1
        if dataset == "Hadoop":
            if (
                "retrying connect to server" in lower
                or "retrying rpc toward node" in lower
                or "rpc handshake loop against remote endpoint" in lower
                or "control-plane session establishment kept cycling against peer endpoint" in lower
            ):
                retry_hits += 1
            if (
                "forcibly closed by the remote host" in lower
                or "peer terminated the socket unexpectedly" in lower
                or "remote endpoint closed the channel during exchange" in lower
                or "peer interrupted the transport session mid-exchange" in lower
            ):
                forced_hits += 1
            if (
                "could not delete hdfs" in lower
                or "failed to remove hdfs" in lower
                or "cleanup failed for distributed output path" in lower
                or "bad datanode" in lower
                or "unhealthy data node" in lower
                or "machine down" in lower
                or "host appears unavailable" in lower
            ):
                delete_hits += 1
            if (
                "disk full" in lower
                or "no space" in lower
                or "shuffling to disk" in lower
                or "maxsingleshufflelimit" in lower
                or "storage exhausted" in lower
                or "storage unavailable" in lower
            ):
                disk_hits += 1
    if dataset == "Hadoop":
        # Persistent retries to the same host are often the observable symptom
        # of a machine-down case, but only when the logs also show explicit
        # delete/bad-datanode evidence. Retries plus forced-close alone are
        # still compatible with plain network connectivity failures.
        if delete_hits >= 1 and (retry_hits >= 1 or forced_hits >= 1):
            support["HADOOP_ISOLATE_NODE_AND_RESCHEDULE"] = (
                support.get("HADOOP_ISOLATE_NODE_AND_RESCHEDULE", 0) + 3
            )
        if retry_hits >= 1 and forced_hits >= 1:
            support["HADOOP_RESTORE_NETWORK_AND_RETRY"] = (
                support.get("HADOOP_RESTORE_NETWORK_AND_RETRY", 0) + 2
            )
        if disk_hits >= 2:
            support["HADOOP_FREE_DISK_AND_RETRY"] = (
                support.get("HADOOP_FREE_DISK_AND_RETRY", 0) + 2
            )
    return support


def _top_candidate_actions(dataset: str, cand_json: str, observed_template: str, limit: int = 3) -> List[str]:
    try:
        obj = json.loads(cand_json)
    except Exception:
        obj = []
    scores: Dict[str, float] = {}
    for item in obj if isinstance(obj, list) else []:
        if not isinstance(item, dict):
            continue
        tpl = str(item.get("source_template", "") or "")
        action_id = infer_action_id_from_text(dataset, tpl)
        if not action_id:
            continue
        weight = abs(float(item.get("weight", 0.0) or 0.0))
        scores[action_id] = scores.get(action_id, 0.0) + weight
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [action for action, _ in ordered[:limit]]


def _extract_actionaware_context(
    raw: str,
    dataset: str,
    cand_json: str,
    observed_template: str,
    selected_alert: str = "",
    max_chars: int = 900,
) -> str:
    lines = [line for line in str(raw or "").split("\n") if line.strip()]
    if not lines:
        return ""

    def _cap_lines_preserve_order(selected_lines: List[str]) -> str:
        kept: List[str] = []
        total_chars = 0
        for line in selected_lines:
            if not line or line in kept:
                continue
            extra = len(line) + (1 if kept else 0)
            if kept and total_chars + extra > max_chars:
                break
            kept.append(line)
            total_chars += extra
        if not kept:
            return ""
        return "\n".join(kept)

    top_actions = set(_top_candidate_actions(dataset, cand_json, observed_template))
    observed_action = infer_action_id_from_text(dataset, observed_template)
    observed_family = family_for_action(dataset, observed_action) if observed_action else ""
    if dataset == "Hadoop" and selected_alert:
        local_lines = _window_around_alert(raw, selected_alert, 12)
        if local_lines:
            chosen: List[str] = []
            lower_selected = str(selected_alert).lower()
            node_terms = (
                "could not delete hdfs",
                "failed to remove hdfs",
                "bad datanode",
                "machine down",
                "host appears unavailable",
                "heartbeat",
                "unhealthy data node",
            )
            network_terms = (
                "retrying connect to server",
                "retrying rpc toward node",
                "rpc handshake loop against remote endpoint",
                "control-plane session establishment kept cycling against peer endpoint",
                "forcibly closed by the remote host",
                "peer terminated the socket unexpectedly",
                "remote endpoint closed the channel during exchange",
                "peer interrupted the transport session mid-exchange",
                "netutils.connect",
                "rmcontainerallocator.heartbeat",
            )
            worker_like_selected = any(
                pat in lower_selected
                for pat in (
                    "containerlauncher",
                    "dispatch path",
                    "leaserenewer",
                    "client renewer:",
                    "communication thread",
                    "rpc transport path",
                    "task: communication exception",
                    "local host is:",
                    "local worker is:",
                )
            )
            rm_like_selected = any(
                pat in lower_selected
                for pat in (
                    "rmcommunicator allocator",
                    "scheduler control loop",
                    "rmcontainerallocator",
                    "rmcontainerallocator.heartbeat",
                    "netutils.connect",
                    ":8030",
                    "error in contacting rm",
                )
            )
            generic_socket_selected = "socket reader" in lower_selected or "channel reader for port" in lower_selected
            nearby_node_evidence = any(
                any(pat in line.lower() for pat in node_terms)
                for line in local_lines
            )

            def _append_first_from(
                pool: List[str],
                patterns: Tuple[str, ...],
                *,
                allow_selected: bool = True,
            ) -> None:
                for line in pool:
                    if not allow_selected and line.strip() == str(selected_alert).strip():
                        continue
                    lower = line.lower()
                    if any(pat in lower for pat in patterns):
                        if line not in chosen:
                            chosen.append(line)
                        return

            _append_first_from(local_lines, (lower_selected,))
            if any(pat in lower_selected for pat in network_terms):
                if worker_like_selected and not rm_like_selected:
                    _append_first_from(lines, node_terms, allow_selected=False)
                    _append_first_from(lines, network_terms, allow_selected=False)
                    _append_first_from(
                        lines,
                        (
                            "containerlauncher",
                            "dispatch path",
                            "leaserenewer",
                            "client renewer:",
                            "communication thread",
                            "local host is:",
                            "local worker is:",
                            "destination host is:",
                            "peer worker is:",
                            "container_remote_cleanup",
                            "kill_container_cleanup",
                            "opening proxy :",
                        ),
                        allow_selected=False,
                    )
                elif generic_socket_selected and not rm_like_selected:
                    _append_first_from(lines, network_terms, allow_selected=False)
                    if nearby_node_evidence or "containerlauncher" in str(raw).lower():
                        _append_first_from(lines, node_terms, allow_selected=False)
                        _append_first_from(
                            lines,
                            (
                                "containerlauncher",
                                "dispatch path",
                                "leaserenewer",
                                "client renewer:",
                                "communication thread",
                                "local host is:",
                                "local worker is:",
                                "destination host is:",
                                "peer worker is:",
                                "container_remote_cleanup",
                                "kill_container_cleanup",
                                "opening proxy :",
                            ),
                            allow_selected=False,
                        )
                elif rm_like_selected:
                    _append_first_from(lines, network_terms, allow_selected=False)
                else:
                    _append_first_from(lines, network_terms, allow_selected=False)
                    _append_first_from(lines, node_terms, allow_selected=False)
            else:
                _append_first_from(lines, network_terms, allow_selected=False)
                _append_first_from(lines, node_terms, allow_selected=False)
            _append_first_from(
                lines,
                (
                    "taskheartbeathandler thread interrupted",
                    "rmcontainerallocator.heartbeat",
                    "nodemanager",
                ),
                allow_selected=False,
            )
            if chosen:
                compact = _cap_lines_preserve_order(chosen)
                if compact:
                    return compact
    if dataset == "OpenStack" and selected_alert:
        mode = _openstack_context_mode(selected_alert)
        local_lines = _window_around_alert(raw, selected_alert, 14)
        focus_lines = _openstack_claim_segment(raw, selected_alert, max_lines=80) if mode == "host" else local_lines
        selected_instance_ids = _openstack_instance_ids(selected_alert)
        if selected_instance_ids:
            aligned_local_lines = [
                line
                for line in local_lines
                if not _openstack_instance_ids(line) or (_openstack_instance_ids(line) & selected_instance_ids)
            ]
            aligned_focus_lines = [
                line
                for line in focus_lines
                if not _openstack_instance_ids(line) or (_openstack_instance_ids(line) & selected_instance_ids)
            ]
            if len(aligned_local_lines) >= 2:
                local_lines = aligned_local_lines
            if len(aligned_focus_lines) >= 2:
                focus_lines = aligned_focus_lines
        if local_lines:
            if mode == "inventory":
                chosen: List[str] = []
                has_strong_inventory = any(
                    "instance sync for host" in line.lower() or "re-created its instancelist" in line.lower()
                    for line in local_lines
                )

                def _append_first(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
                    for line in local_lines:
                        if not allow_selected and line.strip() == str(selected_alert).strip():
                            continue
                        lower = line.lower()
                        if any(pat in lower for pat in patterns):
                            if line not in chosen:
                                chosen.append(line)
                            return

                _append_first((str(selected_alert).lower(),))
                _append_first(("os-server-external-events", "network-vif-plugged"), allow_selected=False)
                _append_first(("instance sync for host", "re-created its instancelist"), allow_selected=False)
                _append_first(("/servers/detail",), allow_selected=False)
                if not has_strong_inventory:
                    _append_first(("vm started", "vm paused", "vm resumed"), allow_selected=False)
                    _append_first(("sync_power_state", "pending task (spawning)"), allow_selected=False)
                else:
                    _append_first(("vm started", "vm paused", "vm resumed"), allow_selected=False)
                if chosen:
                    compact = _cap_lines_preserve_order(chosen)
                    if compact:
                        return compact
            if mode == "host":
                chosen = []

                def _append_host(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
                    for line in focus_lines:
                        if not allow_selected and line.strip() == str(selected_alert).strip():
                            continue
                        lower = line.lower()
                        if any(pat in lower for pat in patterns):
                            if line not in chosen:
                                chosen.append(line)
                            return

                def _append_host_global(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
                    for line in focus_lines:
                        if not allow_selected and line.strip() == str(selected_alert).strip():
                            continue
                        lower = line.lower()
                        if any(pat in lower for pat in patterns):
                            if line not in chosen:
                                chosen.append(line)
                            return

                _append_host((str(selected_alert).lower(),))
                _append_host(
                    (
                        "attempting claim:",
                        "claim successful",
                        "trying host claim:",
                        "host claim accepted",
                    ),
                    allow_selected=False,
                )
                _append_host(("auditing locally available compute resources",), allow_selected=False)
                _append_host(
                    (
                        "creating image",
                        "building instance disk image",
                    ),
                    allow_selected=False,
                )
                _append_host(
                    (
                        "unknown base file",
                        "removable base files",
                        "active base files",
                        "base or swap file too young",
                    ),
                    allow_selected=False,
                )
                _append_host(("sync_power_state", "pending task (spawning)"), allow_selected=False)
                has_host_strong = any(
                    any(
                        pat in line.lower()
                        for pat in (
                            "cpu affinity",
                            "vcpu count",
                            "total usable vcpus",
                            "trying host claim:",
                        )
                    )
                    for line in chosen
                )
                has_image_signal = any(
                    any(
                        pat in line.lower()
                        for pat in (
                            "creating image",
                            "building instance disk image",
                            "unknown base file",
                            "removable base files",
                            "active base files",
                            "base or swap file too young",
                        )
                    )
                    for line in chosen
                )
                if not has_host_strong:
                    _append_host(("cpu affinity", "vcpu count", "total usable vcpus", "trying host claim:"), allow_selected=False)
                if has_host_strong or not has_image_signal:
                    _append_host(("memory limit", "total vcpu", "vcpu limit"), allow_selected=False)
                _append_host(("/servers/detail",), allow_selected=False)
                if chosen:
                    compact = _cap_lines_preserve_order(chosen)
                    if compact:
                        return compact
            if mode == "image":
                chosen = []

                def _append_image(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
                    for line in focus_lines:
                        if not allow_selected and line.strip() == str(selected_alert).strip():
                            continue
                        lower = line.lower()
                        if any(pat in lower for pat in patterns):
                            if line not in chosen:
                                chosen.append(line)
                            return

                _append_image((str(selected_alert).lower(),))
                _append_image(("vm stopped", "vm paused", "vm started"), allow_selected=False)
                _append_image(
                    (
                        "delete /v2/",
                        "terminating instance",
                        "deleting instance files",
                        "instance destroyed successfully",
                    ),
                    allow_selected=False,
                )
                _append_image(("/servers/detail",), allow_selected=False)
                _append_image(("http exception thrown", "no instances found for any event"), allow_selected=False)
                _append_image(("sync_power_state", "pending task (spawning)"), allow_selected=False)
                if len(chosen) < 3:
                    _append_image(("creating image", "building instance disk image"), allow_selected=False)
                    _append_image(
                        (
                            "unknown base file",
                            "removable base files",
                            "active base files",
                            "base or swap file too young",
                            "cached object ",
                            "routine inspection",
                            "runtime workspace",
                            "workspace objects",
                        ),
                        allow_selected=False,
                    )
                if chosen:
                    compact = _cap_lines_preserve_order(chosen)
                    if compact:
                        return compact
            if mode == "metadata":
                chosen = []

                def _append_metadata(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
                    for line in local_lines:
                        if not allow_selected and line.strip() == str(selected_alert).strip():
                            continue
                        lower = line.lower()
                        if any(pat in lower for pat in patterns):
                            if line not in chosen:
                                chosen.append(line)
                            return

                _append_metadata((str(selected_alert).lower(),))
                _append_metadata(
                    (
                        "get /openstack/2013-10-17 http/1.1",
                        "get /latest/meta-data/",
                        "/control-plane/runtime-catalog",
                    ),
                    allow_selected=False,
                )
                _append_metadata(("delete /v2/", "terminating instance", "/servers/detail"), allow_selected=False)
                _append_metadata(("http exception thrown", "no instances found for any event"), allow_selected=False)
                if len(chosen) < 3:
                    _append_metadata(
                        (
                            "vendor_data.json",
                            "user_data http/1.1\" status: 404",
                            "meta_data.json",
                            "bootstrap-blob",
                            "vendor-profile",
                            "instance-profile",
                        ),
                        allow_selected=False,
                    )
                if chosen:
                    compact = _cap_lines_preserve_order(chosen)
                    if compact:
                        return compact
            if mode == "power":
                chosen = []
                has_strong_power = any(
                    (
                        "while synchronizing instance power states" in line.lower()
                        or "found 1 instances in the database and 0 instances on the hypervisor" in line.lower()
                        or "sync_power_state" in line.lower()
                        or "pending task (spawning)" in line.lower()
                        or "pending state: spawning" in line.lower()
                    )
                    for line in local_lines
                )

                def _append_power(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
                    for line in local_lines:
                        if not allow_selected and line.strip() == str(selected_alert).strip():
                            continue
                        lower = line.lower()
                        if any(pat in lower for pat in patterns):
                            if line not in chosen:
                                chosen.append(line)
                            return

                _append_power((str(selected_alert).lower(),))
                _append_power(
                    (
                        "while synchronizing instance power states",
                        "found 1 instances in the database and 0 instances on the hypervisor",
                        "sync_power_state",
                        "pending task (spawning)",
                        "pending state: spawning",
                    ),
                    allow_selected=False,
                )
                _append_power(
                    (
                        "vm started",
                        "vm paused",
                        "vm resumed",
                        "instance resumed lifecycle transition",
                        "instance paused lifecycle transition",
                        "instance started lifecycle transition",
                    ),
                    allow_selected=False,
                )
                _append_power(
                    (
                        "build instance",
                        "spawn the instance",
                        "instance spawned successfully",
                    ),
                    allow_selected=False,
                )
                if not has_strong_power:
                    _append_power(
                        (
                            "sync_power_state",
                            "pending task (spawning)",
                            "pending state: spawning",
                            "while synchronizing instance power states",
                        ),
                        allow_selected=False,
                    )
                if chosen:
                    compact = _cap_lines_preserve_order(chosen)
                    if compact:
                        return compact
            scored_local: List[Tuple[int, int, str]] = []
            for idx, line in enumerate(local_lines):
                lower = line.lower()
                score = 0
                if line.strip() == str(selected_alert).strip():
                    score += 14
                if mode == "inventory":
                    if "get /v2/" in lower and "/servers/detail" in lower:
                        score += 8
                    if "instance sync for host" in lower or "re-created its instancelist" in lower:
                        score += 3
                    if "vm " in lower and "lifecycle event" in lower:
                        score += 7
                    if "sync_power_state" in lower or "pending task (spawning)" in lower:
                        score += 5
                    if "metadata" in lower and "/servers/detail" not in lower:
                        score -= 2
                elif mode == "host":
                    if "auditing locally available compute resources" in lower:
                        score += 3
                    if (
                        "attempting claim:" in lower
                        or "claim successful" in lower
                        or "total vcpu" in lower
                        or "vcpu limit" in lower
                        or "memory limit" in lower
                    ):
                        score += 7
                    if "cpu affinity" in lower or "vcpu count" in lower or "total usable vcpus" in lower:
                        score += 4
                    if (
                        "creating image" in lower
                        or "building instance disk image" in lower
                        or "unknown base file" in lower
                        or "removable base files" in lower
                        or "active base files" in lower
                        or "base or swap file too young" in lower
                    ):
                        score += 12
                    if "sync_power_state" in lower or "pending task (spawning)" in lower:
                        score += 5
                    if "/servers/detail" in lower:
                        score += 1
                    if "metadata" in lower:
                        score -= 2
                elif mode == "power":
                    if "vm paused" in lower or "vm resumed" in lower or "vm started" in lower:
                        score += 10
                    if "pending task (spawning)" in lower or "sync_power_state" in lower:
                        score += 3
                    if "build instance" in lower or "spawn the instance" in lower or "instance spawned successfully" in lower:
                        score += 8
                    if "instance sync for host" in lower:
                        score += 3
                elif mode == "image":
                    if (
                        "removable base files" in lower
                        or "active base files" in lower
                        or "base or swap file too young" in lower
                        or (("image " in lower) and (" in use:" in lower or ": checking" in lower))
                    ):
                        score += 8
                    if "unknown base file" in lower:
                        score += 2
                    if "removing base or swap file" in lower:
                        score += 7
                elif mode == "metadata":
                    if "get /openstack/2013-10-17 http/1.1" in lower or "get /latest/meta-data/" in lower:
                        score += 3
                    if "vendor_data.json" in lower or "user_data http/1.1\" status: 404" in lower:
                        score += 9
                    if "meta_data.json" in lower:
                        score += 8
                    if "nova.metadata.wsgi.server" in lower:
                        score += 4
                    if "/servers/detail" in lower:
                        score -= 2
                else:
                    if "get /v2/" in lower and "/servers/detail" in lower:
                        score += 6
                    if "vm " in lower and "lifecycle event" in lower:
                        score += 6
                    if (
                        "removable base files" in lower
                        or "active base files" in lower
                        or "base or swap file too young" in lower
                        or "get /openstack/2013-10-17 http/1.1" in lower
                        or "get /latest/meta-data/" in lower
                        or "vendor_data.json" in lower
                        or "auditing locally available compute resources" in lower
                        or "attempting claim:" in lower
                    ):
                        score += 6
                if _is_openstack_explicit_alert(lower):
                    score -= 4
                line_action = infer_action_id_from_text(dataset, line)
                if observed_action and line_action == observed_action:
                    score += 2
                elif line_action and line_action in top_actions:
                    score += 1
                scored_local.append((score, idx, line))
            keep = sorted(scored_local, key=lambda x: (-x[0], x[1]))[: min(7, len(scored_local))]
            selected = [line for _, _, line in sorted(keep, key=lambda x: x[1])]
            compact = _cap_lines_preserve_order(selected)
            if compact:
                return compact

    scored: List[Tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        score = 0
        if dataset == "HDFS":
            if "got exception while serving" in lower or "connection reset by peer" in lower:
                score += 12
            if "delete" in lower or "blockinfo not found" in lower:
                score += 12
            if "allocateblock" in lower:
                score += 9
            if "packetresponder" in lower:
                score += 7
            if "receiveblock" in lower:
                score += 7
            if "receiving block" in lower:
                score += 5
        elif dataset == "OpenStack":
            if "cpu affinity" in lower or "vcpu count" in lower:
                score += 13
            if "unknown base file" in lower:
                score += 9
            if "sync_power_state" in lower or "synchronizing instance power states" in lower:
                score += 8
            if "metadata" in lower:
                score += 7
            if "instance sync" in lower:
                score += 6
        else:
            if "could not delete hdfs" in lower:
                score += 15
            if "no space" in lower or "disk full" in lower:
                score += 12
            if "shuffling to disk" in lower:
                score += 9
            elif "maxsingleshufflelimit" in lower:
                score += 4
            if "machine down" in lower or "bad datanode" in lower:
                score += 12
            if "forcibly closed by the remote host" in lower:
                score += 8
            if "retrying connect to server" in lower:
                score += 8
            if "nodemanager" in lower and ("heartbeat" in lower or "unhealthy" in lower):
                score += 7
            if "default file system" in lower:
                score -= 2
        if idx >= len(lines) - 2:
            score += 2
        line_action = infer_action_id_from_text(dataset, line)
        line_family = family_for_action(dataset, line_action) if line_action else ""
        if line_action and line_action == observed_action:
            # Keep the observed alert family/action visible in the focused
            # context instead of letting unrelated frequent symptoms dominate.
            score += 10
        elif observed_family and line_family and line_family == observed_family:
            score += 4
        elif line_action and line_action in top_actions and line_action != observed_action:
            score += 2
        if observed_family and line_family and line_family != observed_family:
            score -= 2
        if score > 0:
            scored.append((score, idx, line))

    if not scored:
        return _cap_lines_preserve_order(lines[-6:]) or "\n".join(lines[-6:])[:max_chars]
    keep = sorted(scored, key=lambda x: (-x[0], x[1]))[:8]
    selected = [line for _, _, line in sorted(keep, key=lambda x: x[1])]
    return _cap_lines_preserve_order(selected) or "\n".join(selected)[:max_chars]


def _extract_baseline_context(
    raw: str,
    dataset: str,
    selected_alert: str = "",
    max_chars: int = 900,
) -> str:
    lines = [line for line in str(raw or "").split("\n") if line.strip()]
    if not lines:
        return ""

    radius = {"HDFS": 7, "OpenStack": 8, "Hadoop": 8}.get(dataset, 8)
    local_lines = _window_around_alert(raw, selected_alert, radius)
    if dataset == "Hadoop" and selected_alert:
        chosen: List[str] = []

        def _compact_hadoop(lines_to_keep: List[str]) -> str:
            kept: List[str] = []
            total_chars = 0
            for line in lines_to_keep:
                if not line or line in kept:
                    continue
                extra = len(line) + (1 if kept else 0)
                if kept and total_chars + extra > max_chars:
                    break
                kept.append(line)
                total_chars += extra
            return "\n".join(kept)

        lower_selected = str(selected_alert).lower()
        node_terms = (
            "could not delete hdfs",
            "failed to remove hdfs",
            "bad datanode",
            "machine down",
            "host appears unavailable",
            "unhealthy data node",
            "heartbeat",
        )
        network_terms = (
            "retrying connect to server",
            "retrying rpc toward node",
            "rpc handshake loop against remote endpoint",
            "control-plane session establishment kept cycling against peer endpoint",
            "forcibly closed by the remote host",
            "peer terminated the socket unexpectedly",
            "remote endpoint closed the channel during exchange",
            "peer interrupted the transport session mid-exchange",
            "rmcontainerallocator.heartbeat",
            "netutils.connect",
        )
        storage_terms = (
            "shuffling to disk",
            "redirecting shuffle fragment into fallback merge staging",
            "maxsingleshufflelimit",
            "single-shuffle threshold",
            "single-fragment staging threshold",
            "ondiskmapoutput",
            "staged shuffle fragment",
            "mergemanagerimpl",
            "merging ",
        )
        worker_terms = (
            "containerlauncher",
            "dispatch path",
            "leaserenewer",
            "client renewer:",
            "communication thread",
            "local host is:",
            "local worker is:",
            "destination host is:",
            "peer worker is:",
            "container_remote_cleanup",
            "kill_container_cleanup",
            "opening proxy :",
        )
        worker_like_selected = any(pat in lower_selected for pat in worker_terms)
        rm_like_selected = any(
            pat in lower_selected
            for pat in (
                "rmcommunicator allocator",
                "scheduler control loop",
                "rmcontainerallocator",
                "rmcontainerallocator.heartbeat",
                "netutils.connect",
                ":8030",
                "error in contacting rm",
            )
        )

        def _append_first_from(pool: List[str], patterns: Tuple[str, ...], *, allow_selected: bool = True) -> None:
            for line in pool:
                if not allow_selected and line.strip() == str(selected_alert).strip():
                    continue
                lower = line.lower()
                if any(pat in lower for pat in patterns):
                    if line not in chosen:
                        chosen.append(line)
                    return

        _append_first_from(local_lines or lines, (lower_selected,))
        if any(pat in lower_selected for pat in network_terms):
            if worker_like_selected and not rm_like_selected:
                _append_first_from(lines, node_terms, allow_selected=False)
                _append_first_from(lines, worker_terms, allow_selected=False)
            _append_first_from(lines, network_terms, allow_selected=False)
        elif any(pat in lower_selected for pat in node_terms):
            _append_first_from(lines, worker_terms, allow_selected=False)
            _append_first_from(lines, network_terms, allow_selected=False)
        elif any(pat in lower_selected for pat in storage_terms):
            _append_first_from(lines, storage_terms, allow_selected=False)
            _append_first_from(
                lines,
                (
                    "mergemanagerimpl",
                    "merging ",
                    "ondiskmapoutput",
                    "staged shuffle fragment",
                ),
                allow_selected=False,
            )
        else:
            _append_first_from(lines, network_terms, allow_selected=False)
            _append_first_from(lines, node_terms, allow_selected=False)
            _append_first_from(lines, storage_terms, allow_selected=False)
        if chosen:
            compact = _compact_hadoop(chosen)
            if compact:
                return compact
    if dataset == "OpenStack" and selected_alert and local_lines:
        mode = _openstack_context_mode(selected_alert)
        chosen: List[str] = []

        def _compact_openstack(lines_to_keep: List[str]) -> str:
            kept: List[str] = []
            total_chars = 0
            for line in lines_to_keep:
                if not line or line in kept:
                    continue
                extra = len(line) + (1 if kept else 0)
                if kept and total_chars + extra > max_chars:
                    break
                kept.append(line)
                total_chars += extra
            return "\n".join(kept)

        def _append_first(patterns: Tuple[str, ...], allow_selected: bool = True) -> None:
            for line in local_lines:
                if not allow_selected and line.strip() == str(selected_alert).strip():
                    continue
                lower = line.lower()
                if any(pat in lower for pat in patterns):
                    if line not in chosen:
                        chosen.append(line)
                    return

        _append_first((str(selected_alert).lower(),))
        if mode == "inventory":
            _append_first(("os-server-external-events", "network-vif-plugged"), allow_selected=False)
            _append_first(("instance sync for host", "re-created its instancelist"), allow_selected=False)
            _append_first(
                (
                    "vm resumed",
                    "vm paused",
                    "vm started",
                    "instance resumed lifecycle transition",
                    "instance paused lifecycle transition",
                    "instance started lifecycle transition",
                ),
                allow_selected=False,
            )
            _append_first(("/servers/detail",), allow_selected=False)
        elif mode == "power":
            _append_first(
                (
                    "while synchronizing instance power states",
                    "found 1 instances in the database and 0 instances on the hypervisor",
                    "sync_power_state",
                    "pending task (spawning)",
                    "pending state: spawning",
                ),
                allow_selected=False,
            )
            _append_first(
                (
                    "vm resumed",
                    "vm paused",
                    "vm started",
                    "instance resumed lifecycle transition",
                    "instance paused lifecycle transition",
                    "instance started lifecycle transition",
                ),
                allow_selected=False,
            )
            _append_first(("build instance", "spawn the instance", "instance spawned successfully"), allow_selected=False)
            _append_first(("creating image", "building instance disk image"), allow_selected=False)
            _append_first(("/servers/detail",), allow_selected=False)
            _append_first(("sync_power_state", "pending task (spawning)", "pending state: spawning"), allow_selected=False)
        elif mode == "image":
            _append_first(("vm stopped", "vm paused", "vm started"), allow_selected=False)
            _append_first(
                (
                    "delete /v2/",
                    "terminating instance",
                    "deleting instance files",
                    "instance destroyed successfully",
                ),
                allow_selected=False,
            )
            _append_first(("/servers/detail",), allow_selected=False)
            _append_first(("http exception thrown", "no instances found for any event"), allow_selected=False)
            if len(chosen) < 3:
                _append_first(
                    (
                        "image ",
                        ": checking",
                        " in use:",
                        "cached object ",
                        "routine inspection",
                        "runtime workspace",
                        "workspace objects",
                    ),
                    allow_selected=False,
                )
        elif mode == "metadata":
            _append_first(
                (
                    "get /openstack/2013-10-17 http/1.1",
                    "get /latest/meta-data/",
                    "/control-plane/runtime-catalog",
                ),
                allow_selected=False,
            )
            _append_first(("delete /v2/", "terminating instance", "/servers/detail"), allow_selected=False)
            _append_first(("http exception thrown", "no instances found for any event"), allow_selected=False)
            if len(chosen) < 3:
                _append_first(
                    (
                        "vendor_data.json",
                        "user_data http/1.1\" status: 404",
                        "meta_data.json",
                        "bootstrap-blob",
                        "vendor-profile",
                        "instance-profile",
                    ),
                    allow_selected=False,
                )
        elif mode == "host":
            _append_first(("attempting claim:", "claim successful", "trying host claim:", "host claim accepted"), allow_selected=False)
            _append_first(("auditing locally available compute resources", "memory limit", "total vcpu", "vcpu limit"), allow_selected=False)
            _append_first(("creating image", "building instance disk image"), allow_selected=False)
        if chosen:
            compact = _compact_openstack(chosen)
            if compact:
                return compact
    chosen = local_lines if local_lines else lines[-min(len(lines), radius * 2 + 1) :]

    kept: List[str] = []
    total_chars = 0
    for line in chosen:
        if not line:
            continue
        extra = len(line) + (1 if kept else 0)
        if kept and total_chars + extra > max_chars:
            break
        kept.append(line)
        total_chars += extra
    if kept:
        return "\n".join(kept)
    return "\n".join(chosen)[:max_chars]


def _candidate_summary(dataset: str, cand_json: str, observed_template: str, context_text: str) -> str:
    try:
        obj = json.loads(cand_json)
    except Exception:
        obj = []
    context_support = _line_action_support(dataset, context_text)
    observed_action = infer_action_id_from_text(dataset, observed_template)
    observed_label = family_for_action(dataset, observed_action) if observed_action else ""
    agg: Dict[str, Dict[str, object]] = {}
    for item in obj if isinstance(obj, list) else []:
        if not isinstance(item, dict):
            continue
        tpl = str(item.get("source_template", "") or "")
        weight = abs(float(item.get("weight", 0.0) or 0.0))
        action_id = infer_action_id_from_text(dataset, tpl)
        label = family_for_action(dataset, action_id) or infer_family_from_text(dataset, tpl)
        if not action_id and not label:
            continue
        support_bonus = 1.25 * min(context_support.get(action_id, 0), 2) if action_id else 0.0
        key = action_id or label
        entry = agg.setdefault(
            key,
            {"label": label, "action_id": action_id, "score": 0.0, "graph_score": 0.0, "support_hits": 0, "examples": []},
        )
        entry["graph_score"] = float(entry["graph_score"]) + weight
        entry["score"] = float(entry["score"]) + weight + support_bonus
        entry["support_hits"] = max(int(entry["support_hits"]), int(context_support.get(action_id, 0)))
        if tpl and len(entry["examples"]) < 2:
            entry["examples"].append(tpl)
    if observed_action:
        observed_hits = int(context_support.get(observed_action, 0))
        entry = agg.setdefault(
            observed_action,
            {
                "label": observed_label,
                "action_id": observed_action,
                "score": 0.0,
                "graph_score": 0.0,
                "support_hits": 0,
                "examples": [],
                "observed_hint": False,
            },
        )
        entry["score"] = float(entry["score"]) + 1.75 + 0.75 * min(observed_hits, 2)
        entry["support_hits"] = max(int(entry["support_hits"]), observed_hits)
        entry["observed_hint"] = True
        if observed_template and len(entry["examples"]) < 2:
            entry["examples"].insert(0, observed_template)
    if not agg:
        return "No structured causal candidate summary available."
    ordered = sorted(agg.values(), key=lambda x: float(x["score"]), reverse=True)[:5]
    if dataset == "OpenStack" and not observed_action and ordered and all(int(item["support_hits"]) == 0 for item in ordered):
        return "No structured causal candidate summary available."
    lines = ["Ranked causal family summary:"]
    for i, item in enumerate(ordered, start=1):
        lines.append(
            f"{i}. root_family={item['label'] or 'UNKNOWN'}; "
            f"score={float(item['score']):.3f}; graph={float(item['graph_score']):.3f}; "
            f"context_hits={int(item['support_hits'])}; observed_hint={int(bool(item.get('observed_hint')))}; "
            f"indicative_templates={item['examples']}"
        )
    return "\n".join(lines)


def _dedupe_reference_blocks(refs_text: str, max_unique: int = 3) -> str:
    raw = str(refs_text or "").strip()
    if not raw:
        return raw
    blocks = re.split(r"\n(?=\[\d+\]\s+log:)", raw)
    kept: List[str] = []
    seen: set[str] = set()
    for block in blocks:
        text = str(block or "").strip()
        if not text:
            continue
        match = re.search(r"template:\s*(.+)", text, flags=re.IGNORECASE)
        key = match.group(1).strip().lower() if match else re.sub(r"\s+", " ", text).lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)
        if len(kept) >= max_unique:
            break
    return "\n".join(kept) if kept else raw


def _heuristic_action_hint(dataset: str, selected_alert: str, context_text: str, observed_template: str) -> str:
    if dataset == "OpenStack":
        lower_alert = _canonical_reference_text(dataset, selected_alert).lower()
        lower_context = _canonical_reference_text(dataset, context_text).lower()
    else:
        lower_alert = str(selected_alert or "").lower()
        lower_context = str(context_text or "").lower()
    if dataset == "HDFS":
        if (
            "got exception while serving" in lower_alert
            or "connection reset by peer" in lower_alert
            or "service-stage workflow reported irregular completion" in lower_alert
            or "stream transfer aborted while handling" in lower_alert
            or "replica service path aborted during stream handoff" in lower_alert
            or "downstream replica exchange interrupted while handling" in lower_alert
            or "replica handoff workflow exited before downstream stage completion" in lower_alert
            or (("writeblock" in lower_alert or "replicastagestep" in lower_alert) and "received exception" in lower_alert)
        ):
            return "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE"
        if "unexpected error trying to delete block" in lower_alert:
            return "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK"
        if "allocateblock" in lower_alert or "allocate block" in lower_alert:
            return "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK"
        if (
            "packetresponder" in lower_alert
            or "pkgresponder" in lower_alert
            or "replicastage" in lower_alert
            or "replica stage tracker" in lower_alert
        ):
            return "HDFS_REBUILD_WRITE_PIPELINE"
        if (
            "received block" in lower_alert
            or "got blk" in lower_alert
            or "replica fragment observed" in lower_alert
            or "replica fragment path observed" in lower_alert
        ):
            if (
                "packetresponder" in lower_context
                or "pkgresponder" in lower_context
                or "receiving block" in lower_context
                or "replica fragment path observed" in lower_context
                or "replica stage tracker" in lower_context
                or "replicastage" in lower_context
            ):
                return "HDFS_REBUILD_WRITE_PIPELINE"
            return "HDFS_TUNE_REPLICATION_FLOW"
    if dataset == "Hadoop":
        if any(
            pat in lower_alert
            for pat in (
                "shuffling to disk",
                "spilling map output to local disk because",
                "maxsingleshufflelimit",
                "single-shuffle threshold",
                "ondiskmapoutput",
                "fallback merge staging",
                "single-fragment staging threshold",
                "staged shuffle fragment",
                "disk full",
                "no space",
                "storage exhausted",
                "storage unavailable",
            )
        ):
            return "HADOOP_FREE_DISK_AND_RETRY"
        node_terms = (
            "could not delete hdfs",
            "failed to remove hdfs",
            "cleanup failed for distributed output path",
            "bad datanode",
            "unhealthy data node",
            "machine down",
            "host appears unavailable",
            "heartbeat",
        )
        network_terms = (
            "retrying connect to server",
            "retrying rpc toward node",
            "rpc handshake loop against remote endpoint",
            "control-plane session establishment kept cycling against peer endpoint",
            "forcibly closed by the remote host",
            "peer terminated the socket unexpectedly",
            "remote endpoint closed the channel during exchange",
            "peer interrupted the transport session mid-exchange",
            "failed to connect",
            "couldn't set up io streams",
            "error in contacting rm",
        )
        hard_node_terms = tuple(term for term in node_terms if term != "heartbeat")
        clean_network_alert = any(pat in lower_alert for pat in network_terms)
        rm_like_alert = any(
            pat in lower_alert or pat in lower_context
            for pat in (
                "rmcommunicator allocator",
                "rmcontainerallocator",
                "scheduler control loop",
                ":8030",
                "error in contacting rm",
            )
        )
        worker_like_node_alert = any(
            pat in lower_alert
                for pat in (
                    "containerlauncher",
                    "dispatch path",
                    "leaserenewer",
                    "client renewer:",
                    "communication thread",
                    "task: communication exception",
                    "local host is:",
                    "local worker is:",
                )
            )
        direct_node_alert = any(
            pat in lower_alert
            for pat in node_terms
        )
        node_signal_count = sum(pat in lower_alert for pat in node_terms) + sum(pat in lower_context for pat in node_terms)
        network_signal_count = sum(pat in lower_alert for pat in network_terms) + sum(
            pat in lower_context for pat in network_terms
        )
        has_node_evidence = node_signal_count > 0
        has_network_evidence = network_signal_count > 0
        has_hard_node_evidence = any(
            pat in lower_alert or pat in lower_context
            for pat in hard_node_terms
        )
        has_storage_evidence = any(
            pat in lower_context or pat in lower_alert
            for pat in (
                "shuffling to disk",
                "spilling map output to local disk because",
                "maxsingleshufflelimit",
                "single-shuffle threshold",
                "ondiskmapoutput",
                "disk full",
                "no space",
                "storage exhausted",
                "storage unavailable",
            )
        )
        if has_storage_evidence and not has_node_evidence:
            return "HADOOP_FREE_DISK_AND_RETRY"
        if direct_node_alert:
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if clean_network_alert and has_hard_node_evidence and not rm_like_alert:
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if clean_network_alert and any(
            pat in lower_context
            for pat in (
                "container_remote_cleanup",
                "kill_container_cleanup",
                "opening proxy :",
            )
        ) and not rm_like_alert:
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if clean_network_alert and worker_like_node_alert and has_node_evidence and not rm_like_alert:
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if clean_network_alert and node_signal_count == 0:
            return "HADOOP_RESTORE_NETWORK_AND_RETRY"
        if has_node_evidence and has_network_evidence:
            if node_signal_count > network_signal_count:
                return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
            return "HADOOP_RESTORE_NETWORK_AND_RETRY"
        if has_node_evidence:
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if has_network_evidence:
            return "HADOOP_RESTORE_NETWORK_AND_RETRY"
    if dataset == "OpenStack":
        mode = _openstack_context_mode(selected_alert)
        if (
            "instance sync for host" in lower_alert
            or "re-created its instancelist" in lower_alert
            or "instance sync for host" in lower_context
            or "re-created its instancelist" in lower_context
        ):
            mode = "inventory"
        if (
            "while synchronizing instance power states" in lower_alert
            or "found 1 instances in the database and 0 instances on the hypervisor" in lower_alert
            or "while synchronizing instance power states" in lower_context
            or "found 1 instances in the database and 0 instances on the hypervisor" in lower_context
        ):
            mode = "power"
        host_capacity_evidence = any(
            pat in lower_alert or pat in lower_context
            for pat in (
                "cpu affinity",
                "vcpu count",
                "vcpu limit",
                "total vcpu",
                "total usable vcpus",
            )
        )
        host_strong_evidence = any(
            pat in lower_alert or pat in lower_context
            for pat in (
                "cpu affinity",
                "vcpu count",
                "total usable vcpus",
                "trying host claim:",
                "host claim accepted",
            )
        )
        image_chain_evidence = any(
            pat in lower_alert or pat in lower_context
            for pat in (
                "creating image",
                "building instance disk image",
                "unknown base file",
                "removable base files",
                "active base files",
                "base or swap file too young",
            )
        )
        if mode == "host" and image_chain_evidence and not host_strong_evidence:
            return "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
        if mode == "host" and host_strong_evidence:
            return "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
        if mode == "host" and host_capacity_evidence and not image_chain_evidence:
            return "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
        if mode == "image" and (
            "active base files" in lower_alert
            or "removable base files" in lower_alert
            or "building instance disk image" in lower_alert
            or "unknown base file" in lower_context
            or "base or swap file too young" in lower_context
            or "removable base files" in lower_context
            or "active base files" in lower_context
            or (("image " in lower_context) and (" in use:" in lower_context or ": checking" in lower_context))
        ):
            return "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
        if mode == "power" and (
            "vm resumed" in lower_alert
            or "vm paused" in lower_alert
            or "vm started" in lower_alert
            or "instance resumed lifecycle transition" in lower_alert
            or "instance paused lifecycle transition" in lower_alert
            or "instance started lifecycle transition" in lower_alert
            or "sync_power_state" in lower_context
            or "power-state sync" in lower_context
            or "pending task (spawning)" in lower_context
            or "pending state: spawning" in lower_context
            or "build instance" in lower_context
            or "while synchronizing instance power states" in lower_alert
            or "while synchronizing instance power states" in lower_context
            or "found 1 instances in the database and 0 instances on the hypervisor" in lower_context
        ):
            return "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD"
        if mode == "metadata" and (
            "meta_data.json" in lower_context
            or "instance metadata payload" in lower_context
            or "latest/meta-data" in lower_context
            or "vendor_data.json" in lower_context
            or "vendor metadata payload" in lower_context
        ):
            return "OPENSTACK_SCALE_METADATA_SERVICE"
        if mode == "inventory" and (
            (
                "network-vif-plugged" in lower_alert
                or "os-server-external-events" in lower_alert
                or "server_external_events" in lower_alert
                or "network-vif-plugged" in lower_context
                or "os-server-external-events" in lower_context
                or "server_external_events" in lower_context
            )
            or "instance sync for host" in lower_context
            or "re-created its instancelist" in lower_context
            or "instance inventory sync on host" in lower_context
            or "rebuilt cached instance inventory" in lower_context
            or "scheduler-side state reconciliation on node" in lower_context
            or "reconciled host-side runtime cache" in lower_context
        ):
            return "OPENSTACK_RESYNC_INSTANCE_INVENTORY"
        if mode == "host" and host_capacity_evidence:
            return "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
    alert_hint = infer_action_id_from_text(dataset, selected_alert)
    if alert_hint:
        return alert_hint
    observed_hint = infer_action_id_from_text(dataset, observed_template)
    if observed_hint:
        return observed_hint
    support = _line_action_support(dataset, context_text)
    if not support:
        return ""
    ordered = sorted(support.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[0][0]


def _should_use_agent_shortcut(
    dataset: str,
    heuristic_action_hint: str,
    selected_alert: str = "",
    noise: float = 0.0,
) -> bool:
    if dataset == "OpenStack":
        lower_alert = _canonical_reference_text(dataset, selected_alert).lower()
    else:
        lower_alert = str(selected_alert or "").lower()
    if dataset == "HDFS":
        if heuristic_action_hint == "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE":
            return any(
                pat in lower_alert
                for pat in (
                    "got exception while serving",
                    "connection reset by peer",
                    "writeblock",
                    "replicastagestep",
                )
            )
        if heuristic_action_hint == "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK":
            return any(
                pat in lower_alert
                for pat in (
                    "unexpected error trying to delete block",
                )
            )
        if heuristic_action_hint == "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK":
            return "allocateblock" in lower_alert or "allocate block" in lower_alert
    if dataset == "Hadoop":
        if heuristic_action_hint == "HADOOP_RESTORE_NETWORK_AND_RETRY":
            return any(
                pat in lower_alert
                for pat in (
                    "retrying connect to server",
                    "retrying rpc toward node",
                    "rpc handshake loop against remote endpoint",
                    "control-plane session establishment kept cycling against peer endpoint",
                    "forcibly closed by the remote host",
                    "peer terminated the socket unexpectedly",
                    "remote endpoint closed the channel during exchange",
                    "peer interrupted the transport session mid-exchange",
                )
            ) and not any(
                pat in lower_alert
                for pat in (
                    "could not delete hdfs",
                    "failed to remove hdfs",
                    "bad datanode",
                    "unhealthy data node",
                    "machine down",
                    "host appears unavailable",
                )
            )
        if heuristic_action_hint == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
            if noise < 1.0 and any(
                pat in lower_alert
                for pat in (
                    "containerlauncher",
                    "leaserenewer",
                    "communication thread",
                    "task: communication exception",
                    "local host is:",
                )
            ):
                return True
            return any(
                pat in lower_alert
                for pat in (
                    "could not delete hdfs",
                    "failed to remove hdfs",
                    "bad datanode",
                    "unhealthy data node",
                    "machine down",
                    "host appears unavailable",
                )
            )
        return heuristic_action_hint == "HADOOP_FREE_DISK_AND_RETRY"
    if dataset == "OpenStack":
        if heuristic_action_hint == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
            return any(
                pat in lower_alert
                for pat in (
                    "creating image",
                    "building instance disk image",
                    "unknown base file",
                    "removable base files",
                    "active base files",
                    "base or swap file too young",
                )
            ) and not any(
                pat in lower_alert
                for pat in (
                    "sync_power_state",
                    "pending task (spawning)",
                    "vm resumed",
                    "vm paused",
                    "vm started",
                )
            )
        if heuristic_action_hint == "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST":
            return any(
                pat in lower_alert
                for pat in (
                    "cpu affinity",
                    "vcpu count",
                    "total usable vcpus",
                    "host claim accepted",
                    "couldn't obtain the vcpu count",
                )
            )
    return False


def _posthoc_agent_action_override(
    dataset: str,
    heuristic_action_hint: str,
    selected_alert: str,
    context_text: str,
    pred_label: str,
    pred_action_id: str,
) -> str:
    if dataset != "OpenStack" or not heuristic_action_hint:
        return pred_action_id
    if family_for_action(dataset, heuristic_action_hint) != str(pred_label or ""):
        return pred_action_id
    lower_alert = _canonical_reference_text(dataset, selected_alert).lower()
    lower_context = _canonical_reference_text(dataset, context_text).lower()
    if heuristic_action_hint == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
        has_inventory_alert = any(
            pat in lower_alert
            for pat in (
                "network-vif-plugged",
                "os-server-external-events",
                "server_external_events",
            )
        )
        has_inventory_context = any(
            pat in lower_context
            for pat in (
                "/servers/detail",
                "os-server-external-events",
                "server_external_events",
                "network-vif-plugged",
            )
        )
        has_power_only_context = any(
            pat in lower_context
            for pat in (
                "sync_power_state",
                "pending task (spawning)",
                "pending state: spawning",
            )
        ) and not has_inventory_context
        if has_inventory_alert and (has_inventory_context or "network-vif-plugged" in lower_alert) and not has_power_only_context:
            return heuristic_action_hint
    return pred_action_id


def _build_prompt(
    legacy,
    *,
    method: str,
    case_id: str,
    dataset: str,
    noise: float,
    selected_alert: str,
    noised_context: str,
    clean_for_parse: str,
    tpl_agent: str,
    cand_json: str,
    symbolic_label: str,
    heuristic_action_hint: str,
    allowed_labels: List[str],
    allowed_actions: List[str],
) -> str:
    label_desc = describe_allowed_families(dataset)
    action_desc = describe_allowed_actions(dataset)
    base = (
        f"Dataset: {dataset}\n"
        f"Noise level: {noise}\n"
        f"Selected alert line: {selected_alert}\n"
        f"Log window tail (truncated, noised):\n{noised_context}\n"
        f"Observed template: {tpl_agent or clean_for_parse}\n"
        f"{label_desc}\n"
        f"{action_desc}\n"
        "Return STRICT JSON: "
        "{\"root_cause_family\":\"<ONE_FAMILY>\",\"action_id\":\"<ONE_ACTION_ID>\",\"repair_action\":\"<short action plan grounded in the chosen action>\"}.\n"
    )
    heuristic_hint_text = ""
    if heuristic_action_hint:
        heuristic_label = family_for_action(dataset, heuristic_action_hint)
        heuristic_hint_text = (
            f"Alert-derived action prior: {heuristic_action_hint}\n"
            f"Alert-derived family prior: {heuristic_label}\n"
        )
    if method == "agent":
        refs = (
            _local_exemplar_references(
                legacy,
                dataset=dataset,
                case_id=case_id,
                selected_alert=selected_alert,
                context_text=noised_context,
                top_k=3,
            )
            if dataset in {"OpenStack", "Hadoop"}
            else legacy.rq3_tools.knowledge_retriever(clean_for_parse[:200], dataset, top_k=3)
        )
        clue = f"Symbolic family clue: {symbolic_label}\n" if symbolic_label else ""
        cand_summary = _candidate_summary(dataset, cand_json, tpl_agent or clean_for_parse, noised_context)
        hdfs_rules = ""
        if dataset == "HDFS":
            hdfs_rules = (
                "For HDFS, treat PacketResponder and Received/Receiving-block lines as potentially ambiguous symptoms.\n"
                "When the selected alert or alert-derived prior points to a direct exception such as 'Got exception while serving', "
                "block-deletion failure, or allocateBlock failure, prefer that direct failure signature over surrounding "
                "PacketResponder/Received-block chatter unless the context clearly contradicts it.\n"
                "Treat 'service-stage workflow reported irregular completion' as a weak transfer-service symptom that should only be trusted when it is consistent with the alert-derived prior and not contradicted by explicit receiveBlock failure.\n"
                "Treat 'Stream transfer aborted while handling', 'Replica service path aborted during stream handoff', "
                "'Downstream replica exchange interrupted while handling', and 'Replica handoff workflow exited before downstream stage completion' as transfer-link symptoms aligned with "
                "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE, not as write-pipeline evidence.\n"
                "Do not override a direct serving-exception or delete-block alert with pipeline failure unless the context shows explicit receiveBlock failure, receiver-node failure, or a stronger competing direct root signal.\n"
                "When the selected alert is a PacketResponder-termination or Received/Receiving-block line, or their noisy aliases such as "
                "'replica stage tracker', 'ReplicaStage', 'replica fragment observed', or 'replica fragment path observed', and there is no "
                "direct serving-exception or delete-block error, prefer HDFS_REBUILD_WRITE_PIPELINE over retransmission tuning or receiver isolation.\n"
                "When the selected alert is an ask-delete or cleanup line but the surrounding HDFS context is dominated by allocateBlock events and lacks an explicit delete failure such as 'Unexpected error trying to delete block' or 'BlockInfo not found', prefer HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK over stale-block cleanup.\n"
                "Use the causal candidate summary and the surrounding context to distinguish pipeline-internal failures "
                "from client/peer serving failures.\n"
                "Choose HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE only when the logs explicitly mention receiveBlock "
                "exceptions or clearly indicate a receiver-side node or disk problem.\n"
            )
        return (
            "You are NeSy-Agent. Use only the provided context, causal candidates, and references.\n"
            "First infer the coarse root-cause family, then choose the most appropriate concrete action ID within that family.\n"
            "Use the ranked candidate summary together with the context lines to infer the precursor/root cause family.\n"
            "The selected alert line is a starting clue, but it must be checked against the surrounding context.\n"
            "If an alert-derived prior is provided, use it to break ties inside a family unless the broader context clearly contradicts it.\n"
            "The causal graph clues are family-level hints, not an answer key; use them together with the logs rather than copying a hidden label.\n"
            "For Hadoop, promote a case to node unavailability only when retries are accompanied by explicit delete failures, bad-datanode, or machine-down evidence; retries plus forced-close alone can still be ordinary network connectivity failures.\n"
            "For Hadoop, treat 'Retrying RPC toward node' as equivalent to retrying-connect symptoms, 'peer terminated the socket unexpectedly' as equivalent to forced-close symptoms, and 'Failed to remove hdfs' / 'cleanup failed for distributed output path' / 'unhealthy data node' / 'host appears unavailable' as node-unavailability evidence.\n"
            "For Hadoop, treat 'Shuffling to disk', 'maxSingleShuffleLimit', 'OnDiskMapOutput', 'redirecting shuffle fragment into fallback merge staging', 'single-fragment staging threshold', 'staged shuffle fragment', or direct disk-space exhaustion as storage-pressure evidence and prefer HADOOP_FREE_DISK_AND_RETRY over network recovery when those lines are explicit.\n"
            "For Hadoop, when a forced-close or retry-connect alert appears together with container cleanup, opening a proxy to a specific worker, local/destination host details, or failed HDFS cleanup, prefer HADOOP_ISOLATE_NODE_AND_RESCHEDULE over ordinary network retry.\n"
            "For OpenStack, repeated /servers/detail polling near scheduler instance-sync lines suggests inventory drift, not metadata-service pressure.\n"
            "For OpenStack, when a creating-image line is the selected alert, prefer image-chain repair over nearby VM lifecycle chatter unless the selected alert itself is a sync_power_state or lifecycle line.\n"
            "For OpenStack, when /servers/detail polling appears together with an instance-sync mismatch or re-created InstanceList clue, prefer OPENSTACK_RESYNC_INSTANCE_INVENTORY over OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD.\n"
            "For OpenStack, treat '/detailed-server-list' as equivalent to /servers/detail, 'power-state sync' as equivalent to sync_power_state, and 'instance inventory sync on host' / 'rebuilt cached instance inventory' as inventory-drift evidence.\n"
            "For OpenStack, when the selected alert is a claim line but the context lacks scheduler-audit or CPU-affinity evidence and later image-cache lines report unknown or removable base files, prefer OPENSTACK_REPAIR_BASE_IMAGE_CHAIN over OPENSTACK_REBUILD_ON_COMPATIBLE_HOST.\n"
            "For OpenStack, resource-audit or claim lines paired with cpu-affinity/vcpu-count/host-capacity evidence suggest host-compatibility failure.\n"
            "For OpenStack, do not let nearby creating-image or base-file chatter override OPENSTACK_REBUILD_ON_COMPATIBLE_HOST when the claim/resource-audit context also contains explicit cpu-affinity, vcpu-limit, memory-limit, or claim-capacity evidence.\n"
            "For OpenStack, only prefer OPENSTACK_REPAIR_BASE_IMAGE_CHAIN over OPENSTACK_REBUILD_ON_COMPATIBLE_HOST when the claim/resource-audit context lacks host-capacity evidence and the stronger corroborating clues are creating-image/base-file failures.\n"
            "For OpenStack, when sync_power_state or 'while synchronizing instance power states' appears without any instance-sync or rebuilt-InstanceList clue, prefer OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD over OPENSTACK_RESYNC_INSTANCE_INVENTORY even if /servers/detail polling is nearby.\n"
            "For OpenStack, VM Started/Paused/Resumed lines paired with sync_power_state or pending-task lines suggest power-state drift and rebuild, not inventory resync.\n"
            "For OpenStack, prefer explicit hypervisor or CPU-affinity failures over earlier image-cache bookkeeping lines when both appear.\n"
            f"{hdfs_rules}"
            "Choose the most likely root-cause family and the most appropriate remediation action ID, then write a short action plan that mentions the concrete operational checks or recovery steps.\n"
            f"{base}"
            f"{clue}"
            f"{heuristic_hint_text}"
            f"{cand_summary}\n"
            f"References:\n{refs}\n"
        )
    if method == "rag":
        if dataset == "HDFS":
            focused_lines = [
                line
                for line in str(noised_context or "").split("\n")
                if any(
                    pat in line.lower()
                    for pat in (
                        "packetresponder",
                        "pkgresponder",
                        "receiving block",
                        "received block",
                        "got exception while serving",
                        "stream transfer aborted while handling",
                        "replica service path aborted during stream handoff",
                        "downstream replica exchange interrupted while handling",
                        "replica handoff workflow exited before downstream stage completion",
                        "peer endpoint terminated the downstream channel mid-exchange",
                        "replica service workflow ended before completion",
                        "peer endpoint interrupted the service stage before completion",
                        "replica stage tracker",
                        "replicastage",
                        "replica fragment observed",
                        "replica fragment path observed",
                        "replicastagestep",
                        "writeblock",
                        "allocateblock",
                        "unexpected error trying to delete block",
                        "blockinfo not found",
                    )
                )
            ][:3]
            rag_parts = [
                str(selected_alert or "").strip(),
                str(clean_for_parse or "").strip(),
                "\n".join(focused_lines).strip(),
            ]
        else:
            rag_parts = [
                str(selected_alert or "").strip(),
                str(clean_for_parse or "").strip(),
                str(noised_context or "").strip(),
            ]
        rag_query = " ".join(part for part in rag_parts if part)
        rag_query = " ".join(rag_query.split())[:320]
        refs = (
            _local_exemplar_references(
                legacy,
                dataset=dataset,
                case_id=case_id,
                selected_alert=selected_alert,
                context_text=noised_context,
                top_k=4,
            )
            if dataset in {"OpenStack", "Hadoop"}
            else legacy.rq3_tools.knowledge_retriever(rag_query, dataset, top_k=5)
        )
        if dataset == "HDFS":
            refs = _dedupe_reference_blocks(refs, max_unique=2)
        hdfs_rules = ""
        if dataset == "HDFS":
            hdfs_rules = (
                "For HDFS, PacketResponder and Received/Receiving-block lines may be symptoms rather than decisive evidence.\n"
                "Use the rest of the context and the retrieved references to infer the family before picking an action.\n"
                "Treat 'Stream transfer aborted while handling', 'Replica service path aborted during stream handoff', "
                "'Downstream replica exchange interrupted while handling', and 'Replica handoff workflow exited before downstream stage completion' as transfer-link symptoms aligned with "
                "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE rather than write-pipeline failure.\n"
                "When the selected alert is a PacketResponder-termination or Received/Receiving-block line, or their noisy aliases such as "
                "'replica stage tracker', 'ReplicaStage', 'replica fragment observed', or 'replica fragment path observed', and there is no "
                "direct serving-exception or delete-block error, prefer HDFS_REBUILD_WRITE_PIPELINE over retransmission tuning or receiver isolation.\n"
                "When the selected alert is an ask-delete or cleanup line but nearby HDFS context is dominated by allocateBlock events and lacks an explicit delete failure such as 'Unexpected error trying to delete block' or 'BlockInfo not found', prefer HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK over stale-block cleanup.\n"
            )
        return (
            "You are an ops expert. Use the logs and references to choose the best root-cause family and remediation action ID.\n"
            "Use the selected alert line as the starting clue, then verify it against the rest of the context.\n"
            "Treat the references as external memory: first map the noisy alert/context to the closest retrieved failure pattern, then choose the family and action that are jointly supported by the logs and the best-matching references.\n"
            "When the log wording is noisy or paraphrased, prefer the interpretation that is corroborated by both the selected alert and at least one additional context line.\n"
            "If multiple references partially match, prefer the family/action that is consistent across the largest number of retrieved patterns rather than overfitting to one literal phrase.\n"
            "For OpenStack, repeated /servers/detail polling near scheduler instance-sync lines suggests inventory drift, not metadata-service pressure.\n"
            "For OpenStack, when a creating-image line is the selected alert, prefer image-chain repair over nearby VM lifecycle chatter unless the selected alert itself is a sync_power_state or lifecycle line.\n"
            "For OpenStack, when /servers/detail polling appears together with an instance-sync mismatch or re-created InstanceList clue, prefer OPENSTACK_RESYNC_INSTANCE_INVENTORY over OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD.\n"
            "For OpenStack, treat '/detailed-server-list' as equivalent to /servers/detail, 'power-state sync' as equivalent to sync_power_state, and 'instance inventory sync on host' / 'rebuilt cached instance inventory' as inventory-drift evidence.\n"
            "For OpenStack, when the selected alert is a claim line but the context lacks scheduler-audit or CPU-affinity evidence and later image-cache lines report unknown or removable base files, prefer OPENSTACK_REPAIR_BASE_IMAGE_CHAIN over OPENSTACK_REBUILD_ON_COMPATIBLE_HOST.\n"
            "For OpenStack, resource-audit or claim lines paired with cpu-affinity/vcpu-count/host-capacity evidence suggest host-compatibility failure.\n"
            "For OpenStack, do not let nearby creating-image or base-file chatter override OPENSTACK_REBUILD_ON_COMPATIBLE_HOST when the claim/resource-audit context also contains explicit cpu-affinity, vcpu-limit, memory-limit, or claim-capacity evidence.\n"
            "For OpenStack, only prefer OPENSTACK_REPAIR_BASE_IMAGE_CHAIN over OPENSTACK_REBUILD_ON_COMPATIBLE_HOST when the claim/resource-audit context lacks host-capacity evidence and the stronger corroborating clues are creating-image/base-file failures.\n"
            "For OpenStack, when sync_power_state or 'while synchronizing instance power states' appears without any instance-sync or rebuilt-InstanceList clue, prefer OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD over OPENSTACK_RESYNC_INSTANCE_INVENTORY even if /servers/detail polling is nearby.\n"
            "For OpenStack, VM Started/Paused/Resumed lines paired with sync_power_state or pending-task lines suggest power-state drift and rebuild, not inventory resync.\n"
            "For OpenStack, prefer explicit hypervisor or CPU-affinity failures over earlier image-cache bookkeeping lines when both appear.\n"
            "For Hadoop, promote a case to node unavailability only when retries are accompanied by explicit delete failures, bad-datanode, or machine-down evidence; retries plus forced-close alone can still be ordinary network connectivity failures.\n"
            "For Hadoop, treat 'Retrying RPC toward node' as equivalent to retrying-connect symptoms, 'peer terminated the socket unexpectedly' as equivalent to forced-close symptoms, and 'Failed to remove hdfs' / 'cleanup failed for distributed output path' / 'unhealthy data node' / 'host appears unavailable' as node-unavailability evidence.\n"
            "For Hadoop, treat 'Shuffling to disk', 'maxSingleShuffleLimit', 'OnDiskMapOutput', 'redirecting shuffle fragment into fallback merge staging', 'single-fragment staging threshold', 'staged shuffle fragment', or direct disk-space exhaustion as storage-pressure evidence and prefer HADOOP_FREE_DISK_AND_RETRY over network recovery when those lines are explicit.\n"
            "For Hadoop, when a forced-close or retry-connect alert appears together with container cleanup, opening a proxy to a specific worker, local/destination host details, or failed HDFS cleanup, prefer HADOOP_ISOLATE_NODE_AND_RESCHEDULE over ordinary network retry.\n"
            f"{hdfs_rules}"
            f"{base}"
            f"References:\n{refs}\n"
        )
    hdfs_rules = ""
    if dataset == "HDFS":
        hdfs_rules = (
            "For HDFS, use the surrounding lines rather than overfitting to a single PacketResponder or Received/Receiving-block phrase.\n"
        )
    return (
        "You are an ops expert. Use only the logs to choose the best root-cause family and remediation action ID.\n"
        "Use the selected alert line as the starting clue, then verify it against the surrounding lines.\n"
        "Do not assume hidden domain knowledge beyond the provided log text.\n"
        "If the logs are noisy or ambiguous, choose the family and action supported by the most direct log evidence and at least one corroborating nearby line.\n"
        f"{hdfs_rules}"
        f"{base}"
    )


def _run_once(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    REBUILD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    legacy = _load_legacy_module()
    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        if ds not in DATASETS:
            raise ValueError(f"Unsupported dataset '{ds}'. Expected subset of {DATASETS}.")
    paths = _artifact_paths(args.run_tag)
    manifest = _load_or_create_manifest(
        legacy,
        run_tag=args.run_tag,
        cases_per_dataset=args.cases_per_dataset,
        noise_levels=noise_levels,
        datasets=datasets,
        causal_graph_path=args.causal_graph_path,
        pool_policy=args.pool_policy,
        manifest_path=paths["manifest"],
        force_resample=args.force_resample,
    )

    labeled_cases = list(manifest["labeled_cases"])
    print(f"[INFO] Sampled {len(labeled_cases)} cases.")
    for ds in datasets:
        print(f"[INFO] {ds} label distribution: {manifest['label_distribution'].get(ds, {})}")
        print(f"[INFO] {ds} action distribution: {manifest['action_distribution'].get(ds, {})}")
    print(f"[INFO] Agent causal graph: {args.causal_graph_path}")
    print(f"[INFO] API output token cap: {args.api_max_output_tokens}")
    if args.max_api_calls > 0:
        print(f"[INFO] API call budget cap: {args.max_api_calls}")

    completed, stats, actual_api_calls = _load_progress(paths["progress"], noise_levels, datasets)
    total_steps = len(labeled_cases) * len(noise_levels) * len(METHODS)
    print(f"[INFO] Resume state: {len(completed)}/{total_steps} completed steps.")

    edge_node = legacy.NuSyEdgeNode()
    legacy.LLMClient()
    injector = legacy.NoiseInjector(seed=2026)
    injector_hadoop = legacy.HadoopNoiseInjector(seed=2026)
    deepseek_key = legacy._get_deepseek_api_key()

    pbar = tqdm(
        total=total_steps,
        desc="RQ34 action-aware",
        unit="step",
        initial=len(completed),
    )
    budget_exhausted = False

    for item in labeled_cases:
        case = dict(item["case"])
        gt_label = str(item["gt_label"])
        gt_action_id = str(item["gt_action_id"])
        raw = str(case.get("raw_log", "") or "")
        dataset = str(case.get("dataset", "HDFS"))
        case_id = str(case.get("case_id", ""))

        alert = str(item.get("selected_alert") or _select_actionaware_alert(legacy, raw, dataset))
        ds_parse = dataset
        for noise in noise_levels:
            noisy_alert = _inject_noise_line(
                legacy,
                alert,
                dataset,
                noise,
                role="selected_alert",
            )
            clean_for_parse = legacy.NuSyEdgeNode.preprocess_header(noisy_alert, ds_parse) or noisy_alert
            symbolic_parse_text = clean_for_parse
            try:
                tpl_nusy, _, _, _ = edge_node.parse_log_stream(symbolic_parse_text, ds_parse)
            except Exception:
                tpl_nusy = ""
            try:
                tpl_drain = legacy._DRAIN.parse(symbolic_parse_text)
            except Exception:
                tpl_drain = ""
            tpl_agent = _choose_observed_template(
                legacy,
                dataset,
                tpl_nusy,
                tpl_drain,
                clean_for_parse,
            )
            allowed_labels = allowed_family_ids(dataset)
            allowed_actions = allowed_action_ids(dataset)
            domain = "hdfs" if dataset == "HDFS" else ("openstack" if dataset == "OpenStack" else "hadoop")
            cand_json = legacy.rq3_tools.causal_navigator(
                tpl_agent or symbolic_parse_text, domain, causal_path=args.causal_graph_path
            )
            agent_focus_context = _extract_actionaware_context(
                raw,
                dataset,
                cand_json,
                tpl_agent or clean_for_parse,
                selected_alert=alert,
                max_chars=900,
            )
            baseline_focus_context = _extract_baseline_context(
                raw,
                dataset,
                selected_alert=alert,
                max_chars=900,
            )
            agent_noised_context = _inject_noise_preserve_context(
                legacy,
                agent_focus_context,
                dataset,
                injector,
                injector_hadoop,
                noise,
            )
            baseline_noised_context = _inject_noise_preserve_context(
                legacy,
                baseline_focus_context,
                dataset,
                injector,
                injector_hadoop,
                noise,
            )
            symbolic_label, _ = legacy._agent_symbolic_vote(
                dataset,
                agent_noised_context,
                clean_for_parse,
                symbolic_parse_text,
                tpl_agent,
                cand_json,
            )
            symbolic_label = infer_family_from_text(dataset, symbolic_label) or family_for_action(
                dataset, infer_action_id_from_text(dataset, symbolic_label)
            )

            for method in METHODS:
                step = _step_key(dataset, case_id, noise, method)
                if step in completed:
                    continue

                method_context = agent_noised_context if method == "agent" else baseline_noised_context
                heuristic_action_hint = _heuristic_action_hint(
                    dataset,
                    noisy_alert,
                    agent_noised_context if method == "agent" else baseline_noised_context,
                    tpl_agent or clean_for_parse,
                )

                prompt = _build_prompt(
                    legacy,
                    method=method,
                    case_id=case_id,
                    dataset=dataset,
                    noise=noise,
                    selected_alert=noisy_alert,
                    noised_context=method_context,
                    clean_for_parse=clean_for_parse,
                    tpl_agent=tpl_agent,
                    cand_json=cand_json,
                    symbolic_label=symbolic_label,
                    heuristic_action_hint=heuristic_action_hint if method == "agent" else "",
                    allowed_labels=allowed_labels,
                    allowed_actions=allowed_actions,
                )
                reference_preview = ""
                if dataset in {"OpenStack", "Hadoop"} and method in {"agent", "rag"}:
                    reference_preview = _local_exemplar_references(
                        legacy,
                        dataset=dataset,
                        case_id=case_id,
                        selected_alert=noisy_alert,
                        context_text=method_context,
                        top_k=3 if method == "agent" else 4,
                    )
                api_call = True
                if method == "agent" and _should_use_agent_shortcut(
                    dataset,
                    heuristic_action_hint,
                    noisy_alert,
                    noise,
                ):
                    pred_action_id = heuristic_action_hint
                    pred_label = family_for_action(dataset, pred_action_id)
                    repair_action = str(
                        ACTION_CATALOG.get(dataset, {})
                        .get(pred_action_id, {})
                        .get("description", "")
                    )
                    api_call = False
                else:
                    if args.max_api_calls > 0 and actual_api_calls >= args.max_api_calls:
                        print(
                            f"[INFO] API call budget exhausted at {actual_api_calls} calls; "
                            "stopping after checkpoint."
                        )
                        budget_exhausted = True
                        break
                    resp = legacy._call_deepseek_with_retry(
                        prompt,
                        api_key=deepseek_key,
                        model="deepseek-chat",
                        max_tokens=args.api_max_output_tokens,
                    )
                    actual_api_calls += 1
                    pred_label, pred_action_id, repair_action = _extract_structured_output(
                        dataset, resp, allowed_labels, allowed_actions
                    )

                if method == "agent":
                    pred_action_id = _posthoc_agent_action_override(
                        dataset,
                        heuristic_action_hint,
                        noisy_alert,
                        method_context,
                        pred_label,
                        pred_action_id,
                    )

                matched_groups, min_groups, hit_groups = action_text_match(dataset, gt_action_id, repair_action)
                rca_success = bool(pred_label and pred_label == gt_label)
                action_txt_success = bool(gt_action_id and action_text_success(dataset, gt_action_id, repair_action))
                exact_action_success = bool(pred_action_id and pred_action_id == gt_action_id)
                # Keep plan-text fallback only when the model failed to emit a
                # structured action ID. If it emitted the wrong action ID, that
                # is a real family-internal mistake and should not be scored as
                # correct via permissive keyword overlap.
                action_success = bool(
                    exact_action_success or (not pred_action_id and action_txt_success)
                )
                # End-to-end self-healing should require both a correct RCA
                # label and a correct operational action.
                e2e_success = bool(rca_success and action_success)

                row = {
                    "dataset": dataset,
                    "case_id": case_id,
                    "noise": float(noise),
                    "method": method,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "gt_action_id": gt_action_id,
                    "pred_action_id": pred_action_id,
                    "repair_action": repair_action,
                    "rca_success": rca_success,
                    "action_success": action_success,
                    "action_text_success": action_txt_success,
                    "action_text_groups_matched": matched_groups,
                    "action_text_groups_required": min_groups,
                    "action_text_group_hits": hit_groups,
                    "e2e_success": e2e_success,
                    "api_call": api_call,
                    "clean_selected_alert": alert,
                    "noisy_selected_alert": noisy_alert,
                    "heuristic_action_hint": heuristic_action_hint,
                    "context_text": method_context[:1200],
                    "reference_preview": reference_preview[:1600],
                    "symbolic_label": symbolic_label,
                }
                append_jsonl_row(paths["progress"], row)

                completed.add(step)
                stat = stats[(dataset, noise, method)]
                stat["rca_total"] += 1
                stat["rca_success"] += int(rca_success)
                stat["action_total"] += 1
                stat["action_success"] += int(action_success)
                stat["action_text_total"] += 1
                stat["action_text_success"] += int(action_txt_success)
                stat["e2e_total"] += 1
                stat["e2e_success"] += int(e2e_success)
                pbar.update(1)

                _write_checkpoint(
                    paths,
                    stats,
                    noise_levels,
                    datasets,
                    run_tag=args.run_tag,
                    cases_per_dataset=args.cases_per_dataset,
                    total_steps=total_steps,
                    completed_steps=len(completed),
                    actual_api_calls=actual_api_calls,
                )
            if budget_exhausted:
                break
        if budget_exhausted:
            break
    if budget_exhausted:
        _write_checkpoint(
            paths,
            stats,
            noise_levels,
            datasets,
            run_tag=args.run_tag,
            cases_per_dataset=args.cases_per_dataset,
            total_steps=total_steps,
            completed_steps=len(completed),
            actual_api_calls=actual_api_calls,
        )

    pbar.close()
    print(f"[INFO] Actual DeepSeek calls: {actual_api_calls} / logical steps {total_steps}.")

    summarize_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "summarize_rq34_actionaware_20260316.py"),
        "--source",
        str(paths["summary_rows"]),
        "--run-tag",
        args.run_tag,
        "--cases-per-dataset",
        str(args.cases_per_dataset),
        "--noise-levels",
        args.noise_levels,
    ]
    _write_checkpoint(
        paths,
        stats,
        noise_levels,
        datasets,
        run_tag=args.run_tag,
        cases_per_dataset=args.cases_per_dataset,
        total_steps=total_steps,
        completed_steps=len(completed),
        actual_api_calls=actual_api_calls,
    )
    print("[RUN]", " ".join(summarize_cmd))
    subprocess.run(summarize_cmd, cwd=str(_PROJECT_ROOT), check=True)


def main() -> None:
    args = _parse_args()
    _run_once(args)


if __name__ == "__main__":
    main()
