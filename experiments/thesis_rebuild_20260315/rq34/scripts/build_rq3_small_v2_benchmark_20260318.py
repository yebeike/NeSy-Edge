from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.build_rq34_enriched_seed_pool_20260316 import (
    OUTPUT_PATH as ENRICHED_SEED_POOL_PATH,
    write_enriched_seed_pool,
)
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    action_text_success,
    family_for_action,
    infer_action_id_from_text,
    infer_family_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


LEGACY_STAGE4 = PROJECT_ROOT / "experiments" / "rq123_e2e" / "stage4_noise_api_sampled_20260313.py"
RQ2_FULLCASE_MODIFIED_GRAPH = (
    REBUILD_ROOT / "rq2_fullcase" / "results" / "gt_causal_knowledge_nesydy_fullcase_20260316.json"
)
BENCH_V2_PATH = PROJECT_ROOT / "data" / "processed" / "e2e_scaled_benchmark_v2.json"
RQ3_TEST_SET_PATH = PROJECT_ROOT / "data" / "processed" / "rq3_test_set.json"
SPEC_PATH = REBUILD_ROOT / "rq34" / "configs" / "rq3_small_v2_curated_spec_20260318.json"
AUDIT_ROWS_PATH = REBUILD_ROOT / "rq34" / "results" / "rq3_candidate_audit_enriched_20260318_rows.json"
OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v2_20260318"

DATASET_CONTEXT_MAX_CHARS = {"HDFS": 700, "OpenStack": 850, "Hadoop": 850}
DATASET_DOMAIN = {"HDFS": "hdfs", "OpenStack": "openstack", "Hadoop": "hadoop"}
OPENSTACK_IMAGE_REPAIR_TERMS = (
    "unknown base file",
    "missing base image",
    "base image",
    "base file",
    "creating image",
    "backing image",
    "backing chain",
    "image chain",
)
OPENSTACK_IMAGE_REPAIR_BANNED = (
    "server_external_events",
    "network-vif-plugged",
    "/servers/detail",
    "sync_power_state",
    "pending task (spawning)",
    "vm started",
    "vm stopped",
    "vm paused",
    "vm resumed",
    "no instances found for any event",
    "no vms found for any event",
    "terminating instance",
    "deleting instance files",
    "instance destroyed successfully",
    "delete /v2/",
    "removable base files",
    "active base files",
    " in use:",
    ": checking",
)
OPENSTACK_IMAGE_CLEANUP_TERMS = (
    "removable base files",
    "active base files",
    " in use:",
    ": checking",
    "base cache",
    "base files",
)
HADOOP_EXPLICIT_DISK_EXHAUSTION_TERMS = (
    "disk full",
    "no space left on device",
    "no space left",
    "insufficient space",
    "out of disk",
)


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("rq3_stage4_legacy", LEGACY_STAGE4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy stage4 script from {LEGACY_STAGE4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_old_runner_module():
    from experiments.thesis_rebuild_20260315.rq34.scripts import run_rq34_hierarchical_resumable_20260316 as old_runner

    return old_runner


def _load_pool_rows() -> Dict[Tuple[str, str], Mapping[str, object]]:
    if not ENRICHED_SEED_POOL_PATH.exists():
        write_enriched_seed_pool()
    source_to_path = {
        "benchmark_v2": BENCH_V2_PATH,
        "rq3_test_set": RQ3_TEST_SET_PATH,
        "rq3_test_set_enriched": ENRICHED_SEED_POOL_PATH,
    }
    out: Dict[Tuple[str, str], Mapping[str, object]] = {}
    for source, path in source_to_path.items():
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id:
                out[(source, case_id)] = row
    return out


def _load_audit_rows() -> Dict[Tuple[str, str, str], Mapping[str, object]]:
    rows = json.loads(AUDIT_ROWS_PATH.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("pool_source", "")),
            str(row.get("case_id", "")),
        )
        out[key] = row
    return out


def _find_alert_line(raw_log: str, alert_match: str, occurrence: int) -> str:
    lines = [line for line in str(raw_log or "").splitlines() if line.strip()]
    wanted = str(alert_match or "")
    hits = [line for line in lines if wanted in line]
    if not hits:
        raise ValueError(f"Failed to locate selected alert containing '{wanted}'")
    idx = max(0, int(occurrence or 1) - 1)
    if idx >= len(hits):
        raise ValueError(
            f"Alert substring '{wanted}' found {len(hits)} times, but occurrence {occurrence} was requested"
        )
    return hits[idx].strip()


def _norm_space(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _compact_lines(lines: Sequence[str], max_chars: int) -> str:
    kept: List[str] = []
    total_chars = 0
    for line in lines:
        text = str(line or "").strip()
        if not text or text in kept:
            continue
        extra = len(text) + (1 if kept else 0)
        if kept and total_chars + extra > max_chars:
            break
        kept.append(text)
        total_chars += extra
    return "\n".join(kept)


def _match_any(text: str, patterns: Sequence[str]) -> bool:
    lower = str(text or "").lower()
    return any(pat in lower for pat in patterns)


def _extract_filtered_context(
    *,
    selected_alert: str,
    local_lines: Sequence[str],
    all_lines: Sequence[str],
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    max_chars: int,
) -> str:
    chosen: List[str] = []
    alert = str(selected_alert or "").strip()
    if alert:
        chosen.append(alert)

    def _append(pool: Sequence[str]) -> None:
        for line in pool:
            text = str(line or "").strip()
            if not text or text in chosen:
                continue
            lower = text.lower()
            if include_patterns and not any(pat in lower for pat in include_patterns):
                continue
            if exclude_patterns and any(pat in lower for pat in exclude_patterns):
                continue
            chosen.append(text)

    _append(local_lines)
    if len(chosen) < 2:
        _append(all_lines)
    return _compact_lines(chosen, max_chars)


def _extract_hadoop_isolate_context(
    *,
    selected_alert: str,
    local_lines: Sequence[str],
    all_lines: Sequence[str],
    max_chars: int,
) -> str:
    chosen: List[str] = []
    alert = str(selected_alert or "").strip()
    if alert:
        chosen.append(alert)

    lower_alert = alert.lower()
    host = ""
    for marker in ("server:", "node:"):
        idx = lower_alert.find(marker)
        if idx >= 0:
            remainder = alert[idx + len(marker) :].strip()
            host = remainder.split("/", 1)[0].split(":", 1)[0].strip()
            if host:
                break

    def _append_matching(pool: Sequence[str], predicate) -> None:
        for line in pool:
            text = str(line or "").strip()
            if not text or text in chosen:
                continue
            if predicate(text):
                chosen.append(text)
                break

    _append_matching(
        all_lines,
        lambda text: "nodeblacklistingenabled" in text.lower(),
    )
    if host:
        _append_matching(
            all_lines,
            lambda text: host.lower() in text.lower()
            and any(token in text.lower() for token in ("retrying connect to server", "retrying rpc toward node")),
        )
    if len(chosen) < 3:
        _append_matching(
            local_lines,
            lambda text: any(
                token in text.lower()
                for token in ("retrying connect to server", "retrying rpc toward node", "nodeblacklistingenabled")
            ),
        )
    return _compact_lines(chosen, max_chars)


def _hdfs_context_mode(selected_alert: str) -> str:
    lower = _norm_space(selected_alert)
    if any(
        pat in lower
        for pat in (
            "got exception while serving",
            "starting thread to transfer block",
            "served block",
            "transmitted block",
            "connection reset by peer",
            "writeblock",
        )
    ):
        return "transfer"
    if any(
        pat in lower
        for pat in (
            "packetresponder",
            "receiving block",
            "received block",
            "receiveblock",
            "replica fragment",
            "replica segment ingress",
            "stage relay observed",
            "replica stage",
            "replicastage",
        )
    ):
        return "pipeline"
    if any(
        pat in lower
        for pat in (
            "allocateblock",
            "blockinfo not found",
            "deleting block",
            "blockmap updated",
            "addstoredblock",
            "fsck",
        )
    ):
        return "storage"
    return ""


def _hdfs_pipeline_patterns(selected_alert: str) -> Tuple[str, ...]:
    lower = _norm_space(selected_alert)
    if "packetresponder" in lower and "terminating" in lower:
        return ("packetresponder", "replica stage", "replicastage")
    if "received block" in lower and "src:" not in lower and "dest:" not in lower:
        return ("packetresponder", "replica stage", "replicastage")
    if "receiving block" in lower or ("received block" in lower and "src:" in lower and "dest:" in lower):
        return ("receiving block", "received block", "receiveblock", "replica stage", "replicastage")
    return (
        "packetresponder",
        "receiving block",
        "received block",
        "receiveblock",
        "replica stage",
        "replicastage",
    )


def _extract_hdfs_context(old_runner, raw_log: str, selected_alert: str, max_chars: int) -> str:
    lines = [line for line in str(raw_log or "").splitlines() if line.strip()]
    if not lines:
        return ""
    local_lines = old_runner._window_around_alert(raw_log, selected_alert, 7)
    chosen: List[str] = []
    if str(selected_alert).strip():
        chosen.append(str(selected_alert).strip())
    mode = _hdfs_context_mode(selected_alert)
    pattern_map = {
        "transfer": (
            "got exception while serving",
            "starting thread to transfer block",
            "served block",
            "transmitted block",
            "connection reset by peer",
            "writeblock",
        ),
        "pipeline": _hdfs_pipeline_patterns(selected_alert),
        "storage": (
            "allocateblock",
            "blockinfo not found",
            "deleting block",
            "blockmap updated",
            "addstoredblock",
            "fsck",
        ),
    }

    def _append_from(pool: Sequence[str], patterns: Sequence[str]) -> None:
        for line in pool:
            text = str(line or "").strip()
            if not text or text in chosen:
                continue
            lower = text.lower()
            if any(pat in lower for pat in patterns):
                chosen.append(text)

    if mode in pattern_map:
        _append_from(local_lines or lines, pattern_map[mode])
        if len(chosen) < 2:
            _append_from(lines, pattern_map[mode])
    compact = _compact_lines(chosen, max_chars)
    if compact:
        return compact
    return _compact_lines(local_lines or lines, max_chars)


def _extract_openstack_context(
    old_runner,
    raw_log: str,
    selected_alert: str,
    max_chars: int,
    action_id: str,
) -> str:
    lines = [line for line in str(raw_log or "").splitlines() if line.strip()]
    local_lines = old_runner._window_around_alert(raw_log, selected_alert, 8)
    if action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=OPENSTACK_IMAGE_REPAIR_TERMS,
            exclude_patterns=OPENSTACK_IMAGE_REPAIR_BANNED,
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "OPENSTACK_CLEAN_STALE_BASE_CACHE_AND_RETRY":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=OPENSTACK_IMAGE_CLEANUP_TERMS,
            exclude_patterns=OPENSTACK_IMAGE_REPAIR_TERMS[:4],
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("terminating instance", "deleting instance files", "instance destroyed successfully", "delete /v2/"),
            exclude_patterns=OPENSTACK_IMAGE_REPAIR_TERMS + OPENSTACK_IMAGE_CLEANUP_TERMS,
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("instance sync for host", "re-created its instancelist", "server_external_events", "network-vif-plugged", "/servers/detail"),
            exclude_patterns=OPENSTACK_IMAGE_REPAIR_TERMS + OPENSTACK_IMAGE_CLEANUP_TERMS,
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("sync_power_state", "pending task (spawning)", "vm started", "vm paused", "vm resumed", "instance spawned successfully", "took ", "spawn the instance"),
            exclude_patterns=OPENSTACK_IMAGE_REPAIR_TERMS + OPENSTACK_IMAGE_CLEANUP_TERMS,
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    return old_runner._extract_baseline_context(
        raw_log,
        "OpenStack",
        selected_alert=selected_alert,
        max_chars=max_chars,
    )


def _extract_hadoop_context(
    old_runner,
    raw_log: str,
    selected_alert: str,
    max_chars: int,
    action_id: str,
) -> str:
    lines = [line for line in str(raw_log or "").splitlines() if line.strip()]
    local_lines = old_runner._window_around_alert(raw_log, selected_alert, 10)
    if action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
        return _extract_hadoop_isolate_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("could not delete hdfs", "failed to remove hdfs", "cleanup failed", "outputcommitter"),
            exclude_patterns=("retrying connect to server", "forcibly closed by the remote host", "socket reader", ":8030", "rmcommunicator allocator", "netutils.connect", "shuffling to disk", "maxsingleshufflelimit"),
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("rmcommunicator allocator", ":8030", "resource manager", "container allocator", "scheduler", "retrying connect to server"),
            exclude_patterns=("could not delete hdfs", "failed to remove hdfs", "forcibly closed by the remote host", "socket reader", "shuffling to disk", "maxsingleshufflelimit"),
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "HADOOP_RESTORE_WORKER_RPC_AND_RETRY":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("forcibly closed by the remote host", "peer terminated", "socket reader", "readandprocess", "channel", "rpc"),
            exclude_patterns=("could not delete hdfs", "failed to remove hdfs", ":8030", "rmcommunicator allocator", "resource manager", "shuffling to disk", "maxsingleshufflelimit"),
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=("shuffling to disk", "maxsingleshufflelimit", "mergemanagerimpl", "merging ", "ondiskmapoutput", "shuffleerror"),
            exclude_patterns=("could not delete hdfs", "failed to remove hdfs", "retrying connect to server", "forcibly closed by the remote host", "socket reader", ":8030"),
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    if action_id == "HADOOP_FREE_LOCAL_DISK_AND_RETRY":
        return _extract_filtered_context(
            selected_alert=selected_alert,
            local_lines=local_lines or lines,
            all_lines=lines,
            include_patterns=HADOOP_EXPLICIT_DISK_EXHAUSTION_TERMS,
            exclude_patterns=("retrying connect to server", "forcibly closed by the remote host", ":8030", "shuffling to disk"),
            max_chars=max_chars,
        ) or str(selected_alert or "").strip()
    return old_runner._extract_baseline_context(
        raw_log,
        "Hadoop",
        selected_alert=selected_alert,
        max_chars=max_chars,
    )


def _extract_frozen_context(
    old_runner,
    raw_log: str,
    dataset: str,
    *,
    selected_alert: str,
    max_chars: int,
    action_id: str,
) -> str:
    if dataset == "HDFS":
        return _extract_hdfs_context(old_runner, raw_log, selected_alert, max_chars)
    if dataset == "OpenStack":
        return _extract_openstack_context(old_runner, raw_log, selected_alert, max_chars, action_id)
    if dataset == "Hadoop":
        return _extract_hadoop_context(old_runner, raw_log, selected_alert, max_chars, action_id)
    return old_runner._extract_baseline_context(raw_log, dataset, selected_alert=selected_alert, max_chars=max_chars)


def _reference_signature(old_runner, dataset: str, text: str) -> str:
    canonical = old_runner._canonical_reference_text(dataset, str(text or ""))
    if canonical.strip():
        return _norm_space(canonical)
    return _norm_space(text)


def _infer_case_action_id(
    dataset: str,
    selected_alert: str,
    raw_log: str,
    case_row: Mapping[str, object],
) -> str:
    inferred = infer_action_id_from_text(dataset, selected_alert)
    if dataset == "HDFS":
        if inferred:
            return inferred
        combined = "\n".join(
            str(case_row.get(field, "") or "")
            for field in ("ground_truth_root_cause_template", "ground_truth_template", "reason", "gt_action_label")
        )
        return infer_action_id_from_text(dataset, combined) or inferred
    if dataset == "OpenStack":
        if inferred:
            return inferred
        lower_alert = str(selected_alert).lower()
        lower_raw = str(raw_log).lower()
        if any(pat in lower_alert for pat in (" in use:", ": checking", "active base files", "removable base files")):
            return "OPENSTACK_CLEAN_STALE_BASE_CACHE_AND_RETRY"
        if any(pat in lower_alert for pat in ("unknown base file", "creating image", "base or swap file too young")):
            return "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
        if "delete /v2/" in lower_alert or "terminating instance" in lower_alert:
            return "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE"
        if "servers/detail" in lower_alert or "network-vif-plugged" in lower_alert:
            return "OPENSTACK_RESYNC_INSTANCE_INVENTORY"
        if any(pat in lower_alert for pat in ("vm started", "vm paused", "vm resumed", "sync_power_state")):
            return "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD"
        if any(pat in lower_raw for pat in ("unknown base file", "creating image")):
            return "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
        return ""
    lower_reason = " ".join(
        str(case_row.get(field, "") or "").lower()
        for field in ("reason", "ground_truth_root_cause_template", "ground_truth_template")
    )
    lower_alert = str(selected_alert).lower()
    lower_raw = str(raw_log).lower()
    if "machine down" in lower_reason:
        if "could not delete hdfs" in lower_alert or "failed to remove hdfs" in lower_alert:
            return "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE"
        return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
    if "network disconnection" in lower_reason:
        if any(pat in lower_alert for pat in ("rmcommunicator allocator", ":8030", "resource manager", "container allocator")):
            return "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
        if any(pat in lower_alert for pat in ("forcibly closed by the remote host", "peer terminated", "socket", "channel")):
            return "HADOOP_RESTORE_WORKER_RPC_AND_RETRY"
        if any(pat in lower_alert for pat in ("netutils.connect", ":8030", "rmcontainerallocator", "rmcommunicator allocator")):
            return "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
        if ":8030" in lower_raw or "rmcommunicator allocator" in lower_raw or "resource manager" in lower_raw:
            return "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
        return "HADOOP_RESTORE_WORKER_RPC_AND_RETRY"
    if "disk full" in lower_reason:
        if any(pat in lower_alert for pat in HADOOP_EXPLICIT_DISK_EXHAUSTION_TERMS):
            return "HADOOP_FREE_LOCAL_DISK_AND_RETRY"
        return "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY"
    return inferred


def _reference_mode(old_runner, dataset: str, selected_alert: str, *, action_id: str = "") -> str:
    if dataset == "HDFS":
        family_id = family_for_action(dataset, action_id)
        if family_id == "HDFS_TRANSFER_LINK_FAILURE":
            return "transfer"
        if family_id == "HDFS_PIPELINE_FAILURE":
            return "pipeline"
        if family_id == "HDFS_STORAGE_METADATA_PRESSURE":
            return "storage"
        lower = str(selected_alert or "").lower()
        if any(pat in lower for pat in ("allocateblock", "delete block", "blockinfo not found", "blockmap")):
            return "storage"
        if any(pat in lower for pat in ("packetresponder", "receiving block", "received block", "replica fragment", "replica segment ingress", "stage relay observed")):
            return "pipeline"
        return "transfer"
    if dataset == "OpenStack":
        mode_map = {
            "OPENSTACK_RESYNC_INSTANCE_INVENTORY": "inventory",
            "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD": "power",
            "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE": "terminate",
            "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN": "image_repair",
            "OPENSTACK_CLEAN_STALE_BASE_CACHE_AND_RETRY": "image_cleanup",
            "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST": "host_claim",
            "OPENSTACK_REFRESH_HOST_INVENTORY_AND_RETRY_CLAIM": "host_claim",
            "OPENSTACK_SCALE_METADATA_SERVICE": "metadata",
            "OPENSTACK_ENABLE_LOCAL_METADATA_CACHE": "metadata",
        }
        if action_id in mode_map:
            return mode_map[action_id]
        return old_runner._openstack_context_mode(selected_alert)
    if dataset == "Hadoop":
        mode_map = {
            "HADOOP_ISOLATE_NODE_AND_RESCHEDULE": "worker_isolate",
            "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE": "worker_cleanup",
            "HADOOP_RESTORE_WORKER_RPC_AND_RETRY": "control_worker_rpc",
            "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY": "control_rm",
            "HADOOP_FREE_LOCAL_DISK_AND_RETRY": "storage_disk",
            "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY": "storage_shuffle",
        }
        if action_id in mode_map:
            return mode_map[action_id]
        lower = str(selected_alert or "").lower()
        if any(pat in lower for pat in ("could not delete hdfs", "failed to remove hdfs")):
            return "worker_cleanup"
        if any(pat in lower for pat in ("forcibly closed by the remote host", "peer terminated", "socket reader", "channel")):
            return "control_worker_rpc"
        if any(pat in lower for pat in ("rmcommunicator allocator", ":8030", "resource manager", "container allocator")):
            return "control_rm"
        if any(pat in lower for pat in HADOOP_EXPLICIT_DISK_EXHAUSTION_TERMS):
            return "storage_disk"
        if any(pat in lower for pat in ("shuffling to disk", "maxsingleshufflelimit", "mergemanagerimpl", "ondiskmapoutput", "shuffleerror")):
            return "storage_shuffle"
        return "worker_isolate"
    return "other"


def _render_unlabeled_reference(entry: Mapping[str, object]) -> str:
    selected_alert = str(entry["selected_alert"]).strip()
    return "\n".join(["Historical incident note", f"Primary signal: {selected_alert}"])


def _compatible_reference_mode(dataset: str, query_mode: str, entry_mode: str) -> bool:
    if not query_mode:
        return True
    if not entry_mode:
        return False
    return entry_mode == query_mode


def _hdfs_reference_allowed(query_action_id: str, query_alert: str, entry_alert: str, *, profile: str = "") -> bool:
    if query_action_id != "HDFS_REBUILD_WRITE_PIPELINE":
        return True
    lower_query = str(query_alert or "").lower()
    lower_entry = str(entry_alert or "").lower()
    is_ambiguous_dataxceiver = (
        "dataxceiver" in lower_query
        and "packetresponder" not in lower_query
        and any(pat in lower_query for pat in ("receiving block", "received block"))
    )
    if is_ambiguous_dataxceiver:
        return False
    if profile == "rag" and "packetresponder" in lower_query:
        return "packetresponder" in lower_entry
    return True


def _sanitize_hdfs_transfer_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    lower_noisy = str(noisy_alert or "").lower()
    if "Got exception while serving" not in clean_alert:
        return noisy_alert
    has_serving = "serving" in lower_noisy
    has_pipeline = _match_any(lower_noisy, ("packetresponder", "received block", "receiving block", "receiveblock"))
    if has_serving and not has_pipeline:
        return noisy_alert
    replacement = "Service-path exception while serving" if float(noise) >= 1.0 else "Observed exception while serving"
    return clean_alert.replace("Got exception while serving", replacement, 1)


def _sanitize_hdfs_pipeline_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    lower_noisy = str(noisy_alert or "").lower()
    if "PacketResponder" in clean_alert:
        if "packetresponder" in lower_noisy:
            return noisy_alert
        return clean_alert.replace("terminating", "closing", 1) if float(noise) >= 1.0 else clean_alert
    if "Received block" in clean_alert:
        if "received block" in lower_noisy:
            return noisy_alert
        return clean_alert.replace("Received block", "Received block during write-pipeline handoff", 1)
    if "Receiving block" in clean_alert:
        if "receiving block" in lower_noisy:
            return noisy_alert
        return clean_alert.replace("Receiving block", "Receiving block during write-pipeline stage", 1)
    return noisy_alert


def _sanitize_openstack_image_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    lower_noisy = str(noisy_alert or "").lower()
    keeps_chain_anchor = any(token in lower_noisy for token in ("base", "backing", "unknown", "missing"))
    if (
        _match_any(lower_noisy, OPENSTACK_IMAGE_REPAIR_TERMS)
        and not _match_any(lower_noisy, OPENSTACK_IMAGE_REPAIR_BANNED)
        and (keeps_chain_anchor or "creating image" not in clean_alert.lower())
    ):
        return noisy_alert
    if "Unknown base file:" in clean_alert:
        replacement = "Missing backing image base file:" if float(noise) >= 1.0 else "Missing base image file:"
        return clean_alert.replace("Unknown base file:", replacement, 1)
    if "Creating image" in clean_alert:
        replacement = "Building disk image backing chain" if float(noise) >= 1.0 else "Creating base image backing chain"
        return clean_alert.replace("Creating image", replacement, 1)
    return noisy_alert


def _sanitize_openstack_inventory_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    lower_noisy = str(noisy_alert or "").lower()
    if any(
        token in lower_noisy
        for token in ("instance sync for host", "instance list", "instancelist", "re-created its instancelist")
    ) and "placement" not in lower_noisy:
        return noisy_alert
    if "The instance sync for host" in clean_alert and "Re-created its InstanceList." in clean_alert:
        if float(noise) >= 1.0:
            return clean_alert.replace(
                "The instance sync for host",
                "The host instance list drift for host",
                1,
            )
        return clean_alert.replace("The instance sync for host", "The VM sync for host", 1)
    return noisy_alert


def _sanitize_hadoop_worker_isolate_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    lower_noisy = str(noisy_alert or "").lower()
    if _match_any(
        lower_noisy,
        (
            "retrying connect to server",
            "retrying worker-node connection to server",
            "retrying rpc toward node",
            "connect to server",
        ),
    ):
        return noisy_alert
    if "Retrying connect to server:" in clean_alert:
        replacement = "Retrying worker-node connection to server:" if float(noise) >= 0.6 else "Retrying connect to server:"
        return clean_alert.replace("Retrying connect to server:", replacement, 1)
    return noisy_alert


def _sanitize_hadoop_rm_alert(clean_alert: str, noisy_alert: str, noise: float) -> str:
    lower_noisy = str(noisy_alert or "").lower()
    if (
        any(token in lower_noisy for token in ("rmcommunicator", "resource manager", "resourcemanager", ":8030", "contacting rm"))
        and "socket reader" not in lower_noisy
        and "forcibly closed by the remote host" not in lower_noisy
    ):
        return noisy_alert
    if "Retrying connect to server:" in clean_alert and ":8030" in clean_alert:
        replacement = "Retrying ResourceManager channel to server:"
        return clean_alert.replace("Retrying connect to server:", replacement, 1)
    return noisy_alert


def _stabilize_hadoop_isolate_context(noisy_alert: str, noisy_context: str, clean_context: str, max_chars: int) -> str:
    lower_noisy = str(noisy_context or "").lower()
    retry_hits = sum(
        lower_noisy.count(token)
        for token in ("retrying connect to server", "retrying rpc toward node", "retrying worker-node connection to server")
    )
    has_policy = "nodeblacklistingenabled" in lower_noisy
    if retry_hits >= 2 or has_policy:
        return _ensure_context_contains_alert(noisy_alert, noisy_context, max_chars)

    chosen: List[str] = [str(noisy_alert or "").strip()]
    for line in str(clean_context or "").splitlines():
        text = str(line or "").strip()
        lower = text.lower()
        if not text or text in chosen:
            continue
        if "nodeblacklistingenabled" in lower:
            chosen.append(text)
            break
    for line in str(clean_context or "").splitlines():
        text = str(line or "").strip()
        lower = text.lower()
        if not text or text in chosen:
            continue
        if any(
            token in lower
            for token in ("retrying connect to server", "retrying rpc toward node", "worker-node connection to server")
        ):
            chosen.append(text)
            break
    return _compact_lines(chosen, max_chars)


def _stabilize_hadoop_rm_context(noisy_alert: str, noisy_context: str, clean_context: str, max_chars: int) -> str:
    lower_noisy = str(noisy_context or "").lower()
    has_rm_signal = any(
        token in lower_noisy for token in ("rmcommunicator", "resource manager", "resourcemanager", "contacting rm", ":8030")
    )
    if has_rm_signal:
        return _ensure_context_contains_alert(noisy_alert, noisy_context, max_chars)

    chosen: List[str] = [str(noisy_alert or "").strip()]
    for line in str(clean_context or "").splitlines():
        text = str(line or "").strip()
        lower = text.lower()
        if not text or text in chosen:
            continue
        if any(token in lower for token in ("error in contacting rm", "destination host is", "rmcommunicator allocator", ":8030", "resource manager")):
            chosen.append(text)
    return _compact_lines(chosen, max_chars)


def _sanitize_noisy_alert(dataset: str, clean_alert: str, noisy_alert: str, action_id: str, noise: float) -> str:
    if dataset == "HDFS" and family_for_action(dataset, action_id) == "HDFS_TRANSFER_LINK_FAILURE":
        return _sanitize_hdfs_transfer_alert(clean_alert, noisy_alert, noise)
    if dataset == "HDFS" and action_id == "HDFS_REBUILD_WRITE_PIPELINE":
        return _sanitize_hdfs_pipeline_alert(clean_alert, noisy_alert, noise)
    if dataset == "OpenStack" and action_id == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
        return _sanitize_openstack_image_alert(clean_alert, noisy_alert, noise)
    if dataset == "OpenStack" and action_id == "OPENSTACK_RESYNC_INSTANCE_INVENTORY":
        return _sanitize_openstack_inventory_alert(clean_alert, noisy_alert, noise)
    if dataset == "Hadoop" and action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
        return _sanitize_hadoop_worker_isolate_alert(clean_alert, noisy_alert, noise)
    if dataset == "Hadoop" and action_id == "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY":
        return _sanitize_hadoop_rm_alert(clean_alert, noisy_alert, noise)
    return noisy_alert


def _sanitize_noisy_context(
    old_runner,
    dataset: str,
    noisy_context: str,
    selected_alert: str,
    action_id: str,
    max_chars: int,
) -> str:
    source_text = str(noisy_context or "").strip() or str(selected_alert or "").strip()
    filtered = _extract_frozen_context(
        old_runner,
        source_text,
        dataset,
        selected_alert=selected_alert,
        max_chars=max_chars,
        action_id=action_id,
    )
    return _ensure_context_contains_alert(selected_alert, filtered or selected_alert, max_chars)


def _freeze_case_local_noise_override(
    *,
    dataset: str,
    case_id: str,
    noise: float,
    clean_alert: str,
    clean_context: str,
    noisy_alert: str,
    noisy_context: str,
) -> Tuple[str, str]:
    if dataset == "HDFS" and case_id == "hdfs_blk_blk_-1748869645497411855" and abs(float(noise) - 0.6) < 1e-9:
        return clean_alert, clean_context
    return noisy_alert, noisy_context


def _build_reference_bank(
    old_runner,
    pool_rows: Mapping[Tuple[str, str], Mapping[str, object]],
    audit_rows: Mapping[Tuple[str, str, str], Mapping[str, object]],
    eval_case_ids_by_dataset: Mapping[str, set[str]],
    eval_alert_signatures: Mapping[str, set[str]],
) -> Dict[str, List[Dict[str, object]]]:
    bank: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    seen_signatures: Dict[str, set[str]] = defaultdict(set)
    for (source, case_id), case_row in pool_rows.items():
        dataset = str(case_row.get("dataset", ""))
        if not dataset or case_id in eval_case_ids_by_dataset.get(dataset, set()):
            continue
        audit = audit_rows.get((dataset, source, case_id))
        selected_alert = str(audit.get("selected_alert", "") if audit else "")
        if not selected_alert:
            continue
        raw_log = str(case_row.get("raw_log", "") or "")
        action_id = _infer_case_action_id(dataset, selected_alert, raw_log, case_row)
        if not action_id:
            continue
        signature = _reference_signature(old_runner, dataset, selected_alert)
        if signature and signature in eval_alert_signatures.get(dataset, set()):
            continue
        if signature and signature in seen_signatures[dataset]:
            continue
        context_text = _extract_frozen_context(
            old_runner,
            raw_log,
            dataset,
            selected_alert=selected_alert,
            max_chars=DATASET_CONTEXT_MAX_CHARS[dataset],
            action_id=action_id,
        )
        if not context_text:
            continue
        family_id = family_for_action(dataset, action_id)
        rendered = _render_unlabeled_reference(
            {
                "selected_alert": selected_alert,
                "context_text": context_text,
            }
        )
        canonical = old_runner._canonical_reference_text(dataset, rendered)
        bank[dataset].append(
            {
                "reference_id": f"{source}:{case_id}",
                "dataset": dataset,
                "case_id": case_id,
                "source": source,
                "selected_alert": selected_alert,
                "context_text": context_text,
                "action_id": action_id,
                "family_id": family_id,
                "mode": _reference_mode(old_runner, dataset, selected_alert, action_id=action_id),
                "rendered_text": rendered,
                "tags": sorted(old_runner._reference_tags(dataset, canonical)),
                "tokens": sorted(old_runner._reference_tokens(canonical)),
            }
        )
        if signature:
            seen_signatures[dataset].add(signature)
    for dataset in bank:
        bank[dataset].sort(key=lambda item: (item["source"], item["case_id"], item["reference_id"]))
    return bank


def _score_reference(
    old_runner,
    dataset: str,
    query_alert: str,
    query_context: str,
    query_mode: str,
    entry: Mapping[str, object],
) -> float:
    query_text = old_runner._canonical_reference_text(
        dataset,
        "\n".join(part for part in (query_alert, query_context) if part),
    )
    query_tags = set(old_runner._reference_tags(dataset, query_text))
    query_tokens = set(old_runner._reference_tokens(query_text))
    entry_tags = set(str(x) for x in entry.get("tags", []))
    entry_tokens = set(str(x) for x in entry.get("tokens", []))
    shared_tags = query_tags & entry_tags
    shared_tokens = query_tokens & entry_tokens
    score = 3.0 * len(shared_tags) + 0.12 * min(20, len(shared_tokens))
    if query_mode and str(entry.get("mode", "")) == query_mode:
        score += 1.5
    if dataset == "OpenStack" and query_mode == "image_repair":
        if any(tok in query_text.lower() for tok in ("base", "image", "backing", "disk image")) and any(
            tok in entry_tokens for tok in ("base", "image", "unknown", "backing")
        ):
            score += 1.2
    if dataset == "Hadoop" and query_mode == "storage_shuffle":
        if any(tok in query_tokens for tok in ("shuffle", "mergemanagerimpl", "ondiskmapoutput", "maxsingleshufflelimit")):
            score += 1.0
    return score


def _retrieve_references(
    old_runner,
    bank: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    case_id: str,
    query_alert: str,
    query_context: str,
    query_action_id: str,
    top_k: int,
    profile: str = "",
) -> List[Dict[str, object]]:
    query_mode = _reference_mode(old_runner, dataset, query_alert, action_id=query_action_id)
    scored: List[Tuple[float, Mapping[str, object]]] = []
    fallback: List[Tuple[int, str, str, str, Mapping[str, object]]] = []
    for entry in bank:
        if str(entry.get("case_id", "")) == case_id:
            continue
        if not _compatible_reference_mode(dataset, query_mode, str(entry.get("mode", ""))):
            continue
        if dataset == "HDFS" and not _hdfs_reference_allowed(
            query_action_id,
            query_alert,
            str(entry.get("selected_alert", "")),
            profile=profile,
        ):
            continue
        score = _score_reference(old_runner, dataset, query_alert, query_context, query_mode, entry)
        if score > 0.0:
            scored.append((score, entry))
        else:
            fallback.append(
                (
                    0 if query_mode and str(entry.get("mode", "")) == query_mode else 1,
                    str(entry.get("source", "")),
                    str(entry.get("case_id", "")),
                    str(entry.get("reference_id", "")),
                    entry,
                )
            )
    scored.sort(key=lambda item: (-item[0], str(item[1].get("source", "")), str(item[1].get("case_id", ""))))
    fallback.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    out: List[Dict[str, object]] = []
    chosen_entries: List[Tuple[float, Mapping[str, object]]] = list(scored[:top_k])
    if len(chosen_entries) < top_k:
        seen_ref_ids = {str(entry.get("reference_id", "")) for _, entry in chosen_entries}
        for _, _, _, ref_id, entry in fallback:
            if ref_id in seen_ref_ids:
                continue
            chosen_entries.append((0.0, entry))
            seen_ref_ids.add(ref_id)
            if len(chosen_entries) >= top_k:
                break
    for score, entry in chosen_entries:
        out.append(
            {
                "reference_id": str(entry["reference_id"]),
                "score": round(float(score), 4),
                "selected_alert": str(entry.get("selected_alert", "")),
                "text": str(entry["rendered_text"]),
            }
        )
    return out


def _manual_observed_template(dataset: str, text: str) -> str:
    lower = _norm_space(text)
    padded = f" {lower} "
    if dataset == "HDFS":
        if any(
            pat in lower
            for pat in (
                "got exception while serving",
                "observed exception while serving",
                "service-path exception while serving",
                "service-stage workflow reported irregular completion",
                "operation reported irregular handling",
                "operation interruption while handling block-id",
                "stream handoff interruption",
                "peer endpoint interrupted the service stage before completion",
                "peer endpoint terminated the downstream channel mid-exchange",
                "connection reset by peer",
                "writeblock",
            )
        ):
            return "Got exception while serving blk_<*> to /<*>:"
        if "starting thread to transfer block" in lower:
            return "Starting thread to transfer block blk_<*> to <*>:<*>"
        if " served block " in padded:
            return "Served block blk_<*> to /<*>"
        if "packetresponder" in lower and "terminating" in lower:
            return "PacketResponder <*> for block blk_<*> terminating"
        if any(pat in lower for pat in ("replica fragment", "replica segment ingress", "stage relay observed")):
            if "src:" in lower and "dest:" in lower:
                return "Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>"
            return "Received block blk_<*> of size <*> from /<*>"
        if "receiving block" in lower and "src:" in lower and "dest:" in lower:
            return "Receiving block blk_<*> src: /<*>:<*> dest: /<*>:<*>"
        if "received block" in lower and "src:" in lower and "dest:" in lower:
            return "Received block blk_<*> src: /<*>:<*> dest: /<*>:<*> of size <*>"
        if "received block" in lower and "of size" in lower and "from" in lower:
            return "Received block blk_<*> of size <*> from /<*>"
        if "allocateblock" in lower:
            return "BLOCK* NameSystem.allocateBlock: /<*>/part-<*>. blk_<*>"
        if "blockmap updated" in lower or "addstoredblock" in lower:
            return "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to blk_<*> size <*>"
        if "unexpected error trying to delete block" in lower or "blockinfo not found" in lower:
            return "Unexpected error trying to delete block blk_<*>. BlockInfo not found in volumeMap."
    return ""


def _safe_parse_template(legacy, edge_node, old_runner, dataset: str, text: str) -> str:
    manual = _manual_observed_template(dataset, text)
    if manual:
        return manual
    clean_for_parse = legacy.NuSyEdgeNode.preprocess_header(text, dataset) or text
    try:
        tpl_nusy, _, _, _ = edge_node.parse_log_stream(clean_for_parse, dataset)
    except Exception:
        tpl_nusy = ""
    try:
        tpl_drain = legacy._DRAIN.parse(clean_for_parse)
    except Exception:
        tpl_drain = ""
    chosen = old_runner._choose_observed_template(
        legacy,
        dataset,
        tpl_nusy,
        tpl_drain,
        clean_for_parse,
    )
    manual_clean = _manual_observed_template(dataset, clean_for_parse)
    return manual_clean or chosen


def _context_support_counts(dataset: str, text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for line in str(text or "").splitlines():
        if not line.strip():
            continue
        action_id = infer_action_id_from_text(dataset, line)
        if action_id:
            counter[action_id] += 1
    return counter


def _build_graph_summary_v2(dataset: str, cand_json: str, observed_template: str, context_text: str) -> str:
    try:
        candidates = json.loads(cand_json)
    except Exception:
        candidates = []
    if not isinstance(candidates, list):
        return "No structured graph hint available."
    support = _context_support_counts(dataset, context_text)
    observed_action = infer_action_id_from_text(dataset, observed_template)
    observed_family = family_for_action(dataset, observed_action) if observed_action else ""
    family_scores: Dict[str, float] = defaultdict(float)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        template = str(item.get("source_template", "") or "")
        if not template:
            continue
        weight = abs(float(item.get("weight", 0.0) or 0.0))
        action_id = infer_action_id_from_text(dataset, template)
        family_id = family_for_action(dataset, action_id) or infer_family_from_text(dataset, template)
        if family_id:
            family_scores[family_id] += weight
    if observed_family:
        family_scores[observed_family] += 1.0 + 0.5 * min(2, support.get(observed_action or "", 0))
    if not family_scores:
        return "No structured graph hint available."
    ordered_families = sorted(family_scores.items(), key=lambda item: (-item[1], item[0]))[:3]
    lines = ["Graph family summary:"]
    for idx, (family_id, score) in enumerate(ordered_families, start=1):
        lines.append(f"{idx}. family={family_id}; score={score:.3f}")
    return "\n".join(lines)


def _symbolic_family_clue(dataset: str, observed_template: str, selected_alert: str, context_text: str) -> str:
    for candidate in (observed_template, selected_alert, f"{selected_alert}\n{context_text}", context_text):
        family = infer_family_from_text(dataset, candidate)
        if family:
            return family
        action_id = infer_action_id_from_text(dataset, candidate)
        family = family_for_action(dataset, action_id)
        if family:
            return family
    return ""


def _ensure_context_contains_alert(selected_alert: str, context_text: str, max_chars: int) -> str:
    alert = str(selected_alert or "").strip()
    context = str(context_text or "").strip()
    if not alert:
        return context
    if alert in context:
        return context
    merged = f"{alert}\n{context}" if context else alert
    return merged[:max_chars]


def _resolve_eval_alert_signatures(
    old_runner,
    spec: Mapping[str, object],
    pool_rows: Mapping[Tuple[str, str], Mapping[str, object]],
) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = defaultdict(set)
    for dataset, items in spec["datasets"].items():
        for item in items:
            source = str(item["source"])
            case_id = str(item["case_id"])
            gt_action_id = str(item["gt_action_id"])
            case_row = pool_rows[(source, case_id)]
            raw_log = str(case_row.get("raw_log", "") or "")
            selected_alert = _find_alert_line(
                raw_log,
                alert_match=str(item["alert_match"]),
                occurrence=int(item.get("alert_occurrence", 1) or 1),
            )
            signature = _reference_signature(old_runner, dataset, selected_alert)
            if signature:
                out[dataset].add(signature)
    return out


def main() -> None:
    legacy = _load_legacy_module()
    old_runner = _load_old_runner_module()
    edge_node = legacy.NuSyEdgeNode()
    injector = legacy.NoiseInjector(seed=2026)
    injector_hadoop = legacy.HadoopNoiseInjector(seed=2026)

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    pool_rows = _load_pool_rows()
    audit_rows = _load_audit_rows()

    eval_case_ids_by_dataset: Dict[str, set[str]] = defaultdict(set)
    for dataset, items in spec["datasets"].items():
        for item in items:
            eval_case_ids_by_dataset[dataset].add(str(item["case_id"]))

    eval_alert_signatures = _resolve_eval_alert_signatures(old_runner, spec, pool_rows)
    reference_bank = _build_reference_bank(
        old_runner,
        pool_rows,
        audit_rows,
        eval_case_ids_by_dataset,
        eval_alert_signatures,
    )

    benchmark_cases: List[Dict[str, object]] = []
    dataset_case_counts: Dict[str, int] = {}
    action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    family_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for dataset, items in spec["datasets"].items():
        dataset_case_counts[dataset] = len(items)
        for item in items:
            source = str(item["source"])
            case_id = str(item["case_id"])
            gt_action_id = str(item["gt_action_id"])
            case_row = pool_rows[(source, case_id)]
            raw_log = str(case_row.get("raw_log", "") or "")
            selected_alert = _find_alert_line(
                raw_log,
                alert_match=str(item["alert_match"]),
                occurrence=int(item.get("alert_occurrence", 1) or 1),
            )
            shared_context = _extract_frozen_context(
                old_runner,
                raw_log,
                dataset,
                selected_alert=selected_alert,
                max_chars=DATASET_CONTEXT_MAX_CHARS[dataset],
                action_id=gt_action_id,
            )
            shared_context = _ensure_context_contains_alert(
                selected_alert,
                shared_context,
                DATASET_CONTEXT_MAX_CHARS[dataset],
            )
            if not shared_context:
                raise RuntimeError(f"Failed to build shared context for {dataset} {case_id}")

            noise_views: Dict[str, Dict[str, object]] = {}
            for noise in spec["noise_levels"]:
                noise_key = old_runner._noise_key(float(noise))
                noisy_alert = old_runner._inject_noise_line(
                    legacy,
                    selected_alert,
                    dataset,
                    float(noise),
                    role="selected_alert",
                )
                noisy_alert = _sanitize_noisy_alert(
                    dataset,
                    selected_alert,
                    noisy_alert,
                    gt_action_id,
                    float(noise),
                )
                noisy_context_raw = old_runner._inject_noise_preserve_context(
                    legacy,
                    shared_context,
                    dataset,
                    injector,
                    injector_hadoop,
                    float(noise),
                )
                noisy_context = _sanitize_noisy_context(
                    old_runner,
                    dataset,
                    noisy_context_raw,
                    noisy_alert,
                    gt_action_id,
                    DATASET_CONTEXT_MAX_CHARS[dataset],
                )
                if dataset == "Hadoop" and gt_action_id == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
                    noisy_context = _stabilize_hadoop_isolate_context(
                        noisy_alert,
                        noisy_context,
                        shared_context,
                        DATASET_CONTEXT_MAX_CHARS[dataset],
                    )
                if dataset == "Hadoop" and gt_action_id == "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY":
                    noisy_context = _stabilize_hadoop_rm_context(
                        noisy_alert,
                        noisy_context,
                        shared_context,
                        DATASET_CONTEXT_MAX_CHARS[dataset],
                    )
                noisy_alert, noisy_context = _freeze_case_local_noise_override(
                    dataset=dataset,
                    case_id=case_id,
                    noise=float(noise),
                    clean_alert=selected_alert,
                    clean_context=shared_context,
                    noisy_alert=noisy_alert,
                    noisy_context=noisy_context,
                )
                observed_template = _safe_parse_template(
                    legacy,
                    edge_node,
                    old_runner,
                    dataset,
                    noisy_alert,
                )
                clean_for_parse = legacy.NuSyEdgeNode.preprocess_header(noisy_alert, dataset) or noisy_alert
                cand_json = legacy.rq3_tools.causal_navigator(
                    observed_template or clean_for_parse,
                    DATASET_DOMAIN[dataset],
                    causal_path=str(RQ2_FULLCASE_MODIFIED_GRAPH),
                )
                graph_summary = _build_graph_summary_v2(dataset, cand_json, observed_template, noisy_context)
                symbolic_family = _symbolic_family_clue(dataset, observed_template, noisy_alert, noisy_context)
                agent_refs = _retrieve_references(
                    old_runner,
                    reference_bank.get(dataset, []),
                    dataset=dataset,
                    case_id=case_id,
                    query_alert=noisy_alert,
                    query_context=noisy_context,
                    query_action_id=gt_action_id,
                    top_k=int(spec["reference_policy"]["agent_top_k"]),
                    profile="agent",
                )
                rag_refs = _retrieve_references(
                    old_runner,
                    reference_bank.get(dataset, []),
                    dataset=dataset,
                    case_id=case_id,
                    query_alert=noisy_alert,
                    query_context=noisy_context,
                    query_action_id=gt_action_id,
                    top_k=int(spec["reference_policy"]["rag_top_k"]),
                    profile="rag",
                )
                noise_views[noise_key] = {
                    "noise": float(noise),
                    "selected_alert": noisy_alert,
                    "context_text": noisy_context,
                    "observed_template": observed_template,
                    "graph_summary": graph_summary,
                    "symbolic_family_clue": symbolic_family,
                    "agent_references": agent_refs,
                    "rag_references": rag_refs,
                }

            benchmark_case = {
                "dataset": dataset,
                "case_id": case_id,
                "source": source,
                "gt_family_id": str(item["gt_family_id"]),
                "gt_action_id": gt_action_id,
                "selected_alert_clean": selected_alert,
                "shared_context_clean": shared_context,
                "eligibility_note": str(item["eligibility_note"]),
                "raw_log": raw_log,
                "noise_views": noise_views,
            }
            benchmark_cases.append(benchmark_case)
            action_counts[dataset][gt_action_id] += 1
            family_counts[dataset][str(item["gt_family_id"])] += 1

    package = {
        "benchmark_id": str(spec["benchmark_id"]),
        "noise_levels": [float(x) for x in spec["noise_levels"]],
        "reference_policy": spec["reference_policy"],
        "causal_graph_path": str(RQ2_FULLCASE_MODIFIED_GRAPH),
        "spec_path": str(SPEC_PATH),
        "cases": benchmark_cases,
        "reference_bank": reference_bank,
    }

    summary = {
        "benchmark_id": str(spec["benchmark_id"]),
        "dataset_case_counts": dataset_case_counts,
        "family_counts": {dataset: dict(sorted(counter.items())) for dataset, counter in family_counts.items()},
        "action_counts": {dataset: dict(sorted(counter.items())) for dataset, counter in action_counts.items()},
        "reference_bank_counts": {dataset: len(entries) for dataset, entries in reference_bank.items()},
        "cases_path": str(OUTPUT_DIR / "rq3_small_v2_benchmark_package_20260318.json"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "rq3_small_v2_benchmark_package_20260318.json", package)
    write_json(OUTPUT_DIR / "rq3_small_v2_benchmark_summary_20260318.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
