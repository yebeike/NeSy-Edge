from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


ACTION_CATALOG: Dict[str, Dict[str, Dict[str, object]]] = {
    "HDFS": {
        "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK": {
            "label": "HDFS_DELETE_BLOCK",
            "description": "Check DataNode disk health, rebalance or migrate data, and clean or expand storage before retrying block deletion.",
            "keyword_groups": [
                ["disk", "storage", "volume", "space", "capacity"],
                ["rebalance", "rebalancer", "migrate", "move", "cleanup", "clean", "expand"],
                ["delete", "deletion", "block"],
            ],
            "min_groups": 2,
        },
        "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE": {
            "label": "HDFS_GOT_EXCEPTION_SERVING",
            "description": "Check the DataNode-to-client network path, restart the failing DataNode, and trigger block re-replication.",
            "keyword_groups": [
                ["network", "connectivity", "link", "client"],
                ["restart", "relaunch", "recover", "recovering"],
                ["replicate", "replication", "re-replicate", "rebuild block"],
            ],
            "min_groups": 2,
        },
        "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE": {
            "label": "HDFS_EXCEPTION_RECEIVEBLOCK",
            "description": "Inspect source DataNode disk I/O and network status, isolate the unhealthy node, and re-replicate the affected block.",
            "keyword_groups": [
                ["disk", "i/o", "io", "network"],
                ["isolate", "quarantine", "remove", "exclude"],
                ["replicate", "replication", "re-replicate", "copy"],
            ],
            "min_groups": 2,
        },
        "HDFS_REBUILD_WRITE_PIPELINE": {
            "label": "HDFS_PACKETRESPONDER",
            "description": "Inspect the write pipeline for downstream DataNode failure, rebuild the pipeline, and resubmit the current write.",
            "keyword_groups": [
                ["pipeline", "write pipeline", "packetresponder", "downstream"],
                ["rebuild", "recreate", "re-establish", "recover"],
                ["resubmit", "retry", "write", "commit"],
            ],
            "min_groups": 2,
        },
        "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK": {
            "label": "HDFS_ALLOCATE_BLOCK",
            "description": "Check NameNode free space and quotas, then expand storage or clean stale files before retrying allocation.",
            "keyword_groups": [
                ["namenode", "quota", "space", "capacity"],
                ["expand", "scale", "clean", "cleanup", "stale"],
                ["retry", "allocate", "allocation"],
            ],
            "min_groups": 2,
        },
        "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE": {
            "label": "HDFS_OTHER",
            "description": "Validate blockMap metadata consistency and run fsck or re-replication if isolated blocks are detected.",
            "keyword_groups": [
                ["blockmap", "metadata", "consistency"],
                ["fsck", "repair", "replicate", "re-replicate"],
            ],
            "min_groups": 1,
        },
        "HDFS_TUNE_REPLICATION_FLOW": {
            "label": "HDFS_RECEIVED_BLOCK",
            "description": "Monitor cross-node replication throughput, limit replication concurrency when needed, and tune DataNode transfer settings.",
            "keyword_groups": [
                ["replication", "throughput", "cross-node", "transfer"],
                ["limit", "throttle", "concurrency", "parallelism", "tune"],
            ],
            "min_groups": 1,
        },
        "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION": {
            "label": "HDFS_OTHER",
            "description": "Track repeated block retransmissions and trigger link-health checks or DataNode migration if retransmissions remain high.",
            "keyword_groups": [
                ["retransmission", "transmit", "served block"],
                ["link", "health", "network", "migrate", "migration"],
            ],
            "min_groups": 1,
        },
    },
    "OpenStack": {
        "OPENSTACK_RESYNC_INSTANCE_INVENTORY": {
            "label": "OS_SYNC_SUCCESS_ROOT",
            "description": "Trigger instance-list resynchronization and restart nova-compute on the affected host if state drift persists.",
            "keyword_groups": [
                ["sync", "resync", "instance list", "state drift"],
                ["nova-compute", "compute host", "restart"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD": {
            "label": "OS_POWER_STATE_SYNC",
            "description": "Check whether the spawning task is stuck, then rebuild or migrate the instance to a healthy compute node.",
            "keyword_groups": [
                ["spawning", "pending task", "power state"],
                ["rebuild", "migrate", "reschedule", "compute node"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE": {
            "label": "OS_OTHER",
            "description": "Verify whether the termination was expected and restore the instance from image or volume if it was accidental.",
            "keyword_groups": [
                ["terminate", "termination", "delete"],
                ["restore", "recover", "volume", "image"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_SCALE_METADATA_SERVICE": {
            "label": "OS_METADATA_SERVER",
            "description": "Inspect metadata service load and latency, then scale metadata workers or enable local caching.",
            "keyword_groups": [
                ["metadata", "service", "latency", "load"],
                ["scale", "workers", "cache", "caching"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN": {
            "label": "OS_UNKNOWN_BASE_FILE",
            "description": "Check whether the base image file is missing or moved, then recreate the base image and remount the instance disk.",
            "keyword_groups": [
                ["base file", "base image", "image", "disk"],
                ["recreate", "remount", "restore", "recover"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST": {
            "label": "OS_VCPU_AFFINITY",
            "description": "Check the hypervisor CPU-affinity capability, refresh the compute resource inventory, and migrate or rebuild the instance on a compatible host.",
            "keyword_groups": [
                ["vcpu", "cpu affinity", "affinity", "hypervisor"],
                ["inventory", "resource", "refresh", "audit"],
                ["migrate", "rebuild", "compatible host", "reschedule"],
            ],
            "min_groups": 2,
        },
    },
    "Hadoop": {
        "HADOOP_ISOLATE_NODE_AND_RESCHEDULE": {
            "label": "HADOOP_MACHINE_DOWN",
            "description": "Check NodeManager heartbeats, isolate the failed node, and reschedule the affected application on healthy nodes.",
            "keyword_groups": [
                ["nodemanager", "heartbeat", "machine", "node"],
                ["isolate", "mark unavailable", "exclude", "failed node"],
                ["reschedule", "retry", "migrate", "healthy node"],
            ],
            "min_groups": 2,
        },
        "HADOOP_RESTORE_NETWORK_AND_RETRY": {
            "label": "HADOOP_NETWORK_DISCONNECTION",
            "description": "Inspect the switch and network links, retry the failed containers, and rerun the task after connectivity is restored.",
            "keyword_groups": [
                ["network", "switch", "link", "connectivity"],
                ["retry", "rerun", "resubmit", "container"],
                ["restore", "reconnect", "recovery"],
            ],
            "min_groups": 2,
        },
        "HADOOP_FREE_DISK_AND_RETRY": {
            "label": "HADOOP_DISK_FULL",
            "description": "Clean or expand disk capacity on the affected node, adjust HDFS or YARN disk thresholds, and retry the write workload.",
            "keyword_groups": [
                ["disk", "space", "capacity", "full"],
                ["clean", "cleanup", "expand", "free"],
                ["threshold", "retry", "write", "yarn", "hdfs"],
            ],
            "min_groups": 2,
        },
    },
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def allowed_action_ids(dataset: str) -> List[str]:
    return list(ACTION_CATALOG.get(dataset, {}).keys())


def label_for_action(dataset: str, action_id: str) -> str:
    meta = ACTION_CATALOG.get(dataset, {}).get(action_id)
    return str(meta.get("label", "")) if meta else ""


def describe_allowed_actions(dataset: str) -> str:
    items = ACTION_CATALOG.get(dataset, {})
    lines = ["Allowed action IDs (choose ONE exactly):"]
    for action_id, meta in items.items():
        lines.append(f"- {action_id}: {meta['description']}")
    return "\n".join(lines)


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    lower = _norm(text)
    return any(tok.lower() in lower for tok in tokens)


def infer_action_id_from_text(dataset: str, text: str) -> str:
    t = _norm(text)
    if dataset == "HDFS":
        if any(
            pat in t
            for pat in (
                "stream transfer aborted while handling",
                "replica service path aborted during stream handoff",
                "downstream replica exchange interrupted while handling",
                "replica handoff workflow exited before downstream stage completion",
                "peer endpoint terminated the downstream channel mid-exchange",
                "replica service workflow ended before completion",
                "peer endpoint interrupted the service stage before completion",
            )
        ):
            return "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE"
        if "encountered network failure when handling" in t:
            return "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE"
        if "failure during cleanup of data chunk" in t:
            return "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK"
        if "pkgresponder" in t or ("packetresponder" in t and "closing" in t):
            return "HDFS_REBUILD_WRITE_PIPELINE"
        if (
            "packetresponder" in t
            or "replicastage" in t
            or "replica stage tracker" in t
            or "replica fragment observed" in t
            or "replica fragment path observed" in t
            or "terminating" in t
        ):
            return "HDFS_REBUILD_WRITE_PIPELINE"
        if "exception in receiveblock" in t:
            return "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE"
        if "allocateblock" in t or "allocate block" in t:
            return "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK"
        if "got exception while serving" in t or "connection reset by peer" in t:
            return "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE"
        if ("writeblock" in t or "replicastagestep" in t) and (
            "received exception" in t
            or "channel wait threshold exceeded" in t
            or "downstream channel ended before stage completion" in t
        ):
            return "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE"
        if "deleting block" in t or "could not delete" in t or "blockinfo not found" in t:
            return "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK"
        if "receiving block" in t or "received block" in t:
            return "HDFS_REBUILD_WRITE_PIPELINE"
        if "addstoredblock" in t or "blockmap updated" in t:
            return "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE"
        if "transmitted block" in t or "served block" in t:
            return "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION"
        return ""
    if dataset == "OpenStack":
        if any(
            pat in t
            for pat in (
                "unknown base file",
                "removable base files",
                "active base files",
                "releasable base-layer cache entries",
                "retained base-layer cache entries",
                "base or swap file too young",
                "removing base or swap file",
                "creating image",
                "building instance disk image",
                "base-layer artifact",
                "cache audit pass",
                "referenced in local cache",
                "cached object",
                "routine inspection",
                "runtime workspace",
                "workspace objects",
                "/runtime/objects/",
            )
        ):
            return "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
        if "pending task (spawning)" in t or "while synchronizing instance power states" in t:
            return "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD"
        if "instance sync for host" in t or "re-created its instancelist" in t:
            return "OPENSTACK_RESYNC_INSTANCE_INVENTORY"
        if (
            "metadata" in t
            or "nova.metadata" in t
            or "/control-plane/bootstrap-cache" in t
            or "/control-plane/runtime-catalog" in t
            or "user-payload" in t
            or "bootstrap-blob" in t
            or "vendor-payload" in t
            or "vendor-profile" in t
            or "instance-manifest" in t
            or "instance-profile" in t
            or "object-missing" in t
        ):
            return "OPENSTACK_SCALE_METADATA_SERVICE"
        if "terminating instance" in t:
            return "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE"
        if "cpu affinity" in t or "vcpu count" in t:
            return "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
        return ""
    if dataset == "Hadoop":
        if (
            "disk full" in t
            or "no space" in t
            or "shuffling to disk" in t
            or "maxsingleshufflelimit" in t
            or "ondiskmapoutput" in t
        ):
            return "HADOOP_FREE_DISK_AND_RETRY"
        if "network disconnection" in t or "connection was forcibly closed" in t or "connect to" in t:
            return "HADOOP_RESTORE_NETWORK_AND_RETRY"
        if "machine down" in t or "bad datanode" in t or "could not delete hdfs" in t:
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        return ""
    return ""


def infer_label_from_text(dataset: str, text: str) -> str:
    action_id = infer_action_id_from_text(dataset, text)
    return label_for_action(dataset, action_id)


def gt_action_id_for_case(case: Mapping[str, object], gt_label: str) -> str:
    dataset = str(case.get("dataset", "HDFS"))
    root_tpl = _norm(str(case.get("ground_truth_root_cause_template", "") or ""))
    effect_tpl = _norm(str(case.get("ground_truth_template", "") or ""))
    raw_tail = _norm(str(case.get("raw_log", "") or "")[-1200:])
    gt_action_label = _norm(str(case.get("gt_action_label", "") or case.get("reason", "") or ""))

    if dataset == "HDFS":
        # Prefer the explicit benchmark root/effect template over incidental
        # symptom lines in the raw window. Otherwise PacketResponder-style
        # tails can incorrectly override a true "Got exception while serving"
        # root cause.
        for text in (root_tpl, gt_action_label, effect_tpl, raw_tail):
            inferred = infer_action_id_from_text("HDFS", text)
            if inferred:
                return inferred
        fallback = {
            "HDFS_DELETE_BLOCK": "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK",
            "HDFS_GOT_EXCEPTION_SERVING": "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE",
            "HDFS_PACKETRESPONDER": "HDFS_REBUILD_WRITE_PIPELINE",
            "HDFS_ALLOCATE_BLOCK": "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK",
            "HDFS_EXCEPTION_RECEIVEBLOCK": "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE",
            "HDFS_RECEIVED_BLOCK": "HDFS_TUNE_REPLICATION_FLOW",
        }
        return fallback.get(gt_label, "")

    if dataset == "OpenStack":
        # OpenStack windows often contain many background image-cache lines,
        # so we must trust the benchmark root/effect template before scanning
        # the full raw tail.
        for text in (root_tpl, gt_action_label, effect_tpl, raw_tail):
            inferred = infer_action_id_from_text("OpenStack", text)
            if inferred:
                return inferred
        fallback = {
            "OS_POWER_STATE_SYNC": "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD",
            "OS_SYNC_SUCCESS_ROOT": "OPENSTACK_RESYNC_INSTANCE_INVENTORY",
            "OS_METADATA_SERVER": "OPENSTACK_SCALE_METADATA_SERVICE",
            "OS_UNKNOWN_BASE_FILE": "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN",
            "OS_VCPU_AFFINITY": "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST",
        }
        return fallback.get(gt_label, "")

    if dataset == "Hadoop":
        if "machine down" in gt_action_label or gt_label == "HADOOP_MACHINE_DOWN":
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if "network disconnection" in gt_action_label or gt_label == "HADOOP_NETWORK_DISCONNECTION":
            return "HADOOP_RESTORE_NETWORK_AND_RETRY"
        if "disk full" in gt_action_label or gt_label == "HADOOP_DISK_FULL":
            return "HADOOP_FREE_DISK_AND_RETRY"
        return ""

    return ""


def action_text_match(dataset: str, action_id: str, repair_action: str) -> Tuple[int, int, List[int]]:
    meta = ACTION_CATALOG.get(dataset, {}).get(action_id)
    if not meta:
        return 0, 0, []
    text = _norm(repair_action)
    groups = meta["keyword_groups"]
    hits: List[int] = []
    matched = 0
    for idx, group in enumerate(groups):
        if _contains_any(text, group):
            matched += 1
            hits.append(idx)
    return matched, int(meta["min_groups"]), hits


def action_text_success(dataset: str, action_id: str, repair_action: str) -> bool:
    matched, minimum, _ = action_text_match(dataset, action_id, repair_action)
    return matched >= minimum and minimum > 0
