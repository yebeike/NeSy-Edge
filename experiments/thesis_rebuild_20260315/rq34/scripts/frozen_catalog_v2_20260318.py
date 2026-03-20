from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


RCA_FAMILY_CATALOG: Dict[str, Dict[str, Dict[str, object]]] = {
    "HDFS": {
        "HDFS_TRANSFER_LINK_FAILURE": {
            "description": "Explicit serving-path or client/peer link interruptions during block service or handoff; not routine PacketResponder or Received/Receiving block pipeline chatter.",
            "action_ids": [
                "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE",
                "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION",
            ],
        },
        "HDFS_PIPELINE_FAILURE": {
            "description": "Write-pipeline stage failures, including PacketResponder termination or PacketResponder-side received-block handoff signals, that require rebuilding or isolating part of the pipeline.",
            "action_ids": [
                "HDFS_REBUILD_WRITE_PIPELINE",
                "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE",
            ],
        },
        "HDFS_STORAGE_METADATA_PRESSURE": {
            "description": "Allocation, stale-block, or block-map consistency failures that require storage and metadata repair.",
            "action_ids": [
                "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK",
                "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK",
                "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE",
            ],
        },
    },
    "OpenStack": {
        "OPENSTACK_INSTANCE_STATE_DRIFT": {
            "description": "Instance lifecycle or state drift that requires resynchronization, rebuild, or restore actions.",
            "action_ids": [
                "OPENSTACK_RESYNC_INSTANCE_INVENTORY",
                "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD",
                "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE",
            ],
        },
        "OPENSTACK_IMAGE_CHAIN_FAILURE": {
            "description": "Image-chain or base-cache inconsistencies that require repair or cache cleanup before retry.",
            "action_ids": [
                "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN",
                "OPENSTACK_CLEAN_STALE_BASE_CACHE_AND_RETRY",
            ],
        },
        "OPENSTACK_HOST_COMPATIBILITY_FAILURE": {
            "description": "Compute-host capability mismatch that requires retry on a compatible host or refreshed host inventory.",
            "action_ids": [
                "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST",
                "OPENSTACK_REFRESH_HOST_INVENTORY_AND_RETRY_CLAIM",
            ],
        },
        "OPENSTACK_CONTROL_SERVICE_PRESSURE": {
            "description": "Control-plane service pressure that requires scaling or local caching.",
            "action_ids": [
                "OPENSTACK_SCALE_METADATA_SERVICE",
                "OPENSTACK_ENABLE_LOCAL_METADATA_CACHE",
            ],
        },
    },
    "Hadoop": {
        "HADOOP_WORKER_NODE_FAILURE": {
            "description": "Worker-node failure or unusable node state, typically shown by repeated connect-to-worker retries plus blacklisting or failed output cleanup before rescheduling.",
            "action_ids": [
                "HADOOP_ISOLATE_NODE_AND_RESCHEDULE",
                "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE",
            ],
        },
        "HADOOP_CONTROL_LINK_DISRUPTION": {
            "description": "Control-channel disruption shown by explicit socket/channel breakage or ResourceManager endpoint failure, requiring worker RPC recovery or ResourceManager channel recovery.",
            "action_ids": [
                "HADOOP_RESTORE_WORKER_RPC_AND_RETRY",
                "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY",
            ],
        },
        "HADOOP_STORAGE_PRESSURE": {
            "description": "Shuffle-pressure or explicit local-disk exhaustion incidents requiring spill reduction or direct disk cleanup before retry.",
            "action_ids": [
                "HADOOP_FREE_LOCAL_DISK_AND_RETRY",
                "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY",
            ],
        },
    },
}


ACTION_CATALOG: Dict[str, Dict[str, Dict[str, object]]] = {
    "HDFS": {
        "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE": {
            "description": "Check the DataNode-to-client or peer transfer path, restart the failing DataNode if needed, and trigger block re-replication.",
            "keyword_groups": [
                ["network", "connectivity", "link", "client", "peer"],
                ["restart", "relaunch", "recover"],
                ["replicate", "re-replicate", "rereplicate", "rebuild block"],
            ],
            "min_groups": 2,
        },
        "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION": {
            "description": "Use this when the logs explicitly show serving interruption, client/peer reset, or retransmission/link churn; inspect link health and retry the handoff more conservatively.",
            "keyword_groups": [
                ["retransmission", "transmit", "transfer", "handoff"],
                ["link", "network", "health"],
                ["limit", "throttle", "concurrency", "retry"],
            ],
            "min_groups": 2,
        },
        "HDFS_REBUILD_WRITE_PIPELINE": {
            "description": "Rebuild the write pipeline after generic PacketResponder-stage failure or PacketResponder-side received-block handoff evidence, then retry the write path without assuming a client-link or receiver-disk fault.",
            "keyword_groups": [
                ["pipeline", "packetresponder", "received block", "receiving block", "write pipeline"],
                ["rebuild", "recreate", "re-establish"],
                ["retry", "resubmit", "write", "transfer"],
            ],
            "min_groups": 2,
        },
        "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE": {
            "description": "Use this only when the logs explicitly show receiver-side receiveBlock, receiver-path, or disk I/O failure; isolate that receiver and then rebuild the pipeline.",
            "keyword_groups": [
                ["receiver", "receiveblock", "receive block handler", "disk", "io", "i/o"],
                ["isolate", "quarantine", "exclude", "remove"],
                ["pipeline", "rebuild", "replicate"],
            ],
            "min_groups": 2,
        },
        "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK": {
            "description": "Check NameNode space or quota constraints, clean or expand storage, then retry block allocation.",
            "keyword_groups": [
                ["namenode", "quota", "space", "capacity", "allocation"],
                ["clean", "cleanup", "expand", "scale"],
                ["retry", "allocate", "allocation"],
            ],
            "min_groups": 2,
        },
        "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK": {
            "description": "Check DataNode storage health, clean stale block metadata, and retry the blocked deletion path.",
            "keyword_groups": [
                ["storage", "disk", "volume", "space"],
                ["delete", "deletion", "stale block", "cleanup"],
                ["retry", "repair", "rebalance"],
            ],
            "min_groups": 2,
        },
        "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE": {
            "description": "Validate block-map consistency and run repair or re-replication for isolated block-map corruption.",
            "keyword_groups": [
                ["blockmap", "metadata", "consistency", "block map"],
                ["repair", "fsck", "replicate", "re-replicate"],
            ],
            "min_groups": 1,
        },
    },
    "OpenStack": {
        "OPENSTACK_RESYNC_INSTANCE_INVENTORY": {
            "description": "Resynchronize the host instance list when the logs explicitly show instance-sync mismatch, recreated InstanceList, or host instance inventory drift.",
            "keyword_groups": [
                ["instance list", "instancelist", "instance sync", "inventory", "resync"],
                ["host", "re-created", "reconcile", "mismatch"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD": {
            "description": "Resolve power-state drift or a stuck spawning task, then rebuild or migrate the instance.",
            "keyword_groups": [
                ["power state", "sync_power_state", "spawning", "pending task"],
                ["rebuild", "migrate", "reschedule"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE": {
            "description": "Verify whether termination was intended, then restore the instance from image or volume if needed.",
            "keyword_groups": [
                ["terminate", "termination", "delete", "destroyed"],
                ["restore", "recover", "volume", "image"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN": {
            "description": "Repair a broken base-image or backing-file chain when the logs show Creating image together with missing or unknown base-file evidence.",
            "keyword_groups": [
                ["base image", "base file", "image chain", "creating image", "unknown base file", "backing image"],
                ["repair", "recreate", "restore", "remount"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_CLEAN_STALE_BASE_CACHE_AND_RETRY": {
            "description": "Clean stale base-cache bookkeeping only when the logs explicitly focus on removable, active, checking, or in-use base-cache entries.",
            "keyword_groups": [
                ["base cache", "base files", "removable", "active base files", "in use", "checking"],
                ["clean", "cleanup", "prune", "stale"],
                ["retry", "retry launch", "rebuild"],
            ],
            "min_groups": 2,
        },
        "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST": {
            "description": "Move or rebuild the instance on a compute host with compatible CPU-affinity and capacity support.",
            "keyword_groups": [
                ["vcpu", "cpu affinity", "hypervisor", "compatible host"],
                ["migrate", "rebuild", "reschedule"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_REFRESH_HOST_INVENTORY_AND_RETRY_CLAIM": {
            "description": "Refresh compute-host capacity or claim inventory only when the logs explicitly mention scheduler claim, compute resources, or host compatibility/capacity checks.",
            "keyword_groups": [
                ["claim", "resource", "inventory", "audit", "capacity"],
                ["refresh", "retry", "host", "scheduler"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_SCALE_METADATA_SERVICE": {
            "description": "Scale metadata service workers to absorb the control-plane metadata load.",
            "keyword_groups": [
                ["metadata", "service", "latency", "load"],
                ["scale", "workers", "capacity"],
            ],
            "min_groups": 1,
        },
        "OPENSTACK_ENABLE_LOCAL_METADATA_CACHE": {
            "description": "Enable or repair local metadata caching to avoid repeated control-plane metadata pressure.",
            "keyword_groups": [
                ["metadata", "cache", "meta_data.json", "vendor_data.json"],
                ["local", "enable", "repair", "fallback"],
            ],
            "min_groups": 1,
        },
    },
    "Hadoop": {
        "HADOOP_ISOLATE_NODE_AND_RESCHEDULE": {
            "description": "Use this when the logs show repeated connect-to-worker retries together with blacklisting or unhealthy-node semantics; isolate that worker node and reschedule on healthy nodes.",
            "keyword_groups": [
                ["node", "worker", "nodemanager", "heartbeat"],
                ["isolate", "exclude", "drain", "mark unavailable"],
                ["reschedule", "retry", "healthy node"],
            ],
            "min_groups": 2,
        },
        "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE": {
            "description": "Clean the failed output path or stale HDFS cleanup state, then reschedule the application on healthy nodes.",
            "keyword_groups": [
                ["output path", "cleanup", "could not delete hdfs", "stale output"],
                ["clean", "remove", "repair"],
                ["reschedule", "retry", "rerun"],
            ],
            "min_groups": 2,
        },
        "HADOOP_RESTORE_WORKER_RPC_AND_RETRY": {
            "description": "Use this only when the logs explicitly show a forced-close, socket-reader, peer-terminated, or established worker-RPC/channel break; recover that RPC channel and retry the affected task.",
            "keyword_groups": [
                ["rpc", "socket", "channel", "worker", "remote host"],
                ["restore", "reconnect", "recover"],
                ["retry", "rerun", "resubmit"],
            ],
            "min_groups": 2,
        },
        "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY": {
            "description": "Restore the ResourceManager control channel and retry container allocation or scheduling.",
            "keyword_groups": [
                ["resource manager", "rm", "allocator", ":8030", "scheduler"],
                ["restore", "reconnect", "recover"],
                ["retry", "allocation", "container"],
            ],
            "min_groups": 2,
        },
        "HADOOP_FREE_LOCAL_DISK_AND_RETRY": {
            "description": "Free explicitly exhausted local disk space before retrying the blocked workload.",
            "keyword_groups": [
                ["disk full", "no space left", "no space left on device", "insufficient space", "local disk"],
                ["free", "cleanup", "remove", "expand"],
                ["retry", "rerun", "resubmit"],
            ],
            "min_groups": 2,
        },
        "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY": {
            "description": "Reduce shuffle spill pressure or threshold stress, then retry the affected reduce path.",
            "keyword_groups": [
                ["shuffle", "spill", "merge", "maxsingleshufflelimit", "ondiskmapoutput"],
                ["reduce", "limit", "threshold", "pressure"],
                ["retry", "rerun", "tune"],
            ],
            "min_groups": 2,
        },
    },
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def allowed_family_ids(dataset: str) -> List[str]:
    return list(RCA_FAMILY_CATALOG.get(dataset, {}).keys())


def family_for_action(dataset: str, action_id: str) -> str:
    for family_id, meta in RCA_FAMILY_CATALOG.get(dataset, {}).items():
        if action_id in meta.get("action_ids", []):
            return family_id
    return ""


def allowed_action_ids(dataset: str) -> List[str]:
    return list(ACTION_CATALOG.get(dataset, {}).keys())


def describe_allowed_families(dataset: str) -> str:
    items = RCA_FAMILY_CATALOG.get(dataset, {})
    lines = ["Allowed root-cause families (choose ONE exactly):"]
    for family_id, meta in items.items():
        lines.append(f"- {family_id}: {meta['description']}")
    return "\n".join(lines)


def describe_allowed_actions(dataset: str) -> str:
    items = ACTION_CATALOG.get(dataset, {})
    lines = ["Allowed action IDs (choose ONE exactly):"]
    for action_id, meta in items.items():
        lines.append(f"- {action_id}: {meta['description']}")
    return "\n".join(lines)


def infer_action_id_from_text(dataset: str, text: str) -> str:
    t = _norm(text)
    if dataset == "HDFS":
        if any(
            pat in t
            for pat in (
                "connection reset by peer",
                "writeblock",
                "client link reset",
                "peer link reset",
                "replica service path aborted during stream handoff",
            )
        ):
            return "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE"
        if (
            any(
                pat in t
                for pat in (
                    "got exception while serving",
                    "observed exception while serving",
                    "service-path exception while serving",
                    "service-stage workflow reported irregular completion while serving",
                    "operation reported irregular handling while serving",
                    "operation interruption while handling block-id",
                    "stream handoff interruption while serving",
                )
            )
            or
            _contains_phrase(t, "served block")
            or "starting thread to transfer block" in t
            or "transmitted block" in t
            or "retransmission" in t
        ):
            return "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION"
        if any(pat in t for pat in ("packetresponder", "received block", "receiving block", "replica stage tracker", "replicastage")):
            return "HDFS_REBUILD_WRITE_PIPELINE"
        if "receiveblock" in t or "receiver" in t:
            return "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE"
        if "allocateblock" in t or "allocate block" in t:
            return "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK"
        if any(pat in t for pat in ("blockinfo not found", "unexpected error trying to delete block", "deleting block")):
            return "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK"
        if any(pat in t for pat in ("blockmap updated", "block map updated", "blockmap", "fsck")):
            return "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE"
        return ""
    if dataset == "OpenStack":
        if any(
            pat in t
            for pat in (
                "instance sync for host",
                "re-created its instancelist",
                "instance list drift",
                "host instance list drift",
                "server_external_events",
                "network-vif-plugged",
                "/servers/detail",
            )
        ):
            return "OPENSTACK_RESYNC_INSTANCE_INVENTORY"
        if any(pat in t for pat in ("sync_power_state", "pending task (spawning)", "while synchronizing instance power states", "vm started", "vm paused", "vm resumed", "instance spawned successfully")):
            return "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD"
        if any(pat in t for pat in ("terminating instance", "delete /v2/", "instance destroyed successfully")):
            return "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE"
        if any(
            pat in t
            for pat in (
                "unknown base file",
                "missing base image",
                "base image file",
                "creating image",
                "creating base image backing chain",
                "building disk image backing chain",
                "backing image",
                "backing chain",
                "image chain",
                "base or swap file too young",
            )
        ):
            return "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN"
        if any(pat in t for pat in ("removable base files", "active base files", " in use:", ": checking", "cache audit pass", "cached object")):
            return "OPENSTACK_CLEAN_STALE_BASE_CACHE_AND_RETRY"
        if any(pat in t for pat in ("cpu affinity", "vcpu count", "compatible host")):
            return "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST"
        if any(pat in t for pat in ("attempting claim", "claim successful", "resource audit", "compute resources")):
            return "OPENSTACK_REFRESH_HOST_INVENTORY_AND_RETRY_CLAIM"
        if any(pat in t for pat in ("metadata.wsgi.server", "get /openstack/2013-10-17", "get /latest/meta-data/")):
            return "OPENSTACK_SCALE_METADATA_SERVICE"
        if any(pat in t for pat in ("meta_data.json", "vendor_data.json", "metadata cache")):
            return "OPENSTACK_ENABLE_LOCAL_METADATA_CACHE"
        return ""
    if dataset == "Hadoop":
        if any(pat in t for pat in ("could not delete hdfs", "failed to remove hdfs", "output path")):
            return "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE"
        if any(pat in t for pat in ("bad datanode", "heartbeat", "host appears unavailable", "unhealthy data node", "machine down")):
            return "HADOOP_ISOLATE_NODE_AND_RESCHEDULE"
        if any(pat in t for pat in ("forcibly closed by the remote host", "peer terminated the socket unexpectedly", "socket reader", "worker rpc", "readandprocess", "channel")):
            return "HADOOP_RESTORE_WORKER_RPC_AND_RETRY"
        if any(pat in t for pat in ("rmcommunicator allocator", ":8030", "resource manager", "container allocator")):
            return "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
        if any(pat in t for pat in ("disk full", "no space left on device", "no space left", "insufficient space", "out of disk")):
            return "HADOOP_FREE_LOCAL_DISK_AND_RETRY"
        if any(pat in t for pat in ("shuffling to disk", "ondiskmapoutput", "maxsingleshufflelimit", "mergemanagerimpl", "merging ", "shuffle pressure", "shuffleerror")):
            return "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY"
        return ""
    return ""


def infer_family_from_text(dataset: str, text: str) -> str:
    action_id = infer_action_id_from_text(dataset, text)
    return family_for_action(dataset, action_id)


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    lower = _norm(text)
    return any(tok.lower() in lower for tok in tokens)


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


__all__ = [
    "ACTION_CATALOG",
    "RCA_FAMILY_CATALOG",
    "action_text_match",
    "action_text_success",
    "allowed_action_ids",
    "allowed_family_ids",
    "describe_allowed_actions",
    "describe_allowed_families",
    "family_for_action",
    "infer_action_id_from_text",
    "infer_family_from_text",
]
