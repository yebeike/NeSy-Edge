from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

from experiments.thesis_rebuild_20260315.rq34.scripts.actionaware_catalog_20260316 import (
    ACTION_CATALOG,
    action_text_match,
    action_text_success,
    allowed_action_ids,
    describe_allowed_actions,
    infer_action_id_from_text,
    label_for_action,
)


RCA_FAMILY_CATALOG: Dict[str, Dict[str, Dict[str, object]]] = {
    "HDFS": {
        "HDFS_TRANSFER_LINK_FAILURE": {
            "description": "Client-to-DataNode transfer or replication-link failures that interrupt block serving or cross-node transfer.",
            "action_ids": [
                "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE",
                "HDFS_TUNE_REPLICATION_FLOW",
                "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION",
            ],
        },
        "HDFS_PIPELINE_FAILURE": {
            "description": "Write-pipeline or receive-block failures that require rebuilding or isolating the affected pipeline segment.",
            "action_ids": [
                "HDFS_ISOLATE_RECEIVER_AND_REBUILD_PIPELINE",
                "HDFS_REBUILD_WRITE_PIPELINE",
            ],
        },
        "HDFS_STORAGE_METADATA_PRESSURE": {
            "description": "Capacity, stale-block, or metadata-consistency failures that require storage cleanup, fsck, or allocation retry.",
            "action_ids": [
                "HDFS_CHECK_STORAGE_AND_DELETE_STALE_BLOCK",
                "HDFS_RETRY_ALLOCATION_AFTER_CAPACITY_CHECK",
                "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE",
            ],
        },
    },
    "OpenStack": {
        "OPENSTACK_INSTANCE_STATE_DRIFT": {
            "description": "Instance power-state or inventory drift that requires resynchronization, rebuild, or restore actions.",
            "action_ids": [
                "OPENSTACK_RESYNC_INSTANCE_INVENTORY",
                "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD",
                "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE",
            ],
        },
        "OPENSTACK_IMAGE_CHAIN_FAILURE": {
            "description": "Missing or broken image-chain state that requires base-image recreation or remounting.",
            "action_ids": [
                "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN",
            ],
        },
        "OPENSTACK_CONTROL_SERVICE_PRESSURE": {
            "description": "Metadata or control-service pressure that requires service scaling or caching.",
            "action_ids": [
                "OPENSTACK_SCALE_METADATA_SERVICE",
            ],
        },
        "OPENSTACK_HOST_COMPATIBILITY_FAILURE": {
            "description": "Hypervisor capability or host-compatibility failures that require migration or rebuild on a compatible host.",
            "action_ids": [
                "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST",
            ],
        },
    },
    "Hadoop": {
        "HADOOP_NODE_UNAVAILABILITY": {
            "description": "Node-level unavailability that requires isolating the failed node and rescheduling work.",
            "action_ids": [
                "HADOOP_ISOLATE_NODE_AND_RESCHEDULE",
            ],
        },
        "HADOOP_NETWORK_CONNECTIVITY": {
            "description": "Connectivity failures that require restoring network links and retrying the affected task.",
            "action_ids": [
                "HADOOP_RESTORE_NETWORK_AND_RETRY",
            ],
        },
        "HADOOP_STORAGE_PRESSURE": {
            "description": "Disk-capacity pressure that requires cleanup or expansion before retrying.",
            "action_ids": [
                "HADOOP_FREE_DISK_AND_RETRY",
            ],
        },
    },
}


WEAK_ACTION_IDS = {
    "HDFS_TUNE_REPLICATION_FLOW",
    "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION",
    "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE",
}


def allowed_family_ids(dataset: str) -> List[str]:
    return list(RCA_FAMILY_CATALOG.get(dataset, {}).keys())


def family_for_action(dataset: str, action_id: str) -> str:
    for family_id, meta in RCA_FAMILY_CATALOG.get(dataset, {}).items():
        if action_id in meta.get("action_ids", []):
            return family_id
    return ""


def describe_allowed_families(dataset: str) -> str:
    items = RCA_FAMILY_CATALOG.get(dataset, {})
    lines = ["Allowed root-cause families (choose ONE exactly):"]
    for family_id, meta in items.items():
        lines.append(f"- {family_id}: {meta['description']}")
    return "\n".join(lines)


def infer_family_from_text(dataset: str, text: str) -> str:
    action_id = infer_action_id_from_text(dataset, text)
    if not action_id:
        return ""
    return family_for_action(dataset, action_id)


def select_gt_action_and_family(
    dataset: str,
    *,
    selected_alert: str,
    raw_log: str,
    raw_log_seed: str,
    gt_root_template: str,
    gt_effect_template: str,
    gt_action_label: str,
    context_support: Mapping[str, int],
) -> Tuple[str, str, Dict[str, object]]:
    authoritative_sources = [
        ("gt_root_template", str(gt_root_template or "").strip(), 12.0),
        ("gt_action_label", str(gt_action_label or "").strip(), 8.0),
        ("gt_effect_template", str(gt_effect_template or "").strip(), 5.0),
    ]
    authoritative_votes: Dict[str, float] = defaultdict(float)
    authoritative_evidence: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    authoritative_hits = 0
    for source, text, weight in authoritative_sources:
        if not text:
            continue
        action_id = infer_action_id_from_text(dataset, text)
        if not action_id:
            continue
        eff_weight = weight * (0.6 if action_id in WEAK_ACTION_IDS else 1.0)
        authoritative_votes[action_id] += eff_weight
        authoritative_evidence[action_id].append((source, round(eff_weight, 3)))
        authoritative_hits += 1
    if authoritative_votes:
        ordered = sorted(authoritative_votes.items(), key=lambda kv: (-kv[1], kv[0]))
        winner, winner_score = ordered[0]
        runner_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = float(winner_score - runner_score)
        family_id = family_for_action(dataset, winner)
        diagnostics = {
            "confidence": "high" if authoritative_hits >= 1 else "medium",
            "winner_score": round(float(winner_score), 3),
            "runner_score": round(float(runner_score), 3),
            "margin": round(margin, 3),
            "votes": {k: round(float(v), 3) for k, v in ordered},
            "evidence": {k: v for k, v in authoritative_evidence.items()},
            "mode": "authoritative_gt",
        }
        return winner, family_id, diagnostics

    votes: Dict[str, float] = defaultdict(float)
    evidence: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def add_vote(text: str, weight: float, source: str) -> None:
        action_id = infer_action_id_from_text(dataset, text)
        if not action_id:
            return
        eff_weight = weight * (0.6 if action_id in WEAK_ACTION_IDS else 1.0)
        votes[action_id] += eff_weight
        evidence[action_id].append((source, round(eff_weight, 3)))

    add_vote(selected_alert, 2.0, "selected_alert")
    add_vote(gt_root_template, 3.5, "gt_root_template")
    add_vote(gt_action_label, 3.0, "gt_action_label")
    add_vote(raw_log_seed, 1.5, "raw_log_seed")
    add_vote(gt_effect_template, 2.0, "gt_effect_template")
    add_vote(raw_log[-1200:], 1.0, "raw_tail")

    for action_id, hits in context_support.items():
        bonus = min(int(hits), 5) * (0.8 if action_id not in WEAK_ACTION_IDS else 0.35)
        if bonus <= 0:
            continue
        votes[action_id] += bonus
        evidence[action_id].append(("context_support", round(bonus, 3)))

    ordered = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ordered:
        return "", "", {"confidence": "none", "votes": {}, "margin": 0.0}

    winner, winner_score = ordered[0]
    runner_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = float(winner_score - runner_score)
    family_id = family_for_action(dataset, winner)
    confidence = "high" if winner_score >= 5.5 and margin >= 1.5 else "medium" if winner_score >= 4.0 and margin >= 0.75 else "low"
    diagnostics = {
        "confidence": confidence,
        "winner_score": round(float(winner_score), 3),
        "runner_score": round(float(runner_score), 3),
        "margin": round(margin, 3),
        "votes": {k: round(float(v), 3) for k, v in ordered},
        "evidence": {k: v for k, v in evidence.items()},
    }
    return winner, family_id, diagnostics


__all__ = [
    "ACTION_CATALOG",
    "RCA_FAMILY_CATALOG",
    "WEAK_ACTION_IDS",
    "action_text_match",
    "action_text_success",
    "allowed_action_ids",
    "allowed_family_ids",
    "describe_allowed_actions",
    "describe_allowed_families",
    "family_for_action",
    "infer_action_id_from_text",
    "infer_family_from_text",
    "label_for_action",
    "select_gt_action_and_family",
]
