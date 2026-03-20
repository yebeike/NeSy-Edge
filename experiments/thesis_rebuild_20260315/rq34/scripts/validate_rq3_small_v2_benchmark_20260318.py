from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    RCA_FAMILY_CATALOG,
    family_for_action,
    infer_action_id_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


PACKAGE_PATH = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v2_20260318" / "rq3_small_v2_benchmark_package_20260318.json"
OUTPUT_PATH = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v2_20260318" / "rq3_small_v2_validation_20260318.json"


def _contains_label_leak(text: str) -> bool:
    lower = str(text or "").lower()
    if "family=" in lower or "action=" in lower:
        return True
    for dataset, families in RCA_FAMILY_CATALOG.items():
        for family_id in families:
            if family_id.lower() in lower:
                return True
    for dataset, actions in ACTION_CATALOG.items():
        for action_id in actions:
            if action_id.lower() in lower:
                return True
    return False


def _contains_remediation_leak(text: str) -> bool:
    lower = str(text or "").lower()
    if "operational note:" in lower:
        return True
    for actions in ACTION_CATALOG.values():
        for meta in actions.values():
            description = str(meta.get("description", "")).strip().lower()
            if description and description in lower:
                return True
    return False


def _contains_graph_leak(text: str) -> bool:
    lower = str(text or "").lower()
    return "graph action hints:" in lower or "action=" in lower or "support_templates=" in lower


def _semantic_contract_errors(dataset: str, case_id: str, gt_action: str, alert: str, context: str, noise_key: str) -> List[str]:
    lower_alert = str(alert or "").lower()
    lower_context = str(context or "").lower()
    combined = f"{lower_alert}\n{lower_context}"
    errors: List[str] = []

    if gt_action == "HDFS_REPAIR_CLIENT_LINK_AND_REREPLICATE":
        transfer_terms = (
            "got exception while serving",
            "observed exception while serving",
            "service-path exception while serving",
            "while serving",
            "starting thread to transfer block",
            "served block",
            "transmitted block",
            "stream handoff",
            "service-stage workflow",
            "operation reported irregular handling while serving",
        )
        if not any(term in combined for term in transfer_terms):
            errors.append(f"{dataset}/{case_id}/{noise_key}: transfer-link case lost serving semantics")

    if gt_action == "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN":
        allowed_terms = (
            "unknown base file",
            "missing base image",
            "base image",
            "base file",
            "creating image",
            "disk image",
            "backing image",
            "backing chain",
            "image chain",
        )
        banned_terms = (
            "runtime object lineage",
            "inventory",
            "power state",
            "vm ",
            "event",
            " in use:",
            ": checking",
            "removable base files",
            "active base files",
        )
        if not any(term in combined for term in allowed_terms):
            errors.append(f"{dataset}/{case_id}/{noise_key}: image-repair case lost image-chain semantics")
        if any(term in lower_alert for term in banned_terms) and not any(term in lower_alert for term in allowed_terms):
            errors.append(f"{dataset}/{case_id}/{noise_key}: image-repair alert drifted into cleanup/state wording")

    if gt_action == "HADOOP_ISOLATE_NODE_AND_RESCHEDULE":
        retry_hits = sum(
            combined.count(term)
            for term in (
                "retrying connect to server",
                "retrying worker-node connection to server",
                "retrying rpc toward node",
                "connect to server",
            )
        )
        has_policy_hint = any(
            term in combined for term in ("nodeblacklistingenabled", "maxtaskfailurespernode", "blacklistdisablepercent")
        )
        if retry_hits < 1:
            errors.append(f"{dataset}/{case_id}/{noise_key}: worker-isolate case lacks direct retry/connect evidence")
        if retry_hits < 2 and not has_policy_hint:
            errors.append(f"{dataset}/{case_id}/{noise_key}: worker-isolate case lacks repeated-retry or blacklisting evidence")
    if gt_action == "HADOOP_CLEAN_OUTPUT_PATH_AND_RESCHEDULE":
        if not any(term in combined for term in ("could not delete hdfs", "failed to remove hdfs", "cleanup failed")):
            errors.append(f"{dataset}/{case_id}/{noise_key}: worker-cleanup case lacks cleanup evidence")
    if gt_action == "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY":
        if not any(term in combined for term in ("rmcommunicator allocator", ":8030", "resource manager", "container allocator")):
            errors.append(f"{dataset}/{case_id}/{noise_key}: RM-channel case lacks RM-specific evidence")
    if gt_action == "HADOOP_RESTORE_WORKER_RPC_AND_RETRY":
        if not any(term in combined for term in ("forcibly closed by the remote host", "peer terminated", "socket reader", "readandprocess", "worker rpc", "channel")):
            errors.append(f"{dataset}/{case_id}/{noise_key}: worker-RPC case lacks worker-channel evidence")
    if gt_action == "HADOOP_REDUCE_SHUFFLE_PRESSURE_AND_RETRY":
        if not any(term in combined for term in ("shuffling to disk", "maxsingleshufflelimit", "mergemanagerimpl", "shuffleerror", "merging ", "ondiskmapoutput")):
            errors.append(f"{dataset}/{case_id}/{noise_key}: shuffle-pressure case lacks shuffle/spill evidence")
    if gt_action == "HADOOP_FREE_LOCAL_DISK_AND_RETRY":
        if not any(term in combined for term in ("disk full", "no space left", "no space left on device", "insufficient space", "out of disk")):
            errors.append(f"{dataset}/{case_id}/{noise_key}: local-disk case lacks explicit disk exhaustion evidence")

    return errors


def main() -> None:
    package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
    cases = list(package.get("cases", []))
    reference_policy = dict(package.get("reference_policy", {}))
    errors: List[str] = []
    warnings: List[str] = []

    dataset_counts: Counter[str] = Counter()
    family_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    action_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    direct_alert_matches: Dict[str, int] = defaultdict(int)
    direct_context_matches: Dict[str, int] = defaultdict(int)
    changed_alert_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    changed_context_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for case in cases:
        dataset = str(case["dataset"])
        case_id = str(case["case_id"])
        gt_family = str(case["gt_family_id"])
        gt_action = str(case["gt_action_id"])
        dataset_counts[dataset] += 1
        family_counts[dataset][gt_family] += 1
        action_counts[dataset][gt_action] += 1

        if family_for_action(dataset, gt_action) != gt_family:
            errors.append(f"{dataset}/{case_id}: gt action/family mismatch")

        alert_clean = str(case.get("selected_alert_clean", ""))
        context_clean = str(case.get("shared_context_clean", ""))
        if not alert_clean:
            errors.append(f"{dataset}/{case_id}: missing clean selected alert")
        if not context_clean:
            errors.append(f"{dataset}/{case_id}: missing clean context")
        if alert_clean and alert_clean not in str(case.get("raw_log", "")):
            errors.append(f"{dataset}/{case_id}: clean selected alert not found in raw log")
        if alert_clean and alert_clean not in context_clean:
            warnings.append(f"{dataset}/{case_id}: clean context does not contain selected alert")

        if infer_action_id_from_text(dataset, alert_clean) == gt_action:
            direct_alert_matches[dataset] += 1
        if infer_action_id_from_text(dataset, context_clean) == gt_action:
            direct_context_matches[dataset] += 1

        noise_views = dict(case.get("noise_views", {}))
        for noise_key, view in noise_views.items():
            alert = str(view.get("selected_alert", ""))
            context = str(view.get("context_text", ""))
            graph_summary = str(view.get("graph_summary", ""))
            agent_refs = list(view.get("agent_references", []))
            rag_refs = list(view.get("rag_references", []))
            if not alert or not context:
                errors.append(f"{dataset}/{case_id}/{noise_key}: missing frozen alert or context")
            if not graph_summary:
                errors.append(f"{dataset}/{case_id}/{noise_key}: missing graph summary")
            if _contains_graph_leak(graph_summary):
                errors.append(f"{dataset}/{case_id}/{noise_key}: graph summary exposes action-level or template-level hint")
            if int(reference_policy.get("agent_top_k", 0)) and len(agent_refs) == 0:
                warnings.append(f"{dataset}/{case_id}/{noise_key}: missing agent references")
            if int(reference_policy.get("rag_top_k", 0)) and len(rag_refs) == 0:
                warnings.append(f"{dataset}/{case_id}/{noise_key}: missing rag references")
            if alert != alert_clean:
                changed_alert_counts[dataset][noise_key] += 1
            if context != context_clean:
                changed_context_counts[dataset][noise_key] += 1
            for ref in agent_refs + rag_refs:
                ref_text = str(ref.get("text", ""))
                ref_id = str(ref.get("reference_id", ""))
                ref_alert = str(ref.get("selected_alert", ""))
                if not ref_text:
                    errors.append(f"{dataset}/{case_id}/{noise_key}: empty reference text in {ref_id}")
                if _contains_label_leak(ref_text):
                    errors.append(f"{dataset}/{case_id}/{noise_key}: label leakage detected in {ref_id}")
                if _contains_remediation_leak(ref_text):
                    errors.append(f"{dataset}/{case_id}/{noise_key}: remediation leakage detected in {ref_id}")
                if case_id and case_id in ref_id:
                    errors.append(f"{dataset}/{case_id}/{noise_key}: self-reference detected in {ref_id}")
                if ref_alert and ref_alert == alert_clean:
                    errors.append(f"{dataset}/{case_id}/{noise_key}: eval/reference selected-alert duplicate in {ref_id}")
            errors.extend(_semantic_contract_errors(dataset, case_id, gt_action, alert, context, noise_key))

    for dataset in sorted(dataset_counts):
        if dataset_counts[dataset] != 9:
            errors.append(f"{dataset}: expected 9 cases, found {dataset_counts[dataset]}")
        if len(family_counts[dataset]) < 2:
            errors.append(f"{dataset}: fewer than 2 families represented")
        if len(action_counts[dataset]) < 4:
            warnings.append(f"{dataset}: fewer than 4 actions represented")
        for noise_key in ("0.6", "1.0"):
            if changed_alert_counts[dataset][noise_key] < max(1, dataset_counts[dataset] // 2):
                warnings.append(
                    f"{dataset}: fewer than half of alerts changed under noise {noise_key}"
                )
            if changed_context_counts[dataset][noise_key] < max(1, dataset_counts[dataset] // 2):
                warnings.append(
                    f"{dataset}: fewer than half of contexts changed under noise {noise_key}"
                )

    summary = {
        "cases": len(cases),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "family_counts": {dataset: dict(sorted(counter.items())) for dataset, counter in family_counts.items()},
        "action_counts": {dataset: dict(sorted(counter.items())) for dataset, counter in action_counts.items()},
        "direct_alert_match_counts": dict(sorted(direct_alert_matches.items())),
        "direct_context_match_counts": dict(sorted(direct_context_matches.items())),
        "changed_alert_counts": {dataset: dict(sorted(counter.items())) for dataset, counter in changed_alert_counts.items()},
        "changed_context_counts": {dataset: dict(sorted(counter.items())) for dataset, counter in changed_context_counts.items()},
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }
    write_json(OUTPUT_PATH, summary)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
