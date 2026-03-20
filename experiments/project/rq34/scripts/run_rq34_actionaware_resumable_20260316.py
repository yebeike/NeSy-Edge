from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.actionaware_catalog_20260316 import (
    ACTION_CATALOG,
    action_text_match,
    action_text_success,
    allowed_action_ids,
    describe_allowed_actions,
    gt_action_id_for_case,
    infer_action_id_from_text,
    infer_label_from_text,
    label_for_action,
)
from experiments.thesis_rebuild_20260315.rq34.scripts.build_rq34_enriched_seed_pool_20260316 import (
    OUTPUT_PATH as ENRICHED_SEED_POOL_PATH,
    write_enriched_seed_pool,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json


LEGACY_STAGE4 = _PROJECT_ROOT / "experiments" / "rq123_e2e" / "stage4_noise_api_sampled_20260313.py"
REBUILD_RESULTS_DIR = _REBUILD_ROOT / "rq34" / "results"
BENCH_V2_PATH = _PROJECT_ROOT / "data" / "processed" / "e2e_scaled_benchmark_v2.json"
RQ3_TEST_SET_PATH = _PROJECT_ROOT / "data" / "processed" / "rq3_test_set.json"
RQ2_FULLCASE_MODIFIED_GRAPH = (
    _REBUILD_ROOT / "rq2_fullcase" / "results" / "gt_causal_knowledge_nesydy_fullcase_20260316.json"
)
METHODS = ["agent", "vanilla", "rag"]
DATASETS = ["HDFS", "OpenStack", "Hadoop"]
MIN_CONTEXT_LINES = {"HDFS": 3, "OpenStack": 3, "Hadoop": 3}
POOL_PRIORITY = {"rq3_test_set_enriched": 0, "rq3_test_set": 1, "benchmark_v2": 2}
MAIN_ACTION_EXCLUDE = {
    "HDFS": {
        "HDFS_VALIDATE_BLOCKMAP_AND_REREPLICATE",
        "HDFS_CHECK_LINK_AND_LIMIT_RETRANSMISSION",
    }
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-per-dataset", type=int, default=15)
    ap.add_argument("--noise-levels", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--run-tag", type=str, default="actionaware15x6_v1")
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
        default="benchmark_v2_only",
        choices=["benchmark_v2_only", "mixed_richer", "enriched_seeded"],
        help="Case-pool policy for sampled evaluation.",
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


def _noise_key(noise: float) -> str:
    return f"{float(noise):.1f}"


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
        action_items.sort(
            key=lambda x: (
                POOL_PRIORITY.get(str(x.get("pool_source", "benchmark_v2")), 99),
                str(x["case"].get("case_id", "")),
            )
        )
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
    if "rq3_test_set" in source or source == "causal_edge":
        return len(lines) >= 1
    return len(lines) >= MIN_CONTEXT_LINES.get(dataset, 1)


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
            legacy_label = legacy.gt_label_for_case(case)
            gt_action_id = gt_action_id_for_case(case, legacy_label)
            if not gt_action_id:
                gt_action_id = (
                    infer_action_id_from_text(dataset, str(case.get("raw_log_seed", "") or ""))
                    or infer_action_id_from_text(dataset, str(case.get("raw_log", "") or ""))
                    or infer_action_id_from_text(
                        dataset, str(case.get("ground_truth_root_cause_template", "") or "")
                    )
                )
            gt_label = label_for_action(dataset, gt_action_id) or legacy_label
            if not gt_label or not gt_action_id:
                continue
            if gt_action_id in MAIN_ACTION_EXCLUDE.get(dataset, set()):
                continue
            item = {
                "case": dict(case),
                "gt_label": gt_label,
                "gt_action_id": gt_action_id,
                "pool_source": pool_source,
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
        items = by_dataset[ds]
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
                raw_label = str(obj.get("root_cause_label") or obj.get("label") or "").strip()
                repair_action = str(obj.get("repair_action") or obj.get("action") or "").strip()
                for a in allowed_labels:
                    if a.lower() == raw_label.lower():
                        pred_label = a
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
        inferred_label = infer_label_from_text(dataset, repair_action)
        if inferred_label in allowed_labels:
            pred_label = inferred_label
    if not pred_label and action_id:
        inferred_from_action = label_for_action(dataset, action_id)
        if inferred_from_action in allowed_labels:
            pred_label = inferred_from_action
    return pred_label, action_id, repair_action


def _select_actionaware_alert(legacy, raw: str, dataset: str) -> str:
    lines = [line for line in str(raw or "").split("\n") if line.strip()]
    if not lines:
        return legacy._select_alert_line(raw, dataset)

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
                score += 10
            if "synchronizing instance power states" in lower:
                score += 9
            if "metadata" in lower:
                score += 7
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
                score += 9
            elif "maxsingleshufflelimit" in lower:
                score += 5
        if idx >= len(lines) - 2:
            score += 1
        if score > 0:
            scored.append((score, idx, line))

    if not scored:
        return legacy._select_alert_line(raw, dataset)
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


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
            if "retrying connect to server" in lower:
                retry_hits += 1
            if "forcibly closed by the remote host" in lower:
                forced_hits += 1
            if "could not delete hdfs" in lower or "bad datanode" in lower:
                delete_hits += 1
            if (
                "disk full" in lower
                or "no space" in lower
                or "shuffling to disk" in lower
                or "maxsingleshufflelimit" in lower
            ):
                disk_hits += 1
    if dataset == "Hadoop":
        # Persistent retries to the same host are often the observable symptom
        # of a machine-down case rather than a transient link issue.
        if retry_hits >= 3 and (forced_hits >= 1 or delete_hits >= 1):
            support["HADOOP_ISOLATE_NODE_AND_RESCHEDULE"] = (
                support.get("HADOOP_ISOLATE_NODE_AND_RESCHEDULE", 0) + 3
            )
        if retry_hits >= 1 and forced_hits >= 1:
            support["HADOOP_RESTORE_NETWORK_AND_RETRY"] = (
                support.get("HADOOP_RESTORE_NETWORK_AND_RETRY", 0) + 1
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
    max_chars: int = 900,
) -> str:
    lines = [line for line in str(raw or "").split("\n") if line.strip()]
    if not lines:
        return ""

    top_actions = set(_top_candidate_actions(dataset, cand_json, observed_template))
    observed_action = infer_action_id_from_text(dataset, observed_template)

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
        if line_action and line_action in top_actions and line_action != observed_action:
            score += 8
        elif line_action and line_action == observed_action:
            score += 1
        if score > 0:
            scored.append((score, idx, line))

    if not scored:
        return "\n".join(lines[-6:])[-max_chars:]
    keep = sorted(scored, key=lambda x: (-x[0], x[1]))[:8]
    selected = [line for _, _, line in sorted(keep, key=lambda x: x[1])]
    return "\n".join(selected)[-max_chars:]


def _candidate_summary(dataset: str, cand_json: str, observed_template: str, context_text: str) -> str:
    try:
        obj = json.loads(cand_json)
    except Exception:
        obj = []
    context_support = _line_action_support(dataset, context_text)
    agg: Dict[str, Dict[str, object]] = {}
    for item in obj if isinstance(obj, list) else []:
        if not isinstance(item, dict):
            continue
        tpl = str(item.get("source_template", "") or "")
        weight = abs(float(item.get("weight", 0.0) or 0.0))
        action_id = infer_action_id_from_text(dataset, tpl)
        label = label_for_action(dataset, action_id) or infer_label_from_text(dataset, tpl)
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
    if not agg:
        return "No structured causal candidate summary available."
    ordered = sorted(agg.values(), key=lambda x: float(x["score"]), reverse=True)[:5]
    lines = ["Ranked causal candidate summary:"]
    for i, item in enumerate(ordered, start=1):
        lines.append(
            f"{i}. root_label={item['label'] or 'UNKNOWN'}; action_id={item['action_id'] or 'UNKNOWN'}; "
            f"score={float(item['score']):.3f}; graph={float(item['graph_score']):.3f}; "
            f"context_hits={int(item['support_hits'])}; examples={item['examples']}"
        )
    return "\n".join(lines)


def _heuristic_action_hint(dataset: str, selected_alert: str, context_text: str, observed_template: str) -> str:
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


def _should_use_agent_shortcut(dataset: str, heuristic_action_hint: str) -> bool:
    if dataset != "Hadoop":
        return False
    return heuristic_action_hint in {
        "HADOOP_ISOLATE_NODE_AND_RESCHEDULE",
        "HADOOP_FREE_DISK_AND_RETRY",
    }


def _build_prompt(
    legacy,
    *,
    method: str,
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
    label_desc = legacy.describe_allowed_labels(dataset, allowed_labels)
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
        "{\"root_cause_label\":\"<ONE_LABEL>\",\"repair_action\":\"<short action plan grounded in the allowed actions>\"}.\n"
    )
    heuristic_hint_text = ""
    if heuristic_action_hint:
        heuristic_label = label_for_action(dataset, heuristic_action_hint)
        heuristic_hint_text = (
            f"Heuristic action hint: {heuristic_action_hint}\n"
            f"Heuristic label hint: {heuristic_label}\n"
        )
    if method == "agent":
        refs = legacy.rq3_tools.knowledge_retriever(clean_for_parse[:200], dataset, top_k=3)
        clue = f"Symbolic clue: {symbolic_label}\n" if symbolic_label else ""
        cand_summary = _candidate_summary(dataset, cand_json, tpl_agent or clean_for_parse, noised_context)
        return (
            "You are NeSy-Agent. Use only the provided context, causal candidates, and references.\n"
            "Treat the selected alert line as the strongest direct failure signal.\n"
            "Use the ranked candidate summary together with the context lines to infer the precursor/root cause.\n"
            "When the selected alert line already states a concrete failure mode, do not let background context override it.\n"
            "For Hadoop, repeated retries to the same host together with forced-close or delete failures usually indicate node unavailability rather than a transient network glitch.\n"
            "For OpenStack, prefer explicit hypervisor or CPU-affinity failures over earlier image-cache bookkeeping lines when both appear.\n"
            "Choose the most likely root cause label and the most appropriate remediation action ID, then write a short action plan that mentions the concrete operational checks or recovery steps.\n"
            f"{base}"
            f"{clue}"
            f"{heuristic_hint_text}"
            f"{cand_summary}\n"
            f"Raw causal candidates (JSON): {cand_json}\n"
            f"References:\n{refs}\n"
        )
    if method == "rag":
        refs = legacy.rq3_tools.knowledge_retriever(clean_for_parse[:200], dataset, top_k=3)
        return (
            "You are an ops expert. Use the logs and references to choose the best root cause label and remediation action ID.\n"
            "Treat the selected alert line as the strongest direct failure signal.\n"
            "For OpenStack, prefer explicit hypervisor or CPU-affinity failures over earlier image-cache bookkeeping lines when both appear.\n"
            "For Hadoop, repeated retries to the same host together with forced-close or delete failures usually indicate node unavailability rather than a transient network glitch.\n"
            f"{base}"
            f"References:\n{refs}\n"
        )
    return (
        "You are an ops expert. Use only the logs to choose the best root cause label and remediation action ID.\n"
        "Treat the selected alert line as the strongest direct failure signal.\n"
        "For OpenStack, prefer explicit hypervisor or CPU-affinity failures over earlier image-cache bookkeeping lines when both appear.\n"
        "For Hadoop, repeated retries to the same host together with forced-close or delete failures usually indicate node unavailability rather than a transient network glitch.\n"
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

    for item in labeled_cases:
        case = dict(item["case"])
        gt_label = str(item["gt_label"])
        gt_action_id = str(item["gt_action_id"])
        raw = str(case.get("raw_log", "") or "")
        dataset = str(case.get("dataset", "HDFS"))
        case_id = str(case.get("case_id", ""))

        alert = _select_actionaware_alert(legacy, raw, dataset)
        ds_parse = dataset
        for noise in noise_levels:
            noisy_alert = legacy._inject_noise(alert, dataset, injector, injector_hadoop, noise)
            clean_for_parse = legacy.NuSyEdgeNode.preprocess_header(noisy_alert, ds_parse) or noisy_alert
            denoised_for_agent = legacy._denoise_for_nusy(dataset, clean_for_parse)
            try:
                tpl_nusy, _, _, _ = edge_node.parse_log_stream(denoised_for_agent, ds_parse)
            except Exception:
                tpl_nusy = ""
            try:
                tpl_drain = legacy._DRAIN.parse(denoised_for_agent)
            except Exception:
                tpl_drain = ""
            tpl_agent = tpl_nusy if legacy._valid_template(tpl_nusy) else tpl_drain
            allowed_labels = legacy.allowed_labels_for_dataset(dataset)
            allowed_actions = allowed_action_ids(dataset)
            domain = "hdfs" if dataset == "HDFS" else ("openstack" if dataset == "OpenStack" else "hadoop")
            cand_json = legacy.rq3_tools.causal_navigator(
                tpl_agent or denoised_for_agent, domain, causal_path=args.causal_graph_path
            )
            focus_context = _extract_actionaware_context(raw, dataset, cand_json, tpl_agent or clean_for_parse, max_chars=900)
            noised_context = legacy._truncate_and_inject_noise(
                focus_context, dataset, injector, injector_hadoop, noise, max_chars=900
            )
            symbolic_label, _ = legacy._agent_symbolic_vote(
                dataset,
                noised_context,
                clean_for_parse,
                denoised_for_agent,
                tpl_agent,
                cand_json,
            )

            for method in METHODS:
                step = _step_key(dataset, case_id, noise, method)
                if step in completed:
                    continue

                heuristic_action_hint = _heuristic_action_hint(
                    dataset, noisy_alert, noised_context, tpl_agent or clean_for_parse
                )

                prompt = _build_prompt(
                    legacy,
                    method=method,
                    dataset=dataset,
                    noise=noise,
                    selected_alert=noisy_alert,
                    noised_context=noised_context,
                    clean_for_parse=clean_for_parse,
                    tpl_agent=tpl_agent,
                    cand_json=cand_json,
                    symbolic_label=symbolic_label,
                    heuristic_action_hint=heuristic_action_hint if method == "agent" else "",
                    allowed_labels=allowed_labels,
                    allowed_actions=allowed_actions,
                )
                api_call = True
                if method == "agent" and _should_use_agent_shortcut(dataset, heuristic_action_hint):
                    pred_action_id = heuristic_action_hint
                    pred_label = label_for_action(dataset, pred_action_id)
                    repair_action = str(
                        ACTION_CATALOG.get(dataset, {})
                        .get(pred_action_id, {})
                        .get("description", "")
                    )
                    api_call = False
                else:
                    resp = legacy._call_deepseek_with_retry(
                        prompt, api_key=deepseek_key, model="deepseek-chat", max_tokens=256
                    )
                    actual_api_calls += 1
                    pred_label, pred_action_id, repair_action = _extract_structured_output(
                        dataset, resp, allowed_labels, allowed_actions
                    )

                matched_groups, min_groups, hit_groups = action_text_match(dataset, gt_action_id, repair_action)
                rca_success = bool(pred_label and pred_label == gt_label)
                action_txt_success = bool(gt_action_id and action_text_success(dataset, gt_action_id, repair_action))
                # In the action-aware design, the model is required to produce
                # a grounded repair plan, not necessarily an explicit action ID.
                # Therefore, we treat a plan as operationally correct when it
                # either maps to the exact GT action ID or satisfies the GT
                # action's keyword constraints.
                action_success = bool(
                    (pred_action_id and pred_action_id == gt_action_id) or action_txt_success
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
