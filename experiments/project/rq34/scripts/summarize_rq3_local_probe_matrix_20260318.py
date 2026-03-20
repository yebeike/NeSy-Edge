from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    family_for_action,
    infer_action_id_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


DEFAULT_BENCHMARK_PATH = (
    REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v2_20260318" / "rq3_small_v2_benchmark_package_20260318.json"
)
DEFAULT_PROGRESS_PATH = (
    REBUILD_ROOT / "rq34" / "analysis" / "rq3_local_probe_20260318" / "rq3_local_probe_matrix_qwen35_9b_20260318_progress.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REBUILD_ROOT / "rq34" / "analysis" / "rq3_local_probe_20260318" / "rq3_local_probe_matrix_qwen35_9b_20260318_report.json"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    ap.add_argument("--progress-path", type=Path, default=DEFAULT_PROGRESS_PATH)
    ap.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return ap.parse_args()


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return round(sum(vals) / len(vals), 4)


def _median(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return round(float(median(vals)), 4)


def _line_count(text: str) -> int:
    return len([line for line in str(text or "").splitlines() if line.strip()])


def _top_action_from_overlap(text: str, dataset: str) -> str:
    lowered = " ".join(str(text or "").lower().split())
    best_action = ""
    best_score = -1.0
    for action_id, meta in ACTION_CATALOG.get(dataset, {}).items():
        desc = " ".join(str(meta.get("description", "")).lower().split())
        overlap = len(set(desc.split()) & set(lowered.split()))
        hits = 0
        for group in meta.get("keyword_groups", []):
            if any(token.lower() in lowered for token in group):
                hits += 1
        score = overlap + 2.5 * hits
        if score > best_score:
            best_score = score
            best_action = action_id
    return best_action


def _build_structural_report(package: Mapping[str, object]) -> Dict[str, object]:
    cases = list(package.get("cases", []))
    dataset_rows: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    same_alert_context = Counter()
    line_counts = defaultdict(list)
    direct_alert_action = Counter()
    direct_context_action = Counter()
    desc_alert_action = Counter()
    desc_context_action = Counter()
    graph_top1_gt = Counter()
    graph_rows = Counter()
    no_graph_hint = Counter()
    stable_reference_sets = Counter()
    agent_subset_rag = Counter()

    for case in cases:
        dataset = str(case["dataset"])
        dataset_rows[dataset].append(case)
        stable_agent_sets = set()
        stable_rag_sets = set()
        subset_ok = True
        for noise_key, view in dict(case["noise_views"]).items():
            alert = str(view.get("selected_alert", ""))
            context = str(view.get("context_text", ""))
            if " ".join(alert.split()) == " ".join(context.split()):
                same_alert_context[dataset] += 1
            line_counts[dataset].append(_line_count(context))

            direct_alert = infer_action_id_from_text(dataset, alert)
            direct_context = infer_action_id_from_text(dataset, context)
            if direct_alert == case["gt_action_id"]:
                direct_alert_action[dataset] += 1
            if direct_context == case["gt_action_id"]:
                direct_context_action[dataset] += 1

            if _top_action_from_overlap(alert, dataset) == case["gt_action_id"]:
                desc_alert_action[dataset] += 1
            if _top_action_from_overlap(context, dataset) == case["gt_action_id"]:
                desc_context_action[dataset] += 1

            graph = str(view.get("graph_summary", ""))
            graph_rows[dataset] += 1
            if "No structured graph hint available." in graph:
                no_graph_hint[dataset] += 1
            else:
                first = ""
                for line in graph.splitlines():
                    if line.strip().startswith("1."):
                        first = line
                        break
                if "family=" in first:
                    top1 = first.split("family=", 1)[1].split(";", 1)[0].strip()
                    if top1 == case["gt_family_id"]:
                        graph_top1_gt[dataset] += 1

            agent_refs = tuple(ref.get("reference_id") for ref in view.get("agent_references", []))
            rag_refs = tuple(ref.get("reference_id") for ref in view.get("rag_references", []))
            stable_agent_sets.add(agent_refs)
            stable_rag_sets.add(rag_refs)
            if list(agent_refs) != list(rag_refs)[: len(agent_refs)]:
                subset_ok = False

        if len(stable_agent_sets) == 1 and len(stable_rag_sets) == 1:
            stable_reference_sets[dataset] += 1
        if subset_ok:
            agent_subset_rag[dataset] += 1

    out: Dict[str, object] = {}
    for dataset, ds_cases in dataset_rows.items():
        total_rows = len(ds_cases) * len(next(iter(ds_cases))["noise_views"])
        out[dataset] = {
            "cases": len(ds_cases),
            "rows": total_rows,
            "same_alert_equals_context_rows": int(same_alert_context[dataset]),
            "context_line_count_distribution": dict(sorted(Counter(line_counts[dataset]).items())),
            "median_context_lines": _median(line_counts[dataset]),
            "avg_context_lines": _mean(line_counts[dataset]),
            "direct_alert_action_match_rows": int(direct_alert_action[dataset]),
            "direct_context_action_match_rows": int(direct_context_action[dataset]),
            "description_overlap_alert_match_rows": int(desc_alert_action[dataset]),
            "description_overlap_context_match_rows": int(desc_context_action[dataset]),
            "graph_top1_gt_rows": int(graph_top1_gt[dataset]),
            "graph_rows": int(graph_rows[dataset]),
            "no_graph_hint_rows": int(no_graph_hint[dataset]),
            "stable_reference_sets_across_noise_cases": int(stable_reference_sets[dataset]),
            "agent_reference_subset_of_rag_cases": int(agent_subset_rag[dataset]),
        }
    return out


def _mode_summary(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    by_mode: Dict[str, Dict[str, object]] = {}
    mode_dataset_noise: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))

    for mode in sorted({str(row["mode"]) for row in rows}):
        part = [row for row in rows if str(row["mode"]) == mode]
        by_mode[mode] = {
            "rows": len(part),
            "family_accuracy": _mean(float(bool(row["rca_success"])) for row in part),
            "action_accuracy": _mean(float(bool(row["action_success"])) for row in part),
            "action_text_success_rate": _mean(float(bool(row["action_text_success"])) for row in part),
        }
        for dataset in sorted({str(row["dataset"]) for row in part}):
            ds_rows = [row for row in part if str(row["dataset"]) == dataset]
            mode_dataset_noise[mode][dataset]["overall"] = {
                "family_accuracy": _mean(float(bool(row["rca_success"])) for row in ds_rows),
                "action_accuracy": _mean(float(bool(row["action_success"])) for row in ds_rows),
            }
            for noise in sorted({float(row["noise"]) for row in ds_rows}):
                noise_rows = [row for row in ds_rows if float(row["noise"]) == noise]
                mode_dataset_noise[mode][dataset][f"{noise:.1f}"] = {
                    "family_accuracy": _mean(float(bool(row["rca_success"])) for row in noise_rows),
                    "action_accuracy": _mean(float(bool(row["action_success"])) for row in noise_rows),
                }
    return {
        "by_mode": by_mode,
        "by_mode_dataset_noise": mode_dataset_noise,
    }


def _difficulty_summary(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    case_profiles = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        for case_id in sorted({str(row["case_id"]) for row in dataset_rows}):
            case_rows = [row for row in dataset_rows if str(row["case_id"]) == case_id]
            probe_successes = sum(
                int(bool(row["rca_success"]))
                for row in case_rows
                if str(row["mode"]) in {"heuristic_alert", "heuristic_context", "open_alert_only", "open_alert_context"}
            )
            total_probes = sum(
                1
                for row in case_rows
                if str(row["mode"]) in {"heuristic_alert", "heuristic_context", "open_alert_only", "open_alert_context"}
            )
            case_profiles.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "gt_action_id": str(case_rows[0]["gt_action_id"]),
                    "gt_family_id": str(case_rows[0]["gt_family_id"]),
                    "shortcut_probe_family_accuracy": round(probe_successes / max(1, total_probes), 4),
                }
            )
    case_profiles.sort(
        key=lambda row: (
            -float(row["shortcut_probe_family_accuracy"]),
            row["dataset"],
            row["gt_action_id"],
            row["case_id"],
        )
    )
    return {
        "too_easy_cases": [row for row in case_profiles if float(row["shortcut_probe_family_accuracy"]) >= 0.75],
        "harder_cases": [row for row in case_profiles if float(row["shortcut_probe_family_accuracy"]) <= 0.25],
    }


def _health_checks(mode_summary: Mapping[str, object], structural: Mapping[str, object]) -> Dict[str, object]:
    by_mode = dict(mode_summary["by_mode"])
    checks = {
        "current_v2_vanilla_not_near_perfect": float(by_mode.get("vanilla_closed_desc", {}).get("family_accuracy", 1.0)) < 0.95,
        "open_alert_only_not_high": float(by_mode.get("open_alert_only", {}).get("family_accuracy", 1.0)) < 0.6,
        "open_alert_context_not_high": float(by_mode.get("open_alert_context", {}).get("family_accuracy", 1.0)) < 0.75,
        "agent_vs_rag_separated": abs(
            float(by_mode.get("agent_closed_desc", {}).get("family_accuracy", 0.0))
            - float(by_mode.get("rag_closed_desc", {}).get("family_accuracy", 0.0))
        ) >= 0.08,
        "agent_vs_vanilla_separated": abs(
            float(by_mode.get("agent_closed_desc", {}).get("family_accuracy", 0.0))
            - float(by_mode.get("vanilla_closed_desc", {}).get("family_accuracy", 0.0))
        ) >= 0.08,
        "hdfs_direct_alert_not_dominant": int(structural["HDFS"]["direct_alert_action_match_rows"]) <= 9,
        "hadoop_direct_alert_not_dominant": int(structural["Hadoop"]["direct_alert_action_match_rows"]) <= 9,
    }
    checks["healthy_for_paid_api"] = all(checks.values())
    return checks


def main() -> None:
    args = parse_args()
    package = json.loads(args.benchmark_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    structural = _build_structural_report(package)
    mode_summary = _mode_summary(rows)
    difficulty = _difficulty_summary(rows)
    health_checks = _health_checks(mode_summary, structural)

    payload = {
        "benchmark_path": str(args.benchmark_path),
        "progress_path": str(args.progress_path),
        "structural_report": structural,
        "mode_report": mode_summary,
        "difficulty_report": difficulty,
        "health_checks": health_checks,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
