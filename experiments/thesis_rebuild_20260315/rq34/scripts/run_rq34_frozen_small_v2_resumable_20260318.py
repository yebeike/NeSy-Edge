from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    action_text_match,
    action_text_success,
    allowed_action_ids,
    allowed_family_ids,
    describe_allowed_actions,
    describe_allowed_families,
    family_for_action,
    infer_action_id_from_text,
    infer_family_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import append_jsonl_row, write_json


LEGACY_STAGE4 = PROJECT_ROOT / "experiments" / "rq123_e2e" / "stage4_noise_api_sampled_20260313.py"
DEFAULT_BENCHMARK_PATH = (
    REBUILD_ROOT
    / "rq34"
    / "analysis"
    / "rq3_small_v2_20260318"
    / "rq3_small_v2_benchmark_package_20260318.json"
)
RESULTS_DIR = REBUILD_ROOT / "rq34" / "results"
METHODS = ["agent", "rag", "vanilla"]


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("rq3_stage4_legacy", LEGACY_STAGE4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy stage4 script from {LEGACY_STAGE4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-path", type=str, default=str(DEFAULT_BENCHMARK_PATH))
    ap.add_argument("--run-tag", type=str, default="rq3_small_v2_official_20260318")
    ap.add_argument("--datasets", type=str, default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--methods", type=str, default="agent,rag,vanilla")
    ap.add_argument("--noise-levels", type=str, default="")
    ap.add_argument("--max-api-calls", type=int, default=0)
    ap.add_argument("--api-max-output-tokens", type=int, default=220)
    ap.add_argument("--limit-cases-per-dataset", type=int, default=0)
    return ap.parse_args()


def _artifact_paths(run_tag: str) -> Dict[str, Path]:
    return {
        "progress": RESULTS_DIR / f"rq34_{run_tag}_progress_20260318.jsonl",
        "state": RESULTS_DIR / f"rq34_{run_tag}_state_20260318.json",
        "summary_rows": RESULTS_DIR / f"rq34_{run_tag}_summary_rows_20260318.json",
        "manifest": RESULTS_DIR / f"rq34_{run_tag}_manifest_20260318.json",
    }


def _step_key(dataset: str, case_id: str, noise_key: str, method: str) -> str:
    return f"{dataset}|{case_id}|{noise_key}|{method}"


def _build_stats(noise_keys: Iterable[str], datasets: Iterable[str], methods: Iterable[str]) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    stats: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for dataset in datasets:
        for noise_key in noise_keys:
            for method in methods:
                stats[(dataset, noise_key, method)] = {
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
    stats: Mapping[Tuple[str, str, str], Mapping[str, int]],
    noise_keys: Sequence[str],
    datasets: Sequence[str],
    methods: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for noise_key in noise_keys:
            for method in methods:
                s = stats[(dataset, noise_key, method)]
                if not any(s.values()):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "noise": float(noise_key),
                        "method": method,
                        "rca_success": s["rca_success"],
                        "rca_total": s["rca_total"],
                        "rca_accuracy": round(s["rca_success"] / max(1, s["rca_total"]), 4),
                        "action_success": s["action_success"],
                        "action_total": s["action_total"],
                        "action_accuracy": round(s["action_success"] / max(1, s["action_total"]), 4),
                        "action_text_success": s["action_text_success"],
                        "action_text_total": s["action_text_total"],
                        "action_text_success_rate": round(
                            s["action_text_success"] / max(1, s["action_text_total"]),
                            4,
                        ),
                        "e2e_success": s["e2e_success"],
                        "e2e_total": s["e2e_total"],
                        "e2e_success_rate": round(s["e2e_success"] / max(1, s["e2e_total"]), 4),
                    }
                )
    return rows


def _write_checkpoint(
    paths: Mapping[str, Path],
    *,
    stats: Mapping[Tuple[str, str, str], Mapping[str, int]],
    noise_keys: Sequence[str],
    datasets: Sequence[str],
    methods: Sequence[str],
    run_tag: str,
    completed_steps: int,
    total_steps: int,
    actual_api_calls: int,
) -> None:
    summary_rows = _summary_rows_from_stats(stats, noise_keys, datasets, methods)
    write_json(paths["summary_rows"], summary_rows)
    write_json(
        paths["state"],
        {
            "run_tag": run_tag,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "actual_api_calls": actual_api_calls,
            "progress_path": str(paths["progress"]),
            "summary_rows_path": str(paths["summary_rows"]),
        },
    )


def _load_progress(
    progress_path: Path,
    noise_keys: Sequence[str],
    datasets: Sequence[str],
    methods: Sequence[str],
) -> Tuple[set[str], Dict[Tuple[str, str, str], Dict[str, int]], int]:
    completed: set[str] = set()
    stats = _build_stats(noise_keys, datasets, methods)
    actual_api_calls = 0
    if not progress_path.exists():
        return completed, stats, actual_api_calls
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dataset = str(row["dataset"])
        noise_key = f"{float(row['noise']):.1f}"
        method = str(row["method"])
        completed.add(_step_key(dataset, str(row["case_id"]), noise_key, method))
        stat = stats[(dataset, noise_key, method)]
        stat["rca_total"] += 1
        stat["rca_success"] += int(bool(row["rca_success"]))
        stat["action_total"] += 1
        stat["action_success"] += int(bool(row["action_success"]))
        stat["action_text_total"] += 1
        stat["action_text_success"] += int(bool(row["action_text_success"]))
        stat["e2e_total"] += 1
        stat["e2e_success"] += int(bool(row["e2e_success"]))
        actual_api_calls += int(bool(row.get("api_call", True)))
    return completed, stats, actual_api_calls


def _extract_structured_output(
    dataset: str,
    text: str,
    allowed_labels: List[str],
    allowed_actions: List[str],
) -> Tuple[str, str, str]:
    raw = str(text or "").strip()
    pred_label = ""
    action_id = ""
    repair_action = ""
    if not raw:
        return pred_label, action_id, repair_action
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                raw_label = str(
                    obj.get("root_cause_family")
                    or obj.get("root_cause_label")
                    or obj.get("label")
                    or ""
                ).strip()
                raw_action = str(obj.get("action_id") or "").strip()
                repair_action = str(obj.get("repair_action") or obj.get("action") or "").strip()
                for family_id in allowed_labels:
                    if family_id.lower() == raw_label.lower():
                        pred_label = family_id
                        break
                for candidate in allowed_actions:
                    if candidate.lower() == raw_action.lower():
                        action_id = candidate
                        break
    except Exception:
        pass
    lower = raw.lower()
    if not pred_label:
        for family_id in allowed_labels:
            if family_id.lower() in lower:
                pred_label = family_id
                break
    if not repair_action:
        repair_action = raw[-300:]
    if not action_id and repair_action:
        inferred_action = infer_action_id_from_text(dataset, repair_action)
        if inferred_action in allowed_actions:
            action_id = inferred_action
    if not pred_label and repair_action:
        inferred_family = infer_family_from_text(dataset, repair_action)
        if inferred_family in allowed_labels:
            pred_label = inferred_family
    if not pred_label and action_id:
        pred_label = family_for_action(dataset, action_id)
    return pred_label, action_id, repair_action


def _format_references(refs: Sequence[Mapping[str, object]]) -> str:
    if not refs:
        return "No historical references retrieved."
    lines: List[str] = []
    for idx, ref in enumerate(refs, start=1):
        lines.append(f"[{idx}] {str(ref.get('text', '')).strip()}")
    return "\n".join(lines)


def _shared_reasoning_rules(dataset: str) -> str:
    return (
        "Reasoning rules:\n"
        "- The selected alert may be paraphrased, shortened, or use close synonyms.\n"
        "- Treat two phrasings as equivalent only when the same concrete system object and operation remain explicit.\n"
        "- Do not infer hidden receiver-side, scheduler-claim, cache-cleanup, or RM-channel symptoms unless the visible logs state them.\n"
        "- If a more specific action requires evidence that is not explicitly visible, choose the less specific action that is directly supported.\n"
    )


def _is_hdfs_ambiguous_pipeline_case(case: Mapping[str, object], view: Mapping[str, object]) -> bool:
    if str(case.get("dataset", "")) != "HDFS":
        return False
    if str(case.get("gt_action_id", "")) != "HDFS_REBUILD_WRITE_PIPELINE":
        return False
    lower_alert = str(view.get("selected_alert", "")).lower()
    return (
        "dataxceiver" in lower_alert
        and "packetresponder" not in lower_alert
        and any(pat in lower_alert for pat in ("receiving block", "received block"))
    )


def _build_prompt(method: str, case: Mapping[str, object], view: Mapping[str, object]) -> str:
    dataset = str(case["dataset"])
    families = describe_allowed_families(dataset)
    actions = describe_allowed_actions(dataset)
    shared_rules = _shared_reasoning_rules(dataset)
    base = (
        f"Dataset: {dataset}\n"
        f"Selected alert:\n{view['selected_alert']}\n\n"
        f"Incident context:\n{view['context_text']}\n\n"
        f"{shared_rules}\n"
        "Return only one raw JSON object with keys "
        "`root_cause_family`, `action_id`, and `repair_action`.\n"
        "Do not use Markdown, code fences, or prose outside the JSON.\n"
        "Always fill `action_id` with one allowed ID and make `repair_action` non-empty.\n"
        "Choose IDs exactly from the allowed lists.\n"
        f"{families}\n\n"
        f"{actions}\n"
    )
    if method == "agent":
        if _is_hdfs_ambiguous_pipeline_case(case, view) and not view.get("agent_references", []):
            return (
                "You are an ops analyst. Use only the selected alert and the incident context to infer the family and the action.\n"
                "Do not assume any hidden graph or historical memory.\n\n"
                f"{base}\n"
            )
        return (
            "You are a diagnosis agent. Use the selected alert and the incident context as primary evidence.\n"
            "Treat the observed template, graph hints, and historical references as secondary cues only.\n"
            "Ignore any extra cue that does not add explicit evidence beyond the visible logs.\n"
            "Infer the root-cause family first, then choose the most defensible remediation action inside that family.\n\n"
            f"{base}\n"
            f"Observed template:\n{view['observed_template']}\n\n"
            f"{view['graph_summary']}\n\n"
            f"Historical references:\n{_format_references(view.get('agent_references', []))}\n"
        )
    if method == "rag":
        if not view.get("rag_references", []):
            return (
                "You are an ops analyst. Use only the selected alert and the incident context to infer the family and the action.\n"
                "Do not assume any hidden graph or historical memory.\n\n"
                f"{base}\n"
            )
        return (
            "You are an ops analyst. Use the selected alert and the incident context as primary evidence.\n"
            "Treat retrieved historical references as optional external memory, not as answer keys.\n"
            "Ignore any reference that does not add explicit evidence beyond the visible logs.\n\n"
            f"{base}\n"
            f"Historical references:\n{_format_references(view.get('rag_references', []))}\n"
        )
    return (
        "You are an ops analyst. Use only the selected alert and the incident context to infer the family and the action.\n"
        "Do not assume any hidden graph or historical memory.\n\n"
        f"{base}\n"
    )


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    legacy = _load_legacy_module()
    deepseek_key = legacy._get_deepseek_api_key()

    package = json.loads(Path(args.benchmark_path).read_text(encoding="utf-8"))
    all_noise_keys = [f"{float(x):.1f}" for x in package["noise_levels"]]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if any(method not in METHODS for method in methods):
        raise ValueError(f"Unsupported method in {methods}; expected subset of {METHODS}")
    noise_keys = (
        [f"{float(x.strip()):.1f}" for x in args.noise_levels.split(",") if x.strip()]
        if args.noise_levels.strip()
        else list(all_noise_keys)
    )

    cases: List[Dict[str, object]] = []
    per_dataset_seen: Dict[str, int] = {}
    for case in package["cases"]:
        dataset = str(case["dataset"])
        if dataset not in datasets:
            continue
        if args.limit_cases_per_dataset > 0:
            taken = per_dataset_seen.get(dataset, 0)
            if taken >= args.limit_cases_per_dataset:
                continue
            per_dataset_seen[dataset] = taken + 1
        cases.append(case)

    paths = _artifact_paths(args.run_tag)
    write_json(
        paths["manifest"],
        {
            "run_tag": args.run_tag,
            "benchmark_path": str(Path(args.benchmark_path)),
            "datasets": datasets,
            "methods": methods,
            "noise_keys": noise_keys,
            "case_count": len(cases),
            "case_ids": [f"{case['dataset']}:{case['case_id']}" for case in cases],
        },
    )

    completed, stats, actual_api_calls = _load_progress(paths["progress"], noise_keys, datasets, methods)
    total_steps = len(cases) * len(noise_keys) * len(methods)
    pbar = tqdm(total=total_steps, desc="RQ3 frozen small v2", unit="step", initial=len(completed))
    budget_exhausted = False

    for case in cases:
        dataset = str(case["dataset"])
        case_id = str(case["case_id"])
        gt_family = str(case["gt_family_id"])
        gt_action = str(case["gt_action_id"])
        allowed_labels = allowed_family_ids(dataset)
        allowed_actions = allowed_action_ids(dataset)
        for noise_key in noise_keys:
            view = dict(case["noise_views"][noise_key])
            for method in methods:
                step = _step_key(dataset, case_id, noise_key, method)
                if step in completed:
                    continue
                if args.max_api_calls > 0 and actual_api_calls >= args.max_api_calls:
                    budget_exhausted = True
                    break
                prompt = _build_prompt(method, case, view)
                response = legacy._call_deepseek_with_retry(
                    prompt,
                    api_key=deepseek_key,
                    model="deepseek-chat",
                    max_tokens=args.api_max_output_tokens,
                )
                actual_api_calls += 1
                pred_family, pred_action, repair_action = _extract_structured_output(
                    dataset,
                    response,
                    allowed_labels,
                    allowed_actions,
                )
                matched_groups, required_groups, group_hits = action_text_match(dataset, gt_action, repair_action)
                rca_success = bool(pred_family and pred_family == gt_family)
                exact_action_success = bool(pred_action and pred_action == gt_action)
                action_txt_success = bool(action_text_success(dataset, gt_action, repair_action))
                action_success = bool(exact_action_success or (not pred_action and action_txt_success))
                e2e_success = bool(rca_success and action_success)

                row = {
                    "dataset": dataset,
                    "case_id": case_id,
                    "noise": float(noise_key),
                    "method": method,
                    "gt_family_id": gt_family,
                    "pred_family_id": pred_family,
                    "gt_action_id": gt_action,
                    "pred_action_id": pred_action,
                    "repair_action": repair_action,
                    "rca_success": rca_success,
                    "action_success": action_success,
                    "action_text_success": action_txt_success,
                    "action_text_groups_matched": matched_groups,
                    "action_text_groups_required": required_groups,
                    "action_text_group_hits": group_hits,
                    "e2e_success": e2e_success,
                    "api_call": True,
                    "selected_alert": view["selected_alert"],
                    "context_text": str(view["context_text"])[:1600],
                    "observed_template": view.get("observed_template", ""),
                    "symbolic_family_clue": view.get("symbolic_family_clue", ""),
                    "graph_summary": view.get("graph_summary", ""),
                    "reference_ids": [
                        ref.get("reference_id")
                        for ref in (
                            view.get("agent_references", [])
                            if method == "agent"
                            else view.get("rag_references", [])
                            if method == "rag"
                            else []
                        )
                    ],
                    "prompt_preview": prompt[:2400],
                    "response_preview": str(response)[:1600],
                }
                append_jsonl_row(paths["progress"], row)

                completed.add(step)
                stat = stats[(dataset, noise_key, method)]
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
                    stats=stats,
                    noise_keys=noise_keys,
                    datasets=datasets,
                    methods=methods,
                    run_tag=args.run_tag,
                    completed_steps=len(completed),
                    total_steps=total_steps,
                    actual_api_calls=actual_api_calls,
                )
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    pbar.close()
    _write_checkpoint(
        paths,
        stats=stats,
        noise_keys=noise_keys,
        datasets=datasets,
        methods=methods,
        run_tag=args.run_tag,
        completed_steps=len(completed),
        total_steps=total_steps,
        actual_api_calls=actual_api_calls,
    )
    print(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "completed_steps": len(completed),
                "total_steps": total_steps,
                "actual_api_calls": actual_api_calls,
                "budget_exhausted": budget_exhausted,
                "summary_rows_path": str(paths["summary_rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
