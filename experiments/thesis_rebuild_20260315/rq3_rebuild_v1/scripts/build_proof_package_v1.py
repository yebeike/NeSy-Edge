from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils


DEFAULT_SPEC_PATH = REBUILD_ROOT / "specs" / "rq3_triad_proof_slice_v1_20260318.json"
DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "analysis" / "rq3_triad_proof_slice_v1_20260318"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--noise-levels", type=str, default="")
    return ap.parse_args()


def build_package(spec: Dict[str, object], *, noise_levels: List[float] | None = None) -> Dict[str, object]:
    noise_levels = list(noise_levels) if noise_levels is not None else [float(x) for x in utils.contract()["noise_levels"]]
    cases: List[Dict[str, object]] = []
    family_counts: Dict[str, Counter] = {}
    action_counts: Dict[str, Counter] = {}
    quality_counts: Dict[str, Counter] = {}

    for dataset, items in spec["datasets"].items():
        family_counts[dataset] = Counter()
        action_counts[dataset] = Counter()
        quality_counts[dataset] = Counter()
        for raw_item in items:
            item = dict(raw_item)
            item["dataset"] = dataset
            source = str(item["source"])
            pool_case_id = str(item["case_id"])
            eval_case_id = str(item.get("eval_case_id", "") or pool_case_id)
            case_row = utils.pool_rows()[(source, pool_case_id)]
            selected_alert = utils.find_selected_alert(item, case_row)
            context_text = utils.build_context_text(item, case_row, selected_alert)
            query_text = "\n".join(part for part in (selected_alert, context_text) if part).strip()
            references = utils.select_references(
                dataset,
                case_id=pool_case_id,
                query_text=query_text,
                top_k=int(utils.contract()["method_budgets"]["reference_top_k"]),
                gt_action_id=str(item["gt_action_id"]),
                gt_family_id=str(item["gt_family_id"]),
            )
            graph_query_text = query_text
            graph_evidence = utils.select_graph_evidence(
                dataset,
                query_text=graph_query_text,
                top_k=int(utils.contract()["method_budgets"]["graph_top_k"]),
                reference_texts=[utils.reference_primary_signal_text(ref) for ref in references],
            )
            noise_views = utils.build_noise_views(
                dataset,
                gt_action_id=str(item["gt_action_id"]),
                selected_alert=selected_alert,
                context_text=context_text,
                noise_levels=noise_levels,
                references=references,
                graph_evidence=graph_evidence,
            )
            family_counts[dataset][str(item["gt_family_id"])] += 1
            action_counts[dataset][str(item["gt_action_id"])] += 1
            quality_counts[dataset][str(item.get("quality_tier", "unlabeled"))] += 1
            cases.append(
                {
                    "dataset": dataset,
                    "case_id": eval_case_id,
                    "pool_case_id": pool_case_id,
                    "source": source,
                    "gt_family_id": str(item["gt_family_id"]),
                    "gt_action_id": str(item["gt_action_id"]),
                    "eligibility_note": str(item.get("eligibility_note", "")),
                    "quality_tier": str(item.get("quality_tier", "")),
                    "selection_bucket": str(item.get("selection_bucket", "")),
                    "toxicity_flags": list(item.get("toxicity_flags", []) or []),
                    "base_incident_id": str(item.get("base_incident_id", "") or pool_case_id),
                    "reanchor_group_id": str(item.get("reanchor_group_id", "") or ""),
                    "is_duplicate_reanchor": bool(item.get("is_duplicate_reanchor", False)),
                    "candidate_origin": str(item.get("candidate_origin", "") or ""),
                    "scored_by_probe": bool(item.get("scored_by_probe", False)),
                    "inclusion_reason": str(item.get("inclusion_reason", "") or ""),
                    "selected_alert_clean": selected_alert,
                    "shared_context_clean": context_text,
                    "provenance": {
                        "eval_case_id": eval_case_id,
                        "pool_case_id": pool_case_id,
                        "ground_truth_root_cause_template": str(case_row.get("ground_truth_root_cause_template", "") or ""),
                        "ground_truth_template": str(case_row.get("ground_truth_template", "") or ""),
                        "reason": str(case_row.get("reason", "") or case_row.get("gt_action_label", "") or ""),
                        "quality_tier": str(item.get("quality_tier", "")),
                        "selection_bucket": str(item.get("selection_bucket", "")),
                        "toxicity_flags": list(item.get("toxicity_flags", []) or []),
                        "base_incident_id": str(item.get("base_incident_id", "") or pool_case_id),
                        "reanchor_group_id": str(item.get("reanchor_group_id", "") or ""),
                        "is_duplicate_reanchor": bool(item.get("is_duplicate_reanchor", False)),
                        "candidate_origin": str(item.get("candidate_origin", "") or ""),
                        "scored_by_probe": bool(item.get("scored_by_probe", False)),
                        "inclusion_reason": str(item.get("inclusion_reason", "") or ""),
                    },
                    "noise_views": noise_views,
                }
            )

    package = {
        "benchmark_id": str(spec["benchmark_id"]),
        "benchmark_kind": str(spec["benchmark_kind"]),
        "purpose": str(spec["purpose"]),
        "contract_path": str(spec["contract_path"]),
        "formal_small_ready": bool(spec.get("formal_small_ready", False)),
        "paid_api_allowed": bool(spec.get("paid_api_allowed", False)),
        "noise_levels": noise_levels,
        "cases": cases,
    }
    summary = {
        "benchmark_id": str(spec["benchmark_id"]),
        "benchmark_kind": str(spec["benchmark_kind"]),
        "dataset_case_counts": {dataset: len(items) for dataset, items in spec["datasets"].items()},
        "family_counts": {dataset: dict(counter) for dataset, counter in family_counts.items()},
        "action_counts": {dataset: dict(counter) for dataset, counter in action_counts.items()},
        "quality_tier_counts": {dataset: dict(counter) for dataset, counter in quality_counts.items()},
        "case_count": len(cases),
        "noise_levels": noise_levels,
    }
    return {"package": package, "summary": summary}


def main() -> None:
    args = parse_args()
    spec = utils.load_json(args.spec)
    noise_levels = (
        [float(token.strip()) for token in str(args.noise_levels).split(",") if token.strip()]
        if str(args.noise_levels).strip()
        else None
    )
    payload = build_package(spec, noise_levels=noise_levels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package_path = args.output_dir / f"{spec['benchmark_id']}_package.json"
    summary_path = args.output_dir / f"{spec['benchmark_id']}_summary.json"
    payload["summary"]["package_path"] = str(package_path)
    write_json(package_path, payload["package"])
    write_json(summary_path, payload["summary"])
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
