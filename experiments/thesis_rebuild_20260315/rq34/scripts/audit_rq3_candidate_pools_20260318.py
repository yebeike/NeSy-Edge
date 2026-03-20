from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts.hierarchical_catalog_20260316 import (
    select_gt_action_and_family,
)
from experiments.thesis_rebuild_20260315.rq34.scripts.build_rq34_enriched_seed_pool_20260316 import (
    OUTPUT_PATH as ENRICHED_SEED_POOL_PATH,
    write_enriched_seed_pool,
)
from experiments.thesis_rebuild_20260315.rq34.scripts.run_rq34_hierarchical_resumable_20260316 import (
    BENCH_V2_PATH,
    LEGACY_STAGE4,
    REBUILD_RESULTS_DIR,
    RQ3_TEST_SET_PATH,
    _action_bucket,
    _difficulty_score,
    _has_enough_context,
    _is_weak_mainline_alert,
    _line_action_support,
    _load_fixed_small_case_ids,
    _local_alert_context,
    _refine_selected_alert_for_action,
    _select_actionaware_alert,
    _selection_score,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("stage4_actionaware_legacy", LEGACY_STAGE4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy stage4 script from {LEGACY_STAGE4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_rows(pool: Iterable[Mapping[str, object]], pool_source: str) -> Iterable[tuple[str, Mapping[str, object]]]:
    for row in pool:
        yield pool_source, row


def _direct_alert_flags(dataset: str, selected_alert: str) -> Dict[str, bool]:
    lower = str(selected_alert or "").lower()
    if dataset == "HDFS":
        return {
            "direct_serving_exception": "got exception while serving" in lower,
            "direct_delete_failure": "blockinfo not found" in lower or "unexpected error trying to delete block" in lower,
            "direct_allocate_block": "allocateblock" in lower or "allocate block" in lower,
            "pipeline_keyword": "packetresponder" in lower or "received block" in lower or "receiving block" in lower,
        }
    if dataset == "OpenStack":
        return {
            "direct_image_cache": any(
                pat in lower
                for pat in (
                    "unknown base file",
                    "creating image",
                    "removable base files",
                    "active base files",
                    ": checking",
                )
            ),
            "direct_power_sync": any(
                pat in lower
                for pat in ("sync_power_state", "pending task (spawning)", "while synchronizing instance power states")
            ),
            "direct_host_claim": any(
                pat in lower for pat in ("attempting claim:", "cpu affinity", "vcpu count", "claim successful")
            ),
            "direct_metadata": any(
                pat in lower for pat in ("get /openstack/2013-10-17", "get /latest/meta-data/", "meta_data.json", "vendor_data.json")
            ),
        }
    return {
        "direct_retry_connect": "retrying connect to server" in lower or "retrying rpc toward node" in lower,
        "direct_forced_close": "forcibly closed by the remote host" in lower or "peer terminated the socket unexpectedly" in lower,
        "direct_delete_hdfs": "could not delete hdfs" in lower or "failed to remove hdfs" in lower,
        "direct_storage": any(pat in lower for pat in ("disk full", "no space", "shuffling to disk", "maxsingleshufflelimit", "ondiskmapoutput")),
    }


def _audit_dataset(
    legacy,
    dataset: str,
    cases_per_dataset: int,
    benchmark_pool: List[Mapping[str, object]],
    rq3_test_pool: List[Mapping[str, object]],
    enriched_seed_pool: List[Mapping[str, object]],
) -> List[Dict[str, object]]:
    fixed_ids = set(_load_fixed_small_case_ids(cases_per_dataset).get(dataset, []))
    out: List[Dict[str, object]] = []
    pooled_rows = (
        list(_pool_rows(benchmark_pool, "benchmark_v2"))
        + list(_pool_rows(rq3_test_pool, "rq3_test_set"))
        + list(_pool_rows(enriched_seed_pool, "rq3_test_set_enriched"))
    )
    for pool_source, case in pooled_rows:
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
        if not gt_action_id or not gt_label:
            continue
        case_id = str(case.get("case_id", ""))
        item = {
            "dataset": dataset,
            "case_id": case_id,
            "pool_source": pool_source,
            "fixed_small_member": case_id in fixed_ids,
            "gt_label": gt_label,
            "gt_action_id": gt_action_id,
            "gt_confidence": str(gt_diag.get("confidence", "")),
            "gt_margin": float(gt_diag.get("margin", 0.0) or 0.0),
            "action_bucket": _action_bucket(
                dataset,
                gt_action_id,
                selected_alert,
                str(case.get("raw_log_seed", "") or ""),
            ),
            "selection_score": round(
                float(_selection_score(dataset, selected_alert, raw_log, gt_action_id, gt_diag)),
                3,
            ),
            "difficulty_score": round(
                float(
                    _difficulty_score(
                        dataset,
                        gt_action_id,
                        selected_alert,
                        str(case.get("raw_log_seed", "") or ""),
                        raw_log,
                        gt_diag,
                    )
                ),
                3,
            ),
            "weak_mainline_alert": bool(_is_weak_mainline_alert(dataset, selected_alert)),
            "selected_alert": selected_alert,
            "selected_alert_flags": _direct_alert_flags(dataset, selected_alert),
            "raw_log_seed": str(case.get("raw_log_seed", "") or ""),
            "ground_truth_root_cause_template": str(case.get("ground_truth_root_cause_template", "") or ""),
            "ground_truth_template": str(case.get("ground_truth_template", "") or ""),
            "reason": str(case.get("reason", "") or case.get("gt_action_label", "") or ""),
        }
        out.append(item)
    out.sort(
        key=lambda row: (
            row["dataset"],
            row["gt_action_id"],
            row["action_bucket"],
            -float(row["difficulty_score"]),
            -float(row["selection_score"]),
            row["case_id"],
        )
    )
    return out


def _summary(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    by_dataset: Dict[str, Dict[str, object]] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ds_rows = [row for row in rows if str(row["dataset"]) == dataset]
        action_counts = Counter(str(row["gt_action_id"]) for row in ds_rows)
        bucket_counts = Counter(str(row["action_bucket"]) for row in ds_rows)
        source_counts = Counter(str(row["pool_source"]) for row in ds_rows)
        fixed_counts = Counter(bool(row["fixed_small_member"]) for row in ds_rows)
        by_dataset[dataset] = {
            "cases": len(ds_rows),
            "action_counts": dict(sorted(action_counts.items())),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "fixed_small_counts": {
                "true": int(fixed_counts.get(True, 0)),
                "false": int(fixed_counts.get(False, 0)),
            },
        }
    return by_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="HDFS,OpenStack,Hadoop")
    ap.add_argument("--cases-per-dataset", type=int, default=9)
    ap.add_argument(
        "--output-prefix",
        type=str,
        default="rq3_candidate_audit_20260318",
    )
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    benchmark_pool = json.loads(BENCH_V2_PATH.read_text(encoding="utf-8"))
    rq3_test_pool = json.loads(RQ3_TEST_SET_PATH.read_text(encoding="utf-8"))
    if not ENRICHED_SEED_POOL_PATH.exists():
        write_enriched_seed_pool()
    enriched_seed_pool = json.loads(ENRICHED_SEED_POOL_PATH.read_text(encoding="utf-8"))
    legacy = _load_legacy_module()

    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        rows.extend(
            _audit_dataset(
                legacy,
                dataset=dataset,
                cases_per_dataset=args.cases_per_dataset,
                benchmark_pool=benchmark_pool,
                rq3_test_pool=rq3_test_pool,
                enriched_seed_pool=enriched_seed_pool,
            )
        )

    summary = _summary(rows)
    REBUILD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REBUILD_RESULTS_DIR / f"{args.output_prefix}_rows.json"
    out_summary = REBUILD_RESULTS_DIR / f"{args.output_prefix}_summary.json"
    write_json(out_json, rows)
    write_json(out_summary, summary)
    print(f"[INFO] Wrote {len(rows)} candidate rows to {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
