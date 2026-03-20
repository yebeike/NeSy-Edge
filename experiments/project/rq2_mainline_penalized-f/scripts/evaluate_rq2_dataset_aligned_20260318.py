from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINAL_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FINAL_ROOT.parents[0]
_PROJECT_ROOT = _FINAL_ROOT.parents[2]

for path in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiments" / "rq123_e2e",
    _REBUILD_ROOT / "rq2_fullcase_audit_20260318" / "scripts",
    _REBUILD_ROOT / "rq2_hadoop_family_audit_20260318" / "scripts",
    _FINAL_ROOT / "scripts",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rq2_fullcase_audit_common_20260318 import build_eval_cases as build_hdfs_eval_cases  # type: ignore
from rq2_fullcase_audit_common_20260318 import calc_rank as calc_hdfs_rank  # type: ignore
from rq2_hadoop_family_audit_common_20260318 import build_eval_cases as build_hadoop_eval_cases  # type: ignore
from rq2_hadoop_family_audit_common_20260318 import calc_family_rank  # type: ignore
from rq2_mainline_completion_common_20260318 import (
    GRAPH_FILES,
    REPORTS_DIR,
    RESULTS_DIR,
    SOURCE_HDFS_BENCH_PATH,
    UNIFIED_BENCHMARK_MAINLINE_PATH,
    calc_edge_rank,
    calc_path2_rank,
    ensure_dirs,
    load_json,
    write_json,
)


SUMMARY_JSON = RESULTS_DIR / "rq2_dataset_aligned_summary_20260318.json"
SUMMARY_MD = REPORTS_DIR / "rq2_dataset_aligned_summary_20260318.md"
ASSESSMENT_MD = REPORTS_DIR / "rq2_dataset_aligned_final_assessment_20260318.md"


def _summarize(rows):
    lines = [
        "> `Avg_Rank` is conditional on `Rankable`; `Sparsity_mean` is the mean edge count and is not interpretable without coverage.",
        "",
        "| Dataset | Method | Evaluator | Cases | Rankable | Sparsity_mean | Avg_Rank |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        avg = "nan" if row["avg_rank"] is None else str(row["avg_rank"])
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {row['cases']} | {row['rankable']} | {row['sparsity_mean']} | {avg} |"
        )
    return "\n".join(lines) + "\n"


def _add_row(rows, dataset, method, evaluator, cases, rankable, sparsity_sum, rank_sum):
    n = len(cases) or 1
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "evaluator": evaluator,
            "cases": len(cases),
            "rankable": rankable,
            "sparsity_mean": round(sparsity_sum / n, 4),
            "avg_rank": None if rankable == 0 else round(rank_sum / rankable, 4),
        }
    )


def main() -> None:
    ensure_dirs()
    print("[1/3] Loading graph artifacts and dataset-aligned benchmarks. Expected: <5s")
    kb_by_name = {name: load_json(path) for name, path in GRAPH_FILES.items()}
    hdfs_cases = [c for c in build_hdfs_eval_cases(SOURCE_HDFS_BENCH_PATH) if c["dataset"] == "HDFS"]
    mainline_rows = load_json(UNIFIED_BENCHMARK_MAINLINE_PATH)
    openstack_rows = [
        row
        for row in mainline_rows
        if str(row.get("dataset", "") or "") == "OpenStack"
        and str(row.get("benchmark_tier", "") or "") == "mainline"
    ]
    hadoop_cases = [
        {
            "dataset": "Hadoop",
            "case_id": str(row["case_id"]),
            "gt_effect": str(row["effect_target_value"]),
            "gt_root_family": str(row["root_target_value"]),
        }
        for row in mainline_rows
        if str(row.get("dataset", "") or "") == "Hadoop"
        and str(row.get("benchmark_tier", "") or "") == "mainline"
    ]

    print("[2/3] Evaluating dataset-aligned semantics. Expected: <10s")
    rows = []
    for method, kb in kb_by_name.items():
        hdfs_rankable = 0
        hdfs_sparsity_sum = 0.0
        hdfs_rank_sum = 0.0
        for case in hdfs_cases:
            sparsity, rank = calc_hdfs_rank(kb, "HDFS", case["gt_root"], case["gt_effect"], match_mode="task_aligned")
            hdfs_sparsity_sum += float(sparsity)
            if rank >= 0:
                hdfs_rankable += 1
                hdfs_rank_sum += float(rank)
        _add_row(rows, "HDFS", method, "audit_task_aligned", hdfs_cases, hdfs_rankable, hdfs_sparsity_sum, hdfs_rank_sum)

        os_edge_rankable = 0
        os_edge_sparsity_sum = 0.0
        os_edge_rank_sum = 0.0
        os_path_rankable = 0
        os_path_sparsity_sum = 0.0
        os_path_rank_sum = 0.0
        for row in openstack_rows:
            sparsity, rank = calc_edge_rank(kb, row, "task_aligned_edge")
            os_edge_sparsity_sum += float(sparsity)
            if rank >= 0:
                os_edge_rankable += 1
                os_edge_rank_sum += float(rank)
            sparsity2, rank2 = calc_path2_rank(kb, row)
            os_path_sparsity_sum += float(sparsity2)
            if rank2 >= 0:
                os_path_rankable += 1
                os_path_rank_sum += float(rank2)
        _add_row(rows, "OpenStack", method, "redesign_task_aligned_edge", openstack_rows, os_edge_rankable, os_edge_sparsity_sum, os_edge_rank_sum)
        _add_row(rows, "OpenStack", method, "redesign_task_aligned_path2", openstack_rows, os_path_rankable, os_path_sparsity_sum, os_path_rank_sum)

        hd_rankable = 0
        hd_sparsity_sum = 0.0
        hd_rank_sum = 0.0
        for case in hadoop_cases:
            sparsity, rank = calc_family_rank(kb, case["gt_root_family"], case["gt_effect"], match_mode="task_aligned")
            hd_sparsity_sum += float(sparsity)
            if rank >= 0:
                hd_rankable += 1
                hd_rank_sum += float(rank)
        _add_row(rows, "Hadoop", method, "family_audit_core80_task_aligned", hadoop_cases, hd_rankable, hd_sparsity_sum, hd_rank_sum)

    print("[3/3] Writing dataset-aligned summary and assessment. Expected: <5s")
    write_json(SUMMARY_JSON, rows)
    SUMMARY_MD.write_text(_summarize(rows), encoding="utf-8")

    by_key = {(row["dataset"], row["method"], row["evaluator"]): row for row in rows}
    lines = [
        "# RQ2 Dataset-Aligned Final Assessment (2026-03-18)",
        "",
        "- HDFS is evaluated with the audited HDFS task-aligned evaluator on the audited HDFS benchmark.",
        "- OpenStack is evaluated on the redesigned 97-case core benchmark with edge and path2 summaries.",
        "- Hadoop is evaluated with the family-audit evaluator on the quality-first core80 benchmark (unique local root within radius 80).",
        "- `Avg_Rank` is only meaningful conditional on `Rankable`; `Sparsity_mean` is descriptive and must be read alongside coverage.",
        "",
        "| Dataset | Method | Evaluator | Cases | Rankable | Sparsity_mean | Avg_Rank |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        avg = "nan" if row["avg_rank"] is None else str(row["avg_rank"])
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {row['cases']} | {row['rankable']} | {row['sparsity_mean']} | {avg} |"
        )
    ASSESSMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {SUMMARY_JSON}")
    print(f"[Saved] {SUMMARY_MD}")
    print(f"[Saved] {ASSESSMENT_MD}")


if __name__ == "__main__":
    main()
