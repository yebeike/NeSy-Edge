from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINAL_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FINAL_ROOT.parents[0]
_PROJECT_ROOT = _FINAL_ROOT.parents[2]

for path in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiments" / "rq123_e2e",
    _AUDIT_SCRIPTS := _REBUILD_ROOT / "rq2_fullcase_audit_20260318" / "scripts",
    _HADOOP_AUDIT_SCRIPTS := _REBUILD_ROOT / "rq2_hadoop_family_audit_20260318" / "scripts",
    _FINAL_ROOT / "scripts",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rq2_fullcase_audit_common_20260318 import build_eval_cases as build_hdfs_eval_cases  # type: ignore
from rq2_fullcase_audit_common_20260318 import calc_rank as calc_hdfs_rank  # type: ignore
from rq2_hadoop_family_audit_common_20260318 import calc_family_rank  # type: ignore
from rq2_mainline_completion_common_20260318 import (  # type: ignore
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


SUMMARY_JSON = RESULTS_DIR / "rq2_paper_metrics_summary_20260318.json"
SUMMARY_MD = REPORTS_DIR / "rq2_paper_metrics_summary_20260318.md"
ASSESSMENT_MD = REPORTS_DIR / "rq2_paper_metrics_assessment_20260318.md"


def _penalized_case_rank(rank: int, sparsity: int) -> float:
    return float(rank if rank >= 0 else sparsity + 1)


def _add_row(rows, dataset, method, evaluator, cases, sparsity_sum, penalized_rank_sum):
    n = len(cases) or 1
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "evaluator": evaluator,
            "sparsity_mean": round(sparsity_sum / n, 4),
            "avg_rank": round(penalized_rank_sum / n, 4),
        }
    )


def _summarize(rows):
    lines = [
        "> `Avg_Rank` is the thesis-facing penalized mean over all cases; any miss is assigned rank `E + 1`, where `E` is the edge count of that method graph on the dataset.",
        "",
        "| Dataset | Method | Evaluator | Sparsity_mean | Avg_Rank |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {row['sparsity_mean']} | {row['avg_rank']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    print("[1/3] Loading graph artifacts and fixed mainline benchmarks. Expected: <5s")
    kb_by_name = {name: load_json(path) for name, path in GRAPH_FILES.items()}
    hdfs_cases = [c for c in build_hdfs_eval_cases(SOURCE_HDFS_BENCH_PATH) if c["dataset"] == "HDFS"]
    mainline_rows = [row for row in load_json(UNIFIED_BENCHMARK_MAINLINE_PATH) if str(row.get("benchmark_tier", "") or "") == "mainline"]
    openstack_rows = [row for row in mainline_rows if str(row.get("dataset", "") or "") == "OpenStack"]
    hadoop_cases = [
        {
            "dataset": "Hadoop",
            "case_id": str(row["case_id"]),
            "gt_effect": str(row["effect_target_value"]),
            "gt_root_family": str(row["root_target_value"]),
        }
        for row in mainline_rows
        if str(row.get("dataset", "") or "") == "Hadoop"
    ]

    print("[2/3] Evaluating paper-facing penalized metrics. Expected: <10s")
    rows = []
    for method, kb in kb_by_name.items():
        hdfs_sparsity_sum = 0.0
        hdfs_penalized_sum = 0.0
        for case in hdfs_cases:
            sparsity, rank = calc_hdfs_rank(kb, "HDFS", case["gt_root"], case["gt_effect"], match_mode="task_aligned")
            hdfs_sparsity_sum += float(sparsity)
            hdfs_penalized_sum += _penalized_case_rank(rank, sparsity)
        _add_row(rows, "HDFS", method, "audit_task_aligned_penalized", hdfs_cases, hdfs_sparsity_sum, hdfs_penalized_sum)

        os_edge_sparsity_sum = 0.0
        os_edge_penalized_sum = 0.0
        os_path_sparsity_sum = 0.0
        os_path_penalized_sum = 0.0
        for row in openstack_rows:
            sparsity, rank = calc_edge_rank(kb, row, "task_aligned_edge")
            os_edge_sparsity_sum += float(sparsity)
            os_edge_penalized_sum += _penalized_case_rank(rank, sparsity)
            sparsity2, rank2 = calc_path2_rank(kb, row)
            os_path_sparsity_sum += float(sparsity2)
            os_path_penalized_sum += _penalized_case_rank(rank2, sparsity2)
        _add_row(rows, "OpenStack", method, "redesign_task_aligned_edge_penalized", openstack_rows, os_edge_sparsity_sum, os_edge_penalized_sum)
        _add_row(rows, "OpenStack", method, "redesign_task_aligned_path2_penalized", openstack_rows, os_path_sparsity_sum, os_path_penalized_sum)

        hd_sparsity_sum = 0.0
        hd_penalized_sum = 0.0
        for case in hadoop_cases:
            sparsity, rank = calc_family_rank(kb, case["gt_root_family"], case["gt_effect"], match_mode="task_aligned")
            hd_sparsity_sum += float(sparsity)
            hd_penalized_sum += _penalized_case_rank(rank, sparsity)
        _add_row(rows, "Hadoop", method, "family_audit_core80_task_aligned_penalized", hadoop_cases, hd_sparsity_sum, hd_penalized_sum)

    print("[3/3] Writing paper-facing summary. Expected: <5s")
    write_json(SUMMARY_JSON, rows)
    SUMMARY_MD.write_text(_summarize(rows), encoding="utf-8")

    lines = [
        "# RQ2 Paper-Facing Metrics Assessment (2026-03-18)",
        "",
        "- This summary keeps only the two thesis-facing metrics: `Sparsity_mean` and penalized `Avg_Rank`.",
        "- Penalized `Avg_Rank` is computed over all cases; any miss receives rank `E + 1`, where `E` is the graph edge count for that dataset/method.",
        "",
        "| Dataset | Method | Evaluator | Sparsity_mean | Avg_Rank |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {row['sparsity_mean']} | {row['avg_rank']} |"
        )
    ASSESSMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {SUMMARY_JSON}")
    print(f"[Saved] {SUMMARY_MD}")
    print(f"[Saved] {ASSESSMENT_MD}")


if __name__ == "__main__":
    main()
