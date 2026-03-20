from __future__ import annotations

import argparse
from typing import Dict, List

from rq2_mainline_completion_common_20260318 import (
    GRAPH_FILES,
    HADOOP_CALIBRATION_PATH,
    PATH_DIAGNOSTIC_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    UNIFIED_BENCHMARK_MAINLINE_PATH,
    build_path_diagnostic_rows,
    ensure_dirs,
    load_json,
    path_diagnostic_markdown,
    summarize_rows,
    write_json,
    evaluate_rows,
)


MODES = ["exact_only_edge", "task_aligned_edge", "task_aligned_path2"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _find(rows: List[Dict[str, object]], dataset: str, method: str) -> Dict[str, object]:
    for row in rows:
        if row["dataset"] == dataset and row["method"] == method:
            return row
    raise KeyError((dataset, method))


def _write_final_assessment(
    benchmark_rows: List[Dict[str, object]],
    result_by_mode: Dict[str, List[Dict[str, object]]],
    path_rows: List[Dict[str, object]],
) -> None:
    report_path = REPORTS_DIR / "rq2_mainline_final_assessment_20260318.md"
    calibration = load_json(HADOOP_CALIBRATION_PATH)
    mainline_counts = {dataset: sum(1 for row in benchmark_rows if row["dataset"] == dataset) for dataset in ["HDFS", "OpenStack", "Hadoop"]}

    hdfs_edge = result_by_mode["task_aligned_edge"]
    os_edge = result_by_mode["task_aligned_edge"]
    os_path = result_by_mode["task_aligned_path2"]
    hadoop_edge = result_by_mode["task_aligned_edge"]

    hdfs_mod = _find(hdfs_edge, "HDFS", "modified")
    hdfs_orig = _find(hdfs_edge, "HDFS", "original_dynotears")
    os_mod_edge = _find(os_edge, "OpenStack", "modified")
    os_orig_edge = _find(os_edge, "OpenStack", "original_dynotears")
    os_mod_path = _find(os_path, "OpenStack", "modified")
    os_orig_path = _find(os_path, "OpenStack", "original_dynotears")
    hd_mod = _find(hadoop_edge, "Hadoop", "modified")
    hd_orig = _find(hadoop_edge, "Hadoop", "original_dynotears")

    os_path_diag = next(row for row in path_rows if row["dataset"] == "OpenStack" and row["method"] == "modified")
    hadoop_path_diag = next(row for row in path_rows if row["dataset"] == "Hadoop" and row["method"] == "modified")

    lines = [
        "# RQ2 Mainline Final Assessment (2026-03-18)",
        "",
        f"- Mainline benchmark rows: HDFS `{mainline_counts['HDFS']}`, OpenStack `{mainline_counts['OpenStack']}`, Hadoop `{mainline_counts['Hadoop']}`.",
        f"- Hadoop is evaluated as a family-level root task with shared feature cap `{calibration['chosen_cap']}` on `{calibration['selected_columns']}` columns.",
        "",
        "## HDFS",
        "",
        f"- modified task_aligned_edge: rankable `{hdfs_mod['rankable']}/{hdfs_mod['cases']}`, sparsity `{hdfs_mod['sparsity_mean']}`, avg_rank `{hdfs_mod['avg_rank']}`",
        f"- original_dynotears task_aligned_edge: rankable `{hdfs_orig['rankable']}/{hdfs_orig['cases']}`, sparsity `{hdfs_orig['sparsity_mean']}`, avg_rank `{hdfs_orig['avg_rank']}`",
        "",
        "## OpenStack",
        "",
        f"- modified task_aligned_edge: rankable `{os_mod_edge['rankable']}/{os_mod_edge['cases']}`, sparsity `{os_mod_edge['sparsity_mean']}`, avg_rank `{os_mod_edge['avg_rank']}`",
        f"- original_dynotears task_aligned_edge: rankable `{os_orig_edge['rankable']}/{os_orig_edge['cases']}`, sparsity `{os_orig_edge['sparsity_mean']}`, avg_rank `{os_orig_edge['avg_rank']}`",
        f"- modified task_aligned_path2: rankable `{os_mod_path['rankable']}/{os_mod_path['cases']}`, avg_rank `{os_mod_path['avg_rank']}`",
        f"- original_dynotears task_aligned_path2: rankable `{os_orig_path['rankable']}/{os_orig_path['cases']}`, avg_rank `{os_orig_path['avg_rank']}`",
        f"- OpenStack modified path diagnostic: direct `{os_path_diag['direct']}`, two_hop `{os_path_diag['two_hop']}`, three_hop `{os_path_diag['three_hop']}`, none `{os_path_diag['none']}`",
        "",
        "## Hadoop",
        "",
        f"- modified task_aligned_edge: rankable `{hd_mod['rankable']}/{hd_mod['cases']}`, sparsity `{hd_mod['sparsity_mean']}`, avg_rank `{hd_mod['avg_rank']}`",
        f"- original_dynotears task_aligned_edge: rankable `{hd_orig['rankable']}/{hd_orig['cases']}`, sparsity `{hd_orig['sparsity_mean']}`, avg_rank `{hd_orig['avg_rank']}`",
        f"- Hadoop modified path diagnostic: direct `{hadoop_path_diag['direct']}`, two_hop `{hadoop_path_diag['two_hop']}`, three_hop `{hadoop_path_diag['three_hop']}`, none `{hadoop_path_diag['none']}`",
        "",
        "## Decision",
        "",
        "- Keep edge-mode and path2-mode as separate reported tables.",
        "- If modified loses on any dataset in these cleaned tables, keep the loss rather than loosening the rules again.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    report_path = REPORTS_DIR / "rq2_mainline_eval_index_20260318.md"
    if (
        PATH_DIAGNOSTIC_PATH.exists()
        and all((RESULTS_DIR / f"rq2_mainline_{mode}_summary_20260318.json").exists() for mode in MODES)
        and report_path.exists()
        and not args.force
    ):
        print(f"[*] Reusing evaluation artifacts under {RESULTS_DIR}")
        print(f"[*] Reusing evaluation index: {report_path}")
        return

    benchmark_rows = [row for row in load_json(UNIFIED_BENCHMARK_MAINLINE_PATH) if str(row["benchmark_tier"]) == "mainline"]

    print("[1/2] Evaluating all methods on all 3 modes. Expected: <30s")
    result_by_mode: Dict[str, List[Dict[str, object]]] = {}
    index_lines = ["# RQ2 Mainline Evaluation Index (2026-03-18)", ""]
    for idx, mode in enumerate(MODES, start=1):
        print(f"[1/2.{idx}] Evaluating mode `{mode}`. Expected: <10s")
        rows = evaluate_rows(GRAPH_FILES, benchmark_rows, mode)
        result_by_mode[mode] = rows
        out_json = RESULTS_DIR / f"rq2_mainline_{mode}_summary_20260318.json"
        out_md = REPORTS_DIR / f"rq2_mainline_{mode}_summary_20260318.md"
        write_json(out_json, rows)
        out_md.write_text(summarize_rows(rows), encoding="utf-8")
        index_lines.append(f"## {mode}")
        index_lines.append("")
        index_lines.append(summarize_rows(rows).rstrip())
        index_lines.append("")
        print(f"[Saved] {out_json}")
        print(f"[Saved] {out_md}")

    print("[2/2] Writing path diagnostic and final assessment. Expected: <10s")
    path_rows = build_path_diagnostic_rows(GRAPH_FILES, benchmark_rows)
    write_json(PATH_DIAGNOSTIC_PATH, path_rows)
    path_md = REPORTS_DIR / "rq2_mainline_path_diagnostic_20260318.md"
    path_md.write_text(path_diagnostic_markdown(path_rows), encoding="utf-8")
    index_lines.append("## path_diagnostic")
    index_lines.append("")
    index_lines.append(path_diagnostic_markdown(path_rows).rstrip())
    index_lines.append("")
    report_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[Saved] {PATH_DIAGNOSTIC_PATH}")
    print(f"[Saved] {path_md}")
    print(f"[Saved] {report_path}")

    _write_final_assessment(benchmark_rows, result_by_mode, path_rows)
    print(f"[Saved] {REPORTS_DIR / 'rq2_mainline_final_assessment_20260318.md'}")


if __name__ == "__main__":
    main()
