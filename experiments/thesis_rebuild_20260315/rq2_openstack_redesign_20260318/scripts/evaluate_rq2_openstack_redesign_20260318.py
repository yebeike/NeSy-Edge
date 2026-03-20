from __future__ import annotations

import argparse
from pathlib import Path

from rq2_openstack_redesign_common_20260318 import (
    COMBINED_EVAL_BENCH_PATH,
    CORE_BENCH_PATH,
    GRAPH_FILES,
    REPORTS_DIR,
    RESULTS_DIR,
    build_path_supplement_rows,
    ensure_dirs,
    evaluate_graph_rows,
    load_json,
    summarize_rows,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact_only", "task_aligned", "both"], default="both")
    return ap.parse_args()


def _path_rows_markdown(rows):
    lines = [
        "| Graph | Cases | Direct | Two_Hop | Three_Hop | None |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['graph']} | {row['cases']} | {row['direct']} | {row['two_hop']} | {row['three_hop']} | {row['none']} |"
        )
    return "\n".join(lines) + "\n"


def _write_final_assessment(exact_rows, task_rows, path_rows) -> None:
    report_path = REPORTS_DIR / "rq2_openstack_redesign_final_assessment_20260318.md"
    core_rows = [row for row in load_json(CORE_BENCH_PATH) if str(row.get("dataset", "")) == "OpenStack"]

    def _find(rows, dataset, graph):
        for row in rows:
            if row["dataset"] == dataset and row["graph"] == graph:
                return row
        raise KeyError((dataset, graph))

    hdfs_mod = _find(task_rows, "HDFS", "modified")
    hdfs_orig = _find(task_rows, "HDFS", "original_dynotears")
    os_mod_exact = _find(exact_rows, "OpenStack", "modified")
    os_orig_exact = _find(exact_rows, "OpenStack", "original_dynotears")
    os_mod_task = _find(task_rows, "OpenStack", "modified")
    os_orig_task = _find(task_rows, "OpenStack", "original_dynotears")
    mod_path = next(row for row in path_rows if row["graph"] == "modified")

    openstack_positive = (
        os_mod_task["avg_rank"] is not None
        and os_orig_task["avg_rank"] is not None
        and float(os_mod_task["sparsity_mean"]) < float(os_orig_task["sparsity_mean"])
        and float(os_mod_task["avg_rank"]) < float(os_orig_task["avg_rank"])
    )

    lines = [
        "# RQ2 OpenStack Redesign Final Assessment (2026-03-18)",
        "",
        f"- HDFS remains frozen from the audit fork and keeps `{hdfs_mod['cases']}` evaluable rows.",
        f"- OpenStack redesign core size: `{len(core_rows)}`",
        f"- OpenStack redesign uses only the core set for thesis-eligible evaluation; the extended set is sensitivity-only.",
        "",
        "## HDFS Frozen Check",
        "",
        f"- modified task_aligned: sparsity `{hdfs_mod['sparsity_mean']}`, avg_rank `{hdfs_mod['avg_rank']}`",
        f"- original_dynotears task_aligned: sparsity `{hdfs_orig['sparsity_mean']}`, avg_rank `{hdfs_orig['avg_rank']}`",
        "- HDFS remains mainline-usable under task_aligned and is intentionally not reworked here.",
        "",
        "## OpenStack Core Check",
        "",
        f"- modified exact_only: rankable `{os_mod_exact['rankable']}/{os_mod_exact['cases']}`, avg_rank `{os_mod_exact['avg_rank']}`",
        f"- original_dynotears exact_only: rankable `{os_orig_exact['rankable']}/{os_orig_exact['cases']}`, avg_rank `{os_orig_exact['avg_rank']}`",
        f"- modified task_aligned: rankable `{os_mod_task['rankable']}/{os_mod_task['cases']}`, sparsity `{os_mod_task['sparsity_mean']}`, avg_rank `{os_mod_task['avg_rank']}`",
        f"- original_dynotears task_aligned: rankable `{os_orig_task['rankable']}/{os_orig_task['cases']}`, sparsity `{os_orig_task['sparsity_mean']}`, avg_rank `{os_orig_task['avg_rank']}`",
        f"- Coverage tradeoff: modified ranks fewer OpenStack core cases than original_dynotears under task_aligned (`{os_mod_task['rankable']}/{os_mod_task['cases']}` vs `{os_orig_task['rankable']}/{os_orig_task['cases']}`), so the claim is about ranking quality and sparsity, not recall.",
        "",
        "## OpenStack Path Supplement",
        "",
        f"- modified direct: `{mod_path['direct']}`",
        f"- modified two_hop: `{mod_path['two_hop']}`",
        f"- modified three_hop: `{mod_path['three_hop']}`",
        f"- modified none: `{mod_path['none']}`",
        "",
        "## Decision",
        "",
    ]
    if openstack_positive:
        lines.append(
            "- Keep OpenStack in thesis only as the redesigned core subset. modified is better than original_dynotears on task_aligned average rank and sparsity, but not on coverage and not on exact_only direct-edge recovery. The thesis claim must be framed with that tradeoff explicitly."
        )
    else:
        lines.append(
            "- Do not claim OpenStack as a clean positive mainline win. The redesigned core set should be reported as a constrained audit subset, and the OpenStack claim should be downgraded rather than loosening the rules."
        )
    lines.append("- Hadoop remains deferred to a separate audit fork and is not upgraded by this redesign.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    modes = ["exact_only", "task_aligned"] if args.mode == "both" else [args.mode]

    exact_rows = None
    task_rows = None
    total_modes = len(modes) or 1
    index_lines = ["# RQ2 OpenStack Redesign Evaluation (2026-03-18)", ""]
    for idx, mode in enumerate(modes, start=1):
        print(f"[{idx}/{total_modes}] Evaluating mode `{mode}` on HDFS-frozen + OpenStack-core. Expected: 5-20s")
        rows = evaluate_graph_rows(GRAPH_FILES, COMBINED_EVAL_BENCH_PATH, match_mode=mode)
        out_json = RESULTS_DIR / f"rq2_openstack_redesign_{mode}_summary_20260318.json"
        out_md = REPORTS_DIR / f"rq2_openstack_redesign_{mode}_summary_20260318.md"
        write_json(out_json, rows)
        md = summarize_rows(rows)
        out_md.write_text(md, encoding="utf-8")
        index_lines.append(f"## {mode}")
        index_lines.append("")
        index_lines.append(md.rstrip())
        index_lines.append("")
        print(f"[Saved] {out_json}")
        print(f"[Saved] {out_md}")
        if mode == "exact_only":
            exact_rows = rows
        if mode == "task_aligned":
            task_rows = rows

    core_rows = [row for row in load_json(CORE_BENCH_PATH) if str(row.get("dataset", "")) == "OpenStack"]
    print("[Path] Building OpenStack path supplement. Expected: <5s")
    path_rows = build_path_supplement_rows(GRAPH_FILES, core_rows)
    path_json = RESULTS_DIR / "rq2_openstack_redesign_path_supplement_20260318.json"
    path_md = REPORTS_DIR / "rq2_openstack_redesign_path_supplement_20260318.md"
    write_json(path_json, path_rows)
    path_md.write_text(_path_rows_markdown(path_rows), encoding="utf-8")
    print(f"[Saved] {path_json}")
    print(f"[Saved] {path_md}")

    index_path = REPORTS_DIR / "rq2_openstack_redesign_eval_index_20260318.md"
    index_lines.append("## path_supplement")
    index_lines.append("")
    index_lines.append(_path_rows_markdown(path_rows).rstrip())
    index_lines.append("")
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[Saved] {index_path}")

    if exact_rows is not None and task_rows is not None:
        _write_final_assessment(exact_rows, task_rows, path_rows)
        print(f"[Saved] {REPORTS_DIR / 'rq2_openstack_redesign_final_assessment_20260318.md'}")


if __name__ == "__main__":
    main()
