from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from rq2_hadoop_family_audit_common_20260318 import (
    GRAPH_FILES,
    HADOOP_BENCHMARK_CORE_PATH,
    HADOOP_BENCHMARK_FULL_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    evaluate_family_graph_rows,
    summarize_rows,
    write_json,
)


BENCHMARKS = {
    "full": HADOOP_BENCHMARK_FULL_PATH,
    "local_core": HADOOP_BENCHMARK_CORE_PATH,
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact_only", "task_aligned", "both"], default="both")
    return ap.parse_args()


def _find(rows: List[Dict[str, object]], graph: str) -> Dict[str, object]:
    for row in rows:
        if str(row.get("graph", "")) == graph:
            return row
    raise KeyError(graph)


def _write_final_assessment(results: Dict[str, Dict[str, List[Dict[str, object]]]]) -> None:
    report_path = REPORTS_DIR / "rq2_hadoop_family_audit_final_assessment_20260318.md"

    full_exact = results["full"]["exact_only"]
    full_task = results["full"]["task_aligned"]
    core_exact = results["local_core"]["exact_only"]
    core_task = results["local_core"]["task_aligned"]

    full_mod_exact = _find(full_exact, "modified")
    full_orig_exact = _find(full_exact, "original_dynotears")
    full_mod_task = _find(full_task, "modified")
    full_orig_task = _find(full_task, "original_dynotears")
    core_mod_exact = _find(core_exact, "modified")
    core_orig_exact = _find(core_exact, "original_dynotears")
    core_mod_task = _find(core_task, "modified")
    core_orig_task = _find(core_task, "original_dynotears")

    modified_positive = (
        full_mod_task["avg_rank"] is not None
        and full_orig_task["avg_rank"] is not None
        and float(full_mod_task["sparsity_mean"]) < float(full_orig_task["sparsity_mean"])
        and float(full_mod_task["avg_rank"]) < float(full_orig_task["avg_rank"])
    )

    lines = [
        "# RQ2 Hadoop Family Audit Final Assessment (2026-03-18)",
        "",
        "- Hadoop is evaluated as an exact-effect plus root-family task.",
        "- This audit intentionally does not claim exact root-template recovery for Hadoop.",
        "- No representative-root forcing and no proxy-root evaluator fallback are used.",
        "",
        "## Full Benchmark",
        "",
        f"- modified exact_only: rankable `{full_mod_exact['rankable']}/{full_mod_exact['cases']}`, avg_rank `{full_mod_exact['avg_rank']}`, sparsity `{full_mod_exact['sparsity_mean']}`",
        f"- original_dynotears exact_only: rankable `{full_orig_exact['rankable']}/{full_orig_exact['cases']}`, avg_rank `{full_orig_exact['avg_rank']}`, sparsity `{full_orig_exact['sparsity_mean']}`",
        f"- modified task_aligned: rankable `{full_mod_task['rankable']}/{full_mod_task['cases']}`, avg_rank `{full_mod_task['avg_rank']}`, sparsity `{full_mod_task['sparsity_mean']}`",
        f"- original_dynotears task_aligned: rankable `{full_orig_task['rankable']}/{full_orig_task['cases']}`, avg_rank `{full_orig_task['avg_rank']}`, sparsity `{full_orig_task['sparsity_mean']}`",
        "",
        "## Local Core Sanity Subset",
        "",
        f"- modified exact_only: rankable `{core_mod_exact['rankable']}/{core_mod_exact['cases']}`, avg_rank `{core_mod_exact['avg_rank']}`, sparsity `{core_mod_exact['sparsity_mean']}`",
        f"- original_dynotears exact_only: rankable `{core_orig_exact['rankable']}/{core_orig_exact['cases']}`, avg_rank `{core_orig_exact['avg_rank']}`, sparsity `{core_orig_exact['sparsity_mean']}`",
        f"- modified task_aligned: rankable `{core_mod_task['rankable']}/{core_mod_task['cases']}`, avg_rank `{core_mod_task['avg_rank']}`, sparsity `{core_mod_task['sparsity_mean']}`",
        f"- original_dynotears task_aligned: rankable `{core_orig_task['rankable']}/{core_orig_task['cases']}`, avg_rank `{core_orig_task['avg_rank']}`, sparsity `{core_orig_task['sparsity_mean']}`",
        "",
        "## Decision",
        "",
    ]
    if modified_positive:
        lines.append(
            "- Hadoop can be reported as a family-level RQ2 subsection. modified is better than original_dynotears on the full family benchmark for sparsity and task_aligned average rank. This claim must stay explicitly family-level, not exact-template."
        )
    else:
        lines.append(
            "- Hadoop should remain provisional for thesis mainline. The cleaned family-level audit does not show a stable modified-over-original win on the full benchmark, so Hadoop should be reported with a caveat or moved out of the main comparative claim."
        )
    lines.append(
        "- The local-core subset is only a sanity check for local evidence and must not replace the full family benchmark."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    ensure_dirs()
    modes = ["exact_only", "task_aligned"] if args.mode == "both" else [args.mode]

    results: Dict[str, Dict[str, List[Dict[str, object]]]] = {name: {} for name in BENCHMARKS}
    index_lines = ["# RQ2 Hadoop Family Audit Evaluation (2026-03-18)", ""]

    total_jobs = len(BENCHMARKS) * len(modes)
    job_idx = 0
    for bench_name, bench_path in BENCHMARKS.items():
        for mode in modes:
            job_idx += 1
            print(
                f"[{job_idx}/{total_jobs}] Evaluating `{bench_name}` benchmark with `{mode}`. Expected: <10s"
            )
            rows = evaluate_family_graph_rows(GRAPH_FILES, bench_path, match_mode=mode)
            results[bench_name][mode] = rows
            out_json = RESULTS_DIR / f"rq2_hadoop_family_audit_{bench_name}_{mode}_summary_20260318.json"
            out_md = REPORTS_DIR / f"rq2_hadoop_family_audit_{bench_name}_{mode}_summary_20260318.md"
            write_json(out_json, rows)
            out_md.write_text(summarize_rows(rows), encoding="utf-8")
            index_lines.append(f"## {bench_name} / {mode}")
            index_lines.append("")
            index_lines.append(summarize_rows(rows).rstrip())
            index_lines.append("")
            print(f"[Saved] {out_json}")
            print(f"[Saved] {out_md}")

    index_path = REPORTS_DIR / "rq2_hadoop_family_audit_eval_index_20260318.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[Saved] {index_path}")

    if all("exact_only" in results[name] and "task_aligned" in results[name] for name in BENCHMARKS):
        _write_final_assessment(results)
        print(f"[Saved] {REPORTS_DIR / 'rq2_hadoop_family_audit_final_assessment_20260318.md'}")


if __name__ == "__main__":
    main()
