from __future__ import annotations

import argparse

from rq2_fullcase_audit_common_20260318 import (
    AUDIT_BENCHMARK_EVAL_PATH,
    GRAPH_FILES,
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    evaluate_graph_rows,
    summarize_rows,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact_only", "task_aligned", "both"], default="both")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()

    modes = ["exact_only", "task_aligned"] if args.mode == "both" else [args.mode]
    total_modes = len(modes) or 1
    report_lines = ["# RQ2 Fullcase Audit Evaluation (2026-03-18)", ""]
    for idx, mode in enumerate(modes, start=1):
        print(f"[{idx}/{total_modes}] Evaluating mode `{mode}`. Expected: 5-20s")
        rows = evaluate_graph_rows(GRAPH_FILES, AUDIT_BENCHMARK_EVAL_PATH, match_mode=mode)
        out_json = RESULTS_DIR / f"rq2_fullcase_audit_{mode}_summary_20260318.json"
        out_md = REPORTS_DIR / f"rq2_fullcase_audit_{mode}_summary_20260318.md"
        write_json(out_json, rows)
        md = summarize_rows(rows)
        out_md.write_text(md, encoding="utf-8")
        report_lines.append(f"## {mode}")
        report_lines.append("")
        report_lines.append(md.rstrip())
        report_lines.append("")
        print(f"[Saved] {out_json}")
        print(f"[Saved] {out_md}")

    index_path = REPORTS_DIR / "rq2_fullcase_audit_eval_index_20260318.md"
    print("[Final] Writing evaluation index. Expected: <5s")
    index_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[Saved] {index_path}")


if __name__ == "__main__":
    main()
