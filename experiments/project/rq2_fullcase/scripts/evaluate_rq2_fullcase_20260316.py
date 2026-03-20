from __future__ import annotations

import argparse
from pathlib import Path

from rq2_fullcase_common_20260316 import (
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    evaluate_graph_rows,
    summarize_rows,
    write_json,
)


DEFAULT_GRAPH_PATHS = {
    "modified": RESULTS_DIR / "gt_causal_knowledge_nesydy_fullcase_20260316.json",
    "original": RESULTS_DIR / "gt_causal_knowledge_dynotears_fullcase_20260316.json",
    "pearson": RESULTS_DIR / "gt_causal_knowledge_pearson_fullcase_20260316.json",
    "pc": RESULTS_DIR / "gt_causal_knowledge_pc_fullcase_20260316.json",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hybrid", "strict", "both"], default="both")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_dirs()

    modes = ["hybrid", "strict"] if args.mode == "both" else [args.mode]
    report_lines = ["# RQ2 Fullcase Evaluation (2026-03-16)", ""]
    for mode in modes:
        rows = evaluate_graph_rows(DEFAULT_GRAPH_PATHS, match_mode=mode, include_hdfs_unknown=True)
        out_json = RESULTS_DIR / f"rq2_fullcase_{mode}_summary_20260316.json"
        out_md = REPORTS_DIR / f"rq2_fullcase_{mode}_summary_20260316.md"
        write_json(out_json, rows)
        md = summarize_rows(rows)
        out_md.write_text(md, encoding="utf-8")
        report_lines.append(f"## {mode}")
        report_lines.append("")
        report_lines.append(md.rstrip())
        report_lines.append("")
        print(f"[Saved] {out_json}")
        print(f"[Saved] {out_md}")

    index_path = REPORTS_DIR / "rq2_fullcase_eval_index_20260316.md"
    index_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[Saved] {index_path}")


if __name__ == "__main__":
    main()
