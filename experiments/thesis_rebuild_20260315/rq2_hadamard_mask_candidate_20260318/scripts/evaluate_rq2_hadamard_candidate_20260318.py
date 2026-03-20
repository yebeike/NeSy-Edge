from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FINAL_ROOT = _SCRIPT_DIR.parent
_REBUILD_ROOT = _FINAL_ROOT.parents[0]
_PROJECT_ROOT = _FINAL_ROOT.parents[2]

for path in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "experiments" / "rq123_e2e",
    _FINAL_ROOT / "scripts",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rq2_hadamard_candidate_common_20260318 import (  # type: ignore
    ASSESSMENT_MD,
    FROZEN_COMPARE_GRAPH_FILES,
    PAPER_SUMMARY_JSON,
    PAPER_SUMMARY_MD,
    evaluate_paper_rows,
    ensure_dirs,
    summarize_rows,
    write_json,
    write_text,
)


METHOD_ORDER = [
    "modified",
    "hadamard_mask_dynotears",
    "original_dynotears",
    "pearson_hypothesis",
    "pc_cpdag_hypothesis",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def _assessment_text(rows):
    row_map = {(row["dataset"], row["evaluator"], row["method"]): row for row in rows}
    lines = [
        "# RQ2 Hadamard-Mask Candidate Assessment (2026-03-18)",
        "",
        "- Comparison methods are frozen from `rq2_mainline_penalized_20260318`.",
        "- Only `hadamard_mask_dynotears` is newly rebuilt in this workspace.",
        "- `Avg_Rank` is the same penalized all-case metric used by the current paper-facing RQ2 summary.",
        "",
        "## Candidate vs Current NeSy",
        "",
        "| Dataset | Evaluator | modified (S, R) | hadamard_mask_dynotears (S, R) | Better Avg_Rank |",
        "|---|---|---|---|---|",
    ]
    compare_keys = [
        ("HDFS", "audit_task_aligned_penalized"),
        ("OpenStack", "redesign_task_aligned_edge_penalized"),
        ("OpenStack", "redesign_task_aligned_path2_penalized"),
        ("Hadoop", "family_audit_core80_task_aligned_penalized"),
    ]
    for dataset, evaluator in compare_keys:
        modified = row_map[(dataset, evaluator, "modified")]
        candidate = row_map[(dataset, evaluator, "hadamard_mask_dynotears")]
        better = "hadamard_mask_dynotears" if float(candidate["avg_rank"]) < float(modified["avg_rank"]) else "modified"
        lines.append(
            f"| {dataset} | {evaluator} | {modified['sparsity_mean']}, {modified['avg_rank']} | {candidate['sparsity_mean']}, {candidate['avg_rank']} | {better} |"
        )
    lines.extend(
        [
            "",
            "## Full Five-Method Table",
            "",
            "| Dataset | Method | Evaluator | Sparsity_mean | Avg_Rank |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['evaluator']} | {row['sparsity_mean']} | {row['avg_rank']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    _parse_args()
    ensure_dirs()
    print("[1/2] Evaluating frozen four-method table plus hadamard candidate. Expected: <15s", flush=True)
    rows = evaluate_paper_rows(FROZEN_COMPARE_GRAPH_FILES)
    rows = sorted(rows, key=lambda row: (str(row["dataset"]), str(row["evaluator"]), _method_sort_key(str(row["method"]))))
    write_json(PAPER_SUMMARY_JSON, rows)
    write_text(PAPER_SUMMARY_MD, summarize_rows(rows))
    print(f"[Saved] {PAPER_SUMMARY_JSON}", flush=True)
    print(f"[Saved] {PAPER_SUMMARY_MD}", flush=True)

    print("[2/2] Writing candidate assessment. Expected: <5s", flush=True)
    write_text(ASSESSMENT_MD, _assessment_text(rows))
    print(f"[Saved] {ASSESSMENT_MD}", flush=True)


if __name__ == "__main__":
    main()
