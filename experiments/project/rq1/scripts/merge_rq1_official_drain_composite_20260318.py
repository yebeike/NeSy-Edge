from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (  # noqa: E402
    NOISE_LEVELS,
    row_key,
    summarize_rows,
    write_markdown_report,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_csv, write_json  # noqa: E402
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import REPORT_DIR, RQ1_RESULTS_DIR, ensure_dirs  # noqa: E402


FIELDNAMES = [
    "dataset",
    "case_id",
    "noise",
    "method",
    "gt_template",
    "prediction",
    "pa_hit",
    "latency_ms",
    "gt_source",
    "route",
    "query_chars",
    "ref_chars",
    "ref_count",
]
METHOD_ORDER = {"Drain": 0, "NuSy": 1, "Qwen": 2}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--manifest-name", required=True)
    ap.add_argument("--canonical-rows", required=True)
    ap.add_argument("--drain-rows", required=True)
    ap.add_argument("--drain-summary", required=True)
    return ap.parse_args()


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _stage_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row["dataset"]),
        str(row["case_id"]),
        f"{float(row['noise']):.1f}",
    )


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            str(row["case_id"]),
            float(row["noise"]),
            METHOD_ORDER.get(str(row["method"]), 99),
        ),
    )


def _validate(canonical_rows: list[dict], drain_rows: list[dict]) -> list[str]:
    errors: list[str] = []
    canonical_drain = [row for row in canonical_rows if row["method"] == "Drain"]
    canonical_nq = [row for row in canonical_rows if row["method"] in {"NuSy", "Qwen"}]

    unexpected_canonical = sorted({row["method"] for row in canonical_rows if row["method"] not in METHOD_ORDER})
    unexpected_drain = sorted({row["method"] for row in drain_rows if row["method"] != "Drain"})
    if unexpected_canonical:
        errors.append(f"canonical rows contain unexpected methods: {unexpected_canonical}")
    if unexpected_drain:
        errors.append(f"drain rows contain unexpected methods: {unexpected_drain}")

    drain_keys = {row_key(row) for row in drain_rows}
    canonical_drain_keys = {row_key(row) for row in canonical_drain}
    if drain_keys != canonical_drain_keys:
        missing = sorted(canonical_drain_keys - drain_keys)[:5]
        extra = sorted(drain_keys - canonical_drain_keys)[:5]
        errors.append(f"drain key mismatch missing={missing} extra={extra}")

    canonical_base = {_stage_key(row) for row in canonical_drain}
    canonical_nq_base = {_stage_key(row) for row in canonical_nq}
    if canonical_base != canonical_nq_base:
        missing = sorted(canonical_base - canonical_nq_base)[:5]
        extra = sorted(canonical_nq_base - canonical_base)[:5]
        errors.append(f"canonical NuSy/Qwen base-key mismatch missing={missing} extra={extra}")

    drain_by_key = {row_key(row): row for row in drain_rows}
    canonical_by_key = {row_key(row): row for row in canonical_drain}
    for key in sorted(canonical_drain_keys & drain_keys):
        left = canonical_by_key[key]
        right = drain_by_key[key]
        for field in ("dataset", "case_id", "gt_template", "gt_source"):
            if str(left[field]) != str(right[field]):
                errors.append(f"{key}: field mismatch {field} canonical={left[field]!r} drain={right[field]!r}")
                break
    return errors


def main() -> int:
    args = _parse_args()
    ensure_dirs()
    canonical_rows = _read_rows(Path(args.canonical_rows))
    drain_rows = _read_rows(Path(args.drain_rows))
    drain_summary = json.loads(Path(args.drain_summary).read_text(encoding="utf-8"))

    errors = _validate(canonical_rows, drain_rows)
    if errors:
        raise SystemExit("\n".join(errors))

    canonical_nq = [row for row in canonical_rows if row["method"] in {"NuSy", "Qwen"}]
    merged_rows = _sorted_rows(drain_rows + canonical_nq)

    out_csv = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_rows_20260317.csv"
    out_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_summary_20260317.json"
    merge_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_merge_20260317.json"

    edge_meta = dict(drain_summary.get("edge_meta", {}))
    edge_meta["expected_rows"] = len(canonical_rows)
    edge_meta["final_rows"] = len(merged_rows)
    edge_meta["drain_only_run_tag"] = drain_summary.get("run_tag")
    edge_meta["canonical_rows_path"] = str(Path(args.canonical_rows))
    edge_meta["drain_rows_path"] = str(Path(args.drain_rows))
    edge_meta["composite_reused_methods"] = ["NuSy", "Qwen"]

    payload = summarize_rows(merged_rows, args.manifest_name, args.run_tag, edge_meta)
    write_csv(out_csv, merged_rows)
    write_json(out_json, payload)
    report_path = write_markdown_report(out_json, payload, f"rq1_{args.run_tag}_report_20260318.md")
    write_json(
        merge_json,
        {
            "run_tag": args.run_tag,
            "manifest": args.manifest_name,
            "canonical_rows": str(Path(args.canonical_rows)),
            "drain_rows": str(Path(args.drain_rows)),
            "merged_rows": str(out_csv),
            "report_path": str(report_path),
            "expected_rows": len(canonical_rows),
            "final_rows": len(merged_rows),
            "reused_rows": len(canonical_nq),
            "drain_rows_count": len(drain_rows),
            "noise_levels": NOISE_LEVELS,
            "validation_errors": [],
        },
    )
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    print(f"[Saved] {merge_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
