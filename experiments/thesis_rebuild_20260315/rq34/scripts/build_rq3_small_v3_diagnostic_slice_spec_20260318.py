from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
SHORTLIST_PATH = (
    REBUILD_ROOT
    / "rq34"
    / "analysis"
    / "rq3_small_v3_seed_shortlist_20260318"
    / "rq3_small_v3_seed_shortlist_20260318.json"
)
OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v3_diagnostic_slice_20260318"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_type(section: Mapping[str, object]) -> str:
    if str(section.get("status")) == "blocked":
        return "blocked"
    action_diversity = int(section.get("action_diversity", 0) or 0)
    if action_diversity <= 1:
        return "single_action_slice"
    if action_diversity == 2:
        return "narrow_mixed_slice"
    return "mixed_action_slice"


def _case_note(row: Mapping[str, object], *, review_only: bool) -> str:
    prefix = "Review-only candidate." if review_only else "Approved local diagnostic case."
    return f"{prefix} {row['short_reason']}; bucket={row['bucket_id']}."


def build_spec() -> Dict[str, object]:
    shortlist = _load_json(SHORTLIST_PATH)
    out: Dict[str, object] = {
        "benchmark_id": "rq3_small_v3_local_diagnostic_slice_20260318",
        "purpose": "Local-only open-text diagnostic slice for RQ3 rebuild. This is not a formal small benchmark and must not be used for paid API runs directly.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "noise_levels": [0.0, 1.0],
        "probe_modes_required": [
            "heuristic_alert",
            "heuristic_context",
            "vanilla_closed_ids",
            "open_alert_only",
            "open_alert_context",
        ],
        "datasets": {},
    }

    for dataset, section in shortlist["datasets"].items():
        approved_cases: List[Dict[str, object]] = []
        review_cases: List[Dict[str, object]] = []
        for row in section.get("approved_shortlist", []):
            approved_cases.append(
                {
                    "case_id": row["case_id"],
                    "source": row["pool_source"],
                    "gt_family_id": row["gt_label"],
                    "gt_action_id": row["gt_action_id"],
                    "alert_match": row["selected_alert"],
                    "alert_occurrence": 1,
                    "eligibility_note": _case_note(row, review_only=False),
                    "seed_score": row["seed_score"],
                    "bucket_id": row["bucket_id"],
                }
            )
        for row in section.get("review_shortlist", []):
            review_cases.append(
                {
                    "case_id": row["case_id"],
                    "source": row["pool_source"],
                    "gt_family_id": row["gt_label"],
                    "gt_action_id": row["gt_action_id"],
                    "alert_match": row["selected_alert"],
                    "alert_occurrence": 1,
                    "eligibility_note": _case_note(row, review_only=True),
                    "seed_score": row["seed_score"],
                    "bucket_id": row["bucket_id"],
                }
            )

        out["datasets"][dataset] = {
            "status": section["status"],
            "slice_type": _slice_type(section),
            "blocked_reason": section.get("blocked_reason", ""),
            "approved_case_count": len(approved_cases),
            "review_case_count": len(review_cases),
            "action_diversity": int(section.get("action_diversity", 0) or 0),
            "approved_cases": approved_cases,
            "review_cases": review_cases,
        }
    return out


def write_outputs(payload: Mapping[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "rq3_small_v3_diagnostic_slice_spec_20260318.json"
    md_path = OUTPUT_DIR / "rq3_small_v3_diagnostic_slice_spec_20260318.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = [
        "# RQ3 Small V3 Local Diagnostic Slice Spec (2026-03-18)",
        "",
        f"- benchmark_id: {payload['benchmark_id']}",
        "- formal_small_ready: false",
        "- paid_api_allowed: false",
        "",
    ]
    for dataset, section in payload["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append(f"- status: {section['status']}")
        lines.append(f"- slice_type: {section['slice_type']}")
        if section.get("blocked_reason"):
            lines.append(f"- blocked_reason: {section['blocked_reason']}")
        lines.append(f"- approved_case_count: {section['approved_case_count']}")
        lines.append(f"- review_case_count: {section['review_case_count']}")
        lines.append("")
        if section["approved_cases"]:
            lines.append("### Approved")
            for row in section["approved_cases"]:
                lines.append(f"- {row['case_id']} | {row['gt_action_id']} | score={row['seed_score']} | bucket={row['bucket_id']}")
        if section["review_cases"]:
            lines.append("")
            lines.append("### Review")
            for row in section["review_cases"]:
                lines.append(f"- {row['case_id']} | {row['gt_action_id']} | score={row['seed_score']} | bucket={row['bucket_id']}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = build_spec()
    write_outputs(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
