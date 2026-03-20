from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REBUILD_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq34.scripts import build_rq3_small_v2_benchmark_20260318 as builder_v2
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    infer_action_id_from_text,
)


SPEC_PATH = (
    REBUILD_ROOT
    / "rq34"
    / "analysis"
    / "rq3_small_v3_diagnostic_slice_20260318"
    / "rq3_small_v3_diagnostic_slice_spec_20260318.json"
)
ADMISSIBILITY_PATH = REBUILD_ROOT / "rq34" / "configs" / "rq3_small_v3_admissibility_rules_20260318.json"
OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v3_diagnostic_slice_20260318"
LOCAL_WINDOW = 4


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in _norm(text).replace("/", " ").replace(":", " ").replace("-", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
        if len(token) >= 3:
            tokens.append(token)
    return tokens


def _overlap_score(dataset: str, action_id: str, text: str) -> float:
    meta = ACTION_CATALOG.get(dataset, {}).get(action_id, {})
    text_tokens = set(_tokenize(text))
    desc_tokens = set(_tokenize(str(meta.get("description", ""))))
    overlap = len(text_tokens & desc_tokens)
    group_hits = 0
    lowered = _norm(text)
    for group in meta.get("keyword_groups", []):
        if any(_norm(token) in lowered for token in group):
            group_hits += 1
    return round(overlap + 2.5 * group_hits, 4)


def _pattern_hits(dataset: str, text: str, admissibility: Mapping[str, object]) -> List[str]:
    patterns = admissibility["case_structure_rules"]["disallow_direct_anchor_patterns"].get(dataset, [])
    lowered = _norm(text)
    return [pattern for pattern in patterns if pattern.lower() in lowered]


def _anchor_score(dataset: str, gt_action_id: str, text: str, admissibility: Mapping[str, object]) -> Dict[str, object]:
    inferred_action = infer_action_id_from_text(dataset, text)
    overlap = _overlap_score(dataset, gt_action_id, text)
    hits = _pattern_hits(dataset, text, admissibility)
    directness = round((8.0 if inferred_action == gt_action_id else 0.0) + overlap + 5.0 * len(hits), 4)
    return {
        "inferred_action_id": inferred_action,
        "gt_overlap_score": overlap,
        "direct_pattern_hits": hits,
        "directness_score": directness,
    }


def build_report() -> Dict[str, object]:
    spec = _load_json(SPEC_PATH)
    admissibility = _load_json(ADMISSIBILITY_PATH)
    pool_rows = builder_v2._load_pool_rows()
    datasets_out: Dict[str, List[Dict[str, object]]] = {}

    for dataset, section in spec["datasets"].items():
        rows: List[Dict[str, object]] = []
        for item in section.get("approved_cases", []):
            source = str(item["source"])
            case_id = str(item["case_id"])
            raw_log = str(pool_rows[(source, case_id)].get("raw_log", "") or "")
            selected_alert = builder_v2._find_alert_line(
                raw_log,
                alert_match=str(item["alert_match"]),
                occurrence=int(item.get("alert_occurrence", 1) or 1),
            )
            lines = [line.strip() for line in raw_log.splitlines() if line.strip()]
            selected_idx = next(
                idx for idx, line in enumerate(lines) if _norm(selected_alert) == _norm(line) or _norm(selected_alert) in _norm(line)
            )
            current_metrics = _anchor_score(dataset, str(item["gt_action_id"]), selected_alert, admissibility)

            candidates: List[Dict[str, object]] = []
            for idx in range(max(0, selected_idx - LOCAL_WINDOW), min(len(lines), selected_idx + LOCAL_WINDOW + 1)):
                if idx == selected_idx:
                    continue
                line = lines[idx]
                metrics = _anchor_score(dataset, str(item["gt_action_id"]), line, admissibility)
                candidate = {
                    "offset": idx - selected_idx,
                    "line": line,
                    **metrics,
                }
                candidates.append(candidate)

            candidates.sort(
                key=lambda row: (
                    float(row["directness_score"]),
                    len(row["direct_pattern_hits"]),
                    abs(int(row["offset"])),
                    row["line"],
                )
            )
            rows.append(
                {
                    "case_id": case_id,
                    "gt_action_id": item["gt_action_id"],
                    "current_alert": selected_alert,
                    "current_alert_metrics": current_metrics,
                    "alternative_candidates": candidates[:5],
                }
            )
        datasets_out[dataset] = rows

    return {
        "benchmark_id": spec["benchmark_id"],
        "purpose": "Mine weaker nearby alert anchors for the RQ3 v3 local diagnostic slice before any next rebuild step.",
        "datasets": datasets_out,
    }


def write_outputs(payload: Mapping[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "rq3_small_v3_alternative_alert_anchors_20260318.json"
    md_path = OUTPUT_DIR / "rq3_small_v3_alternative_alert_anchors_20260318.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = ["# RQ3 Small V3 Alternative Alert Anchors (2026-03-18)", ""]
    for dataset, cases in payload["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        for row in cases:
            current = row["current_alert_metrics"]
            lines.append(
                f"- {row['case_id']} | current_directness={current['directness_score']} | current_inferred_action={current['inferred_action_id'] or 'none'}"
            )
            lines.append(f"  current: {row['current_alert']}")
            for cand in row["alternative_candidates"]:
                lines.append(
                    f"  alt offset={cand['offset']} | directness={cand['directness_score']} | inferred={cand['inferred_action_id'] or 'none'} | hits={','.join(cand['direct_pattern_hits']) if cand['direct_pattern_hits'] else 'none'}"
                )
                lines.append(f"    {cand['line']}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = build_report()
    write_outputs(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
