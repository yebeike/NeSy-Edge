from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


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


CANDIDATE_ROWS_PATH = REBUILD_ROOT / "rq34" / "results" / "rq3_candidate_audit_enriched_20260318_rows.json"
BLACKLIST_PATH = (
    REBUILD_ROOT
    / "rq34"
    / "analysis"
    / "rq3_local_probe_20260318"
    / "rq3_small_v2_shortcut_case_blacklist_20260318.json"
)
ADMISSIBILITY_PATH = REBUILD_ROOT / "rq34" / "configs" / "rq3_small_v3_admissibility_rules_20260318.json"
OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v3_diagnostic_slice_20260318"
WINDOW = 4


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
    lowered = _norm(text)
    group_hits = 0
    for group in meta.get("keyword_groups", []):
        if any(_norm(token) in lowered for token in group):
            group_hits += 1
    return round(overlap + 2.5 * group_hits, 4)


def _pattern_hits(dataset: str, text: str, admissibility: Mapping[str, object]) -> List[str]:
    patterns = admissibility["case_structure_rules"]["disallow_direct_anchor_patterns"].get(dataset, [])
    lowered = _norm(text)
    return [pattern for pattern in patterns if pattern.lower() in lowered]


def _anchor_metrics(dataset: str, gt_action_id: str, text: str, admissibility: Mapping[str, object]) -> Dict[str, object]:
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


def _extract_neighborhood(lines: List[str], selected_alert: str) -> Tuple[int, List[Tuple[int, str]]]:
    wanted = _norm(selected_alert)
    hit_index = -1
    for idx, line in enumerate(lines):
        line_norm = _norm(line)
        if wanted == line_norm or wanted in line_norm:
            hit_index = idx
            break
    if hit_index < 0:
        raise ValueError(f"Failed to locate selected alert in raw log: {selected_alert}")

    out: List[Tuple[int, str]] = []
    for idx in range(max(0, hit_index - WINDOW), min(len(lines), hit_index + WINDOW + 1)):
        if idx == hit_index:
            continue
        out.append((idx - hit_index, lines[idx]))
    return hit_index, out


def build_report() -> Dict[str, object]:
    candidate_rows = _load_json(CANDIDATE_ROWS_PATH)
    blacklist = {str(row["case_id"]) for row in _load_json(BLACKLIST_PATH)["cases"]}
    admissibility = _load_json(ADMISSIBILITY_PATH)
    pool_rows = builder_v2._load_pool_rows()

    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    summaries: Dict[str, Dict[str, object]] = {}

    for row in candidate_rows:
        dataset = str(row["dataset"])
        if dataset not in {"HDFS", "OpenStack"}:
            continue
        case_id = str(row["case_id"])
        source = str(row["pool_source"])
        gt_action_id = str(row["gt_action_id"])
        selected_alert = str(row.get("selected_alert", "") or "").strip()
        raw_log = str(pool_rows[(source, case_id)].get("raw_log", "") or "")
        lines = [line.strip() for line in raw_log.splitlines() if line.strip()]
        if not selected_alert or not raw_log:
            continue

        try:
            hit_index, nearby = _extract_neighborhood(lines, selected_alert)
        except ValueError:
            continue

        current = _anchor_metrics(dataset, gt_action_id, selected_alert, admissibility)
        candidates: List[Dict[str, object]] = []
        for offset, line in nearby:
            metrics = _anchor_metrics(dataset, gt_action_id, line, admissibility)
            reduction = round(float(current["directness_score"]) - float(metrics["directness_score"]), 4)
            candidates.append(
                {
                    "offset": offset,
                    "line": line,
                    "directness_reduction": reduction,
                    **metrics,
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["directness_score"]),
                len(item["direct_pattern_hits"]),
                -float(item["directness_reduction"]),
                abs(int(item["offset"])),
                item["line"],
            )
        )
        best_alternative = candidates[0] if candidates else None
        improvement = bool(
            best_alternative
            and float(best_alternative["directness_score"]) + 1e-9 < float(current["directness_score"])
        )
        weak_opportunity = bool(
            best_alternative
            and float(best_alternative["directness_score"]) <= 3.0
            and str(best_alternative["inferred_action_id"]) != gt_action_id
        )

        grouped[dataset][gt_action_id].append(
            {
                "case_id": case_id,
                "pool_source": source,
                "blacklisted": case_id in blacklist,
                "difficulty_score": float(row.get("difficulty_score", 0.0) or 0.0),
                "selection_score": float(row.get("selection_score", 0.0) or 0.0),
                "selected_alert_flags": row.get("selected_alert_flags", {}),
                "current_alert": selected_alert,
                "current_metrics": current,
                "best_alternative": best_alternative,
                "has_improvement": improvement,
                "has_weak_opportunity": weak_opportunity,
                "candidate_preview": candidates[:4],
                "raw_log_line_count": len(lines),
                "hit_index": hit_index,
            }
        )

    datasets_out: Dict[str, Dict[str, object]] = {}
    for dataset, by_action in grouped.items():
        dataset_actions: Dict[str, object] = {}
        action_summary: Dict[str, object] = {}
        for action_id, items in by_action.items():
            items.sort(
                key=lambda row: (
                    not bool(row["has_weak_opportunity"]),
                    not bool(row["has_improvement"]),
                    bool(row["blacklisted"]),
                    float(row["best_alternative"]["directness_score"]) if row["best_alternative"] else 999.0,
                    -float(row["difficulty_score"]),
                    row["case_id"],
                )
            )
            dataset_actions[action_id] = items[:12]
            action_summary[action_id] = {
                "case_count": len(items),
                "blacklisted_case_count": sum(int(bool(row["blacklisted"])) for row in items),
                "improved_case_count": sum(int(bool(row["has_improvement"])) for row in items),
                "weak_opportunity_case_count": sum(int(bool(row["has_weak_opportunity"])) for row in items),
            }
        datasets_out[dataset] = {
            "action_summary": dict(sorted(action_summary.items())),
            "actions": dict(sorted(dataset_actions.items())),
        }
        summaries[dataset] = {
            "total_cases": sum(section["case_count"] for section in action_summary.values()),
            "total_weak_opportunities": sum(section["weak_opportunity_case_count"] for section in action_summary.values()),
            "total_improved_cases": sum(section["improved_case_count"] for section in action_summary.values()),
        }

    return {
        "purpose": "Mine weaker nearby anchor opportunities across the wider HDFS/OpenStack candidate pool.",
        "window": WINDOW,
        "dataset_summary": summaries,
        "datasets": datasets_out,
    }


def write_outputs(payload: Mapping[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "rq3_small_v3_pool_anchor_opportunities_20260318.json"
    md_path = OUTPUT_DIR / "rq3_small_v3_pool_anchor_opportunities_20260318.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = ["# RQ3 Small V3 Pool Anchor Opportunities (2026-03-18)", ""]
    for dataset, summary in payload["dataset_summary"].items():
        lines.append(f"## {dataset}")
        lines.append(f"- total_cases: {summary['total_cases']}")
        lines.append(f"- total_improved_cases: {summary['total_improved_cases']}")
        lines.append(f"- total_weak_opportunities: {summary['total_weak_opportunities']}")
        lines.append("")
        for action_id, section in payload["datasets"][dataset]["action_summary"].items():
            lines.append(
                f"- {action_id} | cases={section['case_count']} | improved={section['improved_case_count']} | weak={section['weak_opportunity_case_count']}"
            )
        lines.append("")
        for action_id, rows in payload["datasets"][dataset]["actions"].items():
            lines.append(f"### {action_id}")
            for row in rows[:6]:
                best = row["best_alternative"] or {}
                lines.append(
                    f"- {row['case_id']} | blacklisted={str(row['blacklisted']).lower()} | diff={row['difficulty_score']} | current={row['current_metrics']['directness_score']} | best={best.get('directness_score', 'na')}"
                )
                lines.append(f"  current: {row['current_alert']}")
                if best:
                    lines.append(f"  alt: {best['line']}")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = build_report()
    write_outputs(payload)
    print(json.dumps(payload["dataset_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
