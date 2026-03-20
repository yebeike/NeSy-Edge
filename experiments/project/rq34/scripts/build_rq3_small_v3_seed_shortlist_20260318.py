from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parents[1]
CONFIG_PATH = REBUILD_ROOT / "rq34" / "configs" / "rq3_small_v3_seed_rules_20260318.json"
OUTPUT_DIR = REBUILD_ROOT / "rq34" / "analysis" / "rq3_small_v3_seed_shortlist_20260318"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _active_flags(row: Mapping[str, object]) -> List[str]:
    return sorted(k for k, v in dict(row.get("selected_alert_flags", {})).items() if v)


def _risk_penalty(row: Mapping[str, object], *, hard_flags: set[str], soft_flags: set[str]) -> float:
    flags = set(_active_flags(row))
    penalty = 0.0
    for flag in flags:
        if flag in hard_flags:
            penalty += 10.0
        elif flag in soft_flags:
            penalty += 3.0
        else:
            penalty += 1.0
    if bool(row.get("weak_mainline_alert")):
        penalty += 2.0
    return penalty


def _candidate_score(
    row: Mapping[str, object],
    *,
    hard_flags: set[str],
    soft_flags: set[str],
    preferred_actions: set[str],
    disallowed_actions: set[str],
    hard_drop_text_patterns: List[str],
    preferred_text_patterns: List[str],
    shortcut_blacklist_case_ids: set[str],
) -> float:
    case_id = str(row.get("case_id", ""))
    action_id = str(row.get("gt_action_id", ""))
    selected_alert = str(row.get("selected_alert", "") or "")
    if action_id in disallowed_actions:
        return -999.0
    if case_id in shortcut_blacklist_case_ids:
        return -999.0
    lower_alert = selected_alert.lower()
    if any(pattern.lower() in lower_alert for pattern in hard_drop_text_patterns):
        return -999.0
    score = 0.0
    score += 2.0 * float(row.get("difficulty_score", 0.0) or 0.0)
    score += 0.1 * float(row.get("selection_score", 0.0) or 0.0)
    if action_id in preferred_actions:
        score += 4.0
    if any(pattern.lower() in lower_alert for pattern in preferred_text_patterns):
        score += 2.0
    if not bool(row.get("fixed_small_member")):
        score += 2.0
    if str(row.get("pool_source", "")) == "rq3_test_set_enriched":
        score += 1.0
    score -= _risk_penalty(row, hard_flags=hard_flags, soft_flags=soft_flags)
    return round(score, 3)


def _pattern_bucket(selected_alert: str, bucket_rules: Mapping[str, Sequence[str]]) -> str:
    lower_alert = str(selected_alert or "").lower()
    for bucket, patterns in bucket_rules.items():
        if any(str(pattern).lower() in lower_alert for pattern in patterns):
            return str(bucket)
    return "other"


def _serialize_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "case_id": row["case_id"],
        "pool_source": row["pool_source"],
        "gt_action_id": row["gt_action_id"],
        "gt_label": row["gt_label"],
        "difficulty_score": row["difficulty_score"],
        "selection_score": row["selection_score"],
        "seed_score": row["seed_score"],
        "risk_penalty": row["risk_penalty"],
        "bucket_id": row["bucket_id"],
        "selected_alert": row["selected_alert"],
        "selected_alert_flags": row["selected_alert_flags"],
        "active_flags": row["active_flags"],
        "short_reason": row["short_reason"],
    }


def _short_reason(row: Mapping[str, object], *, score: float) -> str:
    flags = _active_flags(row)
    parts = [f"score={score}", f"diff={row.get('difficulty_score')}"]
    if flags:
        parts.append("flags=" + ",".join(flags))
    if bool(row.get("fixed_small_member")):
        parts.append("was_fixed_small=true")
    parts.append(f"source={row.get('pool_source')}")
    return "; ".join(parts)


def _select_with_caps(
    rows: Sequence[Mapping[str, object]],
    *,
    max_cases: int,
    max_per_action: Mapping[str, int],
    max_per_bucket: Mapping[str, int],
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    for row in rows:
        if len(selected) >= max_cases:
            break
        action_id = str(row.get("gt_action_id", ""))
        bucket_id = str(row.get("bucket_id", "other"))
        action_cap = int(max_per_action.get(action_id, max_per_action.get("*", max_cases)))
        bucket_cap = int(max_per_bucket.get(bucket_id, max_per_bucket.get("*", max_cases)))
        if action_counts[action_id] >= action_cap:
            continue
        if bucket_counts[bucket_id] >= bucket_cap:
            continue
        selected.append(dict(row))
        action_counts[action_id] += 1
        bucket_counts[bucket_id] += 1
    return selected


def build_shortlist() -> Dict[str, object]:
    config = _load_json(CONFIG_PATH)
    candidate_rows = _load_json(Path(config["sources"]["candidate_audit_rows"]))
    blacklist = _load_json(Path(config["sources"]["shortcut_blacklist"]))
    shortcut_blacklist_case_ids = {str(row["case_id"]) for row in blacklist.get("cases", [])}

    out: Dict[str, object] = {
        "config_path": str(CONFIG_PATH),
        "shortcut_blacklist_case_ids": sorted(shortcut_blacklist_case_ids),
        "formal_small_ready": False,
        "datasets": {},
    }

    for dataset, ds_cfg in config["datasets"].items():
        ds_rows = [row for row in candidate_rows if str(row.get("dataset")) == dataset]
        hard_flags = set(ds_cfg.get("hard_drop_flags", []))
        soft_flags = set(ds_cfg.get("soft_drop_flags", []))
        preferred_actions = set(ds_cfg.get("preferred_actions", []))
        disallowed_actions = set(ds_cfg.get("disallowed_actions", []))
        hard_drop_text_patterns = list(ds_cfg.get("hard_drop_text_patterns", []))
        preferred_text_patterns = list(ds_cfg.get("preferred_text_patterns", []))
        review_only_flags = set(ds_cfg.get("review_only_flags", []))
        bucket_rules = dict(ds_cfg.get("pattern_buckets", {}))
        min_seed_score = float(ds_cfg.get("min_seed_score", 0.0) or 0.0)
        min_difficulty_score = float(ds_cfg.get("min_difficulty_score", 0.0) or 0.0)
        min_approved_cases = int(ds_cfg.get("minimum_approved_cases", 0) or 0)
        min_action_diversity = int(ds_cfg.get("minimum_action_diversity", 0) or 0)
        max_approved_cases = int(ds_cfg.get("max_approved_cases", config["global_rules"]["max_seeds_per_dataset"]))
        max_review_cases = int(ds_cfg.get("max_review_cases", 0) or 0)
        max_per_action = dict(ds_cfg.get("max_per_action", {"*": max_approved_cases}))
        max_per_bucket = dict(ds_cfg.get("max_per_bucket", {"*": max_approved_cases}))

        case_best_rows: Dict[str, Dict[str, object]] = {}
        hard_blocked = 0
        flag_counter = Counter()
        for row in ds_rows:
            flags = _active_flags(row)
            flag_counter.update(flags)
            if any(flag in hard_flags for flag in flags):
                hard_blocked += 1
            risk_penalty = _risk_penalty(row, hard_flags=hard_flags, soft_flags=soft_flags)
            score = _candidate_score(
                row,
                hard_flags=hard_flags,
                soft_flags=soft_flags,
                preferred_actions=preferred_actions,
                disallowed_actions=disallowed_actions,
                hard_drop_text_patterns=hard_drop_text_patterns,
                preferred_text_patterns=preferred_text_patterns,
                shortcut_blacklist_case_ids=shortcut_blacklist_case_ids,
            )
            if score <= -999.0:
                continue
            enriched: Dict[str, object] = dict(row)
            enriched["seed_score"] = score
            enriched["risk_penalty"] = risk_penalty
            enriched["active_flags"] = flags
            enriched["bucket_id"] = _pattern_bucket(str(row.get("selected_alert", "") or ""), bucket_rules)
            enriched["short_reason"] = _short_reason(row, score=score)
            case_id = str(row.get("case_id", ""))
            incumbent = case_best_rows.get(case_id)
            if incumbent is None:
                case_best_rows[case_id] = enriched
                continue
            incumbent_key = (
                float(incumbent["seed_score"]),
                -float(incumbent["risk_penalty"]),
                float(incumbent.get("difficulty_score", 0.0) or 0.0),
                float(incumbent.get("selection_score", 0.0) or 0.0),
                str(incumbent.get("selected_alert", "")),
            )
            current_key = (
                float(enriched["seed_score"]),
                -float(enriched["risk_penalty"]),
                float(enriched.get("difficulty_score", 0.0) or 0.0),
                float(enriched.get("selection_score", 0.0) or 0.0),
                str(enriched.get("selected_alert", "")),
            )
            if current_key > incumbent_key:
                case_best_rows[case_id] = enriched

        scored_rows = sorted(
            case_best_rows.values(),
            key=lambda row: (
                -float(row["seed_score"]),
                float(row["risk_penalty"]),
                -float(row.get("difficulty_score", 0.0) or 0.0),
                row.get("case_id", ""),
            ),
        )

        status = str(ds_cfg.get("target_status", "review"))
        blocked_reason = ""
        dropped_counts = Counter()
        approved_candidates: List[Dict[str, object]] = []
        review_candidates: List[Dict[str, object]] = []

        for row in scored_rows:
            if bool(config["global_rules"].get("drop_non_positive_scores", False)) and float(row["seed_score"]) <= 0.0:
                dropped_counts["non_positive_score"] += 1
                continue
            if float(row.get("seed_score", 0.0) or 0.0) < min_seed_score:
                dropped_counts["below_min_seed_score"] += 1
                continue
            if float(row.get("difficulty_score", 0.0) or 0.0) < min_difficulty_score:
                dropped_counts["below_min_difficulty"] += 1
                continue
            flags = set(row["active_flags"])
            if any(flag in review_only_flags for flag in flags):
                review_candidates.append(row)
            else:
                approved_candidates.append(row)

        approved_shortlist: List[Dict[str, object]] = []
        review_shortlist: List[Dict[str, object]] = []
        if status == "blocked":
            blocked_reason = "Dataset remains shortcut-dominated under current pool and should be reset before any new small selection."
            if max_review_cases > 0:
                review_shortlist = _select_with_caps(
                    review_candidates or approved_candidates,
                    max_cases=max_review_cases,
                    max_per_action=max_per_action,
                    max_per_bucket=max_per_bucket,
                )
        else:
            approved_shortlist = _select_with_caps(
                approved_candidates,
                max_cases=max_approved_cases,
                max_per_action=max_per_action,
                max_per_bucket=max_per_bucket,
            )
            approved_case_ids = {str(row["case_id"]) for row in approved_shortlist}
            residual_review_pool = [row for row in review_candidates if str(row["case_id"]) not in approved_case_ids]
            if max_review_cases > 0:
                review_shortlist = _select_with_caps(
                    residual_review_pool,
                    max_cases=max_review_cases,
                    max_per_action=max_per_action,
                    max_per_bucket=max_per_bucket,
                )

            action_diversity = len({str(row["gt_action_id"]) for row in approved_shortlist})
            if not approved_shortlist:
                status = "blocked"
                blocked_reason = "No candidates survived the tightened v3 admissibility screen."
            elif len(approved_shortlist) < min_approved_cases:
                status = "review"
                blocked_reason = (
                    f"Only {len(approved_shortlist)} approved cases survived; minimum required is {min_approved_cases}."
                )
            elif action_diversity < min_action_diversity:
                status = "review"
                blocked_reason = (
                    f"Approved slice has action_diversity={action_diversity}, below the minimum {min_action_diversity}."
                )
            else:
                blocked_reason = str(ds_cfg.get("status_note", ""))

        out["datasets"][dataset] = {
            "status": status,
            "blocked_reason": blocked_reason,
            "candidate_count": len(ds_rows),
            "case_count_after_dedup": len(case_best_rows),
            "hard_flag_rows": hard_blocked,
            "flag_counts": dict(sorted(flag_counter.items())),
            "dropped_counts": dict(sorted(dropped_counts.items())),
            "approved_case_count": len(approved_shortlist),
            "review_case_count": len(review_shortlist),
            "action_diversity": len({str(row["gt_action_id"]) for row in approved_shortlist}),
            "approved_shortlist": [_serialize_row(row) for row in approved_shortlist],
            "review_shortlist": [_serialize_row(row) for row in review_shortlist],
        }
    return out


def write_outputs(payload: Mapping[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "rq3_small_v3_seed_shortlist_20260318.json"
    md_path = OUTPUT_DIR / "rq3_small_v3_seed_shortlist_20260318.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = ["# RQ3 Small V3 Seed Shortlist (2026-03-18)", ""]
    for dataset, section in payload["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append(f"- status: {section['status']}")
        if section.get("blocked_reason"):
            lines.append(f"- blocked_reason: {section['blocked_reason']}")
        lines.append(f"- candidate_count: {section['candidate_count']}")
        lines.append(f"- case_count_after_dedup: {section['case_count_after_dedup']}")
        lines.append(f"- hard_flag_rows: {section['hard_flag_rows']}")
        lines.append(f"- approved_case_count: {section['approved_case_count']}")
        lines.append(f"- review_case_count: {section['review_case_count']}")
        lines.append(f"- action_diversity: {section['action_diversity']}")
        lines.append("")
        if section["approved_shortlist"]:
            lines.append("### Approved")
        for row in section["approved_shortlist"]:
            flags = row["active_flags"]
            lines.append(
                f"- {row['case_id']} | {row['gt_action_id']} | score={row['seed_score']} | diff={row['difficulty_score']} | bucket={row['bucket_id']} | flags={','.join(flags) if flags else 'none'}"
            )
            lines.append(f"  alert: {row['selected_alert']}")
        if section["review_shortlist"]:
            lines.append("")
            lines.append("### Review")
        for row in section["review_shortlist"]:
            flags = row["active_flags"]
            lines.append(
                f"- {row['case_id']} | {row['gt_action_id']} | score={row['seed_score']} | diff={row['difficulty_score']} | bucket={row['bucket_id']} | flags={','.join(flags) if flags else 'none'}"
            )
            lines.append(f"  alert: {row['selected_alert']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = build_shortlist()
    write_outputs(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
