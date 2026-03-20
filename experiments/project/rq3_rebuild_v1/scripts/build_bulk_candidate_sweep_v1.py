from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import family_for_action


DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "specs"
DEFAULT_SKIP_FLAGS = {
    "HDFS": ["direct_serving_exception", "direct_allocate_block", "direct_delete_failure"],
    "Hadoop": ["direct_retry_connect", "direct_forced_close", "direct_delete_hdfs", "direct_storage"],
    "OpenStack": ["direct_power_sync", "direct_host_claim", "direct_image_cache", "direct_metadata"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--top-k-per-action", type=int, default=6)
    ap.add_argument("--max-total", type=int, default=0)
    ap.add_argument("--min-selection-score", type=float, default=0.0)
    ap.add_argument("--actions", type=str, default="")
    ap.add_argument("--keep-direct", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--benchmark-id", type=str, default="")
    return ap.parse_args()


def chosen_actions(dataset: str, raw: str) -> set[str]:
    if raw.strip():
        return {utils.canonical_action_id(dataset, token.strip()) for token in raw.split(",") if token.strip()}
    return set()


def direct_flag_penalty(flags: Mapping[str, object], skip_flags: Sequence[str]) -> int:
    return sum(1 for name in skip_flags if bool(flags.get(name)))


def make_note(row: Mapping[str, object], penalty: int) -> str:
    flags = [name for name, value in (row.get("selected_alert_flags") or {}).items() if bool(value)]
    parts = []
    reason = str(row.get("reason", "") or "").strip()
    if reason:
        parts.append(f"reason={reason}")
    parts.append(f"selection_score={float(row.get('selection_score', 0.0) or 0.0):.1f}")
    parts.append(f"difficulty_score={float(row.get('difficulty_score', 0.0) or 0.0):.1f}")
    parts.append(f"weak_mainline={bool(row.get('weak_mainline_alert', False))}")
    parts.append(f"fixed_small_member={bool(row.get('fixed_small_member', False))}")
    parts.append(f"direct_flag_penalty={penalty}")
    if flags:
        parts.append("flags=" + ",".join(sorted(flags)))
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset).strip()
    if dataset not in {"HDFS", "OpenStack", "Hadoop"}:
        raise SystemExit(f"Unsupported dataset: {dataset}")

    action_filter = chosen_actions(dataset, args.actions)
    skip_flags = [] if args.keep_direct else list(DEFAULT_SKIP_FLAGS.get(dataset, []))
    candidates: Dict[str, Dict[tuple[str, str], Mapping[str, object]]] = defaultdict(dict)

    for row in utils.candidate_audit_rows():
        if str(row.get("dataset", "")) != dataset:
            continue
        action_id = utils.canonical_action_id(dataset, str(row.get("gt_action_id", "") or ""))
        if not action_id:
            continue
        if action_filter and action_id not in action_filter:
            continue
        selected_alert = str(row.get("selected_alert", "") or "").strip()
        if not selected_alert:
            continue
        score = float(row.get("selection_score", 0.0) or 0.0)
        if score < float(args.min_selection_score):
            continue
        flags = dict(row.get("selected_alert_flags", {}) or {})
        penalty = direct_flag_penalty(flags, skip_flags)
        if penalty > 0:
            continue
        source = str(row.get("pool_source", "") or "")
        case_id = str(row.get("case_id", "") or "")
        item = {
            "case_id": case_id,
            "source": source,
            "gt_family_id": str(family_for_action(dataset, action_id) or row.get("gt_label", "") or ""),
            "gt_action_id": action_id,
            "alert_match": selected_alert,
            "eligibility_note": make_note(row, penalty),
            "selection_score": score,
            "difficulty_score": float(row.get("difficulty_score", 0.0) or 0.0),
            "weak_mainline_alert": bool(row.get("weak_mainline_alert", False)),
            "fixed_small_member": bool(row.get("fixed_small_member", False)),
            "selected_alert_flags": flags,
            "action_bucket": str(row.get("action_bucket", "") or ""),
        }
        key = (source, case_id)
        existing = candidates[action_id].get(key)
        if existing is None:
            candidates[action_id][key] = item
            continue
        replace = (
            int(bool(item.get("weak_mainline_alert", False))),
            float(item.get("selection_score", 0.0) or 0.0),
            float(item.get("difficulty_score", 0.0) or 0.0),
            len(str(item.get("alert_match", "") or "")),
        ) > (
            int(bool(existing.get("weak_mainline_alert", False))),
            float(existing.get("selection_score", 0.0) or 0.0),
            float(existing.get("difficulty_score", 0.0) or 0.0),
            len(str(existing.get("alert_match", "") or "")),
        )
        if replace:
            candidates[action_id][key] = item

    chosen: List[Mapping[str, object]] = []
    for action_id, item_map in sorted(candidates.items()):
        items = list(item_map.values())
        ranked = sorted(
            items,
            key=lambda item: (
                -int(bool(item.get("weak_mainline_alert", False))),
                -float(item.get("selection_score", 0.0) or 0.0),
                -float(item.get("difficulty_score", 0.0) or 0.0),
                str(item.get("source", "")),
                str(item.get("case_id", "")),
            ),
        )
        chosen.extend(ranked[: int(args.top_k_per_action)])

    chosen = sorted(
        chosen,
        key=lambda item: (
            str(item["gt_action_id"]),
            -int(bool(item.get("weak_mainline_alert", False))),
            -float(item.get("selection_score", 0.0) or 0.0),
            str(item["source"]),
            str(item["case_id"]),
        ),
    )
    if int(args.max_total) > 0:
        chosen = chosen[: int(args.max_total)]

    case_id_counts = defaultdict(int)
    for item in chosen:
        case_id_counts[str(item["case_id"])] += 1
    case_id_seen = defaultdict(int)
    for item in chosen:
        base_case_id = str(item["case_id"])
        if case_id_counts[base_case_id] <= 1:
            item["eval_case_id"] = base_case_id
            continue
        case_id_seen[base_case_id] += 1
        item["eval_case_id"] = f"{base_case_id}__{case_id_seen[base_case_id]:02d}"

    benchmark_id = (
        str(args.benchmark_id).strip()
        or f"rq3_{dataset.lower()}_bulk_candidate_sweep_v1_20260319"
    )
    spec = {
        "benchmark_id": benchmark_id,
        "benchmark_kind": f"{dataset.lower()}_bulk_candidate_sweep",
        "purpose": f"Bulk local screening sweep for {dataset} candidate cases under the rebuilt RQ3 contract.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": str(utils.CONTRACT_PATH),
        "datasets": {
            dataset: chosen,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{benchmark_id}.json"
    write_json(output_path, spec)
    print(output_path)
    print(f"dataset={dataset} cases={len(chosen)} actions={len({item['gt_action_id'] for item in chosen})}")
    for action_id in sorted({item["gt_action_id"] for item in chosen}):
        count = sum(1 for item in chosen if item["gt_action_id"] == action_id)
        print(f"{action_id} {count}")


if __name__ == "__main__":
    main()
