from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import family_for_action


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "thesis_rebuild_20260315"
    / "rq34"
    / "analysis"
    / "rq3_small_v3_diagnostic_slice_20260318"
    / "rq3_small_v3_pool_anchor_opportunities_20260318.json"
)
DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "specs"
DEFAULT_ACTIONS = (
    "OPENSTACK_REBUILD_ON_COMPATIBLE_HOST,"
    "OPENSTACK_REPAIR_BASE_IMAGE_CHAIN,"
    "OPENSTACK_RESYNC_INSTANCE_INVENTORY,"
    "OPENSTACK_RESYNC_POWER_STATE_AND_REBUILD,"
    "OPENSTACK_SCALE_METADATA_SERVICE,"
    "OPENSTACK_VERIFY_TERMINATION_AND_RESTORE"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    ap.add_argument("--actions", type=str, default=DEFAULT_ACTIONS)
    ap.add_argument("--top-k-per-action", type=int, default=3)
    ap.add_argument("--max-directness", type=float, default=1.0)
    ap.add_argument("--include-blacklisted", action="store_true")
    ap.add_argument("--benchmark-id", type=str, default="rq3_openstack_reanchor_bulk_v1_20260319")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes"}


def as_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    args = parse_args()
    payload = load_json(args.report_path)
    wanted_actions = {token.strip() for token in str(args.actions).split(",") if token.strip()}
    per_action: Dict[str, List[Mapping[str, object]]] = defaultdict(list)

    for action_id, rows in payload["datasets"]["OpenStack"]["actions"].items():
        if wanted_actions and action_id not in wanted_actions:
            continue
        for row in rows:
            best = row.get("best_alternative") or {}
            if not best:
                continue
            if (not args.include_blacklisted) and as_bool(row.get("blacklisted", False)):
                continue
            directness = as_float(best.get("directness_score", 999.0), 999.0)
            if directness > float(args.max_directness):
                continue
            per_action[action_id].append(row)

    chosen: List[Dict[str, object]] = []
    for action_id, rows in sorted(per_action.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                as_float((row.get("best_alternative") or {}).get("directness_score", 999.0), 999.0),
                -float(row.get("difficulty_score", 0.0) or 0.0),
                -float(row.get("selection_score", 0.0) or 0.0),
                str(row.get("case_id", "")),
            ),
        )
        for idx, row in enumerate(ranked[: int(args.top_k_per_action)], start=1):
            best = row["best_alternative"]
            case_id = str(row["case_id"])
            chosen.append(
                {
                    "case_id": case_id,
                    "eval_case_id": f"{case_id}__reanchor{idx:02d}",
                    "source": str(row.get("pool_source", "") or ""),
                    "gt_family_id": str(family_for_action("OpenStack", action_id) or ""),
                    "gt_action_id": action_id,
                    "alert_match": str(best["line"]),
                    "eligibility_note": (
                        "auto_reanchor from pool_anchor_opportunities: "
                        f"current_directness={row['current_metrics']['directness_score']} "
                        f"alt_directness={best['directness_score']} "
                        f"offset={best['offset']} "
                        f"selection_score={float(row.get('selection_score', 0.0) or 0.0):.1f} "
                        f"difficulty_score={float(row.get('difficulty_score', 0.0) or 0.0):.1f}"
                    ),
                }
            )

    spec = {
        "benchmark_id": args.benchmark_id,
        "benchmark_kind": "openstack_reanchor_bulk",
        "purpose": "OpenStack reanchor bulk built from nearby weaker alert opportunities to expand the mid-first RQ3 pool.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": str(REBUILD_ROOT / "configs" / "contract_v1_20260318.json"),
        "datasets": {
            "OpenStack": chosen,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.benchmark_id}.json"
    write_json(output_path, spec)
    print(output_path)
    print(f"OpenStack cases={len(chosen)} actions={len({item['gt_action_id'] for item in chosen})}")
    for action_id in sorted({item['gt_action_id'] for item in chosen}):
        count = sum(1 for item in chosen if item["gt_action_id"] == action_id)
        print(f"{action_id} {count}")


if __name__ == "__main__":
    main()
