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
    REBUILD_ROOT
    / "analysis"
    / "rq3_hadoop_control_anchor_opportunities_v1_20260319"
    / "rq3_hadoop_control_anchor_opportunities_v1_20260319.json"
)
DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "specs"
DEFAULT_ACTIONS = (
    "HADOOP_RESTORE_WORKER_RPC_AND_RETRY,"
    "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    ap.add_argument("--actions", type=str, default=DEFAULT_ACTIONS)
    ap.add_argument("--top-k-per-action", type=int, default=6)
    ap.add_argument("--max-directness", type=float, default=4.0)
    ap.add_argument("--benchmark-id", type=str, default="rq3_hadoop_control_reanchor_bulk_v1_20260319")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return ap.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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

    for action_id, rows in payload["actions"].items():
        if wanted_actions and action_id not in wanted_actions:
            continue
        for row in rows:
            best = row.get("best_alternative") or {}
            if not best:
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
                abs(int((row.get("best_alternative") or {}).get("offset", 999))),
                str(row.get("case_id", "")),
            ),
        )
        for idx, row in enumerate(ranked[: int(args.top_k_per_action)], start=1):
            best = row["best_alternative"]
            case_id = str(row["case_id"])
            chosen.append(
                {
                    "case_id": case_id,
                    "eval_case_id": f"{case_id}__ctrl{idx:02d}",
                    "source": str(row.get("pool_source", "") or "benchmark_v2"),
                    "gt_family_id": str(family_for_action("Hadoop", action_id) or ""),
                    "gt_action_id": action_id,
                    "alert_match": str(best["line"]),
                    "eligibility_note": (
                        "auto_reanchor from hadoop_control_anchor_opportunities: "
                        f"current_directness={row['current_metrics']['directness_score']} "
                        f"alt_directness={best['directness_score']} "
                        f"offset={best['offset']}"
                    ),
                }
            )

    spec = {
        "benchmark_id": args.benchmark_id,
        "benchmark_kind": "hadoop_control_reanchor_bulk",
        "purpose": "Hadoop control-link reanchor bulk built from weaker nearby control-anchor opportunities.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": str(REBUILD_ROOT / "configs" / "contract_v1_20260318.json"),
        "datasets": {
            "Hadoop": chosen,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.benchmark_id}.json"
    write_json(output_path, spec)
    print(output_path)
    print(f"Hadoop cases={len(chosen)} actions={len({item['gt_action_id'] for item in chosen})}")
    for action_id in sorted({item['gt_action_id'] for item in chosen}):
        count = sum(1 for item in chosen if item["gt_action_id"] == action_id)
        print(f"{action_id} {count}")


if __name__ == "__main__":
    main()
