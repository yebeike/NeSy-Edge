from __future__ import annotations

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

from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils
from experiments.thesis_rebuild_20260315.rq34.scripts.audit_rq3_candidate_pools_20260318 import _direct_alert_flags
from experiments.thesis_rebuild_20260315.rq34.scripts.frozen_catalog_v2_20260318 import (
    ACTION_CATALOG,
    infer_action_id_from_text,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


OUTPUT_DIR = REBUILD_ROOT / "analysis" / "rq3_hadoop_control_anchor_opportunities_v1_20260319"
WINDOW = 6
TARGET_ACTIONS = {
    "HADOOP_RESTORE_WORKER_RPC_AND_RETRY",
    "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY",
}


def norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def tokenize(text: str) -> List[str]:
    raw = norm(text).replace("/", " ").replace(":", " ").replace("-", " ")
    out: List[str] = []
    for token in raw.split():
        clean = "".join(ch for ch in token if ch.isalnum() or ch == "_")
        if len(clean) >= 3:
            out.append(clean)
    return out


def overlap_score(action_id: str, text: str) -> float:
    meta = ACTION_CATALOG.get("Hadoop", {}).get(action_id, {})
    text_tokens = set(tokenize(text))
    desc_tokens = set(tokenize(str(meta.get("description", ""))))
    overlap = len(text_tokens & desc_tokens)
    lowered = norm(text)
    group_hits = 0
    for group in meta.get("keyword_groups", []):
        if any(norm(token) in lowered for token in group):
            group_hits += 1
    return round(overlap + 2.5 * group_hits, 4)


def line_metrics(action_id: str, text: str) -> Dict[str, object]:
    inferred = infer_action_id_from_text("Hadoop", text)
    flags = _direct_alert_flags("Hadoop", text)
    overlap = overlap_score(action_id, text)
    direct_penalty = sum(int(bool(v)) for v in flags.values())
    score = round((8.0 if inferred == action_id else 0.0) + overlap + 5.0 * direct_penalty, 4)
    return {
        "inferred_action_id": inferred,
        "overlap_score": overlap,
        "direct_flags": flags,
        "directness_score": score,
    }


def main() -> None:
    rows = [
        row
        for row in utils.candidate_audit_rows()
        if str(row.get("dataset", "")) == "Hadoop"
        and utils.canonical_action_id("Hadoop", str(row.get("gt_action_id", "") or "")) in TARGET_ACTIONS
    ]
    by_action: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in rows:
        action_id = utils.canonical_action_id("Hadoop", str(row.get("gt_action_id", "") or ""))
        source = str(row.get("pool_source", "") or "")
        case_id = str(row.get("case_id", "") or "")
        selected_alert = str(row.get("selected_alert", "") or "")
        raw_log = str(utils.pool_rows()[(source, case_id)].get("raw_log", "") or "")
        lines = [line.strip() for line in raw_log.splitlines() if line.strip()]
        wanted = norm(selected_alert)
        hit_index = next((i for i, line in enumerate(lines) if wanted == norm(line) or wanted in norm(line)), -1)
        if hit_index < 0:
            continue
        current = line_metrics(action_id, selected_alert)
        candidates: List[Dict[str, object]] = []
        for idx in range(max(0, hit_index - WINDOW), min(len(lines), hit_index + WINDOW + 1)):
            if idx == hit_index:
                continue
            line = lines[idx]
            metrics = line_metrics(action_id, line)
            candidates.append(
                {
                    "offset": idx - hit_index,
                    "line": line,
                    "directness_reduction": round(float(current["directness_score"]) - float(metrics["directness_score"]), 4),
                    **metrics,
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["directness_score"]),
                len([k for k, v in item["direct_flags"].items() if v]),
                -float(item["directness_reduction"]),
                abs(int(item["offset"])),
                item["line"],
            )
        )
        best = candidates[0] if candidates else None
        by_action[action_id].append(
            {
                "case_id": case_id,
                "pool_source": source,
                "selected_alert": selected_alert,
                "current_metrics": current,
                "best_alternative": best,
                "candidate_preview": candidates[:6],
            }
        )

    payload = {
        "purpose": "Mine nearby weaker Hadoop control-link anchors for worker/RM retry cases under the rebuilt RQ3 contract.",
        "window": WINDOW,
        "actions": dict(sorted(by_action.items())),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "rq3_hadoop_control_anchor_opportunities_v1_20260319.json"
    write_json(output_path, payload)
    print(output_path)
    for action_id, items in sorted(by_action.items()):
        print(action_id, len(items))
        for item in items[:5]:
            best = item.get("best_alternative") or {}
            print(
                item["case_id"],
                "current=",
                item["current_metrics"]["directness_score"],
                "best=",
                best.get("directness_score"),
                "offset=",
                best.get("offset"),
            )


if __name__ == "__main__":
    main()
