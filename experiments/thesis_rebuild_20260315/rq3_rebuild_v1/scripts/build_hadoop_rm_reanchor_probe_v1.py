from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REBUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REBUILD_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.rq3_rebuild_v1.scripts import rebuild_utils_v1 as utils


DEFAULT_OUTPUT_DIR = REBUILD_ROOT / "specs"
TARGET_CASES = [
    "hadoop_application_1445144423722_0020",
    "hadoop_application_1445144423722_0023",
    "hadoop_application_1445175094696_0003",
    "hadoop_application_1445087491445_0002",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-id", type=str, default="rq3_hadoop_rm_reanchor_probe_v1_20260319")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return ap.parse_args()


def find_rm_anchor(raw_log: str) -> str:
    patterns = [
        r"Retrying connect to server: .*:8030\.",
        r"Address change detected\. Old: .*:8030 New: .*:8030",
        r"Connecting to ResourceManager at .*:8030",
    ]
    lines = [line.strip() for line in str(raw_log or "").splitlines() if line.strip()]
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for line in lines:
            if regex.search(line):
                return line
    return ""


def main() -> None:
    chosen: List[Dict[str, object]] = []
    for case_id in TARGET_CASES:
        row = utils.pool_rows().get(("benchmark_v2", case_id))
        if row is None:
            continue
        raw_log = str(row.get("raw_log", "") or "")
        anchor = find_rm_anchor(raw_log)
        if not anchor:
            continue
        chosen.append(
            {
                "case_id": case_id,
                "eval_case_id": f"{case_id}__rm01",
                "source": "benchmark_v2",
                "gt_family_id": "HADOOP_CONTROL_LINK_DISRUPTION",
                "gt_action_id": "HADOOP_RESTORE_RM_CHANNEL_AND_RETRY",
                "alert_match": anchor,
                "eligibility_note": "RM-specific reanchor probe built from :8030 / ResourceManager retry lines.",
            }
        )

    payload = {
        "benchmark_id": args.benchmark_id,
        "benchmark_kind": "hadoop_rm_reanchor_probe",
        "purpose": "Probe RM-channel-specific Hadoop control-link anchors under the rebuilt RQ3 contract.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": str(REBUILD_ROOT / "configs" / "contract_v1_20260318.json"),
        "datasets": {"Hadoop": chosen},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.benchmark_id}.json"
    write_json(output_path, payload)
    print(output_path)
    print(f"Hadoop cases={len(chosen)}")
    for item in chosen:
        print(item["case_id"], item["gt_action_id"])


if __name__ == "__main__":
    args = parse_args()
    main()
