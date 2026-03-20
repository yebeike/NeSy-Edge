from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def base_case_id(text: str) -> str:
    return re.sub(r"__[^_]+$", "", str(text))


BASE_SPEC = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/specs/rq3_mid_seed_v1_filtered_from_graphfix_20260319.json"
)
OPENSTACK_RANKED = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/analysis/local_probe_openstack_reanchor_bulk_v2_highnoise_20260319/rq3_openstack_reanchor_bulk_v2_highnoise_open_20260319_ranked_graphfix.json"
)
OPENSTACK_SPEC = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/specs/rq3_openstack_reanchor_bulk_v2_20260319.json"
)
HDFS_RANKED = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/analysis/local_probe_hdfs_reanchor_bulk_v1_highnoise_20260319/rq3_hdfs_reanchor_bulk_v1_highnoise_open_20260319_ranked_graphfix.json"
)
HDFS_SPEC = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/specs/rq3_hdfs_reanchor_bulk_v1_20260319.json"
)
HADOOP_RANKED = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/analysis/local_probe_hadoop_control_reanchor_bulk_v1_highnoise_20260319/rq3_hadoop_control_reanchor_bulk_v1_highnoise_open_20260319_ranked_graphfix.json"
)
HADOOP_SPEC = Path(
    "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/specs/rq3_hadoop_control_reanchor_bulk_v1_20260319.json"
)


DROP_EVAL_CASE_IDS = {
    ("Hadoop", "hadoop_application_1445062781478_0012"),
    ("Hadoop", "hadoop_application_1445062781478_0014"),
}

ADD_EVAL_CASE_IDS = {
    ("OpenStack", "openstack_49__reanchor02"),
    ("HDFS", "hdfs_20__reanchor02"),
    ("Hadoop", "hadoop_application_1445175094696_0004__ctrl07"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-id", type=str, default="rq3_mid_seed_v3_mixed_20260319")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/peihanye/Desktop/Projects/NuSy-Edge/experiments/thesis_rebuild_20260315/rq3_rebuild_v1/specs/rq3_mid_seed_v3_mixed_20260319.json"
        ),
    )
    return ap.parse_args()


def load_spec_lookup(path: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    obj = load_json(path)
    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for dataset, items in obj["datasets"].items():
        for item in items:
            key = (str(dataset), str(item.get("eval_case_id", item["case_id"])))
            out[key] = {
                "case_id": str(item["case_id"]),
                "eval_case_id": str(item.get("eval_case_id", item["case_id"])),
                "source": str(item.get("source", "") or ""),
                "gt_family_id": str(item.get("gt_family_id", "") or ""),
                "gt_action_id": str(item.get("gt_action_id", "") or ""),
                "alert_match": str(item.get("alert_match", "") or ""),
                "eligibility_note": str(item.get("eligibility_note", "") or ""),
            }
    return out


def main() -> None:
    args = parse_args()
    base = load_json(BASE_SPEC)
    lookups = {}
    for path in (OPENSTACK_SPEC, HDFS_SPEC, HADOOP_SPEC):
        lookups.update(load_spec_lookup(path))

    datasets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for dataset, items in base["datasets"].items():
        for item in items:
            key = (dataset, str(item.get("eval_case_id", item["case_id"])))
            if key in DROP_EVAL_CASE_IDS:
                continue
            datasets[dataset].append(dict(item))

    existing = {(dataset, item["eval_case_id"]) for dataset, items in datasets.items() for item in items}
    added: List[Dict[str, str]] = []
    for key in sorted(ADD_EVAL_CASE_IDS):
        if key in existing:
            continue
        item = lookups.get(key)
        if item is None:
            continue
        datasets[key[0]].append(item)
        added.append({"dataset": key[0], "eval_case_id": key[1]})

    payload = {
        "benchmark_id": args.benchmark_id,
        "benchmark_kind": "mid_seed_candidate_pool",
        "purpose": "Mixed mid seed built from graphfix-filtered pool plus new hard shortlist replacements.",
        "formal_small_ready": False,
        "paid_api_allowed": False,
        "contract_path": base["contract_path"],
        "datasets": dict(datasets),
        "filter_report": {
            "base_spec": str(BASE_SPEC),
            "dropped_eval_case_ids": [
                {"dataset": dataset, "eval_case_id": case_id}
                for dataset, case_id in sorted(DROP_EVAL_CASE_IDS)
            ],
            "added_eval_case_ids": added,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({k: len(v) for k, v in datasets.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
