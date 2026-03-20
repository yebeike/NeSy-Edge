from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--benchmark-id", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    out: Dict[str, object] = dict(spec)
    out["benchmark_id"] = args.benchmark_id
    out["benchmark_kind"] = "rq3_core_subset_local"
    out["purpose"] = "RQ3 core subset derived from relaxed full spec"

    datasets: Dict[str, List[Dict[str, object]]] = {}
    for dataset, items in spec["datasets"].items():
        kept: List[Dict[str, object]] = []
        seen_base: set[str] = set()
        for item in items:
            quality = str(item.get("quality_tier", "") or "")
            if quality not in {"core_hard", "core_usable"}:
                continue
            base_id = str(item.get("base_incident_id", "") or item.get("case_id", ""))
            if base_id in seen_base:
                continue
            seen_base.add(base_id)
            kept.append(dict(item))
        datasets[dataset] = kept
    out["datasets"] = datasets
    write_json(args.output, out)
    print(args.output)
    print(
        json.dumps(
            {dataset: len(items) for dataset, items in datasets.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
