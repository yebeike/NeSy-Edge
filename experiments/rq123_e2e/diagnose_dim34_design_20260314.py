"""
Offline diagnosis for Dim3/Dim4 experimental design issues.

What it checks:
- How many benchmark cases are actually labelable / rankable per dataset.
- Why OpenStack collapses to one label in the current benchmark.
- Why Hadoop agent candidates do not contain the benchmark GT root templates.
- Why "first 10 cases per dataset" sampling hides method differences.
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e.stage4_noise_api_sampled_20260313 import (  # type: ignore
    BENCH_V2_PATH,
    CAUSAL_KB_DYNOTEARS,
    _map_label_to_sop_id,
    _sample_cases,
    gt_label_for_case,
)
from experiments.rq3 import tools as rq3_tools  # type: ignore


def _load_cases() -> List[Dict[str, object]]:
    with open(BENCH_V2_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _label_dist(cases: List[Dict[str, object]]) -> Dict[str, int]:
    return dict(Counter(gt_label_for_case(c) for c in cases))


def _print_dataset_overview(cases: List[Dict[str, object]]) -> None:
    print("=== Benchmark Coverage ===")
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        ds_cases = [c for c in cases if str(c.get("dataset", "")) == ds]
        known_root = [
            c
            for c in ds_cases
            if str(c.get("ground_truth_root_cause_template", "") or "").strip().lower() not in ("", "unknown")
        ]
        sop_covered = sum(1 for c in ds_cases if _map_label_to_sop_id(ds, gt_label_for_case(c)))
        print(
            f"{ds}: total={len(ds_cases)}, known_root={len(known_root)}, "
            f"sop_mappable={sop_covered}, label_dist={_label_dist(ds_cases)}"
        )


def _print_sampling_comparison(cases: List[Dict[str, object]]) -> None:
    print("\n=== Sampling Check ===")
    sampled = _sample_cases()
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        ds_cases = [c for c in cases if str(c.get("dataset", "")) == ds]
        first10 = ds_cases[:10]
        balanced10 = [c for c in sampled if str(c.get("dataset", "")) == ds]
        print(f"{ds}: first10={_label_dist(first10)} | balanced10={_label_dist(balanced10)}")


def _print_openstack_graph_coverage(cases: List[Dict[str, object]]) -> None:
    print("\n=== OpenStack Graph Coverage ===")
    with open(CAUSAL_KB_DYNOTEARS, "r", encoding="utf-8") as f:
        kb = json.load(f)
    targets = {
        str(e.get("target_template", "") or "").strip()
        for e in kb
        if str(e.get("domain", "")).lower() == "openstack"
    }
    ds_cases = [c for c in cases if str(c.get("dataset", "")) == "OpenStack"]
    covered = 0
    tpl_dist = Counter()
    for c in ds_cases:
        tpl = str(c.get("ground_truth_template", "") or "").strip()
        tpl_dist[tpl] += 1
        if tpl in targets:
            covered += 1
    print(f"OpenStack gt_template covered by DYNOTEARS target set: {covered}/{len(ds_cases)}")
    print(f"OpenStack gt_template distribution: {dict(tpl_dist)}")


def _print_hadoop_candidate_hit(cases: List[Dict[str, object]]) -> None:
    print("\n=== Hadoop Candidate Hit ===")
    known = [
        c
        for c in cases
        if str(c.get("dataset", "")) == "Hadoop"
        and str(c.get("ground_truth_root_cause_template", "") or "").strip().lower() not in ("", "unknown")
    ]
    nonempty = 0
    exact_hit = 0
    for c in known:
        gt_tpl = str(c.get("ground_truth_template", "") or "")
        gt_root = str(c.get("ground_truth_root_cause_template", "") or "")
        cand = json.loads(rq3_tools.causal_navigator(gt_tpl, "hadoop", causal_path=CAUSAL_KB_DYNOTEARS))
        if isinstance(cand, list) and cand:
            nonempty += 1
            if any((x.get("source_template") or "").strip() == gt_root.strip() for x in cand):
                exact_hit += 1
    print(f"Hadoop known-root cases: {len(known)}")
    print(f"Hadoop non-empty candidate lists: {nonempty}/{len(known)}")
    print(f"Hadoop exact GT root in candidate list: {exact_hit}/{len(known)}")


def main() -> None:
    cases = _load_cases()
    _print_dataset_overview(cases)
    _print_sampling_comparison(cases)
    _print_openstack_graph_coverage(cases)
    _print_hadoop_candidate_hit(cases)


if __name__ == "__main__":
    main()
