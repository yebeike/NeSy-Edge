"""
Compare modified DYNOTEARS vs original DYNOTEARS on a small subset of benchmark cases.

Design:
- Sample 20 cases per dataset (HDFS / OpenStack / Hadoop), total 60 cases.
- For each case, compute Dim2 sparsity & rank using:
  - Modified DYNOTEARS graph: gt_causal_knowledge_dynotears.json
  - Original DYNOTEARS graph: gt_causal_knowledge_dynotears_original_20260312.json
- Use _calc_causal_sparsity_and_rank to stay consistent with main pipeline.
- Print a compact table summarizing:
  - Sparsity_mean (per dataset, per graph)
  - Avg_Rank (only rank>=0) (per dataset, per graph)

This script is offline-only; do NOT use DeepSeek / LLM.
"""

import os
import sys
import json
import random
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from experiments.rq123_e2e.run_rq123_e2e_massive import _calc_causal_sparsity_and_rank  # type: ignore

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")

DY_MODIFIED_PATH = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
DY_ORIGINAL_PATH = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears_original_20260312.json")


def _domain_from_dataset(dataset: str) -> str:
    ds = (dataset or "").upper()
    if ds == "HDFS":
        return "hdfs"
    if ds == "OPENSTACK":
        return "openstack"
    if ds == "HADOOP":
        return "hadoop"
    return "hdfs"


def _load_kb(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_cases(cases: List[Dict[str, object]], per_ds: int = 20) -> List[Dict[str, object]]:
    by_ds: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds in by_ds:
            by_ds[ds].append(c)
    sampled: List[Dict[str, object]] = []
    for ds, lst in by_ds.items():
        if not lst:
            continue
        if len(lst) <= per_ds:
            sampled.extend(lst)
        else:
            sampled.extend(random.sample(lst, per_ds))
    return sampled


def main() -> None:
    random.seed(2026)

    cases = _load_benchmark(BENCH_V2_PATH)
    if not cases:
        print("[ERROR] No cases in benchmark.")
        return

    mini_cases = _sample_cases(cases, per_ds=20)
    print(f"[*] Sampled {len(mini_cases)} cases (target 20 per dataset).")

    kb_mod = _load_kb(DY_MODIFIED_PATH)
    kb_orig = _load_kb(DY_ORIGINAL_PATH)

    # stats[(dataset, version)] = {'s_sum':..., 'r_sum':..., 'r_cnt':..., 'n':...}
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        for ver in ["modified", "original"]:
            stats[(ds, ver)] = {"s_sum": 0.0, "r_sum": 0.0, "r_cnt": 0.0, "n": 0.0}

    pbar = tqdm(total=len(mini_cases) * 2, desc="Compare DYNOTEARS (modified vs original)", unit="case")

    for case in mini_cases:
        dataset = str(case.get("dataset", "HDFS"))
        gt_tpl = str(case.get("ground_truth_template", "") or "")
        gt_root = str(case.get("ground_truth_root_cause_template", "") or "")
        if not gt_tpl or not gt_root:
            # skip cases without proper GT
            continue
        domain = _domain_from_dataset(dataset)

        # Modified
        s_mod, r_mod = _calc_causal_sparsity_and_rank(kb_mod, domain, gt_root, gt_tpl)
        key_mod = (dataset, "modified")
        stats[key_mod]["n"] += 1
        stats[key_mod]["s_sum"] += float(s_mod)
        if isinstance(r_mod, (int, float)) and r_mod >= 0:
            stats[key_mod]["r_sum"] += float(r_mod)
            stats[key_mod]["r_cnt"] += 1
        pbar.update(1)

        # Original
        s_org, r_org = _calc_causal_sparsity_and_rank(kb_orig, domain, gt_root, gt_tpl)
        key_org = (dataset, "original")
        stats[key_org]["n"] += 1
        stats[key_org]["s_sum"] += float(s_org)
        if isinstance(r_org, (int, float)) and r_org >= 0:
            stats[key_org]["r_sum"] += float(r_org)
            stats[key_org]["r_cnt"] += 1
        pbar.update(1)

    pbar.close()

    print("\n=== DYNOTEARS Modified vs Original (Sparsity & Avg_Rank, mini 60-case subset) ===")
    header = "| Dataset | Version   | #Cases | Sparsity_mean | Avg_Rank (rank>=0) |"
    sep = "|---------|----------|--------|----------------|---------------------|"
    print(header)
    print(sep)
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        for ver in ["modified", "original"]:
            st = stats[(ds, ver)]
            n = st["n"] or 1.0
            s_mean = st["s_sum"] / n
            if st["r_cnt"] > 0:
                r_mean = st["r_sum"] / st["r_cnt"]
            else:
                r_mean = float("nan")
            print(
                f"| {ds} | {ver:8} | {int(st['n']):6d} | {s_mean:14.2f} | {r_mean:19.2f} |"
            )


if __name__ == "__main__":
    main()

