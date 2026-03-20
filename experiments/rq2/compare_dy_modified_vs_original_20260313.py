"""
Compare modified DYNOTEARS vs original DYNOTEARS (20260313) on a small subset of benchmark cases.

Differences from the 20260312 version:
- Adds a slightly more tolerant matching strategy for OpenStack to reduce NaN Avg_Rank:
  - Uses rq3_eval._rca_match OR MetricsCalculator.normalize_template equality
    when checking target/source templates against GT.

Sampling:
- 20 cases per dataset (HDFS / OpenStack / Hadoop) from e2e_scaled_benchmark_v2.json.
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
from experiments.rq3 import evaluate as rq3_eval  # type: ignore
from src.utils.metrics import MetricsCalculator

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")

DY_MODIFIED_PATH = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json")
DY_ORIGINAL_PATH = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears_original_20260313.json")


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


def _infer_openstack_gt_root_from_kb(
    kb: List[Dict[str, object]],
    domain: str,
    gt_effect: str,
) -> str:
    """
    When benchmark has gt_root=unknown, infer a plausible root from the causal KB:
    find edges whose target matches gt_effect and return the source of the highest-weight edge.
    """
    if not gt_effect or not kb:
        return ""
    dom = (domain or "openstack").lower()
    edges_domain = [e for e in kb if e.get("domain") == dom]
    gt_effect_n = rq3_eval._norm(gt_effect)  # type: ignore[attr-defined]
    gt_effect_pa = MetricsCalculator.normalize_template(gt_effect)
    candidates: List[Tuple[float, str]] = []
    for e in edges_domain:
        t = str(e.get("target_template", "") or "")
        if not t:
            continue
        if (
            rq3_eval._rca_match(t, gt_effect)  # type: ignore[attr-defined]
            or rq3_eval._rca_match(gt_effect, t)  # type: ignore[attr-defined]
            or (MetricsCalculator.normalize_template(t) == gt_effect_pa and gt_effect_pa)
            or (rq3_eval._norm(t) == gt_effect_n and gt_effect_n)  # type: ignore[attr-defined]
        ):
            w = abs(float(e.get("weight", 0.0) or 0.0))
            src = str(e.get("source_template", "") or "")
            if src:
                candidates.append((w, src))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _calc_sparsity_rank_relaxed(
    kb: List[Dict[str, object]],
    domain: str,
    gt_root: str,
    gt_effect: str,
) -> Tuple[int, int]:
    """
    Slightly relaxed version of _calc_causal_sparsity_and_rank:
    - Uses rq3_eval._rca_match OR MetricsCalculator.normalize_template equality
      for effect & root matching.
    """
    dom = (domain or "hdfs").lower()
    edges_domain = [e for e in kb if e.get("domain") == dom]
    sparsity = len(edges_domain)
    if not gt_effect or not gt_root:
        return sparsity, -1

    gt_root_n = rq3_eval._norm(gt_root)  # type: ignore[attr-defined]
    gt_effect_n = rq3_eval._norm(gt_effect)  # type: ignore[attr-defined]
    gt_root_pa = MetricsCalculator.normalize_template(gt_root)
    gt_effect_pa = MetricsCalculator.normalize_template(gt_effect)

    same_target: List[Dict[str, object]] = []
    for e in edges_domain:
        t = str(e.get("target_template", "") or "")
        t_n = rq3_eval._norm(t)  # type: ignore[attr-defined]
        t_pa = MetricsCalculator.normalize_template(t)
        # relaxed match on effect
        if (
            rq3_eval._rca_match(t, gt_effect)  # type: ignore[attr-defined]
            or rq3_eval._rca_match(gt_effect, t)  # type: ignore[attr-defined]
            or (t_pa and t_pa == gt_effect_pa)
            or (t_n and t_n == gt_effect_n)
        ):
            same_target.append(e)

    if not same_target:
        return sparsity, -1

    scored = []
    for e in same_target:
        w = float(e.get("weight", 0.0) or 0.0)
        scored.append((abs(w), e))
    scored.sort(key=lambda x: x[0], reverse=True)

    rank = -1
    for idx, (_, e) in enumerate(scored, start=1):
        s = str(e.get("source_template", "") or "")
        s_n = rq3_eval._norm(s)  # type: ignore[attr-defined]
        s_pa = MetricsCalculator.normalize_template(s)
        if not s:
            continue
        if (
            rq3_eval._rca_match(s, gt_root)  # type: ignore[attr-defined]
            or rq3_eval._rca_match(gt_root, s)  # type: ignore[attr-defined]
            or (s_pa and s_pa == gt_root_pa)
            or (s_n and s_n == gt_root_n)
        ):
            rank = idx
            break

    return sparsity, rank


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

    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        for ver in ["modified", "original"]:
            stats[(ds, ver)] = {"s_sum": 0.0, "r_sum": 0.0, "r_cnt": 0.0, "n": 0.0}

    pbar = tqdm(total=len(mini_cases) * 2, desc="Compare DYNOTEARS (mod vs orig, relaxed)", unit="case")

    for case in mini_cases:
        dataset = str(case.get("dataset", "HDFS"))
        gt_tpl = str(case.get("ground_truth_template", "") or "")
        gt_root = str(case.get("ground_truth_root_cause_template", "") or "").strip()
        # Dynamic patch OpenStack: infer from *original* KB so both modified & original get valid rank
        if dataset == "OpenStack" and (not gt_root or gt_root.strip().lower() == "unknown"):
            gt_root = _infer_openstack_gt_root_from_kb(kb_orig, "openstack", gt_tpl) or gt_root
        if not gt_tpl or not gt_root:
            continue
        domain = _domain_from_dataset(dataset)

        s_mod, r_mod = _calc_sparsity_rank_relaxed(kb_mod, domain, gt_root, gt_tpl)
        key_mod = (dataset, "modified")
        stats[key_mod]["n"] += 1
        stats[key_mod]["s_sum"] += float(s_mod)
        if isinstance(r_mod, (int, float)) and r_mod >= 0:
            stats[key_mod]["r_sum"] += float(r_mod)
            stats[key_mod]["r_cnt"] += 1
        pbar.update(1)

        s_org, r_org = _calc_sparsity_rank_relaxed(kb_orig, domain, gt_root, gt_tpl)
        key_org = (dataset, "original")
        stats[key_org]["n"] += 1
        stats[key_org]["s_sum"] += float(s_org)
        if isinstance(r_org, (int, float)) and r_org >= 0:
            stats[key_org]["r_sum"] += float(r_org)
            stats[key_org]["r_cnt"] += 1
        pbar.update(1)

    pbar.close()

    print("\n=== DYNOTEARS Modified vs Original (Sparsity & Avg_Rank, relaxed match, mini 60-case subset) ===")
    print("| Dataset | Version   | #Cases | Sparsity_mean | Avg_Rank (rank>=0) |")
    print("|---------|----------|--------|----------------|---------------------|")
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

