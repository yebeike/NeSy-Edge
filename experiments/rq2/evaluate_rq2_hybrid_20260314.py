"""
Evaluate RQ2 causal graphs on a graph-evaluable hybrid case pool.

Pool design:
- HDFS / OpenStack: prefer rq3_test_set cases because they are richer and more diverse.
- Hadoop: use the benchmark v2 cases that already contain template/root pairs.

Metrics:
- Sparsity: number of edges in the domain graph.
- Avg_Rank: average rank of the true root among edges targeting the GT effect.
"""

import json
import math
import os
import sys
from typing import Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.metrics import MetricsCalculator

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
RQ3_TEST_SET_PATH = os.path.join(DATA_PROCESSED, "rq3_test_set.json")
CAUSAL_SYMBOLIC_PATH = os.path.join(DATA_PROCESSED, "causal_knowledge.json")

GRAPH_PATHS = {
    "modified": os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears.json"),
    "original": os.path.join(DATA_PROCESSED, "gt_causal_knowledge_dynotears_original_20260313.json"),
    "pearson": os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson.json"),
    "pc": os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc.json"),
}


def _load_json_list(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _relaxed_match(a: str, b: str) -> bool:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    pa = MetricsCalculator.normalize_template(a)
    pb = MetricsCalculator.normalize_template(b)
    return bool(pa and pb and pa == pb)


def _domain(dataset: str) -> str:
    return "hdfs" if dataset == "HDFS" else dataset.lower()


def _infer_root_from_kb(kb: List[Dict[str, object]], domain: str, gt_effect: str) -> str:
    candidates: List[Tuple[float, str]] = []
    for edge in kb:
        if str(edge.get("domain", "")).lower() != domain:
            continue
        tgt = str(edge.get("target_template", "") or "")
        src = str(edge.get("source_template", "") or "")
        if not src or not tgt:
            continue
        if _relaxed_match(tgt, gt_effect):
            candidates.append((abs(float(edge.get("weight", 0.0) or 0.0)), src))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _calc_rank(kb: List[Dict[str, object]], domain: str, gt_root: str, gt_effect: str) -> Tuple[int, int]:
    edges_domain = [e for e in kb if str(e.get("domain", "")).lower() == domain]
    sparsity = len(edges_domain)
    if not gt_root or not gt_effect:
        return sparsity, -1

    same_target = [
        e for e in edges_domain if _relaxed_match(str(e.get("target_template", "") or ""), gt_effect)
    ]
    if not same_target:
        return sparsity, -1

    scored = sorted(
        ((abs(float(e.get("weight", 0.0) or 0.0)), e) for e in same_target),
        key=lambda x: x[0],
        reverse=True,
    )
    for idx, (_, edge) in enumerate(scored, start=1):
        if _relaxed_match(str(edge.get("source_template", "") or ""), gt_root):
            return sparsity, idx
    return sparsity, -1


def _build_eval_cases() -> List[Dict[str, str]]:
    symbolic_kb = _load_json_list(CAUSAL_SYMBOLIC_PATH)
    bench = _load_json_list(BENCH_V2_PATH)
    rq3 = _load_json_list(RQ3_TEST_SET_PATH)

    eval_cases: List[Dict[str, str]] = []

    for case in rq3:
        dataset = str(case.get("dataset", "") or "")
        if dataset not in ("HDFS", "OpenStack"):
            continue
        gt_effect = str(case.get("ground_truth_template", "") or "").strip()
        gt_root = str(case.get("ground_truth_root_cause_template", "") or "").strip()
        dom = _domain(dataset)
        if (not gt_root or gt_root.lower() == "unknown") and gt_effect:
            gt_root = _infer_root_from_kb(symbolic_kb, dom, gt_effect)
        if gt_effect and gt_root and gt_root.lower() != "unknown":
            eval_cases.append(
                {
                    "dataset": dataset,
                    "case_id": str(case.get("case_id", "") or case.get("source", "")),
                    "gt_effect": gt_effect,
                    "gt_root": gt_root,
                }
            )

    for case in bench:
        dataset = str(case.get("dataset", "") or "")
        if dataset != "Hadoop":
            continue
        gt_effect = str(case.get("ground_truth_template", "") or "").strip()
        gt_root = str(case.get("ground_truth_root_cause_template", "") or "").strip()
        if gt_effect and gt_root and gt_root.lower() != "unknown":
            eval_cases.append(
                {
                    "dataset": dataset,
                    "case_id": str(case.get("case_id", "") or ""),
                    "gt_effect": gt_effect,
                    "gt_root": gt_root,
                }
            )
    return eval_cases


def main() -> None:
    eval_cases = _build_eval_cases()
    print(f"[*] Built hybrid eval pool: {len(eval_cases)} cases")
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        print(f"    - {ds}: {sum(1 for c in eval_cases if c['dataset'] == ds)}")

    kb_by_name = {name: _load_json_list(path) for name, path in GRAPH_PATHS.items()}
    rows: List[Dict[str, object]] = []

    print("\n=== RQ2 Hybrid Evaluation (Sparsity & Avg_Rank) ===")
    print("| Dataset | Graph | #Cases | Rankable | Sparsity_mean | Avg_Rank |")
    print("|---------|-------|--------|----------|---------------|----------|")

    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        ds_cases = [c for c in eval_cases if c["dataset"] == ds]
        dom = _domain(ds)
        for name, kb in kb_by_name.items():
            sparsity_sum = 0.0
            rank_sum = 0.0
            rankable = 0
            for case in ds_cases:
                sparsity, rank = _calc_rank(kb, dom, case["gt_root"], case["gt_effect"])
                sparsity_sum += float(sparsity)
                if rank >= 0:
                    rankable += 1
                    rank_sum += float(rank)
            n = len(ds_cases) or 1
            sparsity_mean = sparsity_sum / n
            avg_rank = rank_sum / rankable if rankable else math.nan
            rows.append(
                {
                    "dataset": ds,
                    "graph": name,
                    "cases": len(ds_cases),
                    "rankable": rankable,
                    "sparsity_mean": round(sparsity_mean, 4),
                    "avg_rank": None if math.isnan(avg_rank) else round(avg_rank, 4),
                }
            )
            avg_rank_text = "nan" if math.isnan(avg_rank) else f"{avg_rank:.2f}"
            print(f"| {ds} | {name} | {len(ds_cases)} | {rankable} | {sparsity_mean:.2f} | {avg_rank_text} |")

    out_path = os.path.join(_PROJECT_ROOT, "results", "rq2", "hybrid_eval_20260314.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved hybrid RQ2 evaluation to: {out_path}")


if __name__ == "__main__":
    main()
