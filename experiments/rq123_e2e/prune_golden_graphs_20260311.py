"""
Prune golden causal graphs (Pearson / PC) to reduce sparsity and make Avg_Rank more meaningful.

This script does NOT overwrite existing golden files. It writes new pruned versions:
- data/processed/gt_causal_knowledge_pearson_pruned_20260311.json
- data/processed/gt_causal_knowledge_pc_pruned_20260311.json

Pruning strategy (simple, stable):
- For each (domain, target_template), keep top-K incoming edges by abs(weight).
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")

PEARSON_IN = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson.json")
PC_IN = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc.json")

PEARSON_OUT = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pearson_pruned_20260311.json")
PC_OUT = os.path.join(DATA_PROCESSED, "gt_causal_knowledge_pc_pruned_20260311.json")

TOP_K_PER_TARGET = 5


def _load(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump(path: str, data: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _key(e: Dict[str, object]) -> Tuple[str, str]:
    dom = str(e.get("domain", "") or "")
    tgt = str(e.get("target_template", "") or "")
    return dom, tgt


def prune_edges(edges: List[Dict[str, object]], top_k: int = TOP_K_PER_TARGET) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for e in edges:
        buckets[_key(e)].append(e)

    pruned: List[Dict[str, object]] = []
    for (dom, tgt), es in tqdm(buckets.items(), desc=f"Pruning (top_k={top_k})", unit="target"):
        scored = []
        for e in es:
            try:
                w = float(e.get("weight", 0.0) or 0.0)
            except Exception:
                w = 0.0
            scored.append((abs(w), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [e for _, e in scored[:top_k]]
        pruned.extend(keep)
    return pruned


def _summarize(label: str, edges: List[Dict[str, object]]) -> None:
    by_dom: Dict[str, int] = defaultdict(int)
    for e in edges:
        by_dom[str(e.get("domain", "") or "")] += 1
    total = len(edges)
    print(f"\n[{label}] total_edges={total}")
    for dom in sorted(by_dom.keys()):
        print(f"  - {dom}: {by_dom[dom]}")


def main() -> None:
    print(f"[INFO] TOP_K_PER_TARGET={TOP_K_PER_TARGET}")

    pear = _load(PEARSON_IN)
    pc = _load(PC_IN)

    _summarize("Pearson (original)", pear)
    _summarize("PC (original)", pc)

    pear_p = prune_edges(pear, top_k=TOP_K_PER_TARGET)
    pc_p = prune_edges(pc, top_k=TOP_K_PER_TARGET)

    _summarize("Pearson (pruned)", pear_p)
    _summarize("PC (pruned)", pc_p)

    _dump(PEARSON_OUT, pear_p)
    _dump(PC_OUT, pc_p)

    print(f"\n[DONE] Wrote:\n  - {PEARSON_OUT}\n  - {PC_OUT}\n")


if __name__ == "__main__":
    main()

