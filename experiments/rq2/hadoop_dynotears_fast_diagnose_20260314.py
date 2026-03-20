"""
Quick offline sweep for Hadoop DYNOTEARS settings.

It measures:
- rows used after downsampling
- build time
- edge count
- Avg_Rank on Hadoop benchmark cases with known fine-grained GT roots

The goal is to pick a setting that stays in the ~10 minute envelope when rebuilding
the full graph, without collapsing rank quality.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq2.generate_golden_graphs_all import (  # type: ignore
    _downsample_rows,
    _load_hadoop_timeseries,
    _safe_standardize,
    _trim_top_variance_columns,
)
from experiments.rq2.compare_dy_modified_vs_original_20260313 import (  # type: ignore
    _calc_sparsity_rank_relaxed,
)
from src.reasoning.dynotears import dynotears as fast_dynotears

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")


def _load_hadoop_cases() -> List[Dict[str, object]]:
    with open(BENCH_V2_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return [
        c
        for c in cases
        if str(c.get("dataset", "")) == "Hadoop"
        and str(c.get("ground_truth_root_cause_template", "") or "").strip().lower() not in ("", "unknown")
    ]


def _build_edges(
    df,
    tpl_map: Dict[str, str],
    max_rows: int,
    max_cols: int,
    lambda_w: float,
    lambda_a: float,
    thr: float,
    max_iter: int,
) -> Tuple[List[Dict[str, object]], int, float]:
    df_small = _downsample_rows(df, max_rows)
    df_small = _trim_top_variance_columns(df_small, max_cols)
    X, cols = _safe_standardize(df_small)
    if X.size == 0 or not cols:
        return [], len(df_small), 0.0
    start = time.perf_counter()
    W, A = fast_dynotears(
        X,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        h_tol=1e-6,
        w_threshold=0.0,
    )
    secs = time.perf_counter() - start
    edges: List[Dict[str, object]] = []
    for mat, rel in ((A, "temporally_causes"), (W, "instantly_triggers")):
        mat = np.array(mat, copy=True)
        mat[np.abs(mat) < thr] = 0.0
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            src_id = cols[s]
            tgt_id = cols[t]
            edges.append(
                {
                    "domain": "hadoop",
                    "source_template": tpl_map.get(src_id, "Unknown"),
                    "relation": rel,
                    "target_template": tpl_map.get(tgt_id, "Unknown"),
                    "weight": float(round(float(mat[s, t]), 4)),
                }
            )
    return edges, len(df_small), secs


def _evaluate(edges: List[Dict[str, object]], cases: List[Dict[str, object]]) -> Tuple[float, int]:
    ranks: List[float] = []
    for c in cases:
        gt_tpl = str(c.get("ground_truth_template", "") or "")
        gt_root = str(c.get("ground_truth_root_cause_template", "") or "")
        _, rank = _calc_sparsity_rank_relaxed(edges, "hadoop", gt_root, gt_tpl)
        if rank >= 0:
            ranks.append(float(rank))
    if not ranks:
        return float("nan"), 0
    return float(sum(ranks) / len(ranks)), len(ranks)


def main() -> None:
    df, tpl_map = _load_hadoop_timeseries()
    cases = _load_hadoop_cases()

    configs = [
        {"name": "fast_400x32", "max_rows": 400, "max_cols": 32, "lambda_w": 0.060, "lambda_a": 0.120, "thr": 0.45, "max_iter": 2},
        {"name": "balanced_600x48", "max_rows": 600, "max_cols": 48, "lambda_w": 0.055, "lambda_a": 0.100, "thr": 0.40, "max_iter": 3},
    ]

    print("=== Hadoop DYNOTEARS Fast Sweep ===")
    print("| Config | Rows | Cols | Secs | Edges | RankableCases | Avg_Rank |")
    print("|--------|------|------|------|-------|---------------|----------|")
    for cfg in configs:
        edges, rows_used, secs = _build_edges(
            df,
            tpl_map,
            max_rows=cfg["max_rows"],
            max_cols=cfg["max_cols"],
            lambda_w=cfg["lambda_w"],
            lambda_a=cfg["lambda_a"],
            thr=cfg["thr"],
            max_iter=cfg["max_iter"],
        )
        avg_rank, rankable = _evaluate(edges, cases)
        avg_rank_str = f"{avg_rank:.2f}" if rankable else "nan"
        print(
            f"| {cfg['name']} | {rows_used} | {cfg['max_cols']} | {secs:.2f} | {len(edges)} | {rankable} | {avg_rank_str} |"
        )


if __name__ == "__main__":
    main()
