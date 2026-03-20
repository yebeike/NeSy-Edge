import os
import sys
import json
import statistics
from typing import Dict, Any, List

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import experiments.run_rq2_os_final_benchmark as _osbench  # type: ignore


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _aggregate_nusy(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """Compute Avg_Rank / Sparsity / Precision@Top3 for NuSy-Edge."""
    nusy_runs = results.get("NuSy-Edge", [])
    if not nusy_runs:
        return {
            "avg_rank": float("inf"),
            "avg_sparsity": 0.0,
            "precision_at_3": 0.0,
            "runs": 0,
        }

    ranks = [float(r["rank"]) for r in nusy_runs]
    spars = [float(r["sparsity"]) for r in nusy_runs]
    avg_rank = float(statistics.fmean(ranks))
    avg_sparsity = float(statistics.fmean(spars))
    p_at_3 = sum(1 for r in ranks if r <= 3.0) / len(ranks)

    return {
        "avg_rank": avg_rank,
        "avg_sparsity": avg_sparsity,
        "precision_at_3": float(p_at_3),
        "runs": len(ranks),
    }


def run_config(lambda_w: float, w_threshold: float, runs: int = 5) -> Dict[str, Any]:
    """Run OpenStack single-edge benchmark for a given (lambda_w, w_threshold)."""
    print(
        f"[INFO] Running OS single-edge benchmark | "
        f"lambda_w={lambda_w}, W_THRESHOLD={w_threshold}, runs={runs}"
    )

    bench = _osbench.OSFinalBench(
        "data/processed/openstack_refined_ts.csv",
        lambda_w=lambda_w,
        w_threshold=w_threshold,
    )
    results = bench.execute(runs=runs)
    agg = _aggregate_nusy(results)

    print(
        f"[RESULT] lambda_w={lambda_w}, W_THRESHOLD={w_threshold} | "
        f"Avg_Rank={agg['avg_rank']:.2f}, "
        f"Sparsity={agg['avg_sparsity']:.1f}, "
        f"P@3={agg['precision_at_3']:.2%}"
    )

    return {
        "lambda_w": lambda_w,
        "w_threshold": w_threshold,
        **agg,
    }


def main():
    # Grid for (lambda_w, W_THRESHOLD) focusing on smaller regularisation,
    # as requested in the optimisation task.
    lambda_ws = [0.005, 0.01, 0.02, 0.03]
    w_thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    runs_per_config = 5

    print(
        "[INFO] OpenStack RQ2 single-edge grid search over:\n"
        f"       lambda_w={lambda_ws}\n"
        f"       W_THRESHOLD={w_thresholds}\n"
        f"       runs_per_config={runs_per_config}"
    )

    rows: List[Dict[str, Any]] = []

    for lw in lambda_ws:
        for th in w_thresholds:
            row = run_config(lw, th, runs=runs_per_config)
            rows.append(row)

    if not rows:
        print("[ERROR] No grid results collected; aborting.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    grid_path = os.path.join(RESULTS_DIR, "rq2_os_single_edge_grid.csv")
    pd.DataFrame(rows).to_csv(grid_path, index=False)
    print(f"[INFO] OS single-edge grid summary written to {grid_path}")

    # Select best configuration: minimise Avg_Rank, then prefer sparser graphs.
    def sort_key(r: Dict[str, Any]):
        return (r["avg_rank"], r["avg_sparsity"])

    best = sorted(rows, key=sort_key)[0]

    best_path = os.path.join(RESULTS_DIR, "rq2_os_single_edge_grid_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(
        "\n=== Best Config (OpenStack Single-Edge Grid) ===\n"
        f"lambda_w={best['lambda_w']}, W_THRESHOLD={best['w_threshold']} | "
        f"Avg_Rank={best['avg_rank']:.2f}, "
        f"Sparsity={best['avg_sparsity']:.1f}, "
        f"P@3={best['precision_at_3']:.2%} "
        f"(runs={best['runs']})"
    )


if __name__ == "__main__":
    main()

