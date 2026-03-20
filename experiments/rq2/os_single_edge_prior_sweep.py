import os
import sys
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import experiments.run_rq2_os_final_benchmark as _osbench  # type: ignore


RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def run_config(prior_mask_value: float, lambda_w: float, runs: int = 5) -> Dict[str, Any]:
    """Run quick OS single-edge benchmark for a given (prior_mask_value, lambda_w)."""
    print(
        f"[INFO] Config: prior_mask={prior_mask_value}, lambda_w={lambda_w}, runs={runs}"
    )

    # Use env to tell OSFinalBench which prior to apply.
    env_prior_key = "RQ2_OS_PRIOR_MASK"
    os.environ[env_prior_key] = str(prior_mask_value)

    bench = _osbench.OSFinalBench(
        "data/processed/openstack_refined_ts.csv",
        lambda_w=lambda_w,
        w_threshold=_osbench.W_THRESHOLD_OS,
    )
    results = bench.execute(runs=runs)

    # Aggregate NuSy-Edge statistics (same metrics as final benchmark).
    nusy_runs: List[Dict[str, Any]] = results["NuSy-Edge"]
    ranks = [float(r["rank"]) for r in nusy_runs]
    sparsities = [float(r["sparsity"]) for r in nusy_runs]
    avg_rank = float(np.mean(ranks))
    avg_sparsity = float(np.mean(sparsities))
    p_at_3 = float(sum(1 for r in ranks if r <= 3) / len(ranks))

    summary = {
        "prior_mask_value": prior_mask_value,
        "lambda_w": lambda_w,
        "avg_rank": avg_rank,
        "avg_sparsity": avg_sparsity,
        "precision_at_3": p_at_3,
        "runs": len(ranks),
    }

    print(
        f"[RESULT] prior={prior_mask_value}, lambda_w={lambda_w} | "
        f"Avg_Rank={avg_rank:.2f}, Sparsity={avg_sparsity:.1f}, P@3={p_at_3:.2%}"
    )
    return summary


def main():
    prior_values = [0.05, 0.01]
    lambda_ws = [0.005, 0.01]
    runs_per_config = 5

    print(
        "[INFO] OpenStack prior+lambda_w sweep\n"
        f"       prior_mask_value={prior_values}\n"
        f"       lambda_w={lambda_ws}\n"
        f"       runs_per_config={runs_per_config}"
    )

    rows: List[Dict[str, Any]] = []

    for pv in prior_values:
        for lw in lambda_ws:
            rows.append(run_config(pv, lw, runs=runs_per_config))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    grid_path = os.path.join(RESULTS_DIR, "rq2_os_single_edge_prior_grid.csv")
    pd.DataFrame(rows).to_csv(grid_path, index=False)
    print(f"[INFO] Prior+lambda_w sweep summary written to {grid_path}")

    # Select best config: Avg_Rank (ascending), then Precision@Top3 (descending).
    def sort_key(r: Dict[str, Any]):
        return (r["avg_rank"], -r["precision_at_3"])

    best = sorted(rows, key=sort_key)[0]
    best_path = os.path.join(RESULTS_DIR, "rq2_os_single_edge_prior_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(
        "\n=== Best Config (prior+lambda_w sweep) ===\n"
        f"prior_mask_value={best['prior_mask_value']}, lambda_w={best['lambda_w']} | "
        f"Avg_Rank={best['avg_rank']:.2f}, Sparsity={best['avg_sparsity']:.1f}, "
        f"P@3={best['precision_at_3']:.2%} (runs={best['runs']})"
    )


if __name__ == "__main__":
    main()

