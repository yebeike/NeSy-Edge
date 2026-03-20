"""
RQ2 OpenStack – Single-Edge Benchmark
-------------------------------------
Wrapper around `run_rq2_os_final_benchmark.py`.

Provides a clean entrypoint under `experiments/rq2/` while preserving
the original benchmark logic (NuSy-Edge vs Pearson vs PC_Algo).
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import run_rq2_os_final_benchmark as _osbench  # type: ignore


def main():
    bench = _osbench.OSFinalBench("data/processed/openstack_refined_ts.csv")
    final_res = bench.execute()

    print("\n" + "=" * 80)
    print("RQ2 FINAL BENCHMARK (OPENSTACK) | DIMENSIONS: 50 | SAMPLES: 3864")
    print("=" * 80)
    print(f"{'Algorithm':<20} | {'Avg_Rank':<12} | {'Sparsity':<12} | {'Precision@Top3'}")
    print("-" * 80)
    for algo in ["NuSy-Edge", "Pearson", "PC_Algo"]:
        avg_r = np.mean([r["rank"] for r in final_res[algo]])
        avg_s = np.mean([r["sparsity"] for r in final_res[algo]])
        p_at_3 = sum(1 for r in final_res[algo] if r["rank"] <= 3) / len(final_res[algo])
        print(f"{algo:<20} | #{avg_r:<11.2f} | {avg_s:<12.1f} | {p_at_3:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()


