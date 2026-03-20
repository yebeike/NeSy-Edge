"""
RQ2 metric refactor (non-destructive): redefine FPRR as Reachability.

Instead of requiring exact edge direction/rank, we treat success as:
  exists a directed path from root_cause_node -> target_alert_node
in the learned directed causal graph (DYNOTEARS intra-slice W matrix).

This script runs the existing 15-run OpenStack single-edge benchmark kernel
in-memory, builds a DiGraph from thresholded W each run, and reports the
Reachability success rate across runs.
"""

import os
import sys
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

# Project root: experiments/ -> root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _import_networkx():
    try:
        import networkx as nx  # type: ignore
        return nx
    except Exception as e:
        raise RuntimeError(
            "networkx is required for reachability. Install it in your venv, e.g. `pip install networkx`.\n"
            f"Import error: {e}"
        )


def _load_os_final_bench():
    # Reuse existing stable benchmark code without modification.
    from experiments.run_rq2_os_final_benchmark import OSFinalBench

    return OSFinalBench


def _build_graph_from_W(nx, W: np.ndarray, thresholded_nonzero: bool = True):
    # Nodes are indices 0..d-1; edges are directed i->j for nonzero weights.
    G = nx.DiGraph()
    d = int(W.shape[0])
    G.add_nodes_from(range(d))
    if thresholded_nonzero:
        rows, cols = np.where(np.abs(W) > 0)
    else:
        rows, cols = np.where(np.abs(W) > 1e-12)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        G.add_edge(i, j, weight=float(W[i, j]))
    return G


def _run_reachability_15run(
    runs: int = 15,
    data_path: str = "data/processed/openstack_refined_ts.csv",
) -> Tuple[float, int, int]:
    nx = _import_networkx()
    OSFinalBench = _load_os_final_bench()

    bench = OSFinalBench(data_path)
    df = bench.df
    feats = bench.feats
    d = bench.d
    src = bench.src
    tgt = bench.tgt
    s_idx = bench.s_idx
    t_idx = bench.t_idx

    # Import kernel from existing file
    from experiments.run_rq2_os_final_benchmark import dynotears_os_optimal_kernel

    success = 0
    for i in range(int(runs)):
        np.random.seed(9000 + i)
        data = df.copy()
        data[tgt] += data[tgt].std() * bench.strength_mult * data[src] + np.random.normal(0, 0.1, len(data))
        X_norm = StandardScaler().fit_transform(data)

        w_mask = np.ones((d, d))
        a_mask = np.ones((d, d))
        w_mask[s_idx, t_idx] = bench.prior_mask
        w_mask[t_idx, s_idx] = 100.0
        for j in range(d):
            if j != s_idx and j != t_idx:
                w_mask[t_idx, j] = 2.0

        W_n, _ = dynotears_os_optimal_kernel(X_norm, bench.lambda_w, bench.lambda_w * 2, w_mask, a_mask)
        W_n[np.abs(W_n) < bench.w_threshold] = 0

        G = _build_graph_from_W(nx, W_n, thresholded_nonzero=True)
        if nx.has_path(G, s_idx, t_idx):
            success += 1

    rate = success / float(runs) if runs else 0.0
    return rate, success, runs


def main():
    runs = 100
    data_path = "data/processed/openstack_refined_ts.csv"
    if not os.path.exists(os.path.join(_PROJECT_ROOT, data_path)):
        print(f"[ERROR] Missing data file: {data_path}")
        sys.exit(1)

    print("=" * 90)
    print("RQ2 Reachability Metric (OpenStack single-edge benchmark)")
    print("Definition: success=1 iff exists directed path root_cause -> target_alert in learned DYNOTEARS graph")
    print(f"Runs: {runs} | Data: {data_path}")
    print("=" * 90)

    rate, succ, total = _run_reachability_15run(runs=runs, data_path=data_path)
    print(f"Reachability FPRR success: {succ}/{total} = {rate:.2%}")
    print("=" * 90)


if __name__ == "__main__":
    main()

