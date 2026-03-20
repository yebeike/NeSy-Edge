"""
Profile edge-device footprint for NeSy-Edge (experiments-only).

Goals:
- Measure memory footprint of instantiating NuSyEdgeNode (Qwen 0.6B + KB).
- Compare cloud upload payload between:
  - Vanilla: sending full raw_log window.
  - NeSy-Agent: sending only (parsed effect template + causal candidates).

This script is offline and DOES NOT call DeepSeek; it only estimates sizes.
"""

import os
import sys
from typing import Dict, List, Tuple

import psutil

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from experiments.rq3 import tools as rq3_tools  # type: ignore
from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from src.utils.metrics import MetricsCalculator

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")


def _estimate_tokens(text: str) -> int:
    # Reuse MetricsCalculator's simple heuristic if available
    return MetricsCalculator.estimate_tokens(text)


def _profile_memory_nusy() -> None:
    """Profile RSS before/after instantiating NuSyEdgeNode."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024 ** 2)  # MB
    print(f"[MEM] RSS before NuSyEdgeNode: {rss_before:.2f} MB")

    _ = NuSyEdgeNode()

    rss_after = proc.memory_info().rss / (1024 ** 2)
    print(f"[MEM] RSS after  NuSyEdgeNode: {rss_after:.2f} MB")
    print(f"[MEM] Delta (NuSyEdgeNode): {rss_after - rss_before:.2f} MB")


def _sample_cases_for_payload(cases: List[Dict[str, object]], per_ds: int = 3) -> List[Dict[str, object]]:
    by_ds: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for c in cases:
        ds = str(c.get("dataset", "HDFS"))
        if ds in by_ds:
            by_ds[ds].append(c)
    sampled: List[Dict[str, object]] = []
    for ds, lst in by_ds.items():
        sampled.extend(lst[:per_ds])
    return sampled


def _profile_payload() -> None:
    """Compare Vanilla vs NeSy-Agent payload sizes on a few benchmark cases."""
    cases = _load_benchmark(BENCH_V2_PATH)
    if not cases:
        print("[ERROR] No benchmark cases.")
        return
    sample = _sample_cases_for_payload(cases, per_ds=3)
    print(f"[INFO] Profiling payload on {len(sample)} cases (3 per dataset where available).")

    edge_node = NuSyEdgeNode()

    rows: List[Dict[str, object]] = []

    pbar = tqdm(total=len(sample), desc="Profile payload", unit="case")
    for case in sample:
        raw = str(case.get("raw_log", "") or "")
        dataset = str(case.get("dataset", "HDFS"))
        lines = [l for l in raw.split("\n") if l.strip()]
        alert = lines[-1] if lines else raw
        ds_header = "HDFS" if dataset == "Hadoop" else dataset
        clean_alert = NuSyEdgeNode.preprocess_header(alert, ds_header) or alert

        # Vanilla payload: full window text
        vanilla_payload = raw
        vanilla_chars = len(vanilla_payload)
        vanilla_tokens = _estimate_tokens(vanilla_payload)

        # NeSy-Agent payload: effect template + causal candidates
        try:
            tpl_nusy, _, _, _ = edge_node.parse_log_stream(clean_alert, ds_header)
        except Exception:
            tpl_nusy = ""

        domain = "hdfs" if dataset == "HDFS" else ("openstack" if dataset == "OpenStack" else "hadoop")
        cand_json = rq3_tools.causal_navigator(tpl_nusy or clean_alert, domain)
        payload_agent = f"template: {tpl_nusy}\ncausal_candidates: {cand_json}"
        agent_chars = len(payload_agent)
        agent_tokens = _estimate_tokens(payload_agent)

        rows.append(
            {
                "dataset": dataset,
                "vanilla_chars": vanilla_chars,
                "vanilla_tokens": vanilla_tokens,
                "agent_chars": agent_chars,
                "agent_tokens": agent_tokens,
            }
        )
        pbar.update(1)

    pbar.close()

    # Aggregate per dataset
    from collections import defaultdict

    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "v_chars": 0.0, "v_tok": 0.0, "a_chars": 0.0, "a_tok": 0.0})
    for r in rows:
        ds = r["dataset"]
        g = agg[ds]
        g["n"] += 1
        g["v_chars"] += r["vanilla_chars"]
        g["v_tok"] += r["vanilla_tokens"]
        g["a_chars"] += r["agent_chars"]
        g["a_tok"] += r["agent_tokens"]

    print("\n=== Cloud payload comparison (avg per case) ===")
    print("| Dataset | n | Vanilla_chars | Agent_chars | Reduction_chars | Vanilla_tokens | Agent_tokens | Reduction_tokens |")
    print("|---------|---|---------------|------------|-----------------|----------------|--------------|------------------|")
    for ds, g in agg.items():
        n = g["n"] or 1.0
        v_chars = g["v_chars"] / n
        a_chars = g["a_chars"] / n
        v_tok = g["v_tok"] / n
        a_tok = g["a_tok"] / n
        red_chars = 1.0 - (a_chars / v_chars) if v_chars > 0 else 0.0
        red_tok = 1.0 - (a_tok / v_tok) if v_tok > 0 else 0.0
        print(
            f"| {ds} | {int(g['n']):d} | {v_chars:13.1f} | {a_chars:10.1f} | {red_chars:15.3f} | "
            f"{v_tok:14.1f} | {a_tok:12.1f} | {red_tok:16.3f} |"
        )


def main() -> None:
    print("=== Edge Memory Footprint ===")
    _profile_memory_nusy()
    print("\n=== Cloud Payload Profiling ===")
    _profile_payload()


if __name__ == "__main__":
    main()

