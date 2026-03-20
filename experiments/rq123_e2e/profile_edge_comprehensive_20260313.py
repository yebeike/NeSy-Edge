"""
StageX: Edge Environment Comprehensive Profiling (2026-03-13).

This script is for *experiments only* and is not part of the main pipeline.
It performs three groups of measurements:

1. Memory / GPU footprint of instantiating NuSyEdgeNode (Qwen3-0.6B + KB).
2. Local CPU throughput & latency of Qwen3-0.6B on a small set of prompts.
3. Cloud upload payload comparison between:
   - Vanilla: sending full raw_log window to cloud.
   - NeSy-Agent: sending (parsed template + causal candidates subset).

At the end, it prints a Markdown summary block: "Edge Simulation Physical Profiling".
"""

import os
import sys
import time
from typing import Dict, List

import psutil

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.utils.llm_client import LLMClient
from src.utils.metrics import MetricsCalculator
from experiments.rq3 import tools as rq3_tools  # type: ignore
from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")


def _estimate_tokens(text: str) -> int:
    return MetricsCalculator.estimate_tokens(text)


def _profile_memory_and_gpu() -> Dict[str, float]:
    """Profile RSS (and GPU if available) before/after NuSyEdgeNode instantiation."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024 ** 2)  # MB

    gpu_before_mb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_before_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        gpu_before_mb = 0.0

    _ = NuSyEdgeNode()

    rss_after = proc.memory_info().rss / (1024 ** 2)

    gpu_after_mb = gpu_before_mb
    try:
        import torch

        if torch.cuda.is_available():
            gpu_after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        gpu_after_mb = gpu_before_mb

    return {
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "gpu_before_mb": gpu_before_mb,
        "gpu_after_mb": gpu_after_mb,
        "gpu_delta_mb": gpu_after_mb - gpu_before_mb,
    }


def _profile_local_qwen_throughput(n_calls: int = 10) -> Dict[str, float]:
    """
    Run a few Qwen3-0.6B generations locally to estimate latency and TPS.
    We use parse_with_multi_rag on simple synthetic inputs to avoid external deps.
    """
    client = LLMClient()
    prompt = "HDFS DataNode got exception while serving block to client."
    refs = [("HDFS DataNode got exception while serving blk_123 to /10.0.0.1", "[*]Got exception while serving[*]to[*]")]

    latencies: List[float] = []
    tokens_out = 0
    for _ in tqdm(range(n_calls), desc="Local Qwen throughput", unit="call"):
        t0 = time.perf_counter()
        try:
            tpl = client.parse_with_multi_rag(prompt, refs)
        except Exception:
            tpl = ""
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        tokens_out += _estimate_tokens(tpl)

    if not latencies:
        return {"avg_latency_ms": 0.0, "min_latency_ms": 0.0, "max_latency_ms": 0.0, "tps": 0.0}

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    total_s = sum(latencies) / 1000.0
    tps = (tokens_out / total_s) if total_s > 0 else 0.0

    return {
        "avg_latency_ms": avg_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "tps": tps,
        "calls": len(latencies),
        "tokens_generated": tokens_out,
    }


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


def _profile_payload_reduction() -> List[Dict[str, object]]:
    """Compare Vanilla vs NeSy-Agent payload sizes (chars / tokens) on a few cases."""
    cases = _load_benchmark(BENCH_V2_PATH)
    if not cases:
        print("[ERROR] No benchmark cases.")
        return []

    sample = _sample_cases_for_payload(cases, per_ds=3)
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
    return rows


def _aggregate_payload(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    from collections import defaultdict

    agg = defaultdict(lambda: {"n": 0.0, "v_chars": 0.0, "v_tok": 0.0, "a_chars": 0.0, "a_tok": 0.0})
    for r in rows:
        ds = str(r.get("dataset", "HDFS"))
        g = agg[ds]
        g["n"] += 1
        g["v_chars"] += float(r["vanilla_chars"])
        g["v_tok"] += float(r["vanilla_tokens"])
        g["a_chars"] += float(r["agent_chars"])
        g["a_tok"] += float(r["agent_tokens"])

    out: List[Dict[str, object]] = []
    for ds, g in agg.items():
        n = g["n"] or 1.0
        v_chars = g["v_chars"] / n
        a_chars = g["a_chars"] / n
        v_tok = g["v_tok"] / n
        a_tok = g["a_tok"] / n
        red_chars = 1.0 - (a_chars / v_chars) if v_chars > 0 else 0.0
        red_tok = 1.0 - (a_tok / v_tok) if v_tok > 0 else 0.0
        out.append(
            {
                "dataset": ds,
                "n": int(g["n"]),
                "vanilla_chars": v_chars,
                "agent_chars": a_chars,
                "reduction_chars": red_chars,
                "vanilla_tokens": v_tok,
                "agent_tokens": a_tok,
                "reduction_tokens": red_tok,
            }
        )
    return out


def main() -> None:
    print("# Edge Simulation Physical Profiling\n")

    print("## 1. Memory / GPU Footprint\n")
    mem = _profile_memory_and_gpu()
    print(
        f"- RSS before NuSyEdgeNode: **{mem['rss_before_mb']:.2f} MB**  \n"
        f"- RSS after  NuSyEdgeNode: **{mem['rss_after_mb']:.2f} MB**  \n"
        f"- ΔRSS (NuSyEdgeNode): **{mem['rss_delta_mb']:.2f} MB**"
    )
    if mem["gpu_after_mb"] or mem["gpu_before_mb"]:
        print(
            f"- GPU mem before: **{mem['gpu_before_mb']:.2f} MB**, "
            f"after: **{mem['gpu_after_mb']:.2f} MB**, "
            f"ΔGPU: **{mem['gpu_delta_mb']:.2f} MB**"
        )
    else:
        print("- GPU: CUDA not available or not used.\n")

    print("\n## 2. Local Qwen3-0.6B Throughput (no cloud, edge-only)\n")
    thr = _profile_local_qwen_throughput(n_calls=10)
    print(
        f"- Calls: **{thr['calls']}**, Tokens generated (approx): **{thr['tokens_generated']}**  \n"
        f"- Avg latency: **{thr['avg_latency_ms']:.2f} ms** "
        f"(min: {thr['min_latency_ms']:.2f} ms, max: {thr['max_latency_ms']:.2f} ms)  \n"
        f"- Throughput: **{thr['tps']:.2f} tokens/s**"
    )

    print("\n## 3. Cloud Upload Payload Reduction (Vanilla vs NeSy-Agent)\n")
    rows = _profile_payload_reduction()
    agg = _aggregate_payload(rows)

    print(
        "| Dataset | n | Vanilla_chars | Agent_chars | Reduction_chars | "
        "Vanilla_tokens | Agent_tokens | Reduction_tokens |"
    )
    print(
        "|---------|---|---------------|------------|-----------------|"
        "----------------|--------------|------------------|"
    )
    for r in agg:
        print(
            f"| {r['dataset']} | {r['n']} | {r['vanilla_chars']:.1f} | {r['agent_chars']:.1f} | "
            f"{r['reduction_chars']:.3f} | {r['vanilla_tokens']:.1f} | {r['agent_tokens']:.1f} | "
            f"{r['reduction_tokens']:.3f} |"
        )


if __name__ == "__main__":
    main()

