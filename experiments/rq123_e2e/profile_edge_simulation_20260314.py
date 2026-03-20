"""
Comprehensive edge-simulation profiling for NuSy-Edge.

This script is designed to support the "edge suitability" claim with
reproducible, offline measurements:

1. Host/device profile: CPU, memory, OS, Python.
2. NuSyEdgeNode cold-start profile: init latency, RSS delta, peak RSS.
3. Local Qwen3-0.6B profile: latency / throughput on small parsing prompts.
4. Parsing-service profile: NuSy vs Drain vs local Qwen on a benchmark subset.
5. Cloud-payload reduction: raw-log upload vs template+causal upload.

Outputs:
- results/edge_profile_simulation_20260314.json
- Markdown summary printed to stdout
"""

import json
import os
import platform
import statistics
import sys
import threading
import time
from typing import Callable, Dict, List, Tuple

import psutil
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from experiments.rq3 import tools as rq3_tools  # type: ignore
from src.perception.drain_parser import DrainParser
from src.system.edge_node import NuSyEdgeNode
from src.utils.llm_client import LLMClient
from src.utils.metrics import MetricsCalculator

DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
OUT_PATH = os.path.join(RESULTS_DIR, "edge_profile_simulation_20260314.json")

EDGE_DEVICE_PRESETS = [
    {
        "name": "rpi5_8gb_profile",
        "label": "Raspberry Pi 5 (8GB-class)",
        "cpu_cores": 4,
        "ram_gb": 8.0,
        "safe_memory_ratio": 0.60,
        "notes": "memory-first fit check only; no claim about equal runtime on ARM SBC",
    },
    {
        "name": "jetson_nano_4gb_profile",
        "label": "Jetson Nano (4GB-class)",
        "cpu_cores": 4,
        "ram_gb": 4.0,
        "safe_memory_ratio": 0.55,
        "notes": "tight memory budget; useful as a lower-bound edge scenario",
    },
    {
        "name": "orin_nx_16gb_profile",
        "label": "Jetson Orin NX (16GB-class)",
        "cpu_cores": 8,
        "ram_gb": 16.0,
        "safe_memory_ratio": 0.65,
        "notes": "practical edge accelerator class for local Qwen3-0.6B deployment",
    },
]


def _estimate_tokens(text: str) -> int:
    return MetricsCalculator.estimate_tokens(text)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * pct)))
    return float(ordered[idx])


class ProcessSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.rss_samples_mb: List[float] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss_mb = self.proc.memory_info().rss / (1024 ** 2)
                self.rss_samples_mb.append(rss_mb)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self) -> "ProcessSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_rss_mb(self) -> float:
        return max(self.rss_samples_mb) if self.rss_samples_mb else 0.0


def _measure_operation(fn: Callable[[], object]) -> Tuple[object, Dict[str, float]]:
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024 ** 2)
    cpu_before = proc.cpu_times()
    t0 = time.perf_counter()
    with ProcessSampler() as sampler:
        result = fn()
    wall_s = time.perf_counter() - t0
    cpu_after = proc.cpu_times()
    rss_after = proc.memory_info().rss / (1024 ** 2)
    cpu_time_s = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    cpu_count = max(1, psutil.cpu_count(logical=True) or 1)
    cpu_util_pct = (cpu_time_s / max(wall_s, 1e-9)) * 100.0 / cpu_count
    metrics = {
        "wall_ms": wall_s * 1000.0,
        "cpu_time_s": cpu_time_s,
        "avg_cpu_pct_process": cpu_util_pct,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "rss_peak_mb": max(rss_after, sampler.peak_rss_mb),
    }
    return result, metrics


def _host_profile() -> Dict[str, object]:
    vm = psutil.virtual_memory()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "cpu_freq_mhz": getattr(psutil.cpu_freq(), "max", 0.0) if psutil.cpu_freq() else 0.0,
        "total_memory_gb": vm.total / (1024 ** 3),
        "available_memory_gb": vm.available / (1024 ** 3),
    }


def _profile_edge_node_init() -> Dict[str, float]:
    _, metrics = _measure_operation(lambda: NuSyEdgeNode())
    return metrics


def _profile_local_qwen(n_calls: int = 8) -> Dict[str, float]:
    client = LLMClient()
    prompt = "OpenStack nova-compute reports unknown base file while building the instance."
    refs = [
        ("Unknown base file: ami-0001", "Unknown base file: <*>"),
        ("During sync_power_state the instance has a pending task (spawning). Skip.", "During sync_power_state the instance has a pending task (spawning). Skip."),
    ]

    latencies: List[float] = []
    cpu_pcts: List[float] = []
    peak_rss: List[float] = []
    tokens_generated = 0
    for _ in tqdm(range(n_calls), desc="Profile local Qwen", unit="call"):
        result, metrics = _measure_operation(lambda: client.parse_with_multi_rag(prompt, refs))
        latencies.append(metrics["wall_ms"])
        cpu_pcts.append(metrics["avg_cpu_pct_process"])
        peak_rss.append(metrics["rss_peak_mb"])
        tokens_generated += _estimate_tokens(str(result or ""))

    total_s = sum(latencies) / 1000.0
    return {
        "calls": n_calls,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_cpu_pct_process": statistics.mean(cpu_pcts) if cpu_pcts else 0.0,
        "peak_rss_mb": max(peak_rss) if peak_rss else 0.0,
        "tokens_generated": tokens_generated,
        "throughput_toks_per_s": (tokens_generated / total_s) if total_s > 0 else 0.0,
    }


def _sample_cases(per_dataset: int = 5) -> List[Dict[str, object]]:
    cases = _load_benchmark(BENCH_V2_PATH)
    by_ds: Dict[str, List[Dict[str, object]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for case in cases:
        ds = str(case.get("dataset", "HDFS"))
        if ds in by_ds:
            by_ds[ds].append(case)
    sampled: List[Dict[str, object]] = []
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        sampled.extend(by_ds[ds][:per_dataset])
    return sampled


def _profile_parsing_service(cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    drain = DrainParser()

    refs = {
        "HDFS": [("Got exception while serving blk_123 to /10.0.0.1", "[*]Got exception while serving[*]to[*]")],
        "OpenStack": [("Unknown base file: ami-001", "Unknown base file: <*>")],
        "Hadoop": [("Container killed on request. Exit code is 137", "Container killed on request. Exit code is <*>")],
    }

    stats: Dict[str, Dict[str, List[float]]] = {
        "NuSy": {"lat_ms": [], "cpu_pct": [], "peak_rss": []},
        "Drain": {"lat_ms": [], "cpu_pct": [], "peak_rss": []},
        "Qwen": {"lat_ms": [], "cpu_pct": [], "peak_rss": []},
    }

    for case in tqdm(cases, desc="Profile parsing service", unit="case"):
        dataset = str(case.get("dataset", "HDFS"))
        raw = str(case.get("raw_log", "") or "")
        lines = [x for x in raw.split("\n") if x.strip()]
        alert = lines[-1] if lines else raw
        ds_header = "HDFS" if dataset == "Hadoop" else dataset
        clean = NuSyEdgeNode.preprocess_header(alert, ds_header) or alert

        _, m_nusy = _measure_operation(lambda: edge_node.parse_log_stream(raw if dataset != "Hadoop" else clean, dataset))
        _, m_drain = _measure_operation(lambda: drain.parse(clean))
        _, m_qwen = _measure_operation(lambda: qwen.parse_with_multi_rag(clean, refs[dataset]))

        for key, metrics in [("NuSy", m_nusy), ("Drain", m_drain), ("Qwen", m_qwen)]:
            stats[key]["lat_ms"].append(metrics["wall_ms"])
            stats[key]["cpu_pct"].append(metrics["avg_cpu_pct_process"])
            stats[key]["peak_rss"].append(metrics["rss_peak_mb"])

    rows: List[Dict[str, object]] = []
    for method, st in stats.items():
        rows.append(
            {
                "method": method,
                "avg_latency_ms": statistics.mean(st["lat_ms"]) if st["lat_ms"] else 0.0,
                "p95_latency_ms": _percentile(st["lat_ms"], 0.95),
                "avg_cpu_pct_process": statistics.mean(st["cpu_pct"]) if st["cpu_pct"] else 0.0,
                "peak_rss_mb": max(st["peak_rss"]) if st["peak_rss"] else 0.0,
            }
        )
    return rows


def _profile_payload(cases: List[Dict[str, object]]) -> List[Dict[str, object]]:
    edge_node = NuSyEdgeNode()
    rows: List[Dict[str, object]] = []
    for case in tqdm(cases, desc="Profile payload", unit="case"):
        dataset = str(case.get("dataset", "HDFS"))
        raw = str(case.get("raw_log", "") or "")
        lines = [x for x in raw.split("\n") if x.strip()]
        alert = lines[-1] if lines else raw
        ds_header = "HDFS" if dataset == "Hadoop" else dataset
        clean = NuSyEdgeNode.preprocess_header(alert, ds_header) or alert
        try:
            tpl_nusy, _, _, _ = edge_node.parse_log_stream(clean, ds_header)
        except Exception:
            tpl_nusy = ""
        domain = "hdfs" if dataset == "HDFS" else ("openstack" if dataset == "OpenStack" else "hadoop")
        cand_json = rq3_tools.causal_navigator(tpl_nusy or clean, domain)
        vanilla_payload = raw
        agent_payload = f"template: {tpl_nusy}\ncausal_candidates: {cand_json}"
        rows.append(
            {
                "dataset": dataset,
                "vanilla_chars": len(vanilla_payload),
                "vanilla_tokens": _estimate_tokens(vanilla_payload),
                "agent_chars": len(agent_payload),
                "agent_tokens": _estimate_tokens(agent_payload),
            }
        )
    return rows


def _aggregate_payload(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        ds = str(row["dataset"])
        bucket = out.setdefault(ds, {"n": 0.0, "vanilla_chars": 0.0, "agent_chars": 0.0, "vanilla_tokens": 0.0, "agent_tokens": 0.0})
        bucket["n"] += 1
        bucket["vanilla_chars"] += float(row["vanilla_chars"])
        bucket["agent_chars"] += float(row["agent_chars"])
        bucket["vanilla_tokens"] += float(row["vanilla_tokens"])
        bucket["agent_tokens"] += float(row["agent_tokens"])
    rows_out: List[Dict[str, object]] = []
    for ds, bucket in out.items():
        n = bucket["n"] or 1.0
        vanilla_chars = bucket["vanilla_chars"] / n
        agent_chars = bucket["agent_chars"] / n
        vanilla_tokens = bucket["vanilla_tokens"] / n
        agent_tokens = bucket["agent_tokens"] / n
        rows_out.append(
            {
                "dataset": ds,
                "n": int(bucket["n"]),
                "vanilla_chars": vanilla_chars,
                "agent_chars": agent_chars,
                "reduction_chars": 1.0 - (agent_chars / vanilla_chars) if vanilla_chars else 0.0,
                "vanilla_tokens": vanilla_tokens,
                "agent_tokens": agent_tokens,
                "reduction_tokens": 1.0 - (agent_tokens / vanilla_tokens) if vanilla_tokens else 0.0,
            }
        )
    return rows_out


def _device_feasibility(report: Dict[str, object]) -> List[Dict[str, object]]:
    init_peak = float(report["edge_node_init"]["rss_peak_mb"])
    qwen_peak = float(report["local_qwen_profile"]["peak_rss_mb"])
    parsing_rows = {str(row["method"]): row for row in report["parsing_service_profile"]}
    nusy_peak = float(parsing_rows["NuSy"]["peak_rss_mb"])
    qwen_service_peak = float(parsing_rows["Qwen"]["peak_rss_mb"])

    rows: List[Dict[str, object]] = []
    for preset in EDGE_DEVICE_PRESETS:
        budget_mb = preset["ram_gb"] * 1024.0 * preset["safe_memory_ratio"]
        rows.append(
            {
                "device": preset["label"],
                "cpu_cores": preset["cpu_cores"],
                "ram_gb": preset["ram_gb"],
                "safe_budget_mb": budget_mb,
                "fits_edge_init": init_peak <= budget_mb,
                "fits_local_qwen": qwen_peak <= budget_mb,
                "fits_nusy_service": nusy_peak <= budget_mb,
                "fits_qwen_service": qwen_service_peak <= budget_mb,
                "max_parallel_nusy": int(max(0.0, budget_mb // max(nusy_peak, 1.0))),
                "max_parallel_qwen": int(max(0.0, budget_mb // max(qwen_service_peak, 1.0))),
                "notes": preset["notes"],
            }
        )
    return rows


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_cases = _sample_cases(per_dataset=5)

    report = {
        "host_profile": _host_profile(),
        "edge_node_init": _profile_edge_node_init(),
        "local_qwen_profile": _profile_local_qwen(n_calls=8),
        "parsing_service_profile": _profile_parsing_service(sample_cases),
        "payload_profile": _aggregate_payload(_profile_payload(sample_cases)),
        "device_feasibility": [],
        "sample_cases": len(sample_cases),
    }
    report["device_feasibility"] = _device_feasibility(report)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("# Edge Simulation Report\n")
    host = report["host_profile"]
    print(
        f"- Host: `{host['platform']}` on `{host['machine']}`\n"
        f"- CPU: `{host['physical_cores']}` physical / `{host['logical_cores']}` logical cores\n"
        f"- RAM: `{host['total_memory_gb']:.2f} GB` total, `{host['available_memory_gb']:.2f} GB` available\n"
    )

    init = report["edge_node_init"]
    print("## NuSyEdgeNode Cold Start\n")
    print(
        f"- Init latency: `{init['wall_ms']:.2f} ms`\n"
        f"- RSS delta: `{init['rss_delta_mb']:.2f} MB`\n"
        f"- Peak RSS: `{init['rss_peak_mb']:.2f} MB`\n"
        f"- Avg CPU usage (process, normalized): `{init['avg_cpu_pct_process']:.2f}%`\n"
    )

    qwen = report["local_qwen_profile"]
    print("## Local Qwen3-0.6B\n")
    print(
        f"- Calls: `{qwen['calls']}`\n"
        f"- Avg latency: `{qwen['avg_latency_ms']:.2f} ms`, p95: `{qwen['p95_latency_ms']:.2f} ms`\n"
        f"- Throughput: `{qwen['throughput_toks_per_s']:.2f} tok/s`\n"
        f"- Peak RSS during local Qwen calls: `{qwen['peak_rss_mb']:.2f} MB`\n"
    )

    print("## Parsing Service Profile\n")
    print("| Method | Avg Latency (ms) | P95 Latency (ms) | Avg CPU % | Peak RSS (MB) |")
    print("|--------|------------------|------------------|-----------|---------------|")
    for row in report["parsing_service_profile"]:
        print(
            f"| {row['method']} | {row['avg_latency_ms']:.2f} | {row['p95_latency_ms']:.2f} | "
            f"{row['avg_cpu_pct_process']:.2f} | {row['peak_rss_mb']:.2f} |"
        )

    print("\n## Payload Reduction\n")
    print("| Dataset | n | Vanilla Tokens | Agent Tokens | Reduction |")
    print("|---------|---|----------------|--------------|-----------|")
    for row in report["payload_profile"]:
        print(
            f"| {row['dataset']} | {row['n']} | {row['vanilla_tokens']:.1f} | "
            f"{row['agent_tokens']:.1f} | {row['reduction_tokens']:.3f} |"
        )

    print("\n## Device Feasibility (Memory-Budget Simulation)\n")
    print("| Device | RAM (GB) | Safe Budget (MB) | Edge Init | Local Qwen | NuSy Service | Qwen Service | Max Parallel NuSy |")
    print("|--------|----------|------------------|-----------|------------|--------------|--------------|-------------------|")
    for row in report["device_feasibility"]:
        print(
            f"| {row['device']} | {row['ram_gb']:.1f} | {row['safe_budget_mb']:.0f} | "
            f"{'yes' if row['fits_edge_init'] else 'no'} | "
            f"{'yes' if row['fits_local_qwen'] else 'no'} | "
            f"{'yes' if row['fits_nusy_service'] else 'no'} | "
            f"{'yes' if row['fits_qwen_service'] else 'no'} | "
            f"{row['max_parallel_nusy']} |"
        )

    print(f"\n[Saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
