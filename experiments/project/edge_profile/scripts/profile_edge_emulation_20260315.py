from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import psutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from experiments.rq3 import tools as rq3_tools  # type: ignore
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import (  # type: ignore
    EDGE_PROFILE_REPORT_DIR,
    EDGE_PROFILE_RESULTS_DIR,
    PROCESSED_DATA_DIR,
    ensure_dirs,
)
from src.system.edge_node import NuSyEdgeNode
from src.utils.metrics import MetricsCalculator


BENCH_V2_PATH = PROCESSED_DATA_DIR / "e2e_scaled_benchmark_v2.json"


def _estimate_tokens(text: str) -> int:
    return MetricsCalculator.estimate_tokens(text)


def _system_info() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "memory_gb": round(vm.total / (1024**3), 2),
        "cpu": cpu,
    }


def _profile_memory() -> Dict[str, float]:
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024**2)
    t0 = time.perf_counter()
    node = NuSyEdgeNode()
    startup_s = time.perf_counter() - t0
    rss_after = proc.memory_info().rss / (1024**2)
    return {
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "startup_seconds": round(startup_s, 3),
        "node": node,
    }


def _sample_cases(cases: List[Dict[str, Any]], per_ds: int) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    by_ds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        ds = str(case.get("dataset", ""))
        by_ds[ds].append(case)
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        picked.extend(by_ds.get(ds, [])[:per_ds])
    return picked


def _profile_payload_and_latency(node: NuSyEdgeNode, per_ds: int = 3) -> List[Dict[str, Any]]:
    cases = _load_benchmark(str(BENCH_V2_PATH))
    sample = _sample_cases(cases, per_ds=per_ds)
    rows: List[Dict[str, Any]] = []
    for case in sample:
        raw = str(case.get("raw_log", "") or "")
        dataset = str(case.get("dataset", "HDFS"))
        domain = "hdfs" if dataset == "HDFS" else ("openstack" if dataset == "OpenStack" else "hadoop")
        ds_header = "HDFS" if dataset == "Hadoop" else dataset
        lines = [line for line in raw.split("\n") if line.strip()]
        alert = lines[-1] if lines else raw
        clean_alert = NuSyEdgeNode.preprocess_header(alert, ds_header) or alert

        t0 = time.perf_counter()
        parsed_template, _, _, route = node.parse_log_stream(clean_alert, ds_header)
        local_parse_ms = (time.perf_counter() - t0) * 1000.0
        candidates = rq3_tools.causal_navigator(parsed_template or clean_alert, domain)

        vanilla_payload = raw
        agent_payload = f"template: {parsed_template}\ncausal_candidates: {candidates}"
        rows.append(
            {
                "dataset": dataset,
                "case_id": str(case.get("case_id", "")),
                "window_chars": len(vanilla_payload),
                "window_tokens": _estimate_tokens(vanilla_payload),
                "agent_chars": len(agent_payload),
                "agent_tokens": _estimate_tokens(agent_payload),
                "reduction_chars": round(1.0 - (len(agent_payload) / max(len(vanilla_payload), 1)), 4),
                "reduction_tokens": round(1.0 - (_estimate_tokens(agent_payload) / max(_estimate_tokens(vanilla_payload), 1)), 4),
                "local_parse_latency_ms": round(local_parse_ms, 3),
                "route": route,
            }
        )
    return rows


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "window_chars": 0.0,
            "window_tokens": 0.0,
            "agent_chars": 0.0,
            "agent_tokens": 0.0,
            "reduction_chars": 0.0,
            "reduction_tokens": 0.0,
            "local_parse_latency_ms": 0.0,
        }
    )
    for row in rows:
        ds = row["dataset"]
        agg[ds]["n"] += 1
        for key in [
            "window_chars",
            "window_tokens",
            "agent_chars",
            "agent_tokens",
            "reduction_chars",
            "reduction_tokens",
            "local_parse_latency_ms",
        ]:
            agg[ds][key] += float(row[key])

    out: Dict[str, Dict[str, float]] = {}
    for ds, vals in agg.items():
        n = max(vals["n"], 1.0)
        out[ds] = {
            "sample_cases": int(vals["n"]),
            "avg_window_chars": round(vals["window_chars"] / n, 1),
            "avg_window_tokens": round(vals["window_tokens"] / n, 1),
            "avg_agent_chars": round(vals["agent_chars"] / n, 1),
            "avg_agent_tokens": round(vals["agent_tokens"] / n, 1),
            "avg_reduction_chars": round(vals["reduction_chars"] / n, 4),
            "avg_reduction_tokens": round(vals["reduction_tokens"] / n, 4),
            "avg_local_parse_latency_ms": round(vals["local_parse_latency_ms"] / n, 3),
        }
    return out


def _write_report(system_info: Dict[str, Any], memory_info: Dict[str, Any], aggregates: Dict[str, Any], out_json: Path) -> Path:
    report_path = EDGE_PROFILE_REPORT_DIR / "edge_profile_status_20260315.md"
    lines = [
        "# Edge-Oriented Emulation Status (2026-03-15)",
        "",
        "This profiling run characterizes the rebuilt pipeline under an edge-oriented local execution assumption.",
        "",
        "## Scope",
        "",
        "- Single-node local workstation profiling only",
        "- No explicit CPU throttling, memory caps, or external edge board",
        "- Intended to support an edge-oriented deployment claim rather than a hardware-constrained claim",
        "",
        "## System",
        "",
        f"- Platform: `{system_info['platform']}`",
        f"- Machine: `{system_info['machine']}`",
        f"- Python: `{system_info['python_version']}`",
        f"- Memory: `{system_info['memory_gb']}` GB",
        f"- CPU physical/logical: `{system_info['cpu']['physical_cores']}` / `{system_info['cpu']['logical_cores']}`",
        "",
        "## Memory / startup",
        "",
        f"- RSS before NuSyEdgeNode: `{memory_info['rss_before_mb']}` MB",
        f"- RSS after NuSyEdgeNode: `{memory_info['rss_after_mb']}` MB",
        f"- RSS delta: `{memory_info['rss_delta_mb']}` MB",
        f"- Startup time: `{memory_info['startup_seconds']}` s",
        "",
        "## Payload / local parse summary",
        "",
    ]
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        if ds not in aggregates:
            continue
        row = aggregates[ds]
        lines.extend(
            [
                f"### {ds}",
                f"- Sample cases: `{row['sample_cases']}`",
                f"- Avg full-window chars/tokens: `{row['avg_window_chars']}` / `{row['avg_window_tokens']}`",
                f"- Avg agent payload chars/tokens: `{row['avg_agent_chars']}` / `{row['avg_agent_tokens']}`",
                f"- Avg payload reduction chars/tokens: `{row['avg_reduction_chars']}` / `{row['avg_reduction_tokens']}`",
                f"- Avg local parse latency: `{row['avg_local_parse_latency_ms']}` ms",
                "",
            ]
        )
    lines.append(f"Source JSON: `{out_json}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()
    system_info = _system_info()
    memory_info = _profile_memory()
    node = memory_info.pop("node")
    rows = _profile_payload_and_latency(node, per_ds=3)
    aggregates = _aggregate(rows)

    out_json = EDGE_PROFILE_RESULTS_DIR / "edge_profile_summary_20260315.json"
    write_json(
        out_json,
        {
            "mode": "edge_oriented_emulation",
            "system": system_info,
            "memory_startup": memory_info,
            "aggregates": aggregates,
            "rows": rows,
        },
    )
    report_path = _write_report(system_info, memory_info, aggregates, out_json)
    print(f"[Saved] {out_json}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
