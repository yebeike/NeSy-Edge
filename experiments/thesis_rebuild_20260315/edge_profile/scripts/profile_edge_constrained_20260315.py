from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict

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

from experiments.thesis_rebuild_20260315.edge_profile.scripts.profile_edge_emulation_20260315 import (  # type: ignore
    _aggregate,
    _profile_payload_and_latency,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import (
    EDGE_PROFILE_REPORT_DIR,
    EDGE_PROFILE_RESULTS_DIR,
    ensure_dirs,
)
from src.system.edge_node import NuSyEdgeNode


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--memory-limit-mb", type=int, default=4096)
    ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--run-tag", type=str, default="cpu1_mem4096_sample2")
    return ap.parse_args()


def _apply_thread_budget(cpu_threads: int) -> Dict[str, Any]:
    env_updates = {
        "OMP_NUM_THREADS": str(cpu_threads),
        "OPENBLAS_NUM_THREADS": str(cpu_threads),
        "MKL_NUM_THREADS": str(cpu_threads),
        "VECLIB_MAXIMUM_THREADS": str(cpu_threads),
        "NUMEXPR_NUM_THREADS": str(cpu_threads),
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(env_updates)
    applied = {"env": env_updates}
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        applied["torch_num_threads"] = cpu_threads
        applied["torch_interop_threads"] = 1
    except Exception as exc:  # pragma: no cover
        applied["torch_error"] = str(exc)
    return applied


def _apply_memory_budget(memory_limit_mb: int) -> Dict[str, Any]:
    limit_bytes = int(memory_limit_mb * 1024 * 1024)
    results: Dict[str, Any] = {"requested_limit_mb": memory_limit_mb}
    for name in ["RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"]:
        if not hasattr(resource, name):
            continue
        res = getattr(resource, name)
        try:
            before_soft, before_hard = resource.getrlimit(res)
            target_soft = limit_bytes
            if before_hard not in (-1, resource.RLIM_INFINITY):
                target_soft = min(target_soft, before_hard)
            resource.setrlimit(res, (target_soft, before_hard))
            after = resource.getrlimit(res)
            results[name] = {"before": [before_soft, before_hard], "after": list(after), "applied": True}
        except Exception as exc:
            results[name] = {"applied": False, "error": str(exc)}
    return results


def _child_worker(out_path: str, cpu_threads: int, memory_limit_mb: int, per_dataset: int) -> None:
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024**2)
    thread_info = _apply_thread_budget(cpu_threads)
    memory_info = _apply_memory_budget(memory_limit_mb)

    t0 = time.perf_counter()
    node = NuSyEdgeNode()
    startup_s = time.perf_counter() - t0
    rss_after = proc.memory_info().rss / (1024**2)

    rows = _profile_payload_and_latency(node, per_ds=per_dataset)
    aggregates = _aggregate(rows)

    payload = {
        "mode": "constrained_edge_profile",
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "logical_cores_host": psutil.cpu_count(logical=True),
            "physical_cores_host": psutil.cpu_count(logical=False),
            "memory_gb_host": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "constraints": {
            "cpu_threads": cpu_threads,
            "memory_limit_mb": memory_limit_mb,
            "thread_budget": thread_info,
            "memory_budget": memory_info,
        },
        "memory_startup": {
            "rss_before_mb": round(rss_before, 2),
            "rss_after_mb": round(rss_after, 2),
            "rss_delta_mb": round(rss_after - rss_before, 2),
            "startup_seconds": round(startup_s, 3),
        },
        "aggregates": aggregates,
        "rows": rows,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _monitor_process(pid: int) -> Dict[str, Any]:
    proc = psutil.Process(pid)
    max_rss_mb = 0.0
    cpu_samples = []
    started = time.perf_counter()
    while proc.is_running():
        try:
            rss_mb = proc.memory_info().rss / (1024**2)
            max_rss_mb = max(max_rss_mb, rss_mb)
            cpu_samples.append(proc.cpu_percent(interval=0.2))
            if proc.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.Error:
            break
    return {
        "max_observed_rss_mb": round(max_rss_mb, 2),
        "avg_cpu_percent": round(sum(cpu_samples) / max(len(cpu_samples), 1), 2),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def _write_report(out_json: Path, payload: Dict[str, Any], monitor: Dict[str, Any]) -> Path:
    report_path = EDGE_PROFILE_REPORT_DIR / f"edge_profile_constrained_{payload['constraints']['cpu_threads']}t_{payload['constraints']['memory_limit_mb']}mb_20260315.md"
    memory_budget = payload["constraints"]["memory_budget"]
    any_memory_limit_applied = any(
        isinstance(details, dict) and details.get("applied")
        for key, details in memory_budget.items()
        if key != "requested_limit_mb"
    )
    lines = [
        "# Constrained Edge Profile (2026-03-15)",
        "",
        "This run explicitly evaluates the local online path under a bounded thread budget and a target memory budget.",
        "",
        "## Constraint settings",
        "",
        f"- Requested CPU thread budget: `{payload['constraints']['cpu_threads']}`",
        f"- Requested memory budget: `{payload['constraints']['memory_limit_mb']}` MB",
        f"- Host platform: `{payload['system']['platform']}`",
        f"- Host memory: `{payload['system']['memory_gb_host']}` GB",
        "",
        "## Observed process behavior",
        "",
        f"- Startup time: `{payload['memory_startup']['startup_seconds']}` s",
        f"- RSS before/after load: `{payload['memory_startup']['rss_before_mb']}` / `{payload['memory_startup']['rss_after_mb']}` MB",
        f"- RSS delta: `{payload['memory_startup']['rss_delta_mb']}` MB",
        f"- Max observed RSS: `{monitor['max_observed_rss_mb']}` MB",
        f"- Average sampled CPU percent: `{monitor['avg_cpu_percent']}`",
        f"- Child wall time: `{monitor['wall_seconds']}` s",
        f"- Hard memory cap applied by OS: `{any_memory_limit_applied}`",
        "",
        "## Dataset summaries",
        "",
    ]
    if not any_memory_limit_applied:
        meets_target = monitor["max_observed_rss_mb"] <= payload["constraints"]["memory_limit_mb"]
        budget_note = (
            f"- Even without a hard OS cap, the observed peak RSS remained below the target budget (`{monitor['max_observed_rss_mb']}` MB < `{payload['constraints']['memory_limit_mb']}` MB)."
            if meets_target
            else f"- Even without a hard OS cap, the observed peak RSS exceeded the target budget (`{monitor['max_observed_rss_mb']}` MB > `{payload['constraints']['memory_limit_mb']}` MB)."
        )
        lines.extend(
            [
                "## Notes",
                "",
                "- The single-thread budget was applied via explicit environment and Torch thread settings.",
                "- macOS rejected Python-level `setrlimit` calls for the requested memory cap during this run.",
                budget_note,
                "",
            ]
        )
    for ds in ["HDFS", "OpenStack", "Hadoop"]:
        if ds not in payload["aggregates"]:
            continue
        row = payload["aggregates"][ds]
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
    args = _parse_args()
    temp_path = EDGE_PROFILE_RESULTS_DIR / f"edge_profile_constrained_{args.run_tag}_tmp_20260315.json"
    out_json = EDGE_PROFILE_RESULTS_DIR / f"edge_profile_constrained_{args.run_tag}_20260315.json"

    proc = mp.Process(
        target=_child_worker,
        args=(str(temp_path), args.cpu_threads, args.memory_limit_mb, args.per_dataset),
        daemon=False,
    )
    proc.start()
    monitor = _monitor_process(proc.pid)
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Constrained edge profile failed with exit code {proc.exitcode}")

    payload = json.loads(temp_path.read_text(encoding="utf-8"))
    payload["monitor"] = monitor
    payload["run_tag"] = args.run_tag
    payload["assessment"] = {
        "meets_target_memory_budget": monitor["max_observed_rss_mb"] <= args.memory_limit_mb,
        "memory_headroom_mb": round(args.memory_limit_mb - monitor["max_observed_rss_mb"], 2),
        "smallest_thread_budget_tested": args.cpu_threads,
    }
    write_json(out_json, payload)
    report_path = _write_report(out_json, payload, monitor)
    print(f"[Saved] {out_json}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
