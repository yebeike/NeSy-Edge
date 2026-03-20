from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_csv, write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import EDGE_PROFILE_REPORT_DIR, EDGE_PROFILE_RESULTS_DIR, ensure_dirs

_PROFILE_SCRIPT = _SCRIPT_DIR / "profile_edge_constrained_20260315.py"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-threads", type=str, default="1,2")
    ap.add_argument("--memory-budgets-mb", type=str, default="2048,2304,2560,3072,4096")
    ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--run-tag", type=str, default="limit_sweep")
    return ap.parse_args()


def _recommended_safe_budget(max_rss: float) -> int:
    return int(math.ceil((max_rss * 1.10) / 256.0) * 256)


def main() -> None:
    ensure_dirs()
    args = _parse_args()
    cpu_threads = [int(x.strip()) for x in args.cpu_threads.split(",") if x.strip()]
    memory_budgets = [int(x.strip()) for x in args.memory_budgets_mb.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    json_paths: List[str] = []

    for cpu in cpu_threads:
        for memory_mb in memory_budgets:
            child_tag = f"{args.run_tag}_t{cpu}_m{memory_mb}"
            cmd = [
                sys.executable,
                str(_PROFILE_SCRIPT),
                "--cpu-threads",
                str(cpu),
                "--memory-limit-mb",
                str(memory_mb),
                "--per-dataset",
                str(args.per_dataset),
                "--run-tag",
                child_tag,
            ]
            print("[RUN]", " ".join(cmd))
            subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)
            out_json = EDGE_PROFILE_RESULTS_DIR / f"edge_profile_constrained_{child_tag}_20260315.json"
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            json_paths.append(str(out_json))
            rows.append(
                {
                    "cpu_threads": cpu,
                    "target_memory_mb": memory_mb,
                    "meets_target_memory_budget": payload["assessment"]["meets_target_memory_budget"],
                    "memory_headroom_mb": payload["assessment"]["memory_headroom_mb"],
                    "max_observed_rss_mb": payload["monitor"]["max_observed_rss_mb"],
                    "startup_seconds": payload["memory_startup"]["startup_seconds"],
                    "wall_seconds": payload["monitor"]["wall_seconds"],
                    "avg_cpu_percent": payload["monitor"]["avg_cpu_percent"],
                    "avg_hdfs_parse_ms": payload["aggregates"]["HDFS"]["avg_local_parse_latency_ms"],
                    "avg_openstack_parse_ms": payload["aggregates"]["OpenStack"]["avg_local_parse_latency_ms"],
                    "avg_hadoop_parse_ms": payload["aggregates"]["Hadoop"]["avg_local_parse_latency_ms"],
                    "hard_memory_cap_applied": any(
                        isinstance(details, dict) and details.get("applied")
                        for key, details in payload["constraints"]["memory_budget"].items()
                        if key != "requested_limit_mb"
                    ),
                    "json_path": str(out_json),
                }
            )

    rows.sort(key=lambda row: (row["cpu_threads"], row["target_memory_mb"]))
    csv_path = EDGE_PROFILE_RESULTS_DIR / f"edge_profile_limit_sweep_{args.run_tag}_20260315.csv"
    write_csv(csv_path, rows)

    feasible = [row for row in rows if row["meets_target_memory_budget"]]
    best_extreme = min(feasible, key=lambda row: (row["cpu_threads"], row["target_memory_mb"])) if feasible else None
    best_by_thread: Dict[int, Dict[str, Any]] = {}
    for cpu in cpu_threads:
        options = [row for row in feasible if row["cpu_threads"] == cpu]
        if options:
            best_by_thread[cpu] = min(options, key=lambda row: row["target_memory_mb"])

    max_rss_observed = max((row["max_observed_rss_mb"] for row in rows), default=0.0)
    summary = {
        "run_tag": args.run_tag,
        "cpu_threads_tested": cpu_threads,
        "memory_budgets_mb_tested": memory_budgets,
        "per_dataset": args.per_dataset,
        "best_extreme_config": best_extreme,
        "best_by_thread": best_by_thread,
        "max_rss_observed_mb": max_rss_observed,
        "recommended_safe_memory_budget_mb": _recommended_safe_budget(max_rss_observed) if max_rss_observed else None,
        "rows": rows,
        "json_paths": json_paths,
    }
    json_summary = EDGE_PROFILE_RESULTS_DIR / f"edge_profile_limit_sweep_{args.run_tag}_20260315.json"
    write_json(json_summary, summary)

    report_path = EDGE_PROFILE_REPORT_DIR / f"edge_profile_limit_sweep_{args.run_tag}_20260315.md"
    lines = [
        "# Edge Limit Sweep (2026-03-15)",
        "",
        f"- CPU thread budgets tested: `{cpu_threads}`",
        f"- Memory targets tested (MB): `{memory_budgets}`",
        f"- Sample cases per dataset: `{args.per_dataset}`",
        "",
        "## Summary",
        "",
    ]
    if best_extreme:
        lines.extend(
            [
                f"- Smallest feasible tested config: `{best_extreme['cpu_threads']} thread(s)` + `{best_extreme['target_memory_mb']}` MB",
                f"- Observed peak RSS under that config: `{best_extreme['max_observed_rss_mb']}` MB",
                f"- Memory headroom under that config: `{best_extreme['memory_headroom_mb']}` MB",
                f"- Recommended safer memory budget (10% headroom, 256 MB rounded): `{summary['recommended_safe_memory_budget_mb']}` MB",
                "",
            ]
        )
    else:
        lines.extend(["- No tested configuration met its target memory budget.", ""])

    lines.extend(["## Tested configurations", ""])
    for row in rows:
        lines.append(
            f"- t={row['cpu_threads']}, mem={row['target_memory_mb']} MB: "
            f"peak RSS `{row['max_observed_rss_mb']}` MB, "
            f"meets target `{row['meets_target_memory_budget']}`, "
            f"HDFS/OpenStack/Hadoop parse `{row['avg_hdfs_parse_ms']}` / `{row['avg_openstack_parse_ms']}` / `{row['avg_hadoop_parse_ms']}` ms"
        )
    lines.extend(
        [
            "",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_summary}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {json_summary}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
