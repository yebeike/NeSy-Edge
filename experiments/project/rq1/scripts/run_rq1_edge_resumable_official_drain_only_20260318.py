from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.drain_official_replay_20260318 import (  # noqa: E402
    OfficialDrainBaseline,
)
from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (  # noqa: E402
    NOISE_LEVELS,
    build_noise_injectors,
    configure_edge_budget,
    current_rss_mb,
    exact_match_hit,
    inject_noise,
    load_manifest,
    prepare_runtime_alert,
    read_existing_rows,
    row_key,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json  # noqa: E402
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import RQ1_RESULTS_DIR, ensure_dirs  # noqa: E402


FIELDNAMES = [
    "dataset",
    "case_id",
    "noise",
    "method",
    "gt_template",
    "prediction",
    "pa_hit",
    "latency_ms",
    "gt_source",
    "route",
    "query_chars",
    "ref_chars",
    "ref_count",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", type=str, required=True)
    ap.add_argument("--manifest-name", type=str, required=True)
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--noise-levels", type=str, default="")
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--memory-target-mb", type=int, default=2304)
    ap.add_argument("--max-cases-per-dataset", type=int, default=0)
    ap.add_argument("--case-start", type=int, default=0)
    ap.add_argument("--case-stop", type=int, default=0)
    ap.add_argument("--skip-finalize", action="store_true")
    ap.add_argument("--chunk-meta-path", type=str, default="")
    return ap.parse_args()


def _noise_list(raw: str) -> List[float]:
    if not raw.strip():
        return list(NOISE_LEVELS)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _selected_datasets(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def _case_rows_keyset(rows: List[dict]) -> set[tuple[str, str, str, str]]:
    return {row_key(row) for row in rows}


def _append_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _slice_eval_cases(eval_cases: List[dict], args: argparse.Namespace) -> List[dict]:
    start = max(0, int(args.case_start or 0))
    stop = int(args.case_stop or 0)
    sliced = eval_cases[start:stop or None]
    if args.max_cases_per_dataset > 0:
        sliced = sliced[: args.max_cases_per_dataset]
    return sliced


def _summarize_drain_only(rows: List[dict], manifest_name: str, run_tag: str, edge_meta: dict) -> dict:
    grouped: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), f"{float(row['noise']):.1f}")].append(row)

    summary: Dict[str, Dict[str, dict]] = defaultdict(dict)
    clean_sanity: Dict[str, Dict[str, float]] = defaultdict(dict)
    route_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for (dataset, noise), part in sorted(grouped.items()):
        latencies = [float(r["latency_ms"]) for r in part]
        pa = sum(float(r["pa_hit"]) for r in part) / len(part)
        summary[dataset][f"{noise}:Drain"] = {
            "cases": len(part),
            "pa": round(pa, 4),
            "latency_ms": round(sum(latencies) / len(latencies), 3),
        }
        if noise == "0.0":
            clean_sanity[dataset]["Drain"] = round(pa, 4)
        route_counts[dataset][f"{noise}:Drain"] = dict(Counter(str(r.get("route", "")) for r in part if r.get("route")))

    acceptance_flags = []
    expected_rows = int(edge_meta.get("expected_rows", 0) or 0)
    final_rows = int(edge_meta.get("final_rows", len(rows)) or 0)
    if expected_rows > 0 and final_rows != expected_rows:
        acceptance_flags.append(f"Run incomplete: final rows {final_rows} != expected {expected_rows}")
    peak_rss = float(edge_meta.get("peak_rss_mb", 0.0) or 0.0)
    memory_target = float(edge_meta.get("memory_target_mb", 0.0) or 0.0)
    edge_meta["memory_target_ok"] = (memory_target <= 0.0) or (peak_rss <= memory_target)
    if memory_target > 0.0 and peak_rss > memory_target:
        acceptance_flags.append(
            f"Edge budget: peak RSS {peak_rss:.2f} MB exceeded memory target {memory_target:.0f} MB"
        )

    return {
        "manifest": manifest_name,
        "run_tag": run_tag,
        "noise_levels": NOISE_LEVELS,
        "clean_sanity": clean_sanity,
        "route_counts": route_counts,
        "acceptance_flags": acceptance_flags,
        "summary": summary,
        "edge_meta": edge_meta,
        "drain_only": True,
    }


def main() -> str:
    args = _parse_args()
    ensure_dirs()
    thread_meta = configure_edge_budget(args.cpu_threads, args.memory_target_mb)
    manifest = load_manifest(args.manifest_name)
    selected = _selected_datasets(args.datasets)
    noise_levels = _noise_list(args.noise_levels)

    out_csv = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_rows_20260317.csv"
    out_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_summary_20260317.json"

    existing_rows = read_existing_rows(out_csv)
    done = _case_rows_keyset(existing_rows)
    expected_rows = 0
    for dataset, meta in manifest["datasets"].items():
        if selected and dataset not in selected:
            continue
        eval_cases = _slice_eval_cases(meta["eval_cases"], args)
        expected_rows += len(eval_cases) * len(noise_levels)

    print(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "manifest": args.manifest_name,
                "existing_rows": len(existing_rows),
                "expected_rows": expected_rows,
                "cpu_threads": args.cpu_threads,
                "memory_target_mb": args.memory_target_mb,
                "edge_setup": thread_meta,
                "drain_only": True,
            },
            indent=2,
        )
    )

    injectors = build_noise_injectors(seed=manifest.get("seed", 2026))
    peak_rss_mb = current_rss_mb()
    appended_rows = 0

    for dataset, meta in manifest["datasets"].items():
        if selected and dataset not in selected:
            continue
        ref_cases = meta["reference_cases"]
        eval_cases = _slice_eval_cases(meta["eval_cases"], args)
        drain = OfficialDrainBaseline(
            reference_logs=[prepare_runtime_alert(case["clean_alert"], dataset) for case in ref_cases],
            dataset=dataset,
        )
        print(f"[RQ1-DRAIN] dataset={dataset} refs={len(ref_cases)} eval={len(eval_cases)}")

        for noise in noise_levels:
            noise_key = f"{noise:.1f}"
            print(f"[RQ1-DRAIN] dataset={dataset} noise={noise_key}")
            for case_idx, case in enumerate(eval_cases, start=1):
                key = (dataset, case["case_id"], noise_key, "Drain")
                if key in done:
                    peak_rss_mb = max(peak_rss_mb, current_rss_mb())
                    continue

                noisy_alert = inject_noise(dataset, case["clean_alert"], noise, injectors)
                parse_alert = prepare_runtime_alert(noisy_alert, dataset)
                t0 = time.perf_counter()
                pred = drain.parse(parse_alert)
                row = {
                    "dataset": dataset,
                    "case_id": case["case_id"],
                    "noise": noise_key,
                    "method": "Drain",
                    "gt_template": case["gt_template"],
                    "prediction": pred,
                    "pa_hit": exact_match_hit(pred, case["gt_template"]),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "gt_source": case["gt_source"],
                    "route": "official_drain_replayed_refs",
                    "query_chars": len(parse_alert),
                    "ref_chars": 0,
                    "ref_count": len(ref_cases),
                }
                _append_rows(out_csv, [row])
                done.add(key)
                appended_rows += 1
                peak_rss_mb = max(peak_rss_mb, current_rss_mb())
                print(
                    f"[RQ1-DRAIN] dataset={dataset} noise={noise_key} case={case_idx}/{len(eval_cases)} "
                    f"pa={row['pa_hit']} pred={pred[:120]}"
                )

    edge_meta = {
        "cpu_threads": args.cpu_threads,
        "memory_target_mb": args.memory_target_mb,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "expected_rows": expected_rows,
        "final_rows": len(read_existing_rows(out_csv)),
        "edge_setup": thread_meta,
        "appended_rows_this_run": appended_rows,
        "drain_only": True,
    }
    if args.chunk_meta_path:
        write_json(Path(args.chunk_meta_path), {"manifest": args.manifest_name, "run_tag": args.run_tag, "edge_meta": edge_meta})
    if args.skip_finalize:
        return str(out_csv)

    payload = _summarize_drain_only(read_existing_rows(out_csv), args.manifest_name, args.run_tag, edge_meta)
    write_json(out_json, payload)
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    return str(out_json)


if __name__ == "__main__":
    main()
