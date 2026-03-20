from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (
    NOISE_LEVELS,
    BudgetedLLMAdapter,
    DrainBaseline,
    LightweightNuSyParser,
    ReferenceIndex,
    build_noise_injectors,
    configure_edge_budget,
    current_rss_mb,
    exact_match_hit,
    group_rows,
    inject_noise,
    load_manifest,
    prepare_runtime_alert,
    read_existing_rows,
    row_key,
    summarize_rows,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import RQ1_RESULTS_DIR, ensure_dirs
from src.utils.llm_client import LLMClient


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
    ap.add_argument("--qwen-mode", type=str, default="direct", choices=["direct", "top1_ref"])
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
        expected_rows += len(eval_cases) * len(noise_levels) * 3

    print(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "manifest": args.manifest_name,
                "existing_rows": len(existing_rows),
                "expected_rows": expected_rows,
                "cpu_threads": args.cpu_threads,
                "memory_target_mb": args.memory_target_mb,
                "qwen_mode": args.qwen_mode,
                "edge_setup": thread_meta,
            },
            indent=2,
        )
    )

    shared_llm = LLMClient()
    llm = BudgetedLLMAdapter(shared_llm)
    index = ReferenceIndex(manifest)
    nusy = LightweightNuSyParser(index, llm)
    injectors = build_noise_injectors(seed=manifest.get("seed", 2026))

    peak_rss_mb = current_rss_mb()
    appended_rows = 0

    for dataset, meta in manifest["datasets"].items():
        if selected and dataset not in selected:
            continue
        llm.warmup(dataset, index)
        ref_cases = meta["reference_cases"]
        eval_cases = _slice_eval_cases(meta["eval_cases"], args)
        drain = DrainBaseline(
            reference_logs=[prepare_runtime_alert(case["clean_alert"], dataset) for case in ref_cases],
            dataset=dataset,
        )
        print(f"[RQ1-EDGE] dataset={dataset} refs={len(ref_cases)} eval={len(eval_cases)}")

        for noise in noise_levels:
            noise_key = f"{noise:.1f}"
            print(f"[RQ1-EDGE] dataset={dataset} noise={noise_key} case_local_cache_only=1")
            for case_idx, case in enumerate(eval_cases, start=1):
                noisy_alert = inject_noise(dataset, case["clean_alert"], noise, injectors)
                gt = case["gt_template"]

                pending = [
                    method
                    for method in ("Drain", "NuSy", "Qwen")
                    if (dataset, case["case_id"], noise_key, method) not in done
                ]
                if not pending:
                    peak_rss_mb = max(peak_rss_mb, current_rss_mb())
                    continue

                parse_alert = prepare_runtime_alert(noisy_alert, dataset)
                batch_rows: List[dict] = []

                if "Drain" in pending:
                    t0 = time.perf_counter()
                    pred = drain.parse(parse_alert)
                    batch_rows.append(
                        {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise_key,
                            "method": "Drain",
                            "gt_template": gt,
                            "prediction": pred,
                            "pa_hit": exact_match_hit(pred, gt),
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                            "gt_source": case["gt_source"],
                            "route": "drain_replayed_refs",
                            "query_chars": len(parse_alert),
                            "ref_chars": 0,
                            "ref_count": len(ref_cases),
                        }
                    )

                if "NuSy" in pending:
                    nusy.reset_cache()
                    pred, lat_ms, route, meta_nusy = nusy.parse(noisy_alert, dataset)
                    batch_rows.append(
                        {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise_key,
                            "method": "NuSy",
                            "gt_template": gt,
                            "prediction": pred,
                            "pa_hit": exact_match_hit(pred, gt),
                            "latency_ms": round(lat_ms, 3),
                            "gt_source": case["gt_source"],
                            "route": route,
                            "query_chars": meta_nusy.get("query_chars", len(parse_alert)),
                            "ref_chars": meta_nusy.get("ref_chars", 0),
                            "ref_count": meta_nusy.get("ref_count", 0),
                        }
                    )

                if "Qwen" in pending:
                    t0 = time.perf_counter()
                    if args.qwen_mode == "top1_ref":
                        refs = index.search(parse_alert, dataset, top_k=1)
                        pred, meta_qwen = llm.parse(
                            parse_alert,
                            dataset,
                            [(ref.raw_log, ref.template) for ref in refs],
                            semantic_cleanup=False,
                        )
                        route = "llm_top1_ref"
                    else:
                        pred, meta_qwen = llm.parse_direct(
                            parse_alert,
                            dataset,
                            semantic_cleanup=False,
                        )
                        route = "llm_direct"
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    batch_rows.append(
                        {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise_key,
                            "method": "Qwen",
                            "gt_template": gt,
                            "prediction": pred,
                            "pa_hit": exact_match_hit(pred, gt),
                            "latency_ms": round(lat_ms, 3),
                            "gt_source": case["gt_source"],
                            "route": route,
                            "query_chars": meta_qwen.get("query_chars", len(parse_alert)),
                            "ref_chars": meta_qwen.get("ref_chars", 0),
                            "ref_count": meta_qwen.get("ref_count", 0),
                        }
                    )

                _append_rows(out_csv, batch_rows)
                appended_rows += len(batch_rows)
                for row in batch_rows:
                    done.add(row_key(row))
                peak_rss_mb = max(peak_rss_mb, current_rss_mb())
                print(
                    f"[RQ1-EDGE] dataset={dataset} noise={noise_key} "
                    f"case={case_idx}/{len(eval_cases)} id={case['case_id']} "
                    f"written={','.join(row['method'] for row in batch_rows)}"
                )

    final_rows = read_existing_rows(out_csv)
    edge_meta = {
        **thread_meta,
        "memory_target_mb": args.memory_target_mb,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "existing_rows_before_resume": len(existing_rows),
        "appended_rows": appended_rows,
        "expected_rows": expected_rows,
        "final_rows": len(final_rows),
        "case_start": int(args.case_start or 0),
        "case_stop": int(args.case_stop or 0),
    }
    if args.chunk_meta_path:
        chunk_meta_path = Path(args.chunk_meta_path)
        chunk_meta_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            chunk_meta_path,
            {
                "run_tag": args.run_tag,
                "manifest": args.manifest_name,
                "datasets": sorted(selected) if selected else sorted(manifest["datasets"].keys()),
                "noise_levels": noise_levels,
                "qwen_mode": args.qwen_mode,
                "edge_meta": edge_meta,
            },
        )
        print(f"[Saved] {chunk_meta_path}")
    if args.skip_finalize:
        print(f"[Saved] {out_csv}")
        return str(out_csv)
    payload = summarize_rows(final_rows, args.manifest_name, args.run_tag, edge_meta)
    write_json(out_json, payload)
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    return str(out_json)


if __name__ == "__main__":
    main()
