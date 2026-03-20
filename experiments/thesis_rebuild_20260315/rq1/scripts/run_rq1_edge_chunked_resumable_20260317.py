from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.components.edge_protocol_20260317 import (  # noqa: E402
    NOISE_LEVELS,
    load_manifest,
    read_existing_rows,
    row_key,
    summarize_rows,
)
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json  # noqa: E402
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import RQ1_RESULTS_DIR, ensure_dirs  # noqa: E402


RUNNER = _SCRIPT_DIR / "run_rq1_edge_resumable_20260317.py"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", type=str, required=True)
    ap.add_argument("--manifest-name", type=str, required=True)
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--noise-levels", type=str, default="")
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--memory-target-mb", type=int, default=2304)
    ap.add_argument("--cases-per-chunk", type=int, default=0)
    ap.add_argument("--cases-per-chunk-spec", type=str, default="")
    ap.add_argument("--qwen-mode", type=str, default="direct", choices=["direct", "top1_ref"])
    return ap.parse_args()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _selected_datasets(raw: str, manifest: dict) -> list[str]:
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    if not selected:
        return sorted(manifest["datasets"].keys())
    return [dataset for dataset in sorted(manifest["datasets"].keys()) if dataset in set(selected)]


def _noise_list(raw: str) -> list[float]:
    if not raw.strip():
        return list(NOISE_LEVELS)
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _group_counts(rows: list[dict]) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counter[(str(row["dataset"]), f"{float(row['noise']):.1f}", str(row["method"]))] += 1
    return counter


def _done_keys(rows: list[dict]) -> set[tuple[str, str, str, str]]:
    return {row_key(row) for row in rows}


def _cases_per_chunk_map(raw: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in raw.split(","):
        piece = item.strip()
        if not piece:
            continue
        dataset, sep, size = piece.partition(":")
        if not sep:
            raise ValueError(f"invalid cases-per-chunk-spec item: {piece}")
        mapping[dataset.strip()] = max(0, int(size.strip()))
    return mapping


def _log(path: Path, message: str) -> None:
    line = f"[{_timestamp()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> str:
    args = _parse_args()
    ensure_dirs()
    manifest = load_manifest(args.manifest_name)
    datasets = _selected_datasets(args.datasets, manifest)
    noise_levels = _noise_list(args.noise_levels)
    cases_per_chunk_by_dataset = _cases_per_chunk_map(args.cases_per_chunk_spec)

    out_csv = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_rows_20260317.csv"
    out_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_summary_20260317.json"
    state_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_chunk_state_20260317.json"
    chunk_dir = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_chunks_20260317"
    log_md = _PROJECT_ROOT / "experiments" / "thesis_rebuild_20260315" / "reports" / f"rq1_{args.run_tag}_chunked_20260317.md"

    expected_rows = 0
    expected_cases_by_dataset: dict[str, int] = {}
    for dataset in datasets:
        eval_cases = manifest["datasets"][dataset]["eval_cases"]
        expected_cases_by_dataset[dataset] = len(eval_cases)
        expected_rows += len(eval_cases) * len(noise_levels) * 3

    if state_json.exists():
        state = json.loads(state_json.read_text(encoding="utf-8"))
    else:
        state = {
            "run_tag": args.run_tag,
            "manifest": args.manifest_name,
            "datasets": datasets,
            "noise_levels": [f"{noise:.1f}" for noise in noise_levels],
            "chunks": {},
            "expected_rows": expected_rows,
        }

    log_md.write_text("# RQ1 Chunked Runner\n\n", encoding="utf-8")
    _log(log_md, f"start run_tag={args.run_tag} expected_rows={expected_rows}")

    rows = read_existing_rows(out_csv)
    group_counts = _group_counts(rows)
    done_keys = _done_keys(rows)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        eval_cases = manifest["datasets"][dataset]["eval_cases"]
        expected_cases = expected_cases_by_dataset[dataset]
        chunk_size = cases_per_chunk_by_dataset.get(dataset, args.cases_per_chunk)
        if chunk_size <= 0 or chunk_size >= expected_cases:
            case_ranges = [(0, expected_cases)]
        else:
            case_ranges = [(start, min(start + chunk_size, expected_cases)) for start in range(0, expected_cases, chunk_size)]
        for noise in noise_levels:
            noise_key = f"{noise:.1f}"
            for case_start, case_stop in case_ranges:
                chunk_case_ids = [case["case_id"] for case in eval_cases[case_start:case_stop]]
                if len(case_ranges) == 1:
                    chunk_id = f"{dataset}_{noise_key}"
                else:
                    chunk_id = f"{dataset}_{noise_key}_{case_start:04d}_{case_stop:04d}"
                chunk_meta = chunk_dir / f"{chunk_id}.json"
                chunk_complete = all(
                    (dataset, case_id, noise_key, method) in done_keys
                    for case_id in chunk_case_ids
                    for method in ("Drain", "NuSy", "Qwen")
                )
                if chunk_complete and chunk_meta.exists():
                    _log(log_md, f"skip completed chunk={chunk_id}")
                    continue

                cmd = [
                    sys.executable,
                    str(RUNNER),
                    "--run-tag",
                    args.run_tag,
                    "--manifest-name",
                    args.manifest_name,
                    "--datasets",
                    dataset,
                    "--noise-levels",
                    noise_key,
                    "--cpu-threads",
                    str(args.cpu_threads),
                    "--memory-target-mb",
                    str(args.memory_target_mb),
                    "--qwen-mode",
                    args.qwen_mode,
                    "--skip-finalize",
                    "--chunk-meta-path",
                    str(chunk_meta),
                    "--case-start",
                    str(case_start),
                    "--case-stop",
                    str(case_stop),
                ]
                _log(log_md, "RUN " + " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=str(_PROJECT_ROOT))
                chunk_payload = json.loads(chunk_meta.read_text(encoding="utf-8"))
                state["chunks"][chunk_id] = {
                    "dataset": dataset,
                    "noise": noise_key,
                    "case_start": case_start,
                    "case_stop": case_stop,
                    "peak_rss_mb": chunk_payload["edge_meta"]["peak_rss_mb"],
                    "final_rows_after_chunk": chunk_payload["edge_meta"]["final_rows"],
                }
                write_json(state_json, state)
                rows = read_existing_rows(out_csv)
                group_counts = _group_counts(rows)
                done_keys = _done_keys(rows)
                _log(
                    log_md,
                    f"completed chunk={chunk_id} peak_rss_mb={chunk_payload['edge_meta']['peak_rss_mb']}"
                    f" total_rows={len(rows)}",
                )

    final_rows = read_existing_rows(out_csv)
    peak_rss_mb = 0.0
    for meta in state["chunks"].values():
        peak_rss_mb = max(peak_rss_mb, float(meta.get("peak_rss_mb", 0.0) or 0.0))
    edge_meta = {
        "cpu_threads": args.cpu_threads,
        "memory_target_mb": args.memory_target_mb,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "chunk_mode": "dataset_noise_case_window" if args.cases_per_chunk > 0 or cases_per_chunk_by_dataset else "dataset_noise",
        "chunk_count": len(state["chunks"]),
        "completed_chunks": len(state["chunks"]),
        "expected_rows": expected_rows,
        "final_rows": len(final_rows),
        "qwen_mode": args.qwen_mode,
        "cases_per_chunk": args.cases_per_chunk,
        "cases_per_chunk_by_dataset": cases_per_chunk_by_dataset,
    }
    payload = summarize_rows(final_rows, args.manifest_name, args.run_tag, edge_meta)
    write_json(out_json, payload)
    _log(log_md, f"saved summary={out_json.name} acceptance_flags={payload.get('acceptance_flags', [])}")
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    return str(out_json)


if __name__ == "__main__":
    main()
