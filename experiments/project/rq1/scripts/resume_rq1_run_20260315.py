from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REBUILD_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _REBUILD_ROOT.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.thesis_rebuild_20260315.rq1.scripts.run_rq1_pilot_20260315 import (
    NOISE_LEVELS,
    _choose_top_refs,
    _inject_noise,
    _load_manifest,
    _parse_with_nusy_rebuild,
)
from experiments.thesis_rebuild_20260315.shared.components.drain_baseline import DrainBaseline
from experiments.thesis_rebuild_20260315.shared.evaluators.normalized_pa import exact_match_hit
from experiments.thesis_rebuild_20260315.shared.utils.io_utils import write_json
from experiments.thesis_rebuild_20260315.shared.utils.project_paths import MANIFEST_DIR, RQ1_RESULTS_DIR, ensure_dirs
from src.system.edge_node import NuSyEdgeNode
from src.system.knowledge_base import KnowledgeBase
from src.utils.llm_client import LLMClient
from src.utils.noise_injector import NoiseInjector
from experiments.rq123_e2e.noise_hadoop_aggressive_20260311 import get_hadoop_aggressive_injector
from experiments.rq123_e2e.noise_injector_rq1_20260311 import get_rq1_injector
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--manifest-name", type=str, default="rq1_manifest_full_20260315.json")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--noise-levels", type=str, default="")
    return parser.parse_args()


def _read_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _existing_keys(rows: Iterable[dict]) -> Set[Tuple[str, str, str, str]]:
    return {
        (
            str(row["dataset"]),
            str(row["case_id"]),
            f"{float(row['noise']):.1f}",
            str(row["method"]),
        )
        for row in rows
    }


def _summarize(rows: List[dict], noise_levels: List[float], manifest_name: str, run_tag: str) -> dict:
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    clean_sanity: Dict[str, Dict[str, float]] = defaultdict(dict)

    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], f"{float(row['noise']):.1f}", row["method"])].append(row)

    for (dataset, noise, method), part in sorted(grouped.items()):
        pa = sum(float(r["pa_hit"]) for r in part) / len(part)
        lat = sum(float(r["latency_ms"]) for r in part) / len(part)
        summary.setdefault(dataset, {})[f"{noise}:{method}"] = {
            "cases": len(part),
            "pa": round(pa, 4),
            "latency_ms": round(lat, 3),
        }
        if noise == "0.0":
            clean_sanity.setdefault(dataset, {})[method] = round(pa, 4)

    degenerate_flags = []
    for dataset, methods in clean_sanity.items():
        for method, pa in methods.items():
            if method != "NuSy" and pa <= 0.05:
                degenerate_flags.append({"dataset": dataset, "method": method, "clean_pa": pa})

    return {
        "manifest": manifest_name,
        "run_tag": run_tag,
        "noise_levels": noise_levels,
        "clean_sanity": clean_sanity,
        "degenerate_flags": degenerate_flags,
        "summary": summary,
    }


def main() -> str:
    args = _parse_args()
    ensure_dirs()
    manifest = _load_manifest(args.manifest_name)
    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    noise_levels = NOISE_LEVELS
    if args.noise_levels.strip():
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]

    out_csv = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_rows_20260315.csv"
    out_json = RQ1_RESULTS_DIR / f"rq1_{args.run_tag}_summary_20260315.json"
    fieldnames = [
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
    ]

    existing_rows = _read_existing_rows(out_csv)
    done = _existing_keys(existing_rows)

    expected_rows = 0
    for dataset, meta in manifest["datasets"].items():
        if selected and dataset not in selected:
            continue
        expected_rows += len(meta["eval_cases"]) * len(noise_levels) * 3

    print(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "manifest": args.manifest_name,
                "existing_rows": len(existing_rows),
                "expected_rows": expected_rows,
            },
            indent=2,
        )
    )

    if len(existing_rows) >= expected_rows and expected_rows > 0:
        payload = _summarize(existing_rows, noise_levels, args.manifest_name, args.run_tag)
        write_json(out_json, payload)
        print(f"[Resume] existing rows already complete for {args.run_tag}")
        print(f"[Saved] {out_json}")
        return str(out_json)

    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    generic_injector = NoiseInjector(seed=2026)
    hdfs_injector = get_rq1_injector(seed=2026)
    hadoop_injector = get_hadoop_aggressive_injector(seed=2026)

    appended = 0
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if out_csv.stat().st_size == 0:
            writer.writeheader()

        for dataset, meta in manifest["datasets"].items():
            if selected and dataset not in selected:
                continue

            ref_cases = meta["reference_cases"]
            eval_cases = meta["eval_cases"]
            drain = DrainBaseline(reference_logs=[c["clean_alert"] for c in ref_cases])
            qwen_refs = [(c["clean_alert"], c["gt_template"]) for c in ref_cases]
            edge_node.cache = {}
            edge_node.kb = KnowledgeBase(
                collection_name=f"thesis_rebuild_rq1_{args.run_tag}_{dataset.lower()}",
                persist_path=f"experiments/thesis_rebuild_20260315/rq1/results/chroma_{args.run_tag}_{dataset.lower()}",
            )
            if qwen_refs:
                edge_node.kb.add_knowledge(
                    raw_logs=[ref[0] for ref in qwen_refs],
                    templates=[ref[1] for ref in qwen_refs],
                    dataset_type=dataset,
                )

            print(f"[RQ1-RESUME] dataset={dataset} refs={len(ref_cases)} eval={len(eval_cases)}")
            for case_idx, case in enumerate(eval_cases, start=1):
                pending_keys = [
                    (dataset, case["case_id"], f"{noise:.1f}", method)
                    for noise in noise_levels
                    for method in ("Drain", "NuSy", "Qwen")
                    if (dataset, case["case_id"], f"{noise:.1f}", method) not in done
                ]
                if not pending_keys:
                    continue
                print(f"[RQ1-RESUME] case {case_idx}/{len(eval_cases)} {case['case_id']} pending={len(pending_keys)}")
                for noise in noise_levels:
                    noise_str = f"{noise:.1f}"
                    needed = {
                        method
                        for method in ("Drain", "NuSy", "Qwen")
                        if (dataset, case["case_id"], noise_str, method) not in done
                    }
                    if not needed:
                        continue

                    noisy_alert = _inject_noise(
                        dataset=dataset,
                        clean_alert=case["clean_alert"],
                        noise_level=noise,
                        generic_injector=generic_injector,
                        hdfs_injector=hdfs_injector,
                        hadoop_injector=hadoop_injector,
                    )
                    clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, dataset) or noisy_alert
                    gt = case["gt_template"]

                    predictions: Dict[str, dict] = {}
                    if "Drain" in needed:
                        t0 = time.perf_counter()
                        pred = drain.parse(clean_for_parse)
                        predictions["Drain"] = {
                            "prediction": pred,
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                            "route": "drain_replayed_refs",
                        }

                    if "NuSy" in needed:
                        t0 = time.perf_counter()
                        pred, _, _, route = _parse_with_nusy_rebuild(edge_node, noisy_alert, dataset)
                        predictions["NuSy"] = {
                            "prediction": pred,
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                            "route": route,
                        }

                    if "Qwen" in needed:
                        t0 = time.perf_counter()
                        top_refs = _choose_top_refs(clean_for_parse, qwen_refs, top_k=3)
                        if len(top_refs) == 1:
                            pred = qwen.parse_with_rag(clean_for_parse, top_refs[0][0], top_refs[0][1])
                        elif len(top_refs) > 1:
                            pred = qwen.parse_with_multi_rag(clean_for_parse, top_refs)
                        else:
                            pred = qwen.parse_with_multi_rag(clean_for_parse, qwen_refs)
                        predictions["Qwen"] = {
                            "prediction": pred,
                            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                            "route": "qwen_refs",
                        }

                    batch_rows = []
                    for method, payload in predictions.items():
                        row = {
                            "dataset": dataset,
                            "case_id": case["case_id"],
                            "noise": noise_str,
                            "method": method,
                            "gt_template": gt,
                            "prediction": payload["prediction"],
                            "pa_hit": exact_match_hit(payload["prediction"], gt),
                            "latency_ms": payload["latency_ms"],
                            "gt_source": case["gt_source"],
                            "route": payload["route"],
                        }
                        batch_rows.append(row)
                        done.add((dataset, case["case_id"], noise_str, method))

                    for row in batch_rows:
                        writer.writerow(row)
                    if batch_rows:
                        f.flush()
                        appended += len(batch_rows)
                        pa_map = {row["method"]: row["pa_hit"] for row in batch_rows}
                        print(
                            f"[RQ1-RESUME] dataset={dataset} case={case['case_id']} noise={noise_str} "
                            f"written={','.join(sorted(predictions))} "
                            f"PA={pa_map}"
                        )

    final_rows = _read_existing_rows(out_csv)
    payload = _summarize(final_rows, noise_levels, args.manifest_name, args.run_tag)
    payload["resume_meta"] = {
        "existing_rows_before_resume": len(existing_rows),
        "appended_rows": appended,
        "expected_rows": expected_rows,
        "final_rows": len(final_rows),
    }
    write_json(out_json, payload)
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_json}")
    return str(out_json)


if __name__ == "__main__":
    main()
