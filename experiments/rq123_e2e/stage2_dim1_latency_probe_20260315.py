"""
Latency probe for the Stage2 Dim1 protocol.

This script mirrors the Stage2 midscale parsing setup but focuses only on
steady-state latency. It uses a smaller representative subset per dataset to
keep runtime reasonable while preserving the same prompt / parser configuration.
"""

import csv
import os
import sys
import time
from statistics import mean
from typing import Dict, List, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e.noise_hadoop_aggressive_20260311 import get_hadoop_aggressive_injector
from experiments.rq123_e2e.noise_injector_rq1_20260311 import get_rq1_injector
from experiments.rq123_e2e.stage2_midscale_20260311 import (  # type: ignore
    HDFS_MINI_RQ1_SIZE,
    MIDCASE_PER_DATASET,
    NOISE_LEVELS,
    _build_qwen_refs,
    _choose_best_ref,
    _inject_noise,
    _inject_noise_rq1_hdfs,
    _load_hdfs_mini_rq1,
    _select_midscale_cases,
)
from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from experiments.rq1_hadoop.run_rq1_hadoop_benchmark_20260311 import (  # type: ignore
    _build_qwen_refs_from_cases as _build_hadoop_qwen_refs,
    _select_abnormal_cases as _select_hadoop_cases,
)
from src.perception.drain_parser import DrainParser
from src.system.edge_node import NuSyEdgeNode
from src.utils.llm_client import LLMClient
from src.utils.noise_injector import NoiseInjector


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
OUT_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
OUT_CSV = os.path.join(OUT_DIR, "stage2_dim1_latency_probe_20260315.csv")


def _load_openstack_cases(limit: int) -> List[Dict[str, object]]:
    cases = _load_benchmark(BENCH_V2_PATH)
    selected = [c for c in _select_midscale_cases(cases, per_ds=MIDCASE_PER_DATASET) if str(c.get("dataset", "")) == "OpenStack"]
    return selected[:limit]


def _load_hadoop_cases(limit: int) -> List[Dict[str, object]]:
    data_raw = os.path.join(_PROJECT_ROOT, "data", "raw")
    hadoop_root = os.path.join(data_raw, "Hadoop")
    abnormal_label_path = os.path.join(hadoop_root, "abnormal_label.txt")
    return _select_hadoop_cases(hadoop_root, abnormal_label_path, max_apps=limit)


def run(cases_per_dataset: int = 5) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    edge_node = NuSyEdgeNode()
    qwen = LLMClient()
    drain = DrainParser()
    generic_injector = NoiseInjector(seed=2026)
    hdfs_injector = get_rq1_injector(seed=2026)
    hadoop_injector = get_hadoop_aggressive_injector(seed=2026)

    qwen_refs_hdfs, hdfs_cases = _load_hdfs_mini_rq1(drain, per_ds=cases_per_dataset, seed=2026, max_refs=15)
    openstack_cases = _load_openstack_cases(cases_per_dataset)
    refs_openstack = _build_qwen_refs(_load_benchmark(BENCH_V2_PATH)).get("OpenStack", [])
    hadoop_cases = _load_hadoop_cases(cases_per_dataset)
    refs_hadoop = _build_hadoop_qwen_refs(hadoop_cases, drain, max_refs=5)

    rows: List[Dict[str, object]] = []
    total = (len(hdfs_cases) + len(openstack_cases) + len(hadoop_cases)) * len(NOISE_LEVELS)
    pbar = tqdm(total=total, desc="Stage2 latency probe", unit="case")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f_csv:
        fieldnames = ["dataset", "noise", "method", "latency_ms"]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for case in hdfs_cases:
            clean_tail = str(case.get("clean", ""))
            for nl in NOISE_LEVELS:
                noisy_alert = _inject_noise_rq1_hdfs(clean_tail, hdfs_injector, nl)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, "HDFS") or noisy_alert

                t0 = time.perf_counter()
                drain.parse(clean_for_parse)
                rows.append({"dataset": "HDFS", "noise": nl, "method": "Drain", "latency_ms": (time.perf_counter() - t0) * 1000.0})

                t0 = time.perf_counter()
                edge_node.parse_log_stream(noisy_alert, "HDFS")
                rows.append({"dataset": "HDFS", "noise": nl, "method": "NuSy", "latency_ms": (time.perf_counter() - t0) * 1000.0})

                t0 = time.perf_counter()
                best_ref = _choose_best_ref(clean_for_parse, qwen_refs_hdfs) if qwen_refs_hdfs else None
                if best_ref:
                    qwen.parse_with_rag(clean_for_parse, best_ref[0], best_ref[1])
                rows.append({"dataset": "HDFS", "noise": nl, "method": "Qwen", "latency_ms": (time.perf_counter() - t0) * 1000.0})
                pbar.update(1)

        for case in openstack_cases:
            raw = str(case.get("raw_log", "") or "")
            lines = [l for l in raw.split("\n") if l.strip()]
            base_tail = lines[-1] if lines else raw
            clean_tail = NuSyEdgeNode.preprocess_header(base_tail, "OpenStack") or base_tail
            for nl in NOISE_LEVELS:
                noisy_alert = _inject_noise(clean_tail, "OpenStack", generic_injector, nl)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy_alert, "OpenStack") or noisy_alert

                t0 = time.perf_counter()
                drain.parse(clean_for_parse)
                rows.append({"dataset": "OpenStack", "noise": nl, "method": "Drain", "latency_ms": (time.perf_counter() - t0) * 1000.0})

                t0 = time.perf_counter()
                edge_node.parse_log_stream(noisy_alert, "OpenStack")
                rows.append({"dataset": "OpenStack", "noise": nl, "method": "NuSy", "latency_ms": (time.perf_counter() - t0) * 1000.0})

                t0 = time.perf_counter()
                if refs_openstack:
                    qwen.parse_with_multi_rag(clean_for_parse, refs_openstack)
                rows.append({"dataset": "OpenStack", "noise": nl, "method": "Qwen", "latency_ms": (time.perf_counter() - t0) * 1000.0})
                pbar.update(1)

        for case in hadoop_cases:
            alert = str(case.get("alert", "") or "")
            for nl in NOISE_LEVELS:
                hadoop_injector.injection_rate = nl
                noisy = hadoop_injector.inject(alert)
                clean_for_parse = NuSyEdgeNode.preprocess_header(noisy, "Hadoop") or noisy

                t0 = time.perf_counter()
                drain.parse(clean_for_parse)
                rows.append({"dataset": "Hadoop", "noise": nl, "method": "Drain", "latency_ms": (time.perf_counter() - t0) * 1000.0})

                t0 = time.perf_counter()
                edge_node.parse_log_stream(noisy, "Hadoop")
                rows.append({"dataset": "Hadoop", "noise": nl, "method": "NuSy", "latency_ms": (time.perf_counter() - t0) * 1000.0})

                t0 = time.perf_counter()
                best_ref = _choose_best_ref(clean_for_parse, refs_hadoop) if refs_hadoop else None
                if best_ref:
                    qwen.parse_with_rag(clean_for_parse, best_ref[0], best_ref[1])
                rows.append({"dataset": "Hadoop", "noise": nl, "method": "Qwen", "latency_ms": (time.perf_counter() - t0) * 1000.0})
                pbar.update(1)

        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "noise": row["noise"],
                    "method": row["method"],
                    "latency_ms": round(float(row["latency_ms"]), 3),
                }
            )

    pbar.close()

    print("\n## Stage2 Latency Probe Summary\n")
    by_key: Dict[Tuple[str, float, str], List[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["noise"]), str(row["method"]))
        by_key.setdefault(key, []).append(float(row["latency_ms"]))
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        for nl in NOISE_LEVELS:
            parts = []
            for method in ["NuSy", "Drain", "Qwen"]:
                vals = by_key.get((dataset, nl, method), [])
                if vals:
                    parts.append(f"{method}={mean(vals):.1f}ms")
            print(f"{dataset} noise={nl:.1f} " + " ".join(parts))

    print(f"\n[Saved] {OUT_CSV}")
    return OUT_CSV


if __name__ == "__main__":
    run()
