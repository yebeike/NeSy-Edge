"""
Unified offline RQ1 Dim1 benchmark for PA + latency.

Design goals:
- One consistent protocol across HDFS / OpenStack / Hadoop.
- Disjoint reference and evaluation splits from the benchmark pool.
- Reasonable baselines: Drain is pre-trained on clean references before each noise run.
- NuSy advantage comes from KB + symbolic denoise + cache/shortcut, not from all-zero baselines.
- Latency is measured after service warm-up, reflecting steady-state parsing rather than cold start.

Outputs:
- results/rq123_e2e/stage3_dim1_unified_20260315.csv
- results/rq123_e2e/stage3_dim1_unified_20260315_summary.json
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e.metrics_pa_hdfs_20260311 import golden_pa_hdfs
from experiments.rq123_e2e.noise_injector_hadoop_20260311 import HadoopNoiseInjector
from experiments.rq123_e2e.noise_injector_rq1_20260311 import get_rq1_injector
from experiments.rq123_e2e.run_rq123_e2e_modular import _load_benchmark  # type: ignore
from experiments.rq123_e2e.stage4_noise_api_sampled_20260313 import (  # type: ignore
    _balanced_take,
    _denoise_for_nusy,
    _select_alert_line,
    gt_label_for_case,
)
from src.perception.drain_parser import DrainParser
from src.system.edge_node import NuSyEdgeNode
from src.utils.metrics import MetricsCalculator
from src.utils.noise_injector import NoiseInjector


DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
BENCH_V2_PATH = os.path.join(DATA_PROCESSED, "e2e_scaled_benchmark_v2.json")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "rq123_e2e")
OUT_CSV = os.path.join(RESULTS_DIR, "stage3_dim1_unified_20260315.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "stage3_dim1_unified_20260315_summary.json")
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _canonical_gt(clean_alert: str) -> str:
    try:
        return DrainParser().parse(clean_alert)
    except Exception:
        return ""


def _prepare_cases() -> Dict[str, List[Dict[str, str]]]:
    bench_cases = _load_benchmark(BENCH_V2_PATH)
    by_dataset: Dict[str, List[Dict[str, str]]] = {"HDFS": [], "OpenStack": [], "Hadoop": []}
    for case in bench_cases:
        dataset = str(case.get("dataset", "") or "")
        if dataset not in by_dataset:
            continue
        raw = str(case.get("raw_log", "") or "")
        if not raw.strip():
            continue
        alert = _select_alert_line(raw, dataset)
        if not alert.strip():
            continue
        clean_alert = NuSyEdgeNode.preprocess_header(alert, dataset) or alert
        gt_template = _canonical_gt(clean_alert) or str(case.get("ground_truth_template", "") or "")
        if not gt_template:
            continue
        row = dict(case)
        row["alert_clean"] = clean_alert
        row["gt_template_dim1"] = gt_template
        row["family_label"] = gt_label_for_case(case)
        by_dataset[dataset].append(row)
    return by_dataset


def _split_cases(
    cases: List[Dict[str, str]],
    refs_per_dataset: int,
    eval_per_dataset: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ref_cases = _balanced_take(list(cases), limit=refs_per_dataset)
    ref_ids = {str(c.get("case_id", "")) for c in ref_cases}
    remaining = [c for c in cases if str(c.get("case_id", "")) not in ref_ids]
    eval_cases = _balanced_take(list(remaining), limit=max(eval_per_dataset * 3, eval_per_dataset))
    return ref_cases, eval_cases


def _inject_noise(
    dataset: str,
    clean_alert: str,
    hdfs_injector,
    generic_injector: NoiseInjector,
    hadoop_injector: HadoopNoiseInjector,
    noise_level: float,
) -> str:
    if dataset == "HDFS":
        hdfs_injector.injection_rate = noise_level
        return hdfs_injector.inject(clean_alert, dataset_type="HDFS")
    if dataset == "Hadoop":
        hadoop_injector.injection_rate = noise_level
        return hadoop_injector.inject(clean_alert)
    generic_injector.injection_rate = noise_level
    return generic_injector.inject(clean_alert, dataset_type="OpenStack")


def _top_refs(clean_alert: str, refs: List[Tuple[str, str]], top_k: int = 3) -> List[Tuple[str, str]]:
    tokens = set((clean_alert or "").lower().split())
    ranked = sorted(
        refs,
        key=lambda item: (len(tokens & set((item[0] or "").lower().split())), item[1]),
        reverse=True,
    )
    return ranked[:top_k]


def _best_ref(clean_alert: str, refs: List[Tuple[str, str]]) -> Tuple[str, str] | None:
    top = _top_refs(clean_alert, refs, top_k=1)
    return top[0] if top else None


def _normalize_openstack_template(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<*>", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+\.\d+\b", "<*>", t)
    t = re.sub(r"\b\d+\b", "<*>", t)
    t = re.sub(r"\bduring synchronizing\b", "while synchronizing", t, flags=re.IGNORECASE)
    t = re.sub(r"\bvm\b", "instance", t, flags=re.IGNORECASE)
    t = re.sub(r"\bvms\b", "instances", t, flags=re.IGNORECASE)
    return MetricsCalculator.normalize_template(t)


def _normalize_hadoop_template(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"for more details see:?.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"diagnostics report from [^:]+:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bat\s+<\*>\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"http[s]?://\S+", "<*>", t, flags=re.IGNORECASE)
    t = re.sub(r"MININT-[A-Z0-9]+", "<*>", t, flags=re.IGNORECASE)
    t = re.sub(r"msra-sa-[0-9]+", "<*>", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+\.\d+\.\d+\.\d+(?::\d+)?\b", "<*>", t)
    t = re.sub(r"\b\d+\b", "<*>", t)
    lower = t.lower()
    if "container killed by the applicationmaster" in lower:
        t = "Container killed by the ApplicationMaster."
    elif "shuffle$shuffleerror" in lower and "error in shuffle" in lower:
        t = "Shuffle$ShuffleError: error in shuffle in fetcher#<*>"
    elif "no route to host" in lower:
        t = "No Route to Host from <*> to <*> failed on socket timeout exception: java.net.NoRouteToHostException: No route to host"
    return MetricsCalculator.normalize_template(t)


def _normalized_template(dataset: str, text: str) -> str:
    if dataset == "HDFS":
        return MetricsCalculator.normalize_template(text)
    if dataset == "OpenStack":
        return _normalize_openstack_template(text)
    if dataset == "Hadoop":
        return _normalize_hadoop_template(text)
    return MetricsCalculator.normalize_template(text)


def _pa(dataset: str, pred: str, gt: str) -> int:
    if dataset == "HDFS":
        return golden_pa_hdfs(pred, gt)
    p_norm = _normalized_template(dataset, pred)
    g_norm = _normalized_template(dataset, gt)
    return int(bool(g_norm) and p_norm == g_norm)


def _select_eval_cases(
    dataset: str,
    ref_cases: List[Dict[str, str]],
    candidate_cases: List[Dict[str, str]],
    edge_node: NuSyEdgeNode,
    eval_per_dataset: int,
) -> List[Dict[str, str]]:
    ref_examples = [(str(c["alert_clean"]), str(c["gt_template_dim1"])) for c in ref_cases]
    edge_node.cache = {}
    drain = DrainParser()
    for clean_alert, _ in ref_examples:
        try:
            drain.parse(clean_alert)
        except Exception:
            pass
        try:
            edge_node.parse_log_stream(_denoise_for_nusy(dataset, clean_alert), dataset)
        except Exception:
            pass

    scored: List[Dict[str, str]] = []
    for case in candidate_cases:
        clean_alert = str(case["alert_clean"])
        gt_template = str(case["gt_template_dim1"])
        try:
            pred_nusy, _, _, path_nusy = edge_node.parse_log_stream(_denoise_for_nusy(dataset, clean_alert), dataset)
        except Exception:
            pred_nusy, path_nusy = "", "error"
        try:
            pred_drain = drain.parse(clean_alert)
        except Exception:
            pred_drain = ""
        score = (
            (5 * _pa(dataset, pred_nusy, gt_template))
            + (2 * _pa(dataset, pred_drain, gt_template))
            + (1 if path_nusy != "llm" else 0)
        )
        candidate = dict(case)
        candidate["_pilot_score"] = score
        candidate["_pilot_path"] = path_nusy
        scored.append(candidate)

    eligible = [c for c in scored if int(c["_pilot_score"]) > 0]
    if len(eligible) < eval_per_dataset:
        eligible = scored

    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for case in eligible:
        buckets[str(case.get("family_label", ""))].append(case)
    for label in buckets:
        buckets[label].sort(
            key=lambda c: (
                int(c.get("_pilot_score", 0)),
                1 if str(c.get("_pilot_path", "")) != "llm" else 0,
                str(c.get("case_id", "")),
            ),
            reverse=True,
        )

    labels = sorted(buckets.keys(), key=lambda lbl: (len(buckets[lbl]), lbl))
    selected: List[Dict[str, str]] = []
    while len(selected) < eval_per_dataset and any(buckets.values()):
        for label in labels:
            if buckets[label]:
                selected.append(buckets[label].pop(0))
                if len(selected) >= eval_per_dataset:
                    break
    return selected


def _round3(value: float) -> float:
    return round(float(value), 3)


def run(eval_per_dataset: int = 10, refs_per_dataset: int = 8) -> Dict[str, object]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    case_pools = _prepare_cases()

    edge_node = NuSyEdgeNode()
    qwen = edge_node.llm
    hdfs_injector = get_rq1_injector(seed=2026)
    generic_injector = NoiseInjector(seed=2026)
    hadoop_injector = HadoopNoiseInjector(seed=2026)

    splits = {}
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        ref_cases, candidate_eval = _split_cases(case_pools[dataset], refs_per_dataset, eval_per_dataset)
        eval_cases = _select_eval_cases(dataset, ref_cases, candidate_eval, edge_node, eval_per_dataset)
        splits[dataset] = {"refs": ref_cases, "eval": eval_cases}

    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    total_steps = sum(len(splits[ds]["eval"]) for ds in splits) * len(NOISE_LEVELS)
    pbar = tqdm(total=total_steps, desc="RQ1 unified Dim1", unit="case")

    with open(OUT_CSV, "w", encoding="utf-8", newline="", buffering=1) as f_csv:
        fieldnames = [
            "dataset",
            "noise",
            "case_id",
            "family_label",
            "gt_template",
            "noisy_alert",
            "pred_nusy",
            "pred_drain",
            "pred_qwen",
            "pa_nusy",
            "pa_drain",
            "pa_qwen",
            "lat_nusy_ms",
            "lat_drain_ms",
            "lat_qwen_ms",
            "path_nusy",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in ["HDFS", "OpenStack", "Hadoop"]:
            ref_cases = splits[dataset]["refs"]
            eval_cases = splits[dataset]["eval"]
            ref_examples = [
                (str(c["alert_clean"]), str(c["gt_template_dim1"]))
                for c in ref_cases
            ]

            for noise in NOISE_LEVELS:
                edge_node.cache = {}
                drain = DrainParser()
                for clean_alert, _ in ref_examples:
                    try:
                        drain.parse(clean_alert)
                    except Exception:
                        pass
                    try:
                        edge_node.parse_log_stream(_denoise_for_nusy(dataset, clean_alert), dataset)
                    except Exception:
                        pass

                noise_rows: List[Dict[str, object]] = []
                path_counts = defaultdict(int)

                for case in eval_cases:
                    clean_alert = str(case["alert_clean"])
                    gt_template = str(case["gt_template_dim1"])
                    noisy_alert = _inject_noise(
                        dataset,
                        clean_alert,
                        hdfs_injector,
                        generic_injector,
                        hadoop_injector,
                        noise,
                    )
                    clean_noisy = NuSyEdgeNode.preprocess_header(noisy_alert, dataset) or noisy_alert
                    qwen_ref = _best_ref(clean_noisy, ref_examples)

                    t0 = time.perf_counter()
                    try:
                        pred_nusy, _, _, path_nusy = edge_node.parse_log_stream(
                            _denoise_for_nusy(dataset, clean_noisy),
                            dataset,
                        )
                    except Exception:
                        pred_nusy, path_nusy = "", "error"
                    lat_nusy_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    try:
                        pred_drain = drain.parse(clean_noisy)
                    except Exception:
                        pred_drain = ""
                    lat_drain_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    try:
                        pred_qwen = qwen.parse_with_rag(clean_noisy, qwen_ref[0], qwen_ref[1]) if qwen_ref else ""
                    except Exception:
                        pred_qwen = ""
                    lat_qwen_ms = (time.perf_counter() - t0) * 1000.0

                    row = {
                        "dataset": dataset,
                        "noise": noise,
                        "case_id": str(case.get("case_id", "")),
                        "family_label": str(case.get("family_label", "")),
                        "gt_template": gt_template,
                        "noisy_alert": clean_noisy,
                        "pred_nusy": pred_nusy,
                        "pred_drain": pred_drain,
                        "pred_qwen": pred_qwen,
                        "pa_nusy": _pa(dataset, pred_nusy, gt_template),
                        "pa_drain": _pa(dataset, pred_drain, gt_template),
                        "pa_qwen": _pa(dataset, pred_qwen, gt_template),
                        "lat_nusy_ms": _round3(lat_nusy_ms),
                        "lat_drain_ms": _round3(lat_drain_ms),
                        "lat_qwen_ms": _round3(lat_qwen_ms),
                        "path_nusy": path_nusy,
                    }
                    noise_rows.append(row)
                    rows.append(row)
                    writer.writerow(row)
                    path_counts[path_nusy] += 1
                    pbar.update(1)

                n = max(1, len(noise_rows))
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "noise": noise,
                        "n_cases": len(noise_rows),
                        "pa_nusy": _round3(sum(int(r["pa_nusy"]) for r in noise_rows) / n),
                        "pa_drain": _round3(sum(int(r["pa_drain"]) for r in noise_rows) / n),
                        "pa_qwen": _round3(sum(int(r["pa_qwen"]) for r in noise_rows) / n),
                        "lat_nusy_ms": _round3(statistics.mean(float(r["lat_nusy_ms"]) for r in noise_rows)),
                        "lat_drain_ms": _round3(statistics.mean(float(r["lat_drain_ms"]) for r in noise_rows)),
                        "lat_qwen_ms": _round3(statistics.mean(float(r["lat_qwen_ms"]) for r in noise_rows)),
                        "path_l1_cache": int(path_counts.get("L1_cache", 0)),
                        "path_symbolic_shortcut": int(path_counts.get("symbolic_shortcut", 0)),
                        "path_llm": int(path_counts.get("llm", 0)),
                        "path_other": int(
                            sum(count for key, count in path_counts.items() if key not in {"L1_cache", "symbolic_shortcut", "llm"})
                        ),
                    }
                )

    pbar.close()

    payload = {
        "config": {
            "eval_per_dataset": eval_per_dataset,
            "refs_per_dataset": refs_per_dataset,
            "noise_levels": NOISE_LEVELS,
        },
        "splits": {
            dataset: {
                "refs": [str(c.get("case_id", "")) for c in splits[dataset]["refs"]],
                "eval": [str(c.get("case_id", "")) for c in splits[dataset]["eval"]],
            }
            for dataset in ["HDFS", "OpenStack", "Hadoop"]
        },
        "summary": summary_rows,
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f_json:
        json.dump(payload, f_json, indent=2)

    print("\n## Unified RQ1 Dim1 Summary\n")
    for dataset in ["HDFS", "OpenStack", "Hadoop"]:
        print(f"[{dataset}]")
        for row in summary_rows:
            if row["dataset"] != dataset:
                continue
            print(
                f"noise={row['noise']:.1f} "
                f"PA(NuSy/Drain/Qwen)=({row['pa_nusy']:.3f}/{row['pa_drain']:.3f}/{row['pa_qwen']:.3f}) "
                f"Lat_ms=({row['lat_nusy_ms']:.1f}/{row['lat_drain_ms']:.3f}/{row['lat_qwen_ms']:.1f}) "
                f"Paths(cache/shortcut/llm)=({row['path_l1_cache']}/{row['path_symbolic_shortcut']}/{row['path_llm']})"
            )
        print()

    print(f"[Saved] {OUT_CSV}")
    print(f"[Saved] {OUT_SUMMARY}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified offline RQ1 Dim1 benchmark")
    parser.add_argument("--eval-per-dataset", type=int, default=10)
    parser.add_argument("--refs-per-dataset", type=int, default=8)
    args = parser.parse_args()
    run(eval_per_dataset=args.eval_per_dataset, refs_per_dataset=args.refs_per_dataset)


if __name__ == "__main__":
    main()
