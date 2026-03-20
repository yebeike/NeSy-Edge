import os
import sys
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Path fix: experiments/rq1_robustness -> project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.noise_injector import NoiseInjector
from src.utils.metrics import MetricsCalculator
from src.system.knowledge_base import KnowledgeBase


# ================= RQ1 Large-Scale Robustness Config =================
RUN_SEEDS = [42, 1024, 2026, 8888, 9999]  # 5 runs
BASE_POOL_SIZE = 20_000                    # from full logs (per dataset)
TRAIN_SIZE = 10_000                        # first half -> KB / cache warm-up
TEST_POOL_SIZE = 10_000                    # second half
TEST_SAMPLE_PER_RUN = 2_000                # per run sample from test pool
VANILLA_SAMPLE_PER_RUN = 200               # subset for vanilla baseline
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DATASETS = ["HDFS", "OpenStack"]

EXPORT_DIR = os.path.join(_PROJECT_ROOT, "results", "rq1_robustness")
EXPORT_PER_RUN = True
# ====================================================================


def _iter_log_files_for_dataset(dataset: str) -> List[str]:
    """Return ordered list of raw .log files for given dataset."""
    if dataset == "HDFS":
        base_dir = os.path.join(_PROJECT_ROOT, "data", "raw", "HDFS_v1")
        # Use HDFS_v1/HDFS.log as the full log file
        candidates = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".log")]
    else:
        base_dir = os.path.join(_PROJECT_ROOT, "data", "raw", "OpenStack_2")
        # Combine multiple OpenStack_2 logs in sorted name order
        candidates = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".log")]
    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(f"No .log files found for dataset={dataset} under {base_dir}")
    return candidates


def _read_first_n_logs(paths: List[str], n: int) -> List[str]:
    """Sequentially read up to n non-empty lines across a list of log files."""
    logs: List[str] = []
    for p in paths:
        if len(logs) >= n:
            break
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(logs) >= n:
                    break
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                logs.append(line)
    return logs


def _build_base_pool(dataset: str) -> Tuple[List[str], List[str]]:
    """
    Build base pool (first BASE_POOL_SIZE logs) and companion "clean templates" using Drain on clean logs.
    These templates serve as reference GT for robustness (stability) evaluation.
    """
    paths = _iter_log_files_for_dataset(dataset)
    base_logs = _read_first_n_logs(paths, BASE_POOL_SIZE)
    if not base_logs:
        raise RuntimeError(f"Empty base_logs for dataset={dataset}")
    # Generate clean templates via Drain on header-stripped logs
    drain = DrainParser()
    clean_templates: List[str] = []
    for raw in tqdm(base_logs, desc=f"Drain GT ({dataset})", unit="log"):
        clean = NuSyEdgeNode.preprocess_header(raw, dataset)
        if not clean:
            clean = raw
        try:
            tpl = drain.parse(clean)
        except Exception:
            tpl = ""
        clean_templates.append(tpl)
    return base_logs, clean_templates


def _split_train_test(base_logs: List[str], base_templates: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    n = min(len(base_logs), len(base_templates))
    base_logs = base_logs[:n]
    base_templates = base_templates[:n]
    train_n = min(TRAIN_SIZE, n // 2)
    test_n = min(TEST_POOL_SIZE, n - train_n)
    train_logs = base_logs[:train_n]
    train_gt = base_templates[:train_n]
    test_logs = base_logs[train_n:train_n + test_n]
    test_gt = base_templates[train_n:train_n + test_n]
    return train_logs, train_gt, test_logs, test_gt


def _sample_indices(pool_size: int, k: int, seed: int) -> List[int]:
    k_eff = min(k, pool_size)
    rng = random.Random(seed)
    return rng.sample(range(pool_size), k_eff)


def run_rq1_large_scale():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Pre-build a dedicated robustness KB (separate from default NuSy-Edge KB).
    kb = KnowledgeBase(collection_name="rq1_large_scale_kb", persist_path="data/chroma_db_rq1_large")

    # Build base pools and GT templates (Drain on clean logs)
    base_data: Dict[str, Dict[str, List[str]]] = {}
    for ds in DATASETS:
        logs, tpls = _build_base_pool(ds)
        base_data[ds] = {"logs": logs, "tpls": tpls}

    per_run_rows: List[Dict[str, object]] = []

    for run_seed in RUN_SEEDS:
        print("\n" + "=" * 120)
        print(f"[RQ1 Large-Scale] RUN seed={run_seed} | base_pool={BASE_POOL_SIZE} | train={TRAIN_SIZE} | test_pool={TEST_POOL_SIZE} | test_sample={TEST_SAMPLE_PER_RUN}")
        print("=" * 120)

        injector = NoiseInjector(seed=run_seed)
        nusy = NuSyEdgeNode()
        nusy.kb = kb

        for dataset_name in DATASETS:
            logs_all = base_data[dataset_name]["logs"]
            tpls_all = base_data[dataset_name]["tpls"]
            train_logs, train_gt, test_pool_logs, test_pool_gt = _split_train_test(logs_all, tpls_all)
            if not test_pool_logs:
                print(f"[WARN] Empty test pool for dataset={dataset_name}; skip.")
                continue

            # Ingest train half into KB (once per dataset per run)
            train_clean = []
            for r in train_logs:
                c = NuSyEdgeNode.preprocess_header(r, dataset_name)
                train_clean.append(c if c else r)
            kb.add_knowledge(train_clean, train_gt, dataset_name)

            # Pre-sample indices for this run
            test_indices = _sample_indices(len(test_pool_logs), TEST_SAMPLE_PER_RUN, seed=run_seed)
            test_logs = [test_pool_logs[i] for i in test_indices]
            test_gt = [test_pool_gt[i] for i in test_indices]

            # Vanilla subset indices from within 0..len(test_logs)-1
            vanilla_indices = _sample_indices(len(test_logs), VANILLA_SAMPLE_PER_RUN, seed=run_seed + 777)

            # Simple 3-shot references from train half
            vanilla_refs = []
            seen = set()
            for log, tpl in zip(train_logs, train_gt):
                if tpl not in seen and len(vanilla_refs) < 3:
                    seen.add(tpl)
                    vanilla_refs.append((log, tpl))
            if len(vanilla_refs) < 3 and train_logs:
                while len(vanilla_refs) < 3:
                    j = len(vanilla_refs) % len(train_logs)
                    vanilla_refs.append((train_logs[j], train_gt[j]))
            vanilla_refs = vanilla_refs[:3]

            for noise_rate in NOISE_LEVELS:
                injector.injection_rate = noise_rate

                drain = DrainParser()
                nusy.cache = {}

                results: Dict[str, Dict[str, object]] = {
                    "Drain": {"preds": [], "lats": []},
                    "NuSy-Edge": {"preds": [], "lats": [], "tokens": 0},
                    "Vanilla": {"preds": [], "lats": [], "tokens": 0},
                }

                nusy_count_total = 0
                nusy_count_hit = 0
                nusy_count_llm = 0
                nusy_sum_lat_total = 0.0
                nusy_sum_lat_llm = 0.0

                pbar = tqdm(test_logs, desc=f"seed={run_seed} {dataset_name} noise={noise_rate}", unit="log")
                for idx, raw_log in enumerate(pbar):
                    noisy_raw = injector.inject(raw_log, dataset_type=dataset_name)
                    clean_log_content = NuSyEdgeNode.preprocess_header(noisy_raw, dataset_name)
                    if not clean_log_content:
                        clean_log_content = noisy_raw

                    # Drain on noisy log
                    t0 = time.time()
                    try:
                        p_drain = drain.parse(clean_log_content)
                    except Exception:
                        p_drain = ""
                    results["Drain"]["lats"].append((time.time() - t0) * 1000)
                    results["Drain"]["preds"].append(p_drain)

                    # Vanilla only on subset
                    if idx in vanilla_indices:
                        t0 = time.time()
                        try:
                            p_vanilla = nusy.llm.parse_with_multi_rag(clean_log_content, vanilla_refs)
                        except Exception:
                            p_vanilla = ""
                        results["Vanilla"]["lats"].append((time.time() - t0) * 1000)
                        results["Vanilla"]["preds"].append(p_vanilla)
                        t_in = MetricsCalculator.estimate_tokens(clean_log_content) + sum(
                            MetricsCalculator.estimate_tokens(r[0]) for r in vanilla_refs
                        ) + 100
                        t_out = MetricsCalculator.estimate_tokens(p_vanilla)
                        results["Vanilla"]["tokens"] += (t_in + t_out)

                    # NuSy-Edge
                    p_nusy, lat_nusy, is_hit, path = nusy.parse_log_stream(noisy_raw, dataset_name)
                    results["NuSy-Edge"]["lats"].append(lat_nusy)
                    results["NuSy-Edge"]["preds"].append(p_nusy)

                    nusy_count_total += 1
                    nusy_sum_lat_total += float(lat_nusy)
                    if path in ("L1_cache", "symbolic_shortcut"):
                        nusy_count_hit += 1
                    if path == "llm":
                        nusy_count_llm += 1
                        nusy_sum_lat_llm += float(lat_nusy)

                    if path == "llm":
                        t_in_miss = MetricsCalculator.estimate_tokens(clean_log_content) + 300
                        t_out_miss = MetricsCalculator.estimate_tokens(p_nusy)
                        results["NuSy-Edge"]["tokens"] += (t_in_miss + t_out_miss)

                # Summarize metrics (per run, dataset, noise)
                # NOTE: For Vanilla we average over its own subset size.
                for method in ("Drain", "NuSy-Edge", "Vanilla"):
                    preds = results[method]["preds"]
                    if method == "Vanilla":
                        gt = [test_gt[i] for i in vanilla_indices[: len(preds)]]
                        denom = max(1, len(gt))
                        lat_list = results[method]["lats"]
                        avg_lat = float(sum(lat_list) / denom) if lat_list else 0.0
                        avg_tok = float(results[method]["tokens"] / denom) if denom else 0.0
                    else:
                        gt = test_gt
                        denom = max(1, len(test_logs))
                        lat_list = results[method]["lats"]
                        avg_lat = float(sum(lat_list) / denom) if lat_list else 0.0
                        avg_tok = float(results[method].get("tokens", 0) / denom)

                    pa = MetricsCalculator.calculate_pa(preds, gt)
                    ga = MetricsCalculator.calculate_ga(preds, gt)
                    row = {
                        "run_seed": run_seed,
                        "dataset": dataset_name,
                        "noise_rate": noise_rate,
                        "method": method,
                        "PA": float(pa),
                        "GA": float(ga),
                        "Latency_ms": avg_lat,
                        "Tokens": avg_tok,
                    }
                    if method == "NuSy-Edge":
                        row.update(
                            {
                                "nusy_cache_hit_rate": (nusy_count_hit / nusy_count_total) if nusy_count_total else 0.0,
                                "nusy_llm_avg_latency_ms": (nusy_sum_lat_llm / nusy_count_llm) if nusy_count_llm else 0.0,
                                "nusy_total_avg_latency_ms": (nusy_sum_lat_total / nusy_count_total) if nusy_count_total else 0.0,
                                "nusy_count_total": nusy_count_total,
                                "nusy_count_hit": nusy_count_hit,
                                "nusy_count_llm": nusy_count_llm,
                            }
                        )
                    per_run_rows.append(row)

        if EXPORT_PER_RUN:
            df_run = pd.DataFrame([r for r in per_run_rows if r.get("run_seed") == run_seed])
            out_path = os.path.join(EXPORT_DIR, f"rq1_large_scale_seed{run_seed}.csv")
            df_run.to_csv(out_path, index=False)
            print(f"[INFO] Wrote per-run large-scale results: {out_path}")

    # Aggregate 5-run means
    df = pd.DataFrame(per_run_rows)
    if df.empty:
        print("[ERROR] No results collected in large-scale run.")
        return

    group_cols = ["dataset", "noise_rate", "method"]
    metric_cols = ["PA", "GA", "Latency_ms", "Tokens", "nusy_cache_hit_rate", "nusy_llm_avg_latency_ms", "nusy_total_avg_latency_ms"]
    for c in metric_cols:
        if c not in df.columns:
            df[c] = np.nan
    mean_df = df.groupby(group_cols, as_index=False)[metric_cols].mean()

    print("\n" + "=" * 126)
    print("RQ1 LARGE-SCALE ROBUSTNESS (5-run mean) | BasePool=20k (per dataset) | TestSample=2000 | VanillaSub=200")
    print("=" * 126)

    pivot_pa = mean_df.pivot_table(index=["dataset", "noise_rate"], columns="method", values="PA")
    pivot_ga = mean_df.pivot_table(index=["dataset", "noise_rate"], columns="method", values="GA")
    print("\n[PA (5-run mean)]")
    print(pivot_pa.round(4).to_string())
    print("\n[GA (5-run mean)]")
    print(pivot_ga.round(4).to_string())

    nusy_mean = mean_df[mean_df["method"] == "NuSy-Edge"].copy()
    if not nusy_mean.empty:
        show_cols = ["dataset", "noise_rate", "nusy_cache_hit_rate", "nusy_total_avg_latency_ms", "nusy_llm_avg_latency_ms"]
        print("\n[NuSy-Edge Latency Profiling (5-run mean, large-scale)]")
        print(nusy_mean[show_cols].sort_values(["dataset", "noise_rate"]).round(4).to_string(index=False))

        global_hit = float(nusy_mean["nusy_cache_hit_rate"].mean())
        global_lat_total = float(nusy_mean["nusy_total_avg_latency_ms"].mean())
        global_lat_llm = float(nusy_mean["nusy_llm_avg_latency_ms"].mean())
        print("\n[NuSy-Edge Global Averages over (dataset, noise) groups — Large-Scale]")
        print(f"  Cache/Shortcut Hit Rate: {global_hit:.2%}")
        print(f"  Total Avg Latency (ms):  {global_lat_total:.3f}")
        print(f"  LLM-only Avg Latency (ms, miss-only): {global_lat_llm:.3f}")

    out_mean = os.path.join(EXPORT_DIR, "rq1_large_scale_5run_mean.csv")
    mean_df.to_csv(out_mean, index=False)
    print(f"\n[INFO] Wrote large-scale 5-run mean summary: {out_mean}")
    print("=" * 126)


if __name__ == "__main__":
    run_rq1_large_scale()

