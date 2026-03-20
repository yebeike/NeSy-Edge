import os
import sys
import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Path fix: experiments/rq1_robustness -> project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.data_loader import DataLoader
from src.utils.noise_injector import NoiseInjector
from src.utils.metrics import MetricsCalculator
from src.system.knowledge_base import KnowledgeBase


# ================= RQ1 Robustness Config =================
RUN_SEEDS = [42, 1024, 2026, 8888, 9999]  # 5-run
SAMPLE_SIZE = 500                         # random.sample from test half
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DATASETS = ["HDFS", "OpenStack"]

# Export per-run CSV for audit (safe, new output path)
EXPORT_DIR = os.path.join(_PROJECT_ROOT, "results", "rq1_robustness")
EXPORT_PER_RUN = True
# Default: skip Vanilla baseline to keep 5×2×6×500 runtime tractable.
# Set env RQ1_INCLUDE_VANILLA=1 to enable it.
INCLUDE_VANILLA = (os.environ.get("RQ1_INCLUDE_VANILLA") or "").strip() in ("1", "true", "True", "yes", "YES")
# =========================================================


def _split_half(logs: List[str], gt_templates: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    split_idx = int(len(logs) * 0.5)
    train_logs = logs[:split_idx]
    train_gt = gt_templates[:split_idx]
    test_logs = logs[split_idx:]
    test_gt = gt_templates[split_idx:]
    return train_logs, train_gt, test_logs, test_gt


def _sample_test_set(test_logs: List[str], test_gt: List[str], k: int, seed: int) -> Tuple[List[str], List[str]]:
    if not test_logs:
        return [], []
    k_eff = min(int(k), len(test_logs))
    rng = random.Random(seed)
    indices = rng.sample(range(len(test_logs)), k=k_eff)
    logs_s = [test_logs[i] for i in indices]
    gt_s = [test_gt[i] for i in indices]
    return logs_s, gt_s


def run_rq1_5run():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    loader = DataLoader()
    # Build an isolated robustness KB (do NOT touch the default 'data/chroma_db').
    # We ingest the "train half" once per dataset to enable shortcut/cache paths.
    kb = KnowledgeBase(collection_name="rq1_robustness_kb", persist_path="data/chroma_db_rq1_robustness")
    ingested = set()  # dataset_name -> bool

    # Aggregate records:
    # (run_seed, dataset, noise, method) -> metrics dict
    per_run_rows = []

    for run_seed in RUN_SEEDS:
        print("\n" + "=" * 90)
        print(f"[RQ1 Robustness] RUN seed={run_seed} | sample={SAMPLE_SIZE} | noises={NOISE_LEVELS} | datasets={DATASETS}")
        print("=" * 90)

        injector = NoiseInjector(seed=run_seed)
        # Load NuSy-Edge ONCE per run to avoid repeated model loads
        nusy = NuSyEdgeNode()
        # Force NuSy-Edge to use the isolated robustness KB
        nusy.kb = kb

        for dataset_name in DATASETS:
            if dataset_name == "HDFS":
                logs_all, gt_df = loader.get_hdfs_test_data()
            else:
                logs_all, gt_df = loader.get_openstack_test_data()
            gt_templates_all = gt_df["EventTemplate"].tolist()

            train_logs, train_gt, test_logs, test_gt = _split_half(logs_all, gt_templates_all)
            logs, gt_templates = _sample_test_set(test_logs, test_gt, SAMPLE_SIZE, seed=run_seed)

            if not logs:
                print(f"[WARN] Empty logs for dataset={dataset_name}; skipping.")
                continue

            # Ingest "train half" into robustness KB once per dataset
            if dataset_name not in ingested:
                train_clean = []
                for r in train_logs:
                    c = NuSyEdgeNode.preprocess_header(r, dataset_name)
                    train_clean.append(c if c else r)
                kb.add_knowledge(train_clean, train_gt, dataset_name)
                ingested.add(dataset_name)

            # Vanilla 3-shot references: pick 3 distinct templates from train half (deterministic order)
            vanilla_refs = []
            seen = set()
            for i in range(len(train_gt)):
                t = train_gt[i]
                if t not in seen and len(vanilla_refs) < 3:
                    seen.add(t)
                    vanilla_refs.append((train_logs[i], t))
            if len(vanilla_refs) < 3 and train_logs:
                while len(vanilla_refs) < 3:
                    j = len(vanilla_refs) % max(1, len(train_logs))
                    vanilla_refs.append((train_logs[j], train_gt[j]))
            vanilla_refs = vanilla_refs[:3]

            for noise_rate in NOISE_LEVELS:
                injector.injection_rate = noise_rate

                drain = DrainParser()
                # NOTE: We intentionally DO NOT clear NuSy cache per condition.
                # This keeps the benchmark tractable at 5×2×6×500 scale and matches a realistic
                # long-running edge-node setting where cache accumulates over time.

                results: Dict[str, Dict[str, object]] = {
                    "Drain": {"preds": [], "lats": [], "tokens": 0},
                    "NuSy-Edge": {"preds": [], "lats": [], "tokens": 0},
                }
                if INCLUDE_VANILLA:
                    results["Vanilla"] = {"preds": [], "lats": [], "tokens": 0}

                # NuSy-Edge latency profiling (split)
                nusy_count_total = 0
                nusy_count_hit = 0  # L1_cache OR symbolic_shortcut
                nusy_count_llm = 0
                nusy_sum_lat_total = 0.0
                nusy_sum_lat_llm = 0.0

                pbar = tqdm(logs, desc=f"seed={run_seed} {dataset_name} noise={noise_rate}", unit="log", leave=False)
                for raw_log in pbar:
                    # 1) Inject noise (raw log level)
                    noisy_raw = injector.inject(raw_log, dataset_type=dataset_name)

                    # 2) Unified preprocessing for Drain / Vanilla
                    clean_log_content = NuSyEdgeNode.preprocess_header(noisy_raw, dataset_name)
                    if not clean_log_content:
                        clean_log_content = noisy_raw

                    # A) Drain
                    t0 = time.time()
                    try:
                        p_drain = drain.parse(clean_log_content)
                    except Exception:
                        p_drain = ""
                    results["Drain"]["lats"].append((time.time() - t0) * 1000)
                    results["Drain"]["preds"].append(p_drain)

                    # B) Vanilla (optional, slow)
                    if INCLUDE_VANILLA:
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

                    # C) NuSy-Edge
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

                    # Token accounting: only count when LLM path was actually used
                    if path == "llm":
                        t_in_miss = MetricsCalculator.estimate_tokens(clean_log_content) + 300
                        t_out_miss = MetricsCalculator.estimate_tokens(p_nusy)
                        results["NuSy-Edge"]["tokens"] += (t_in_miss + t_out_miss)

                # Summarize metrics for this (run, dataset, noise)
                for method in ("Drain", "Vanilla", "NuSy-Edge"):
                    if method == "Vanilla" and not INCLUDE_VANILLA:
                        continue
                    preds = results[method]["preds"]
                    pa = MetricsCalculator.calculate_pa(preds, gt_templates)
                    ga = MetricsCalculator.calculate_ga(preds, gt_templates)
                    avg_lat = float(sum(results[method]["lats"]) / max(1, len(logs)))
                    avg_tok = float(results[method]["tokens"] / max(1, len(logs)))
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
            out_path = os.path.join(EXPORT_DIR, f"rq1_robustness_seed{run_seed}.csv")
            df_run.to_csv(out_path, index=False)
            print(f"[INFO] Wrote per-run results: {out_path}")

    # =============== Aggregate across runs (5-run mean) ===============
    df = pd.DataFrame(per_run_rows)
    if df.empty:
        print("[ERROR] No results collected.")
        return

    # Mean across runs per (dataset, noise, method)
    group_cols = ["dataset", "noise_rate", "method"]
    metric_cols = ["PA", "GA", "Latency_ms", "Tokens", "nusy_cache_hit_rate", "nusy_llm_avg_latency_ms", "nusy_total_avg_latency_ms"]
    for c in metric_cols:
        if c not in df.columns:
            df[c] = float("nan")
    mean_df = df.groupby(group_cols, as_index=False)[metric_cols].mean()

    # Print requested report focusing on 5-run mean PA/GA and NuSy split latency stats
    print("\n" + "=" * 110)
    print("RQ1 ROBUSTNESS (5-run mean) | Sample=500 (random.sample from test half)")
    print("=" * 110)

    # 1) PA/GA table
    pivot_pa = mean_df.pivot_table(index=["dataset", "noise_rate"], columns="method", values="PA")
    pivot_ga = mean_df.pivot_table(index=["dataset", "noise_rate"], columns="method", values="GA")
    print("\n[PA (5-run mean)]")
    print(pivot_pa.round(4).to_string())
    print("\n[GA (5-run mean)]")
    print(pivot_ga.round(4).to_string())

    # 2) NuSy-Edge latency profiling table
    nusy_mean = mean_df[mean_df["method"] == "NuSy-Edge"].copy()
    if not nusy_mean.empty:
        show_cols = ["dataset", "noise_rate", "nusy_cache_hit_rate", "nusy_total_avg_latency_ms", "nusy_llm_avg_latency_ms"]
        print("\n[NuSy-Edge Latency Profiling (5-run mean)]")
        print(nusy_mean[show_cols].sort_values(["dataset", "noise_rate"]).round(4).to_string(index=False))

        # Global averages across dataset+noise (simple mean over groups)
        global_hit = float(nusy_mean["nusy_cache_hit_rate"].mean())
        global_lat_total = float(nusy_mean["nusy_total_avg_latency_ms"].mean())
        global_lat_llm = float(nusy_mean["nusy_llm_avg_latency_ms"].mean())
        print("\n[NuSy-Edge Global Averages over (dataset, noise) groups]")
        print(f"  Cache/Shortcut Hit Rate: {global_hit:.2%}")
        print(f"  Total Avg Latency (ms):  {global_lat_total:.3f}")
        print(f"  LLM-only Avg Latency (ms, miss-only): {global_lat_llm:.3f}")

    # 3) Save aggregated CSV
    out_mean = os.path.join(EXPORT_DIR, "rq1_robustness_5run_mean.csv")
    mean_df.to_csv(out_mean, index=False)
    print(f"\n[INFO] Wrote 5-run mean summary: {out_mean}")
    print("=" * 110)


if __name__ == "__main__":
    run_rq1_5run()
