import sys
import os
import time
import pandas as pd
from tqdm import tqdm
import logging
import transformers

# --- Silence Warnings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

# 路径修正
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system.edge_node import NuSyEdgeNode
from src.perception.drain_parser import DrainParser
from src.utils.data_loader import DataLoader
from src.utils.noise_injector import NoiseInjector
from src.utils.metrics import MetricsCalculator

# ================= RQ1 Benchmark Config (Thesis: NeSy-Edge) =================
SAMPLE_SIZE = 60
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DATASETS = ["HDFS", "OpenStack"]
EXPORT_PER_LOG = True   # 导出每条 log 的预测，便于分析与复现
EXPORT_NUSY_STATS = True  # 导出 NuSy-Edge 的 L1/shortcut/LLM 路径统计
# =============================================================================

# Vanilla 使用 3-shot：从训练集按模板多样性采样，在 run 里填充
VANILLA_N_SHOT = 3

def run_grand_benchmark():
    print(f"⚔️  STARTING RQ1 MINI-BENCHMARK (N={SAMPLE_SIZE}) ⚔️")
    print(f"📉 Noise Levels: {NOISE_LEVELS}")
    
    loader = DataLoader()
    injector = NoiseInjector(seed=2026)
    
    all_metrics = []
    per_log_rows = []  # for EXPORT_PER_LOG
    nusy_path_counts = []  # for EXPORT_NUSY_STATS: (dataset, noise) -> counts

    for dataset_name in DATASETS:
        print(f"\n{'='*40}")
        print(f"🌍 Dataset: {dataset_name}")
        print(f"{'='*40}")
        
        if dataset_name == "HDFS":
            logs, gt_df = loader.get_hdfs_test_data()
        else:
            logs, gt_df = loader.get_openstack_test_data()
        
        gt_templates = gt_df['EventTemplate'].tolist()


        # [关键修正] 仅使用后 50% 数据作为测试集 (Test Set)
        split_idx = int(len(logs) * 0.5)
        train_logs = logs[:split_idx]
        train_gt = gt_templates[:split_idx]
        test_logs = logs[split_idx:]
        test_gt = gt_templates[split_idx:]
        
        # Vanilla 3-shot：从训练集选 3 个不同模板各取 1 条 (log, template)
        seen = set()
        vanilla_refs = []
        for i in range(len(train_gt)):
            t = train_gt[i]
            if t not in seen and len(vanilla_refs) < VANILLA_N_SHOT:
                seen.add(t)
                vanilla_refs.append((train_logs[i], t))
        if len(vanilla_refs) < VANILLA_N_SHOT:
            # 不足 3 种则重复补足
            while len(vanilla_refs) < VANILLA_N_SHOT and train_logs:
                vanilla_refs.append((train_logs[len(vanilla_refs) % len(train_logs)], train_gt[len(vanilla_refs) % len(train_gt)]))
        vanilla_refs = vanilla_refs[:VANILLA_N_SHOT]
        
        # 应用 Sample Size (从后半部分取 N 条)
        if SAMPLE_SIZE:
            logs = test_logs[:SAMPLE_SIZE]
            gt_templates = test_gt[:SAMPLE_SIZE]
        else:
            logs = test_logs
            gt_templates = test_gt
        
        # tqdm 放在外层，显示总进度
        for noise_rate in NOISE_LEVELS:
            # print(f"   🔊 Noise Level: {noise_rate} ...") # 减少刷屏
            injector.injection_rate = noise_rate
            
            drain = DrainParser()
            nusy = NuSyEdgeNode() 
            
            results = {
                "Drain": {"preds": [], "lats": [], "tokens": 0},
                "Vanilla": {"preds": [], "lats": [], "tokens": 0},
                "NuSy-Edge": {"preds": [], "lats": [], "tokens": 0}
            }
            nusy_paths = {"L1_cache": 0, "symbolic_shortcut": 0, "llm": 0}

            # 这里的 tqdm 改为 leave=False，避免每个噪音等级都留下一行
            pbar = tqdm(logs, desc=f"Noise {noise_rate}", unit="log", leave=False)
            
            for log_idx, raw_log in enumerate(pbar):
                # 1. 注入噪音
                noisy_log_raw = injector.inject(raw_log, dataset_type=dataset_name)
                
                # 2. 统一预处理 (去除 Header)
                clean_log_content = NuSyEdgeNode.preprocess_header(noisy_log_raw, dataset_name)
                if not clean_log_content: clean_log_content = noisy_log_raw

                # === Method A: Drain ===
                t0 = time.time()
                try: p_drain = drain.parse(clean_log_content)
                except: p_drain = ""
                results["Drain"]["lats"].append((time.time() - t0) * 1000)
                results["Drain"]["preds"].append(p_drain)
                
                # === Method B: Vanilla LLM (3-shot from train set) ===
                t0 = time.time()
                try:
                    p_vanilla = nusy.llm.parse_with_multi_rag(clean_log_content, vanilla_refs)
                except Exception:
                    p_vanilla = ""
                results["Vanilla"]["lats"].append((time.time() - t0) * 1000)
                results["Vanilla"]["preds"].append(p_vanilla)
                
                t_in = MetricsCalculator.estimate_tokens(clean_log_content) + sum(MetricsCalculator.estimate_tokens(r[0]) for r in vanilla_refs) + 100
                t_out = MetricsCalculator.estimate_tokens(p_vanilla)
                results["Vanilla"]["tokens"] += (t_in + t_out)
                
                # === Method C: NuSy-Edge ===
                p_nusy, lat_nusy, is_hit, path = nusy.parse_log_stream(noisy_log_raw, dataset_name)
                results["NuSy-Edge"]["lats"].append(lat_nusy)
                results["NuSy-Edge"]["preds"].append(p_nusy)
                nusy_paths[path] = nusy_paths.get(path, 0) + 1
                
                if not is_hit:
                    t_in_miss = MetricsCalculator.estimate_tokens(clean_log_content) + 300 
                    t_out_miss = MetricsCalculator.estimate_tokens(p_nusy)
                    results["NuSy-Edge"]["tokens"] += (t_in_miss + t_out_miss)

                if EXPORT_PER_LOG:
                    gt = gt_templates[log_idx]
                    per_log_rows.append({
                        "Dataset": dataset_name,
                        "Noise_Rate": noise_rate,
                        "Log_Idx": log_idx,
                        "GT": gt,
                        "Drain": p_drain,
                        "Vanilla": p_vanilla,
                        "NuSy_Edge": p_nusy,
                    })
            
            # --- 本轮结算 ---
            for method in results:
                preds = results[method]["preds"]
                pa = MetricsCalculator.calculate_pa(preds, gt_templates)
                ga = MetricsCalculator.calculate_ga(preds, gt_templates)
                token_f1 = MetricsCalculator.calculate_token_f1(preds, gt_templates)
                
                avg_lat = sum(results[method]["lats"]) / len(logs)
                avg_tok = results[method]["tokens"] / len(logs)
                
                all_metrics.append({
                    "Dataset": dataset_name,
                    "Noise_Rate": noise_rate,
                    "Method": method,
                    "PA": pa,
                    "GA": ga,
                    "Latency": avg_lat,
                    "Tokens": avg_tok,
                    "F1_Token": token_f1
                })
            
            if EXPORT_NUSY_STATS:
                nusy_path_counts.append({
                    "Dataset": dataset_name,
                    "Noise_Rate": noise_rate,
                    "L1_cache": nusy_paths["L1_cache"],
                    "symbolic_shortcut": nusy_paths["symbolic_shortcut"],
                    "llm": nusy_paths["llm"],
                })
            
            # === Debug: 只在 0 噪音时检查是否还有 Mismatch ===
            if noise_rate == 0.0:
                mismatch_count = 0
                header_printed = False
                for p, g in zip(results["Drain"]["preds"], gt_templates):
                     if MetricsCalculator.normalize_template(p) != MetricsCalculator.normalize_template(g):
                        if not header_printed:
                            print(f"\n[DEBUG] Remaining Drain Mismatches ({dataset_name}):")
                            header_printed = True
                        print(f"❌ GT  : [{g}]")
                        print(f"   Pred: [{p}]")
                        mismatch_count += 1
                        if mismatch_count >= 2: break 

    # 保存结果
    os.makedirs("results", exist_ok=True)
    final_df = pd.DataFrame(all_metrics)
    output_file = f"results/rq1_benchmark_mini_N{SAMPLE_SIZE}.csv"
    final_df.to_csv(output_file, index=False)
    
    if EXPORT_PER_LOG and per_log_rows:
        per_log_df = pd.DataFrame(per_log_rows)
        per_log_df.to_csv(f"results/rq1_per_log_predictions_N{SAMPLE_SIZE}.csv", index=False)
    if EXPORT_NUSY_STATS and nusy_path_counts:
        pd.DataFrame(nusy_path_counts).to_csv(f"results/rq1_nusy_stats_N{SAMPLE_SIZE}.csv", index=False)

    print("\n" + "="*60)
    print("📊 FINAL RESULT SUMMARY")
    print("="*60)

    # 1. 透视表：按 [Dataset, Noise] 查看各方法的 PA 和 GA 对比
    pivot_res = final_df.pivot_table(
        index=["Dataset", "Noise_Rate"], 
        columns="Method", 
        values=["PA", "GA"]
    )
    print("\n[🔍 Detailed Trend Analysis (PA & GA)]")
    print(pivot_res.round(3))

    # 2. 全局平均榜
    global_avg = final_df.groupby("Method")[["PA", "GA", "Latency", "Tokens", "F1_Token"]].mean()
    print("\n[🏆 Global Performance Average]")
    print(global_avg.sort_values(by="PA", ascending=False).round(3))
    print("="*60)

if __name__ == "__main__":
    run_grand_benchmark()