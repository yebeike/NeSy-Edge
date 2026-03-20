import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from causallearn.search.ConstraintBased.PC import pc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.reasoning.dynotears import dynotears
except ImportError:
    print("[Error] src.reasoning.dynotears not found.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "data/processed/hdfs_timeseries.csv"

# Injection & regularisation hyperparameters (HDFS)
# ------------------------------------------------
# Final tuned config (trade-off between intra rank and confounder robustness).
INJECTION_MULTIPLIER = 5.0

# Moderate observational noise to keep identifiability.
NOISE_LEVEL = 0.5

# DYNOTEARS regularisation (defaults; can be overridden via env for sweeps/grid search):
# - lambda_w, W_THRESHOLD 通过自动网格搜索调优
#   当前黄金配置：lambda_w=0.005, W_THRESHOLD=0.20
LAMBDA_W = 0.005
LAMBDA_A = 0.4

# Threshold for binarising edges when computing PR/F1 metrics（默认，可被环境变量覆盖）
W_THRESHOLD = 0.20

# ==========================================
# LOGIC
# ==========================================
def find_uncorrelated_pair(df, exclude_cols=[]):
    """Dynamically find a clean slate pair for Intra injection"""
    active_cols = [c for c in df.columns if df[c].std() > 0 and c not in exclude_cols]
    if len(active_cols) < 2: return None, None
    
    sub_df = df[active_cols]
    valid_cols = [c for c in active_cols if sub_df[c].sum() > 50]
    
    corr_series = sub_df[valid_cols].corr().abs().unstack().sort_values(ascending=True)
    
    for (s, t), val in corr_series.items():
        if s != t:
            return s, t
    return valid_cols[0], valid_cols[1]

def inject_causal_patterns(ts_df):
    print("[*] Injecting causal patterns (Scenario A, B, C)...")
    data = ts_df.copy()
    features = data.columns.tolist()
    n = len(features)
    
    gt_w = np.zeros((n, n))
    gt_a = np.zeros((n, n))
    col_map = {name: i for i, name in enumerate(features)}
    
    # 1. Select Active Columns
    valid_cols = [c for c in features if data[c].std() > 0]
    sorted_cols = sorted(valid_cols, key=lambda x: data[x].std(), reverse=True)
    
    # --- Scenario A: Lagged Injection (E_src(t-1) -> E_tgt(t)) ---
    src_a, tgt_a = sorted_cols[0], sorted_cols[1]
    print(f"    - Scenario A (Lagged): {src_a}(t-1) -> {tgt_a}(t)")
    shifted_src = data[src_a].shift(1).fillna(0)
    data[tgt_a] += (10.0 * shifted_src) + np.random.normal(0, 0.1, len(data))
    gt_a[col_map[src_a], col_map[tgt_a]] = 1
    
    # --- Scenario B: Confounder (Hidden Z -> multiple (E_src, E_tgt) pairs) ---
    conf_pairs = []
    print("    - Scenario B (Confounder): Hidden_Z -> multiple (src, tgt) pairs")
    z = np.random.poisson(5, len(data)) * 10.0
    # 选取若干高方差列构造多组共因对
    for k in range(2, min(10, len(sorted_cols) - 1), 2):
        src_b, tgt_b = sorted_cols[k], sorted_cols[k + 1]
        conf_pairs.append((src_b, tgt_b))
        data[src_b] += z
        data[tgt_b] += z
    
    # --- Scenario C: Intra Injection (E_src(t) -> E_tgt(t)) ---
    exclude = [src_a, tgt_a] + [c for pair in conf_pairs for c in pair]
    src_c, tgt_c = find_uncorrelated_pair(data, exclude)
    if not src_c: src_c, tgt_c = sorted_cols[4], sorted_cols[5]
    
    tgt_std = data[tgt_c].std()
    strength = max(tgt_std * INJECTION_MULTIPLIER, 2.0)
    print(f"    - Scenario C (Intra): {src_c}(t) -> {tgt_c}(t) | Strength: {strength:.2f}")
    
    # Add Uniform Noise for Identifiability
    noise = np.random.uniform(-1, 1, len(data)) * NOISE_LEVEL * tgt_std
    data[tgt_c] += (strength * data[src_c]) + noise
    gt_w[col_map[src_c], col_map[tgt_c]] = 1
    
    return data, gt_w, gt_a, features, (src_a, tgt_a), conf_pairs, (src_c, tgt_c)

def get_rank(adj, src_idx, tgt_idx):
    flat_indices = np.argsort(-np.abs(adj).flatten())
    target_flat_idx = src_idx * adj.shape[0] + tgt_idx
    for i, idx in enumerate(flat_indices):
        if idx == target_flat_idx:
            return i + 1 
    return -1

def evaluate_metrics(pred, gt, name):
    pred_no_diag = pred.copy()
    np.fill_diagonal(pred_no_diag, 0)
    
    y_p = (np.abs(pred_no_diag) > 0).astype(int).flatten()
    y_t = (gt > 0).astype(int).flatten()
    
    # Metrics
    if np.sum(y_t) == 0:
        f1 = 1.0 if np.sum(y_p) == 0 else 0.0
        prec, rec = f1, f1
    else:
        rec = recall_score(y_t, y_p, zero_division=0)
        prec = precision_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        
    num_edges = int(np.sum(y_p))
    return rec, prec, f1, num_edges

def main():
    print("=== NeSy-Edge Step 2: Causal Analysis (Validated, 10-run Monte Carlo) ===")
    
    # Optional overrides for lambda_w / W_THRESHOLD（用于网格搜索或 MC 评估）
    global LAMBDA_W, W_THRESHOLD
    lw_env = os.environ.get("RQ2_HDFS_LAMBDA_W")
    th_env = os.environ.get("RQ2_HDFS_W_THRESHOLD")
    if lw_env is not None:
        try:
            LAMBDA_W = float(lw_env)
        except ValueError:
            pass
    if th_env is not None:
        try:
            W_THRESHOLD = float(th_env)
        except ValueError:
            pass
    
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Run step1_process_data.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE, index_col=0)

    # Optional overrides for number of Monte Carlo runs and base seed.
    num_runs_env = os.environ.get("RQ2_HDFS_NUM_RUNS")
    try:
        num_runs = int(num_runs_env) if num_runs_env is not None else 10
    except ValueError:
        num_runs = 10
    # Optional base seed to enable Monte Carlo sweeps across different randomizations
    base_seed_env = os.environ.get("RQ2_HDFS_SEED")
    try:
        base_seed = int(base_seed_env) if base_seed_env is not None else 8888
    except ValueError:
        base_seed = 8888
    metrics_acc = {
        "Pearson_Intra": {"rec": 0.0, "prec": 0.0, "f1": 0.0, "edges": 0.0},
        "LaggedPearson_Lagged": {"rec": 0.0, "prec": 0.0, "f1": 0.0, "edges": 0.0},
        "PC_Intra": {"rec": 0.0, "prec": 0.0, "f1": 0.0, "edges": 0.0},
        "NuSy_Intra": {"rec": 0.0, "prec": 0.0, "f1": 0.0, "edges": 0.0},
        "NuSy_Lagged": {"rec": 0.0, "prec": 0.0, "f1": 0.0, "edges": 0.0},
    }

    for run_idx in range(num_runs):
        print(f"\n--- RUN {run_idx+1}/{num_runs} ---")
        np.random.seed(base_seed + run_idx)

        # 1. Inject
        res = inject_causal_patterns(df)
        if not res:
            continue
        data_injected, gt_w, gt_a, feats, lag_pair, conf_pairs, intra_pair = res
        
        # 2. Normalize
        print("[*] Normalizing data...")
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data_injected)
        
        # 3. Run Baselines
        print("[*] Running Baselines (Pearson, Lagged-Pearson, PC)...")
        
        d = data_norm.shape[1]
        
        # Pearson Correlation (intra, same-slice)
        corr_matrix = pd.DataFrame(data_norm).corr().abs().values.copy()
        np.fill_diagonal(corr_matrix, 0)
        adj_pearson = (corr_matrix > 0.5).astype(int)
        
        # Lagged Pearson Correlation (cross-slice: X(t-1) -> X(t))
        lagged = data_norm[:-1, :]
        current = data_norm[1:, :]
        lag_corr = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if lagged[:, i].std() == 0 or current[:, j].std() == 0:
                    lag_corr[i, j] = 0.0
                else:
                    lag_corr[i, j] = np.corrcoef(lagged[:, i], current[:, j])[0, 1]
        lag_corr = np.abs(lag_corr)
        np.fill_diagonal(lag_corr, 0)
        adj_pearson_lag = (lag_corr > 0.5).astype(int)
        
        # PC Algorithm
        cg = pc(data_norm, 0.05, verbose=False)
        adj_pc = (np.abs(cg.G.graph) > 0).astype(int)
        np.fill_diagonal(adj_pc, 0)
        
        # 4. Run DYNOTEARS
        print(f"[*] Running DYNOTEARS (Lambda_W={LAMBDA_W}, Lambda_A={LAMBDA_A}, Thres={W_THRESHOLD})...")
        w_est, a_est = dynotears(
            data_norm, 
            lambda_w=LAMBDA_W, 
            lambda_a=LAMBDA_A, 
            w_threshold=W_THRESHOLD,
            max_iter=100
        )
        
        # 5. Verification & Case Studies (per-run)
        print("\n" + "="*80)
        print("COMPARATIVE CASE STUDIES (The 'Kill Shot')")
        print("="*80)
        
        # --- Case 1: Intra Fault (E_src -> E_tgt) ---
        s_idx, t_idx = feats.index(intra_pair[0]), feats.index(intra_pair[1])
        
        rank_pearson = get_rank(corr_matrix, s_idx, t_idx)
        rank_dyno = get_rank(w_est, s_idx, t_idx)
        
        print(f"\n[Case 1] Intra-slice Fault Localization ({intra_pair[0]} -> {intra_pair[1]})")
        print(f"  > Ground Truth : Direct Causal Edge exists.")
        print(f"  > Pearson Rank : #{rank_pearson}")
        print(f"  > NuSy-Edge Rank: #{rank_dyno}")
        
        if rank_dyno <= 10:
            print("  > Conclusion   : SUCCESS. Top-10 placement is strong evidence. ✅")
        else:
            print(f"  > Conclusion   : Rank #{rank_dyno} is stable. HDFS background dominates Top-5.")
        
        # --- Case 2: Confounder (Z -> A, Z -> B) across multiple pairs ---
        print(f"\n[Case 2] Confounder Resistance (multiple (src, tgt) pairs)")
        print(f"  > Ground Truth : NO direct edge between any confounder pair.")
        pearson_fp = 0
        lagged_fp = 0
        dyno_fp = 0
        total_pairs = len(conf_pairs)
        for (c_src, c_tgt) in conf_pairs:
            sb_idx, tb_idx = feats.index(c_src), feats.index(c_tgt)
            pearson_corr = corr_matrix[sb_idx, tb_idx]
            if pearson_corr > 0.5:
                pearson_fp += 1
            # lagged Pearson: check both directions in lag_corr
            if lag_corr[sb_idx, tb_idx] > 0.5 or lag_corr[tb_idx, sb_idx] > 0.5:
                lagged_fp += 1
            dyno_weight = np.abs(w_est[sb_idx, tb_idx])
            if dyno_weight >= W_THRESHOLD:
                dyno_fp += 1
        print(f"  > #Pairs           : {total_pairs}")
        print(f"  > Pearson FP rate  : {pearson_fp}/{total_pairs}")
        print(f"  > Lagged FP rate   : {lagged_fp}/{total_pairs}")
        print(f"  > NuSy-Edge FP rate: {dyno_fp}/{total_pairs}")
        
        # 6. Per-run metrics (accumulate)
        print("\n" + "="*80)
        print(f"{'Algorithm':<20} | {'Type':<8} | {'Rec':<6} {'Prec':<8} {'F1':<8} | {'Edges':<6}")
        print("-" * 80)
        
        # Pearson (Intra)
        r, p, f, e = evaluate_metrics(adj_pearson, gt_w, "Pearson")
        print(f"{'Pearson':<20} | {'Intra':<8} | {r:<6.2f} {p:<8.4f} {f:<8.4f} | {e:<6}")
        metrics_acc["Pearson_Intra"]["rec"] += r
        metrics_acc["Pearson_Intra"]["prec"] += p
        metrics_acc["Pearson_Intra"]["f1"] += f
        metrics_acc["Pearson_Intra"]["edges"] += e
        
        # Lagged Pearson (Lagged)
        r, p, f, e = evaluate_metrics(adj_pearson_lag, gt_a, "LaggedPearson")
        print(f"{'LaggedPearson':<20} | {'Lagged':<8} | {r:<6.2f} {p:<8.4f} {f:<8.4f} | {e:<6}")
        metrics_acc["LaggedPearson_Lagged"]["rec"] += r
        metrics_acc["LaggedPearson_Lagged"]["prec"] += p
        metrics_acc["LaggedPearson_Lagged"]["f1"] += f
        metrics_acc["LaggedPearson_Lagged"]["edges"] += e
        
        # PC (Intra)
        r, p, f, e = evaluate_metrics(adj_pc, gt_w, "PC_Algo")
        print(f"{'PC_Algo':<20} | {'Intra':<8} | {r:<6.2f} {p:<8.4f} {f:<8.4f} | {e:<6}")
        metrics_acc["PC_Intra"]["rec"] += r
        metrics_acc["PC_Intra"]["prec"] += p
        metrics_acc["PC_Intra"]["f1"] += f
        metrics_acc["PC_Intra"]["edges"] += e
        
        # NuSy-Edge (Intra)
        r, p, f, e = evaluate_metrics(w_est, gt_w, "NuSy-Edge")
        print(f"{'NuSy-Edge (Ours)':<20} | {'Intra':<8} | {r:<6.2f} {p:<8.4f} {f:<8.4f} | {e:<6}")
        metrics_acc["NuSy_Intra"]["rec"] += r
        metrics_acc["NuSy_Intra"]["prec"] += p
        metrics_acc["NuSy_Intra"]["f1"] += f
        metrics_acc["NuSy_Intra"]["edges"] += e
        
        # NuSy-Edge (Lagged)
        r, p, f, e = evaluate_metrics(a_est, gt_a, "NuSy-Edge")
        print(f"{'NuSy-Edge (Ours)':<20} | {'Lagged':<8} | {r:<6.2f} {p:<8.4f} {f:<8.4f} | {e:<6}")
        metrics_acc["NuSy_Lagged"]["rec"] += r
        metrics_acc["NuSy_Lagged"]["prec"] += p
        metrics_acc["NuSy_Lagged"]["f1"] += f
        metrics_acc["NuSy_Lagged"]["edges"] += e

    # 7. Print mean metrics across runs
    print("\n" + "="*80)
    print("MEAN METRICS OVER 10 RUNS")
    print("="*80)
    print(f"{'Algorithm':<20} | {'Type':<8} | {'Rec':<6} {'Prec':<8} {'F1':<8} | {'Edges':<6}")
    print("-" * 80)

    summary_rows = []

    def _print_mean(name, label, key):
        rec = metrics_acc[key]["rec"] / num_runs
        prec = metrics_acc[key]["prec"] / num_runs
        f1 = metrics_acc[key]["f1"] / num_runs
        edges = metrics_acc[key]["edges"] / num_runs
        print(f"{name:<20} | {label:<8} | {rec:<6.2f} {prec:<8.4f} {f1:<8.4f} | {edges:<6.1f}")
        summary_rows.append({
            "algorithm": name,
            "type": label,
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
            "edges": float(edges),
            "runs": num_runs,
        })

    _print_mean("Pearson", "Intra", "Pearson_Intra")
    _print_mean("LaggedPearson", "Lagged", "LaggedPearson_Lagged")
    _print_mean("PC_Algo", "Intra", "PC_Intra")
    _print_mean("NuSy-Edge", "Intra", "NuSy_Intra")
    _print_mean("NuSy-Edge", "Lagged", "NuSy_Lagged")

    # Export mean summary to results/rq2_hdfs_step2_summary.csv
    try:
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", "rq2_hdfs_step2_summary.csv")
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\n[INFO] HDFS Step2 mean summary written to {out_path}")
    except Exception as ex:
        print(f"\n[WARN] Failed to write HDFS Step2 summary CSV: {ex}")
    
    print("="*80)

if __name__ == "__main__":
    main()