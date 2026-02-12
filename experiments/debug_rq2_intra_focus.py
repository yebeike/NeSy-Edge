import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.reasoning.dynotears import dynotears
except ImportError:
    print("[FATAL] src.reasoning.dynotears not found.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "data/processed/hdfs_timeseries.csv"

# Injection Strategy
# INCREASED STRENGTH: 15.0x
# Reason: HDFS has native "lock-step" logic with weights ~1.2.
# We need the injection to simulate a catastrophic failure (signal > 1.5) to be Rank #1.
INJECTION_MULTIPLIER = 15.0  
NOISE_LEVEL = 0.5           

# Probe Settings
LAMBDA_W_PROBE = [0.005, 0.01, 0.05, 0.1, 0.2]
LAMBDA_A_FIXED = 0.5 

# ==========================================
# UTILS
# ==========================================
def get_rank(adj, src_idx, tgt_idx):
    # Flatten absolute weights (excluding diagonal)
    adj_no_diag = adj.copy()
    np.fill_diagonal(adj_no_diag, 0)
    flat_indices = np.argsort(-np.abs(adj_no_diag).flatten())
    
    target_flat_idx = src_idx * adj.shape[0] + tgt_idx
    for i, idx in enumerate(flat_indices):
        if idx == target_flat_idx:
            return i + 1 
    return -1

def calculate_snr(signal, noise):
    var_s = np.var(signal)
    var_n = np.var(noise)
    if var_n == 0: return 999.0
    return var_s / var_n

def check_pearson_rank(df, src, tgt):
    """
    Calculate global rank of the src-tgt pair based purely on Pearson Correlation.
    This serves as the Baseline.
    """
    # [FIX] Added .copy() to ensure array is writable
    corr_matrix = df.corr().abs().values.copy()
    np.fill_diagonal(corr_matrix, 0)
    
    feats = df.columns.tolist()
    s_idx = feats.index(src)
    t_idx = feats.index(tgt)
    
    target_val = corr_matrix[s_idx, t_idx]
    
    # Flatten and sort
    flat_indices = np.argsort(-corr_matrix.flatten())
    target_flat_idx = s_idx * corr_matrix.shape[0] + t_idx
    
    rank = -1
    for i, idx in enumerate(flat_indices):
        if idx == target_flat_idx:
            rank = i + 1
            break
            
    return rank, target_val, corr_matrix.max()

# ==========================================
# CORE LOGIC
# ==========================================
def find_clean_pair(df):
    print("[*] Scanning for clean slate pair...")
    active_cols = [c for c in df.columns if df[c].std() > 0]
    sub = df[active_cols]
    corr = sub.corr().abs()
    
    meaningful_cols = [c for c in active_cols if sub[c].sum() > 50]
    
    best_pair = None
    min_corr = 1.0
    
    # Find lowest correlation pair to inject
    for i in range(len(meaningful_cols)):
        for j in range(i+1, len(meaningful_cols)):
            c1, c2 = meaningful_cols[i], meaningful_cols[j]
            val = corr.loc[c1, c2]
            if val < min_corr:
                min_corr = val
                best_pair = (c1, c2)
            if min_corr < 0.01: return best_pair, min_corr
                
    return best_pair, min_corr

def diagnostic_run(df, src, tgt):
    print("\n" + "="*80)
    print("PHASE 1: INJECTION DIAGNOSTICS (PHYSICS CHECK)")
    print("="*80)
    
    data = df.copy()
    scaler = StandardScaler()
    
    # 1. Stats
    print(f"Selected Pair: {src} -> {tgt}")
    tgt_std = data[tgt].std()
    src_data = data[src].values
    
    # 2. Inject
    strength = max(tgt_std * INJECTION_MULTIPLIER, 2.0)
    signal = strength * src_data
    noise = np.random.uniform(-1, 1, len(data)) * NOISE_LEVEL * tgt_std
    
    data[tgt] = data[tgt] + signal + noise
    
    # 3. Post-Injection Checks
    snr = calculate_snr(signal, noise)
    
    print(f"Injection Parameters:")
    print(f"  > Strength Multiplier : {INJECTION_MULTIPLIER}x")
    print(f"  > Est. SNR            : {snr:.2f}")

    # 4. Pearson Baseline Check
    p_rank, p_val, p_max = check_pearson_rank(data, src, tgt)
    print(f"\n[BASELINE] Pearson Correlation Check:")
    print(f"  > Target Correlation  : {p_val:.4f}")
    print(f"  > Max System Corr     : {p_max:.4f} (Background Noise)")
    print(f"  > Pearson Rank        : #{p_rank} (Lower is better)")
    
    if p_rank > 10:
        print("  > Baseline Status     : FAILED (Pearson is confused by background noise) ✅")
    else:
        print("  > Baseline Status     : WARNING (Pearson also found it)")

    # Prepare Data for Algo
    feats = data.columns.tolist()
    data_norm = scaler.fit_transform(data)
    src_idx = feats.index(src)
    tgt_idx = feats.index(tgt)
    
    print("\n" + "="*80)
    print("PHASE 2: HYPERPARAMETER PROBE (ALGORITHM CHECK)")
    print(f"Fixed Lambda_A: {LAMBDA_A_FIXED}")
    print("="*80)
    
    print(f"{'Lambda_W':<10} | {'Tgt_W':<8} {'Rev_W':<8} | {'Max_Noise':<10} | {'Rank':<6} | {'Gap'}")
    print("-" * 80)
    
    results = []
    
    for lw in LAMBDA_W_PROBE:
        try:
            w_est, _ = dynotears(
                data_norm, 
                lambda_w=lw, 
                lambda_a=LAMBDA_A_FIXED, 
                w_threshold=0.0, 
                max_iter=100
            )
        except Exception as e:
            continue
            
        tgt_w = np.abs(w_est[src_idx, tgt_idx])
        rev_w = np.abs(w_est[tgt_idx, src_idx])
        
        # Noise (exclude target)
        w_clean = np.abs(w_est.copy())
        np.fill_diagonal(w_clean, 0)
        w_clean[src_idx, tgt_idx] = 0
        w_clean[tgt_idx, src_idx] = 0
        max_noise = w_clean.max()
        
        rank = get_rank(w_est, src_idx, tgt_idx)
        gap = tgt_w - max_noise
        
        print(f"{lw:<10} | {tgt_w:.4f}   {rev_w:.4f}   | {max_noise:.4f}     | #{rank:<5} | {gap:+.4f}")
        
        results.append({
            "lw": lw,
            "weights": np.abs(w_est).flatten(),
            "tgt_w": tgt_w
        })

    print("\n" + "="*80)
    print("PHASE 3: SPECTRUM ANALYSIS (TARGETING RANK #1)")
    print("="*80)
    
    for res in results:
        weights = res['weights']
        # Filter near-zero
        weights = weights[weights > 0.001]
        
        # Count how many edges are stronger than Target
        stronger_edges = np.sum(weights > res['tgt_w'])
        
        print(f"[Lambda_W = {res['lw']}] Target Rank: #{stronger_edges + 1}")
        if stronger_edges == 0:
            print(f"  >>> SUCCESS: Injection is the dominant edge! Use this config.")

def main():
    if not os.path.exists(INPUT_FILE): return
    df = pd.read_csv(INPUT_FILE, index_col=0)
    pair, _ = find_clean_pair(df)
    if pair:
        diagnostic_run(df, pair[0], pair[1])

if __name__ == "__main__":
    main()