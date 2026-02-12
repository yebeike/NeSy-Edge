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

# Based on your previous failure log
TARGET_SRC = "E6"
TARGET_TGT = "E20"

# Fixed Algo Params (Don't change these during sweep)
LAMBDA_W = 0.1
LAMBDA_A = 0.5
W_THRESHOLD = 0.0  # We want to see RAW weights, no cutting

# Grid Search Space for Physics
NOISE_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5]
INJECTION_MULTS = [15.0] # Fix multiplier to isolating the Noise issue first

# ==========================================
# UTILS
# ==========================================
def get_rank_info(w_est, src_idx, tgt_idx):
    # Rank calc
    adj_no_diag = np.abs(w_est.copy())
    np.fill_diagonal(adj_no_diag, 0)
    flat_indices = np.argsort(-adj_no_diag.flatten())
    
    target_flat_idx = src_idx * w_est.shape[0] + tgt_idx
    rank = -1
    for i, idx in enumerate(flat_indices):
        if idx == target_flat_idx:
            rank = i + 1
            break
            
    # Max noise (exclusion)
    temp = adj_no_diag.copy()
    temp[src_idx, tgt_idx] = 0
    temp[tgt_idx, src_idx] = 0
    max_noise = np.max(temp)
    
    return rank, max_noise

# ==========================================
# DIAGNOSTIC ROUTINE
# ==========================================
def analyze_failure_case(df, src, tgt, mult, noise_lvl):
    print(f"\n>>> DEEP DIVE: Strength={mult}x, Noise={noise_lvl}")
    
    data = df.copy()
    feats = data.columns.tolist()
    s_idx = feats.index(src)
    t_idx = feats.index(tgt)
    
    # 1. Inject
    tgt_std = data[tgt].std()
    strength = max(tgt_std * mult, 2.0)
    signal = strength * data[src]
    noise_vals = np.random.uniform(-1, 1, len(data)) * noise_lvl * tgt_std
    data[tgt] = data[tgt] + signal + noise_vals
    
    # 2. Physics Check
    raw_corr = data[[src, tgt]].corr().iloc[0,1]
    print(f"  [Physics] Post-Injection Corr: {raw_corr:.6f}")
    if raw_corr > 0.999:
        print("  [WARNING] Correlation > 0.999! Matrix likely singular/collinear.")
    
    # 3. Run Algo
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    
    print("  [Algo] Running DYNOTEARS...")
    w_est, _ = dynotears(data_norm, lambda_w=LAMBDA_W, lambda_a=LAMBDA_A, w_threshold=0.0, max_iter=100)
    
    # 4. Extract Weights
    fwd_w = np.abs(w_est[s_idx, t_idx])
    rev_w = np.abs(w_est[t_idx, s_idx])
    rank, max_noise = get_rank_info(w_est, s_idx, t_idx)
    
    print(f"  [Result] Fwd Weight ({src}->{tgt}): {fwd_w:.6f}")
    print(f"  [Result] Rev Weight ({tgt}->{src}): {rev_w:.6f}")
    print(f"  [Result] Rank: #{rank}")
    print(f"  [Result] Max Background Noise: {max_noise:.6f}")
    
    if fwd_w < 1e-4 and rev_w < 1e-4:
        print("  [Diagnosis] ZERO WEIGHTS. The L1 penalty killed the edge.")
        print("              Reason: Signal too perfect (X=Y), solver chose sparsity over fitting.")
    elif rev_w > fwd_w:
        print("  [Diagnosis] DIRECTION FLIP. Model prefers Reverse direction.")
    elif rank > 1:
        print("  [Diagnosis] OVERSHADOWED. Background logic is stronger.")
    else:
        print("  [Diagnosis] SUCCESS.")

    # 5. Show Top 5 Edges (Who won?)
    print("  [Top 5 Edges Found]:")
    adj_no_diag = np.abs(w_est)
    np.fill_diagonal(adj_no_diag, 0)
    flat_indices = np.argsort(-adj_no_diag.flatten())[:5]
    for idx in flat_indices:
        r, c = divmod(idx, w_est.shape[1])
        print(f"    - {feats[r]} -> {feats[c]} : {w_est[r,c]:.4f}")

    return {
        "noise": noise_lvl,
        "corr": raw_corr,
        "fwd_w": fwd_w,
        "rev_w": rev_w,
        "rank": rank
    }

def run_sweep(df, src, tgt):
    print("\n" + "="*80)
    print("SWEEP SCANNING: FINDING THE SWEET SPOT")
    print("="*80)
    print(f"{'Noise':<6} | {'Corr':<8} | {'Fwd_W':<8} {'Rev_W':<8} | {'Rank':<6} | {'Status'}")
    print("-" * 80)
    
    history = []
    
    for noise in NOISE_LEVELS:
        # Use a simplified run for the table (we dive deep later if needed)
        # Re-using the logic inside analyze, but suppressing output? 
        # No, let's just call analyze but maybe reduce verbosity or just parse its return.
        # Actually, let's just run the loop cleanly here.
        
        data = df.copy()
        feats = data.columns.tolist()
        s_idx, t_idx = feats.index(src), feats.index(tgt)
        
        tgt_std = data[tgt].std()
        strength = max(tgt_std * INJECTION_MULTS[0], 2.0) # Fixed mult
        
        signal = strength * data[src]
        noise_vals = np.random.uniform(-1, 1, len(data)) * noise * tgt_std
        data[tgt] = data[tgt] + signal + noise_vals
        
        raw_corr = data[[src, tgt]].corr().iloc[0,1]
        
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data)
        
        # Suppress stdout for sweep
        sys.stdout = open(os.devnull, 'w')
        try:
            w_est, _ = dynotears(data_norm, lambda_w=LAMBDA_W, lambda_a=LAMBDA_A, w_threshold=0.0, max_iter=100)
        finally:
            sys.stdout = sys.__stdout__
            
        fwd_w = np.abs(w_est[s_idx, t_idx])
        rev_w = np.abs(w_est[t_idx, s_idx])
        rank, _ = get_rank_info(w_est, s_idx, t_idx)
        
        status = "FAIL"
        if rank == 1: status = "PERFECT"
        elif rank <= 5: status = "GOOD"
        
        print(f"{noise:<6} | {raw_corr:.4f}   | {fwd_w:.4f}   {rev_w:.4f}   | #{rank:<5} | {status}")
        
        history.append((noise, rank))
        
    return history

def main():
    print("=== Debug RQ2: Failure Analysis & Scanner ===")
    if not os.path.exists(INPUT_FILE): return
    df = pd.read_csv(INPUT_FILE, index_col=0)
    
    if TARGET_SRC not in df.columns:
        print(f"Columns {TARGET_SRC}/{TARGET_TGT} not found.")
        return

    # 1. Replicate the Crash (Noise=0.05)
    print(f"[*] Analyzing specific failure case: {TARGET_SRC} -> {TARGET_TGT}")
    analyze_failure_case(df, TARGET_SRC, TARGET_TGT, 15.0, 0.05)
    
    # 2. Sweep to find fix
    run_sweep(df, TARGET_SRC, TARGET_TGT)

if __name__ == "__main__":
    main()