"""
RQ2 Causal Reasoning Validation Pipeline for NeSy-Edge
------------------------------------------------------
Methodology: "Oracle Experiment"
1. Flash Matcher: Simulate perfect edge parsing.
2. Aggregation: Convert logs to time-series matrix.
3. Injection: Inject synthetic causal faults (Lagged & Confounder).
4. Evaluation: Compare Pearson, PC (Baseline) vs DYNOTEARS (Ours).
"""

import os
import re
import sys
import time
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom DYNOTEARS implementation
try:
    from src.reasoning.dynotears import dynotears
except ImportError:
    print("[Error] Could not import 'src.reasoning.dynotears'. Make sure the file exists.")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ==========================================
# CONFIGURATION
# ==========================================
DEBUG_MODE = True  # Set False for Full Run
DEBUG_LIMIT = 200000

LOG_FILE = "data/raw/HDFS_v1/HDFS.log"
TEMPLATE_FILE = "data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv"
WINDOW_SIZE = '1min' 

# Injection Settings
INJECTION_STRENGTH = 5.0
NOISE_LEVEL = 0.1

# ==========================================
# 1. FLASH MATCHER
# ==========================================
class FlashMatcher:
    def __init__(self, template_path):
        self.templates = self._load_templates(template_path)
        self.regex_map = self._compile_regex()

    def _load_templates(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template file not found: {path}")
        df = pd.read_csv(path)
        return dict(zip(df['EventId'], df['EventTemplate']))

    def _compile_regex(self):
        regex_map = {}
        for eid, content in self.templates.items():
            pattern = re.escape(content)
            pattern = pattern.replace(r'\[\*\]', r'(.*?)') 
            regex_map[eid] = re.compile(pattern)
        return regex_map

    def parse_log_file(self, log_path):
        print(f"[*] Starting Flash Matching on {log_path}...")
        parsed_events = []
        start_time = time.time()
        matched_count = 0
        
        # Estimate total lines for tqdm
        total_lines = DEBUG_LIMIT if DEBUG_MODE else 11175629
        
        with open(log_path, 'r') as f:
            for idx, line in tqdm(enumerate(f), total=total_lines, unit="line", desc="Parsing"):
                if DEBUG_MODE and idx >= DEBUG_LIMIT:
                    break
                
                parts = line.strip().split()
                if len(parts) < 6: continue
                
                # Timestamp: 081109 203518
                ts_str = f"{parts[0]} {parts[1]}"
                try:
                    dt = datetime.strptime(ts_str, "%y%m%d %H%M%S")
                except ValueError:
                    continue

                # Content extraction
                try:
                    content_start_idx = 5 
                    content = " ".join(parts[content_start_idx:])
                except:
                    continue

                matched_eid = "Unknown"
                for eid, regex in self.regex_map.items():
                    if regex.search(content):
                        matched_eid = eid
                        break
                
                if matched_eid != "Unknown":
                    parsed_events.append({'Timestamp': dt, 'EventId': matched_eid})
                    matched_count += 1

        print(f"[+] Flash Matching done. Processed {idx} lines. Matched {matched_count} events.")
        print(f"[+] Time elapsed: {time.time() - start_time:.2f}s")
        return pd.DataFrame(parsed_events)

# ==========================================
# 2. AGGREGATOR & INJECTOR
# ==========================================
class TimeSeriesBuilder:
    def __init__(self, events_df):
        self.df = events_df

    def aggregate(self):
        print("[*] Aggregating events into time series...")
        if self.df.empty:
            print("[Error] No events to aggregate.")
            return pd.DataFrame()

        self.df.set_index('Timestamp', inplace=True)
        # FIX: Use size() instead of count() to avoid IndexError on single column
        ts_df = self.df.groupby('EventId').resample(WINDOW_SIZE).size().unstack(level=0).fillna(0)
        ts_df.sort_index(inplace=True)
        
        print(f"[+] Aggregation complete. Matrix Shape: {ts_df.shape}")
        return ts_df

    def inject_causal_patterns(self, ts_df):
        print("[*] Injecting causal patterns (Ground Truth)...")
        data = ts_df.copy()
        features = data.columns.tolist()
        n_nodes = len(features)
        
        # Ground Truth Matrices
        gt_w = np.zeros((n_nodes, n_nodes)) # Intra
        gt_a = np.zeros((n_nodes, n_nodes)) # Inter/Lagged
        
        cols = data.columns
        # Safety check for column existence
        if len(cols) < 4:
            print("[Warning] Not enough event types for complex injection. Using available columns.")
        
        src = 'E5' if 'E5' in cols else cols[0]
        tgt = 'E11' if 'E11' in cols else cols[1]
        
        conf_src = 'E22' if 'E22' in cols else cols[min(2, len(cols)-1)]
        conf_tgt1 = 'E26' if 'E26' in cols else cols[min(3, len(cols)-1)]
        
        print(f"    - Scenario A (Lagged): {src}(t-1) -> {tgt}(t)")
        print(f"    - Scenario B (Confounder): Hidden_Z -> {conf_src}(t) & {conf_tgt1}(t)")

        # Scenario A: Time Lagged Injection
        src_idx = features.index(src)
        tgt_idx = features.index(tgt)
        shifted_src = data[src].shift(1).fillna(0)
        data[tgt] = data[tgt] + (INJECTION_STRENGTH * shifted_src) + np.random.normal(0, NOISE_LEVEL, len(data))
        gt_a[src_idx, tgt_idx] = 1

        # Scenario B: Confounder Injection
        z = np.random.poisson(1, len(data)) * INJECTION_STRENGTH
        data[conf_src] = data[conf_src] + z
        data[conf_tgt1] = data[conf_tgt1] + z
        # No edge in GT for confounder
        
        return data, gt_w, gt_a, features

# ==========================================
# 3. ALGORITHM RUNNER
# ==========================================
def run_baselines(data, features):
    print("[*] Running Baselines...")
    results = {}
    
    # 1. Pearson
    corr_matrix = data.corr().abs().values
    threshold = 0.5
    adj_pearson = (corr_matrix > threshold).astype(int)
    np.fill_diagonal(adj_pearson, 0)
    results['Pearson'] = adj_pearson
    
    # 2. PC Algorithm
    try:
        from causalearn.search.ConstraintBased.PC import pc
        data_np = data.values
        # PC usually needs more samples, might be slow on full data
        # Alpha=0.05 is standard
        cg = pc(data_np, 0.05, verbose=False) 
        adj_pc = cg.G.graph
        # PC output: -1 (tail), 1 (head).
        # We simplify to: if there is a link, treat as undirected existence for SHD
        adj_pc = (np.abs(adj_pc) > 0).astype(int)
        np.fill_diagonal(adj_pc, 0)
        results['PC_Algo'] = adj_pc
    except Exception as e:
        print(f"    [!] PC Algorithm failed or not installed: {e}")
        results['PC_Algo'] = np.zeros((len(features), len(features)))
    
    return results

def run_dynotears_wrapper(data, features):
    print("[*] Running DYNOTEARS (NeSy-Edge)...")
    data_np = data.values
    # Call our local implementation
    # Threshold 0.3 to remove weak noise edges
    w_est, a_est = dynotears(data_np, lambda_w=0.05, lambda_a=0.05, w_threshold=0.3)
    return w_est, a_est

def evaluate_results(pred_adj, gt_adj, alg_name, matrix_type="Combined"):
    y_true = gt_adj.flatten()
    y_pred = (np.abs(pred_adj) > 0).astype(int).flatten()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # SHD calculation (simplified adjacency difference)
    diff = np.abs((np.abs(pred_adj) > 0).astype(int) - (gt_adj > 0).astype(int))
    shd = np.sum(diff)
    
    return {
        "Algorithm": alg_name,
        "Matrix": matrix_type,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "SHD": int(shd)
    }

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"=== NeSy-Edge RQ2 Experimental Pipeline ===")
    print(f"Mode: {'DEBUG (Fast)' if DEBUG_MODE else 'FULL (Production)'}")
    
    # 1. Match
    matcher = FlashMatcher(TEMPLATE_FILE)
    if not os.path.exists(LOG_FILE):
        print(f"[Error] Log file not found: {LOG_FILE}")
        return
    events_df = matcher.parse_log_file(LOG_FILE)
    
    # 2. Aggregate
    builder = TimeSeriesBuilder(events_df)
    ts_df = builder.aggregate()
    
    if ts_df.shape[0] < 5:
        print("[Error] Not enough data points. Increase DEBUG_LIMIT.")
        return

    # 3. Inject
    final_data, gt_w, gt_a, features = builder.inject_causal_patterns(ts_df)
    
    # 4. Run Algorithms
    baseline_res = run_baselines(final_data, features)
    dyno_w, dyno_a = run_dynotears_wrapper(final_data, features)
    
    # 5. Report
    report = []
    
    # A-Matrix (Lagged) - The critical test for DCCS
    report.append(evaluate_results(dyno_a, gt_a, "DYNOTEARS", "Lagged (A)"))
    
    # W-Matrix (Intra) - Test for Confounders
    # Compare against GT_W (which is empty in our injection scenario)
    report.append(evaluate_results(dyno_w, gt_w, "DYNOTEARS", "Intra (W)"))
    report.append(evaluate_results(baseline_res['Pearson'], gt_w, "Pearson", "Intra (W)"))
    report.append(evaluate_results(baseline_res['PC_Algo'], gt_w, "PC_Algo", "Intra (W)"))

    print("\n" + "="*70)
    print(f"{'Algorithm':<15} | {'Matrix':<10} | {'F1-Score':<8} | {'SHD':<5} | {'Recall':<6}")
    print("-" * 70)
    for res in report:
        print(f"{res['Algorithm']:<15} | {res['Matrix']:<10} | {res['F1-Score']:<8} | {res['SHD']:<5} | {res['Recall']:<6}")
    print("="*70)

if __name__ == "__main__":
    main()