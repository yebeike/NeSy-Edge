import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import entropy
import os

def run_scientific_audit(input_path):
    print("\n" + "="*70)
    print("SCIENTIFIC AUDIT: OPENSTACK REFINED CAUSAL MATRIX")
    print("="*70)
    
    df = pd.read_csv(input_path, index_col=0)
    
    # 1. 平稳性分析 (ADF Test)
    print(f"[*] Testing Stationarity (ADF)...")
    p_values = []
    for col in df.columns:
        res = adfuller(df[col])
        p_values.append(res[1])
    
    stationarity_pass = sum(1 for p in p_values if p < 0.05)
    print(f"    - Stationary Dimensions: {stationarity_pass}/{len(df.columns)} (p < 0.05)")
    
    # 2. 数值稳定性分析 (Condition Number)
    print(f"[*] Checking Matrix Condition Number...")
    # 使用协方差矩阵的特征值比值
    corr = df.corr().fillna(0).values
    cond_num = np.linalg.cond(corr)
    print(f"    - Condition Number: {cond_num:.4f}")
    
    # 3. 信息熵分析 (Shannon Entropy)
    # 验证特征是否包含足够的“信息量”，而不是死板的重复脉冲
    print(f"[*] Measuring Information Density (Shannon Entropy)...")
    entropies = []
    for col in df.columns:
        counts = df[col].value_counts()
        entropies.append(entropy(counts))
    avg_entropy = np.mean(entropies)
    print(f"    - Average Shannon Entropy: {avg_entropy:.4f}")
    
    # 4. 判定结论
    print("-" * 70)
    is_valid = True
    if stationarity_pass / len(df.columns) < 0.7:
        print("[FAIL] Too many non-stationary series. Consider differencing (差分处理).")
        is_valid = False
    if cond_num > 500:
        print("[FAIL] Matrix is ill-conditioned (多重共线性过强).")
        is_valid = False
    
    if is_valid:
        print("[CONCLUSION] Matrix is scientifically qualified for DYNOTEARS.")
    else:
        print("[CONCLUSION] Matrix requires further preprocessing before RQ2 tuning.")
    print("="*70)
    return is_valid

if __name__ == "__main__":
    AUDIT_FILE = "data/processed/openstack_refined_ts.csv"
    if os.path.exists(AUDIT_FILE):
        run_scientific_audit(AUDIT_FILE)