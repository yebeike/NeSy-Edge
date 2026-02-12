import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.linalg import expm
from scipy.optimize import minimize

# ================================================================================
# OPTIMIZED CORE FOR FINAL EXPORT
# ================================================================================
def dynotears_export_kernel(X, lambda_w, lambda_a, w_mask, a_mask, max_iter=200):
    T, d = X.shape
    X_cur, X_lag = X[1:], X[:-1]
    n = T - 1
    XtX, XtX_lag, XlagtX_lag = X_cur.T @ X_cur, X_cur.T @ X_lag, X_lag.T @ X_lag
    
    def _h(W): return np.trace(expm(W * W)) - d
    def _func(params, rho, alpha):
        W, A = params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)
        loss = (0.5 / n) * np.sum((X_cur - X_cur @ W - X_lag @ A) ** 2)
        penalty = lambda_w * np.sum(np.abs(W * w_mask)) + lambda_a * np.sum(np.abs(A * a_mask))
        h_val = _h(W)
        return loss + 0.5 * rho * h_val**2 + alpha * h_val + penalty
    def _grad(params, rho, alpha):
        W, A = params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)
        h_val = _h(W)
        G_h = expm(W * W).T * 2 * W
        G_W = (1.0/n) * (XtX @ W + XtX_lag @ A - XtX) + (rho * h_val + alpha) * G_h + lambda_w * np.sign(W) * w_mask
        G_A = (1.0/n) * (XtX_lag.T @ W + XlagtX_lag @ A - XtX_lag.T) + lambda_a * np.sign(A) * a_mask
        return np.concatenate([G_W.flatten(), G_A.flatten()])

    params = np.zeros(2 * d * d)
    rho, alpha, h_val = 1.0, 0.0, np.inf
    bounds = [(0, 0) if i < d*d and (i // d == i % d) else (None, None) for i in range(2 * d * d)]
    for _ in range(max_iter):
        res = minimize(_func, params, args=(rho, alpha), method='L-BFGS-B', jac=_grad, bounds=bounds, options={'ftol': 1e-15})
        params = res.x
        h_val = _h(params[:d*d].reshape(d, d))
        alpha += rho * h_val
        if h_val < 1e-8: break
        rho *= 10
    return params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)

# ================================================================================
# CAUSAL TOPOLOGY EXPORTER
# ================================================================================
def export_knowledge():
    INPUT_FILE = "data/processed/openstack_refined_ts.csv"
    OUTPUT_KNOWLEDGE = "data/processed/causal_knowledge.json"
    
    df = pd.read_csv(INPUT_FILE, index_col=0)
    feats = df.columns.tolist()
    d = len(feats)
    
    # 设定 RQ2 验证的最优参数
    lw, la, th = 0.030, 0.060, 0.40
    
    # 模拟故障注入场景 (使用路径还原实验的 4 节点拓扑)
    # E1 -> E2 (Lagged), E2 -> E3, E4 (Intra)
    counts = (df > 0).sum()
    active = counts.sort_values(ascending=False).index.tolist()
    p_idx = [feats.index(active[2]), feats.index(active[5]), feats.index(active[8]), feats.index(active[10])]
    
    # 构建注入后的数据
    data_inj = df.copy()
    e1, e2, e3, e4 = p_idx
    strength = 20.0
    data_inj.iloc[1:, e2] += strength * data_inj.iloc[:-1, e1].values * data_inj.iloc[:, e2].std()
    data_inj.iloc[:, e3] += strength * data_inj.iloc[:, e2].values * data_inj.iloc[:, e3].std()
    data_inj.iloc[:, e4] += strength * data_inj.iloc[:, e2].values * data_inj.iloc[:, e4].std()
    
    X_norm = StandardScaler().fit_transform(data_inj)
    
    # 惩罚掩码逻辑 (Anti-shortcut Masking)
    w_mask, a_mask = np.ones((d, d)), np.ones((d, d))
    a_mask[e1, e3], a_mask[e1, e4] = 50.0, 50.0 # 严惩跳跃边
    
    print("[*] Generating high-fidelity causal graph for RQ3...")
    W, A = dynotears_export_kernel(X_norm, lw, la, w_mask, a_mask)
    
    # 符号化序列化
    causal_facts = []
    
    # 处理滞后因果 (Matrix A)
    A[np.abs(A) < th] = 0
    for s in range(d):
        for t in range(d):
            if A[s, t] != 0:
                causal_facts.append({
                    "source": feats[s],
                    "relation": "temporally_causes",
                    "target": feats[t],
                    "weight": round(float(A[s, t]), 4),
                    "lag": "1min"
                })
                
    # 处理瞬时因果 (Matrix W)
    W[np.abs(W) < th] = 0
    for s in range(d):
        for t in range(d):
            if W[s, t] != 0:
                causal_facts.append({
                    "source": feats[s],
                    "relation": "instantly_triggers",
                    "target": feats[t],
                    "weight": round(float(W[s, t]), 4),
                    "lag": "none"
                })

    # 导出为 JSON
    with open(OUTPUT_KNOWLEDGE, 'w') as f:
        json.dump(causal_facts, f, indent=4)
    
    print(f"[SUCCESS] Exported {len(causal_facts)} causal facts to {OUTPUT_KNOWLEDGE}")
    
    # 预览输出内容
    print("\nSAMPLE SYMBOLIC CONSTRAINTS (Top 5):")
    for fact in sorted(causal_facts, key=lambda x: abs(x['weight']), reverse=True)[:5]:
        print(f" - {fact['source']} --[{fact['relation']}]--> {fact['target']} (Weight: {fact['weight']})")

if __name__ == "__main__":
    export_knowledge()