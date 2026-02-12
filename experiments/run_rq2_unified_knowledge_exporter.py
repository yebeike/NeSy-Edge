import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

def dynotears_engine(X, lambda_w, lambda_a):
    T, d = X.shape
    X_cur, X_lag = X[1:], X[:-1]
    n = T - 1
    XtX, XtX_lag, XlagtX_lag = X_cur.T @ X_cur, X_cur.T @ X_lag, X_lag.T @ X_lag
    def _h(W): return np.trace(expm(W * W)) - d
    def _func(params, rho, alpha):
        W, A = params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)
        loss = (0.5 / n) * np.sum((X_cur - X_cur @ W - X_lag @ A) ** 2)
        penalty = lambda_w * np.sum(np.abs(W)) + lambda_a * np.sum(np.abs(A))
        h_val = _h(W)
        return loss + 0.5 * rho * h_val**2 + alpha * h_val + penalty
    def _grad(params, rho, alpha):
        W, A = params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)
        G_W = (1.0/n) * (XtX @ W + XtX_lag @ A - XtX) + (rho * _h(W) + alpha) * (expm(W * W).T * 2 * W) + lambda_w * np.sign(W)
        G_A = (1.0/n) * (XtX_lag.T @ W + XlagtX_lag @ A - XtX_lag.T) + lambda_a * np.sign(A)
        return np.concatenate([G_W.flatten(), G_A.flatten()])
    params = np.zeros(2 * d * d)
    rho, alpha, h_val = 1.0, 0.0, np.inf
    bounds = [(0, 0) if i < d*d and (i // d == i % d) else (None, None) for i in range(2 * d * d)]
    for _ in range(100):
        res = minimize(_func, params, args=(rho, alpha), method='L-BFGS-B', jac=_grad, bounds=bounds)
        params = res.x
        h_val = _h(params[:d*d].reshape(d, d))
        alpha += rho * h_val
        if h_val < 1e-8: break
        rho *= 10
    return params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)

def main():
    kb = []
    
    # 1. HDFS
    print("[*] Exporting HDFS Knowledge...")
    hdfs_df = pd.read_csv("data/processed/hdfs_timeseries.csv", index_col=0)
    hdfs_map = dict(zip(pd.read_csv("data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv")['EventId'], 
                        pd.read_csv("data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv")['EventTemplate']))
    
    W, A = dynotears_engine(StandardScaler().fit_transform(hdfs_df), 0.025, 0.05)
    for mat, rel in [(A, "temporally_causes"), (W, "instantly_triggers")]:
        mat[np.abs(mat) < 0.30] = 0
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            kb.append({"domain": "hdfs", "source_template": hdfs_map.get(hdfs_df.columns[s], "Unknown"),
                       "relation": rel, "target_template": hdfs_map.get(hdfs_df.columns[t], "Unknown"), "weight": round(float(mat[s,t]), 4)})

    # 2. OpenStack
    print("[*] Exporting OpenStack Knowledge...")
    os_df = pd.read_csv("data/processed/openstack_refined_ts.csv", index_col=0)
    with open("data/processed/openstack_id_map.json", 'r') as f:
        os_map = json.load(f)
    
    W, A = dynotears_engine(StandardScaler().fit_transform(os_df), 0.030, 0.06)
    for mat, rel in [(A, "temporally_causes"), (W, "instantly_triggers")]:
        mat[np.abs(mat) < 0.40] = 0
        s_idx, t_idx = np.where(mat != 0)
        for s, t in zip(s_idx, t_idx):
            kb.append({"domain": "openstack", "source_template": os_map.get(os_df.columns[s], "Unknown"),
                       "relation": rel, "target_template": os_map.get(os_df.columns[t], "Unknown"), "weight": round(float(mat[s,t]), 4)})

    with open("data/processed/causal_knowledge.json", 'w') as f:
        json.dump(kb, f, indent=4)
    print(f"[SUCCESS] Exported {len(kb)} facts. Unknowns: {sum(1 for f in kb if 'Unknown' in str(f))}")

if __name__ == "__main__":
    main()