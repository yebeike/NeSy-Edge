import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.linalg import expm
from scipy.optimize import minimize
from tqdm import tqdm

warnings.filterwarnings("ignore")

def dynotears_os_optimal_kernel(X, lambda_w, lambda_a, penalty_mask, max_iter=150, h_tol=1e-8):
    T, d = X.shape
    X_cur, X_lag = X[1:], X[:-1]
    n = T - 1
    XtX, XtX_lag, XlagtX_lag = X_cur.T @ X_cur, X_cur.T @ X_lag, X_lag.T @ X_lag
    def _h(W): return np.trace(expm(W * W)) - d
    def _func(params, rho, alpha):
        W, A = params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)
        loss = (0.5 / n) * np.sum((X_cur - X_cur @ W - X_lag @ A) ** 2)
        penalty = lambda_w * np.sum(np.abs(W * penalty_mask)) + lambda_a * np.sum(np.abs(A))
        h_val = _h(W)
        return loss + 0.5 * rho * h_val**2 + alpha * h_val + penalty
    def _grad(params, rho, alpha):
        W, A = params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)
        h_val = _h(W)
        G_h = expm(W * W).T * 2 * W
        G_W = (1.0/n) * (XtX @ W + XtX_lag @ A - XtX) + (rho * h_val + alpha) * G_h + lambda_w * np.sign(W) * penalty_mask
        G_A = (1.0/n) * (XtX_lag.T @ W + XlagtX_lag @ A - XtX_lag.T) + lambda_a * np.sign(A)
        return np.concatenate([G_W.flatten(), G_A.flatten()])
    params = np.zeros(2 * d * d)
    rho, alpha, h_val = 1.0, 0.0, np.inf
    bounds = [(0, 0) if i < d*d and (i // d == i % d) else (None, None) for i in range(2 * d * d)]
    for _ in range(max_iter):
        res = minimize(_func, params, args=(rho, alpha), method='L-BFGS-B', jac=_grad, bounds=bounds, options={'maxiter': 60})
        params = res.x
        h_val = _h(params[:d*d].reshape(d, d))
        alpha += rho * h_val
        if h_val < h_tol: break
        rho *= 10
    return params[:d*d].reshape(d, d), params[d*d:].reshape(d, d)

def run_baselines(X_norm):
    corr = np.abs(pd.DataFrame(X_norm).corr().values)
    np.fill_diagonal(corr, 0)
    try:
        from causalearn.search.ConstraintBased.PC import pc
        cg = pc(X_norm, 0.05, verbose=False)
        adj_pc = (np.abs(cg.G.graph) > 0).astype(int)
    except: adj_pc = np.zeros_like(corr)
    return corr, adj_pc

class OSFinalBench:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, index_col=0)
        self.feats = self.df.columns.tolist()
        self.d = len(self.feats)
        self.src, self.tgt = "ad00ceb50c09", "a468a5cab954"
        self.s_idx, self.t_idx = self.feats.index(self.src), self.feats.index(self.tgt)
        
    def execute(self, runs=15):
        best_lw, best_th = 0.030, 0.40
        results = {"NuSy-Edge": [], "Pearson": [], "PC_Algo": []}
        
        for i in tqdm(range(runs), desc="Final OS Benchmark"):
            np.random.seed(9000 + i)
            data = self.df.copy()
            data[self.tgt] += data[self.tgt].std() * 18.0 * data[self.src] + np.random.normal(0, 0.1, len(data))
            X_norm = StandardScaler().fit_transform(data)
            
            W_n, _ = dynotears_os_optimal_kernel(X_norm, best_lw, best_lw*2, np.ones((self.d, self.d)))
            W_n[np.abs(W_n) < best_th] = 0
            W_p, W_pc = run_baselines(X_norm)
            
            for name, W_mat in zip(["NuSy-Edge", "Pearson", "PC_Algo"], [W_n, W_p, W_pc]):
                w_abs = np.abs(W_mat)
                val = w_abs[self.s_idx, self.t_idx]
                flat = w_abs.flatten()
                flat.sort()
                rank = np.where(flat[::-1] == val)[0][0] + 1 if val > 1e-6 else self.d**2
                results[name].append({"rank": rank, "sparsity": np.count_nonzero(w_abs)})
        return results

if __name__ == "__main__":
    bench = OSFinalBench("data/processed/openstack_refined_ts.csv")
    final_res = bench.execute()
    print("\n" + "="*80)
    print(f"RQ2 FINAL BENCHMARK (OPENSTACK) | DIMENSIONS: 50 | SAMPLES: 3864")
    print("="*80)
    print(f"{'Algorithm':<20} | {'Avg_Rank':<12} | {'Sparsity':<12} | {'Precision@Top3'}")
    print("-" * 80)
    for algo in ["NuSy-Edge", "Pearson", "PC_Algo"]:
        avg_r = np.mean([r['rank'] for r in final_res[algo]])
        avg_s = np.mean([r['sparsity'] for r in final_res[algo]])
        p_at_3 = sum(1 for r in final_res[algo] if r['rank'] <= 3) / 15
        print(f"{algo:<20} | #{avg_r:<11.2f} | {avg_s:<12.1f} | {p_at_3:.2%}")
    print("="*80)