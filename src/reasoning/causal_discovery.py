import numpy as np
from scipy.optimize import minimize
import pandas as pd

class Dynotears:
    """
    DYNOTEARS Implementation (Full Version).
    Jointly learns intra-slice (W) and inter-slice (A) dependencies.
    Model: X_t = X_t W + X_{t-1} A + Z_t
    Objective: minimize 0.5/n * ||X - XW - Y A||_F^2 + lambda_w ||W||_1 + lambda_a ||A||_1
    Subject to: h(W) = 0 (Acyclicity on W)
    """
    
    def __init__(self, lambda_w=0.1, lambda_a=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16):
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.W_est = None
        self.A_est = None

    def _loss(self, W, A, X, Y):
        """Least Squares Loss with L1 Regularization"""
        # X: (n, d), Y: (n, d) - Lagged X
        # Prediction: X_pred = X @ W + Y @ A
        n = X.shape[0]
        M = X @ W + Y @ A
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        l1_loss = self.lambda_w * np.abs(W).sum() + self.lambda_a * np.abs(A).sum()
        return loss + l1_loss

    def _h(self, W):
        """Acyclicity Constraint on W"""
        d = W.shape[0]
        M = W * W
        h = np.trace(np.linalg.matrix_power(np.eye(d) + M / d, d)) - d
        return h

    def _func(self, params, X, Y, d, rho, alpha):
        """Augmented Lagrangian Objective"""
        # Unpack params -> W, A
        W = params[:d*d].reshape(d, d)
        A = params[d*d:].reshape(d, d)
        
        loss = self._loss(W, A, X, Y)
        h_val = self._h(W)
        
        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val
        return obj

    def fit(self, dataframe: pd.DataFrame, lag=1):
        """
        :param dataframe: Time series data (Time x Features)
        :param lag: Time lag (default 1)
        """
        data = dataframe.values
        n_samples, d = data.shape
        
        # 构造 X (Current) 和 Y (Past)
        # X = t_1 ... t_n
        # Y = t_0 ... t_{n-1}
        X = data[lag:, :]
        Y = data[:-lag, :]
        
        # Init params
        W_est = np.zeros((d, d))
        A_est = np.zeros((d, d))
        params = np.hstack([W_est.flatten(), A_est.flatten()])
        
        rho, alpha, h = 1.0, 0.0, np.inf
        
        # Diagonal of W must be 0
        bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)] # W bounds
        bnds += [(None, None)] * (d * d) # A bounds (no constraints)
        
        print(f"🚀 Running DYNOTEARS on {d} nodes, {X.shape[0]} samples...")

        for i in range(self.max_iter):
            while rho < self.rho_max:
                res = minimize(
                    self._func, 
                    params, 
                    args=(X, Y, d, rho, alpha),
                    method='L-BFGS-B', 
                    bounds=bnds,
                    options={'maxiter': 500, 'disp': False}
                )
                params_new = res.x
                W_new = params_new[:d*d].reshape(d, d)
                h_new = self._h(W_new)
                
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            
            params = params_new
            W_est = params[:d*d].reshape(d, d)
            h = h_new
            alpha += rho * h
            
            if h <= self.h_tol or rho >= self.rho_max:
                break
                
        self.W_est = params[:d*d].reshape(d, d)
        self.A_est = params[d*d:].reshape(d, d)
        
        # Thresholding
        self.W_est[np.abs(self.W_est) < 0.1] = 0
        self.A_est[np.abs(self.A_est) < 0.1] = 0
        
        return self.W_est, self.A_est