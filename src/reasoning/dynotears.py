import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from tqdm import tqdm

def dynotears(X, lambda_w=0.1, lambda_a=0.1, max_iter=100, h_tol=1e-8, w_threshold=0.0):
    """
    DYNOTEARS Implementation with Analytical Gradient.
    """
    T, d = X.shape
    X_current = X[1:] # n x d
    X_lag = X[:-1]    # n x d
    n = T - 1

    # Pre-compute covariances for efficiency
    # Loss = (1/2n) * || Y - YW - Z A ||^2
    # Gradient computation relies on these products
    XtX = X_current.T @ X_current
    XtX_lag = X_current.T @ X_lag
    XlagtX_lag = X_lag.T @ X_lag
    
    def _loss(W, A):
        # M = Y - YW - Z A
        M = X_current - X_current @ W - X_lag @ A
        loss = (0.5 / n) * np.sum(M ** 2)
        return loss

    def _h(W):
        # Acyclicity: tr(exp(W*W)) - d = 0
        M = W * W
        return np.trace(expm(M)) - d

    def _func(params, rho, alpha):
        W = params[:d*d].reshape(d, d)
        A = params[d*d:].reshape(d, d)
        
        loss = _loss(W, A)
        h_val = _h(W)
        
        # Augmented Lagrangian + L1
        # Note: L-BFGS-B handles smooth parts. L1 is non-smooth. 
        # Ideally we use Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) for L1.
        # Here we approximate L1 gradient or rely on bounds, but standard scipy doesn't support L1 well.
        # For robustness in this thesis experiment, we treat L1 as part of objective but gradient might oscillate around 0.
        # A common trick is to use smooth approximation or just ignore L1 in gradient (sub-gradient).
        
        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + \
              lambda_w * np.sum(np.abs(W)) + \
              lambda_a * np.sum(np.abs(A))
        return obj

    def _grad(params, rho, alpha):
        """Analytical Gradient of the Objective Function"""
        W = params[:d*d].reshape(d, d)
        A = params[d*d:].reshape(d, d)
        
        # 1. Gradient of MSE Loss
        # Residual R = X_current - X_current W - X_lag A
        # Grad_W = -1/n * X_current.T * R
        # Grad_A = -1/n * X_lag.T * R
        # Expanded:
        # Grad_W = 1/n * (XtX W + XtX_lag A - XtX)
        # Grad_A = 1/n * (XtX_lag.T W + XlagtX_lag A - XtX_lag.T)
        
        G_loss_W = (1.0/n) * (XtX @ W + XtX_lag @ A - XtX)
        G_loss_A = (1.0/n) * (XtX_lag.T @ W + XlagtX_lag @ A - XtX_lag.T)
        
        # 2. Gradient of H (Acyclicity)
        # grad_h = (exp(W*W))^T * 2W
        E = expm(W * W)
        G_h = E.T * 2 * W
        
        # 3. Gradient of Augmented Lagrangian Terms
        # term: 0.5*rho*h^2 + alpha*h
        # grad: (rho*h + alpha) * grad_h
        h_val = _h(W)
        G_aug_W = (rho * h_val + alpha) * G_h
        
        # 4. Gradient of L1 Regularization (Sub-gradient)
        # sign(W) * lambda
        G_reg_W = lambda_w * np.sign(W)
        G_reg_A = lambda_a * np.sign(A)
        
        # Combine
        G_W = G_loss_W + G_aug_W + G_reg_W
        G_A = G_loss_A + G_reg_A
        
        return np.concatenate([G_W.flatten(), G_A.flatten()])

    # Initialization
    W_est = np.zeros((d, d))
    A_est = np.zeros((d, d))
    params = np.concatenate([W_est.flatten(), A_est.flatten()])
    
    rho = 1.0
    alpha = 0.0
    h_val = np.inf
    
    # Diagonal bounds logic
    bounds = []
    for i in range(2 * d * d):
        # Determine if this index corresponds to a diagonal element of W
        is_W = i < d*d
        if is_W:
            row = i // d
            col = i % d
            if row == col:
                bounds.append((0, 0)) # Diagonal MUST be 0
                continue
        bounds.append((None, None))
    
    pbar = tqdm(range(max_iter), desc="DYNOTEARS Opt", unit="iter")
    
    for _ in pbar:
        # Check convergence
        if h_val < h_tol and rho > 1e16: # Ensure rho is large enough before quitting
            break
        
        # Pass both func and jac (gradient)
        res = minimize(_func, params, args=(rho, alpha), method='L-BFGS-B', jac=_grad, bounds=bounds)
        
        params = res.x
        W_new = params[:d*d].reshape(d, d)
        h_val = _h(W_new)
        
        # Update Duals
        alpha += rho * h_val
        if h_val > h_tol:
            rho *= 10
        
        pbar.set_postfix({'h': f"{h_val:.2e}", 'rho': f"{rho:.0e}"})
        
    pbar.close()

    W_est = params[:d*d].reshape(d, d)
    A_est = params[d*d:].reshape(d, d)
    
    # Hard Thresholding
    W_est[np.abs(W_est) < w_threshold] = 0
    A_est[np.abs(A_est) < w_threshold] = 0
    np.fill_diagonal(W_est, 0)
    
    return W_est, A_est