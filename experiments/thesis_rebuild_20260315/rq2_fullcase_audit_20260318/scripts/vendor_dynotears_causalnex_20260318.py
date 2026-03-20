from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


def _build_dynamic_design(X: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={X.shape}")
    if p < 1:
        raise ValueError(f"Expected lag order p >= 1, got p={p}")
    if X.shape[0] <= p:
        raise ValueError(f"Need more rows than lag order p={p}, got rows={X.shape[0]}")
    current = X[p:]
    lag_blocks = [X[p - lag : -lag] for lag in range(1, p + 1)]
    lags = np.hstack(lag_blocks)
    return current, lags


def _reshape_wa(wa_vec: np.ndarray, d_vars: int, p_orders: int) -> Tuple[np.ndarray, np.ndarray]:
    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars : 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat


def _learn_dynamic_structure(
    X: np.ndarray,
    Xlags: np.ndarray,
    bnds: List[Tuple[float, float]],
    lambda_w: float,
    lambda_a: float,
    max_iter: int,
    h_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        raise ValueError("Input data X is empty")
    if Xlags.size == 0:
        raise ValueError("Input data Xlags is empty")
    if X.shape[0] != Xlags.shape[0]:
        raise ValueError("X and Xlags must have the same number of rows")
    if Xlags.shape[1] % X.shape[1] != 0:
        raise ValueError("Xlags columns must be a multiple of X columns")

    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    def _h(wa_vec: np.ndarray) -> float:
        w_mat, _ = _reshape_wa(wa_vec, d_vars, p_orders)
        return np.trace(slin.expm(w_mat * w_mat)) - d_vars

    def _func(wa_vec: np.ndarray) -> float:
        w_mat, a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        loss = (
            0.5
            / n
            * np.square(
                np.linalg.norm(
                    X.dot(np.eye(d_vars, d_vars) - w_mat) - Xlags.dot(a_mat),
                    "fro",
                )
            )
        )
        h_value = _h(wa_vec)
        l1_penalty = lambda_w * (wa_vec[: 2 * d_vars**2].sum()) + lambda_a * (
            wa_vec[2 * d_vars**2 :].sum()
        )
        return loss + 0.5 * rho * h_value * h_value + alpha * h_value + l1_penalty

    def _grad(wa_vec: np.ndarray) -> np.ndarray:
        w_mat, a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = slin.expm(w_mat * w_mat)
        loss_grad_w = -1.0 / n * (X.T.dot(X.dot(np.eye(d_vars, d_vars) - w_mat) - Xlags.dot(a_mat)))
        obj_grad_w = loss_grad_w + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * w_mat * 2
        obj_grad_a = -1.0 / n * (Xlags.T.dot(X.dot(np.eye(d_vars, d_vars) - w_mat) - Xlags.dot(a_mat)))

        grad_vec_w = np.append(obj_grad_w, -obj_grad_w, axis=0).flatten() + lambda_w * np.ones(2 * d_vars**2)
        grad_vec_a = obj_grad_a.reshape(p_orders, d_vars**2)
        grad_vec_a = np.hstack((grad_vec_a, -grad_vec_a)).flatten() + lambda_a * np.ones(2 * p_orders * d_vars**2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    wa_est = np.zeros(2 * (p_orders + 1) * d_vars**2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars**2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

    for n_iter in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = sopt.minimize(_func, wa_est, method="L-BFGS-B", jac=_grad, bounds=bnds).x
            h_new = _h(wa_new)
            if h_new > 0.25 * h_value:
                rho *= 10
        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break
        if h_value > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    return _reshape_wa(wa_est, d_vars, p_orders)


def dynotears_from_standardized_matrix(
    X: np.ndarray,
    p: int = 1,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Official-equivalent DYNOTEARS formulation adapted from the Apache-licensed
    CausalNex dynamic structure implementation. This audit fork uses p=1.
    """
    X_current, X_lags = _build_dynamic_design(X, p)
    _, d_vars = X_current.shape
    p_orders = X_lags.shape[1] // d_vars

    bnds_w = 2 * [
        (0, 0) if i == j else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]
    bnds_a: List[Tuple[float, float]] = []
    for _ in range(1, p_orders + 1):
        bnds_a.extend(2 * [(0, None) for _ in range(d_vars**2)])
    bnds = bnds_w + bnds_a

    w_est, a_est = _learn_dynamic_structure(
        X_current,
        X_lags,
        bnds=bnds,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        h_tol=h_tol,
    )
    w_est[np.abs(w_est) < w_threshold] = 0.0
    a_est[np.abs(a_est) < w_threshold] = 0.0
    return w_est, a_est
