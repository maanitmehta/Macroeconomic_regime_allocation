import numpy as np
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RISK_AVERSION, MIN_REGIME_OBS, COVARIANCE_REG


def mean_variance_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = RISK_AVERSION,
) -> np.ndarray:
    """
    Solve:  max  mu' w  -  (lambda/2) * w' Sigma w
            s.t. sum(w) = 1,  w >= 0

    Falls back to equal-weight if the solver fails.
    """
    n = len(mu)
    sigma_reg = sigma + COVARIANCE_REG * np.eye(n)

    def objective(w: np.ndarray) -> float:
        return -(mu @ w) + (risk_aversion / 2.0) * float(w @ sigma_reg @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return -mu + risk_aversion * (sigma_reg @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if res.success:
        w = np.maximum(res.x, 0.0)
        return w / w.sum()

    return w0  # fallback: equal weight


def regime_optimal_weights(
    returns: np.ndarray,        # shape (T, N)  — monthly returns in the lookback window
    regime_labels: np.ndarray,  # shape (T,)    — regime label for each row
    current_regime: int,
    risk_aversion: float = RISK_AVERSION,
    min_obs: int = MIN_REGIME_OBS,
) -> np.ndarray:
    """
    Compute mean-variance weights using only observations from `current_regime`.
    Falls back to the full sample if too few regime observations exist.
    """
    n_assets = returns.shape[1]
    mask = regime_labels == current_regime

    if mask.sum() < min_obs:
        mask = np.ones(len(regime_labels), dtype=bool)

    regime_ret = returns[mask]

    # Annualise: monthly mean * 12, monthly cov * 12
    mu = regime_ret.mean(axis=0) * 12
    sigma = np.cov(regime_ret.T) * 12 if regime_ret.shape[0] > 1 else np.eye(n_assets)

    return mean_variance_weights(mu, sigma, risk_aversion)


def blended_regime_weights(
    returns: np.ndarray,        # shape (T, N)
    regime_labels: np.ndarray,  # shape (T,)
    current_proba: np.ndarray,  # shape (n_regimes,) — HMM posteriors at current date
    risk_aversion: float = RISK_AVERSION,
    min_obs: int = MIN_REGIME_OBS,
) -> np.ndarray:
    """
    Blend regime-conditional MV portfolios by HMM posterior probabilities.

    w_final = sum_r  p_r * w_r

    This replaces the hard "pick one regime" decision with a smooth weighted
    average, so the portfolio transitions gracefully between states rather
    than snapping from one allocation to another.
    """
    n_regimes = len(current_proba)
    w_blend = np.zeros(returns.shape[1])

    for r in range(n_regimes):
        if current_proba[r] < 1e-6:
            continue
        w_r = regime_optimal_weights(returns, regime_labels, r, risk_aversion, min_obs)
        w_blend += current_proba[r] * w_r

    w_blend = np.maximum(w_blend, 0.0)
    return w_blend / w_blend.sum()
