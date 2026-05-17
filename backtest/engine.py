import numpy as np
import pandas as pd
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import LOOKBACK_WINDOW, BACKTEST_START, TICKERS, RISK_AVERSION
from models.mean_variance import regime_optimal_weights, blended_regime_weights


def run_backtest(
    equity_returns: pd.DataFrame,
    regime_series: pd.Series,
    proba_df: Optional[pd.DataFrame] = None,
    lookback_window: int = LOOKBACK_WINDOW,
    risk_aversion: float = RISK_AVERSION,
    backtest_start: str = BACKTEST_START,
) -> dict:
    """
    Walk-forward backtest with monthly rebalancing.

    At end of month t:
      1. Identify current regime (hard label or posterior probabilities).
      2. Pull the [t-lookback, t) window of equity returns and regime labels.
      3. Compute weights — either hard MV for the current regime, or a
         probability-weighted blend of all three regime portfolios.
      4. Apply those weights to month t+1's realised returns.

    Parameters
    ----------
    proba_df : DataFrame (T, n_regimes), optional
        HMM posterior probabilities aligned to macro index.
        When provided, uses probabilistic blending instead of hard labels.
    """
    # Align on common monthly dates
    common = equity_returns.index.intersection(regime_series.index)
    returns = equity_returns.loc[common]
    regimes = regime_series.loc[common]

    # Align proba_df to the same index if supplied
    probas = proba_df.reindex(common) if proba_df is not None else None

    all_dates = returns.index
    start_pos = all_dates.searchsorted(pd.Timestamp(backtest_start))

    port_rets, port_weights, port_dates = [], [], []

    for i in range(start_pos, len(all_dates) - 1):
        t_next = all_dates[i + 1]

        # Lookback window: rows [i-lookback_window, i)
        lb_start = max(0, i - lookback_window)
        ret_win = returns.iloc[lb_start:i].values   # (W, N)
        reg_win = regimes.iloc[lb_start:i].values   # (W,)

        # Drop rows with any NaN in returns
        valid = ~np.any(np.isnan(ret_win), axis=1)
        ret_win = ret_win[valid]
        reg_win = reg_win[valid]

        if len(ret_win) < 2:
            w = np.ones(len(TICKERS)) / len(TICKERS)
        elif probas is not None:
            cur_proba = probas.iloc[i].values   # (n_regimes,)
            w = blended_regime_weights(ret_win, reg_win, cur_proba, risk_aversion)
        else:
            cur_regime = int(regimes.iloc[i])
            w = regime_optimal_weights(ret_win, reg_win, cur_regime, risk_aversion)

        next_ret = returns.loc[t_next].values
        port_rets.append(float(w @ next_ret))
        port_weights.append(w)
        port_dates.append(t_next)

    portfolio = pd.Series(port_rets, index=port_dates, name="regime_portfolio")
    weights_df = pd.DataFrame(port_weights, index=port_dates, columns=TICKERS)

    spy = returns["SPY"].loc[port_dates]
    equal_weight = returns.loc[port_dates].mean(axis=1).rename("equal_weight")

    return {
        "portfolio":    portfolio,
        "spy":          spy,
        "equal_weight": equal_weight,
        "weights":      weights_df,
        "regimes":      regimes.loc[port_dates],
    }
