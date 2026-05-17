import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RISK_FREE_RATE


def compute_metrics(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Compute annualised performance statistics from a monthly return series.
    """
    ret = returns.dropna()
    n_months = len(ret)
    n_years = n_months / 12.0
    monthly_rf = (1.0 + risk_free_rate) ** (1.0 / 12.0) - 1.0

    # CAGR
    total_growth = (1.0 + ret).prod()
    cagr = total_growth ** (1.0 / n_years) - 1.0

    # Annualised volatility
    vol = ret.std() * np.sqrt(12)

    # Sharpe ratio (annualised)
    excess = ret - monthly_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(12) if excess.std() > 0 else np.nan

    # Maximum drawdown
    cum = (1.0 + ret).cumprod()
    rolling_peak = cum.cummax()
    drawdowns = cum / rolling_peak - 1.0
    max_dd = drawdowns.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "CAGR":         cagr,
        "Volatility":   vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
    }


def compare_strategies(results: dict) -> pd.DataFrame:
    """
    Build a performance-summary DataFrame for all three strategies.
    """
    strategies = {
        "Regime Portfolio": results["portfolio"],
        "S&P 500 (SPY)":    results["spy"],
        "Equal Weight":     results["equal_weight"],
    }
    rows = {name: compute_metrics(ret) for name, ret in strategies.items()}
    return pd.DataFrame(rows).T


def print_performance_table(metrics_df: pd.DataFrame) -> None:
    fmt = {
        "CAGR":         "{:.2%}",
        "Volatility":   "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Max Drawdown": "{:.2%}",
        "Calmar Ratio": "{:.2f}",
    }
    display = metrics_df.copy().astype(object)
    for col, f in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f.format(x) if pd.notna(x) else "N/A"
            )

    sep = "=" * 72
    print(f"\n{sep}")
    print("  PERFORMANCE SUMMARY")
    print(sep)
    print(display.to_string())
    print(sep + "\n")
