"""
Macroeconomic Regime Allocation
================================
Entry point.  Run:
    python main.py

Pipeline
--------
1. Load macro indicators from FRED
2. Load equity ETF returns from Yahoo Finance
3. Fit a 3-state Gaussian HMM on macro features
4a. Walk-forward backtest — hard regime label (original)
4b. Walk-forward backtest — probabilistic blend (improved)
5. Print side-by-side performance table and save charts
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # set non-interactive backend before pyplot is imported

import os
import pandas as pd

from config import (
    FRED_API_KEY, START_DATE, END_DATE,
    N_REGIMES, LOOKBACK_WINDOW, RISK_AVERSION,
    TICKERS, OUTPUT_DIR, BACKTEST_START, REGIME_LABELS,
)
from data.fred_loader import load_fred_data
from data.equity_loader import load_equity_data
from models.hmm_regime import RegimeDetector
from backtest.engine import run_backtest
from analysis.metrics import compare_strategies, print_performance_table
from visualization.plots import (
    plot_regime_timeline,
    plot_cumulative_returns_dual,
    plot_drawdowns_dual,
    plot_returns_by_regime,
    plot_weight_evolution,
    plot_regime_heatmap,
)


def main() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Macro data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading macroeconomic data from FRED...")
    macro_features, macro_raw = load_fred_data(FRED_API_KEY, START_DATE, END_DATE)
    print(f"      {macro_features.index[0].date()} → {macro_features.index[-1].date()}  "
          f"({len(macro_features)} months)")

    # ── 2. Equity data ─────────────────────────────────────────────────────
    print("\n[2/6] Loading equity returns from Yahoo Finance...")
    equity_returns = load_equity_data(TICKERS, START_DATE, END_DATE)
    print(f"      {equity_returns.index[0].date()} → {equity_returns.index[-1].date()}  "
          f"({len(equity_returns)} months)")

    # ── 3. Regime detection ────────────────────────────────────────────────
    print("\n[3/6] Fitting Hidden Markov Model...")
    detector = RegimeDetector(n_regimes=N_REGIMES)
    detector.fit(macro_features)
    regime_series = detector.get_regime_series(macro_features)
    proba_df = pd.DataFrame(
        detector.predict_proba(macro_features),
        index=macro_features.index,
    )

    print("\n  Regime summary (macro feature means):")
    summary = detector.regime_summary(macro_features)
    print(summary.to_string())

    counts = regime_series.value_counts().sort_index()
    total = len(regime_series)
    print("\n  Regime frequencies:")
    for r, cnt in counts.items():
        print(f"    Regime {r} ({REGIME_LABELS[r]:9s}): {cnt:3d} months  ({cnt/total:.1%})")

    # ── 4a. Hard-label backtest (original) ────────────────────────────────
    print(f"\n[4/6] Running backtests  (start: {BACKTEST_START})...")
    results_hard = run_backtest(
        equity_returns,
        regime_series,
        proba_df=None,
        lookback_window=LOOKBACK_WINDOW,
        risk_aversion=RISK_AVERSION,
        backtest_start=BACKTEST_START,
    )

    # ── 4b. Probabilistic-blend backtest (improved) ────────────────────────
    results_prob = run_backtest(
        equity_returns,
        regime_series,
        proba_df=proba_df,
        lookback_window=LOOKBACK_WINDOW,
        risk_aversion=RISK_AVERSION,
        backtest_start=BACKTEST_START,
    )

    port = results_prob["portfolio"]
    print(f"      Period: {port.index[0].date()} → {port.index[-1].date()}  ({len(port)} months)")

    # ── 5. Metrics ─────────────────────────────────────────────────────────
    print("\n[5/6] Computing performance metrics...")

    from analysis.metrics import compute_metrics
    metrics_hard = compare_strategies(results_hard)
    metrics_prob = compare_strategies(results_prob)

    # Build a combined table: probabilistic portfolio + benchmarks
    results_combined = {
        "portfolio":    results_prob["portfolio"],
        "spy":          results_prob["spy"],
        "equal_weight": results_prob["equal_weight"],
    }
    metrics_combined = compare_strategies(results_combined)

    # Add hard-label row for direct comparison
    metrics_combined.loc["Hard-label Portfolio"] = compute_metrics(results_hard["portfolio"])
    display_order = [
        "Regime Portfolio",     # probabilistic blend (top line)
        "Hard-label Portfolio", # original hard version
        "S&P 500 (SPY)",
        "Equal Weight",
    ]
    metrics_combined = metrics_combined.reindex(
        [r for r in display_order if r in metrics_combined.index]
    )
    metrics_combined.rename(index={"Regime Portfolio": "Probabilistic Portfolio"}, inplace=True)

    print_performance_table(metrics_combined)

    # Export
    metrics_combined.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

    pd.DataFrame({
        "probabilistic_portfolio": results_prob["portfolio"],
        "hard_portfolio":          results_hard["portfolio"],
        "spy":                     results_prob["spy"],
        "equal_weight":            results_prob["equal_weight"],
        "regime":                  results_prob["regimes"],
    }).to_csv(os.path.join(OUTPUT_DIR, "monthly_returns.csv"))

    results_prob["weights"].to_csv(os.path.join(OUTPUT_DIR, "portfolio_weights_prob.csv"))
    results_hard["weights"].to_csv(os.path.join(OUTPUT_DIR, "portfolio_weights_hard.csv"))

    # ── 6. Visualisations ──────────────────────────────────────────────────
    print("[6/6] Generating charts...")

    plot_regime_timeline(regime_series, macro_raw)
    plot_cumulative_returns_dual(results_hard, results_prob)
    plot_drawdowns_dual(results_hard, results_prob)
    plot_returns_by_regime(equity_returns, regime_series)
    plot_weight_evolution(results_prob["weights"], results_prob["regimes"])
    plot_regime_heatmap(equity_returns, regime_series)

    print(f"\n✓ All outputs saved to '{OUTPUT_DIR}/'")

    return {
        "results_hard":  results_hard,
        "results_prob":  results_prob,
        "metrics":       metrics_combined,
        "detector":      detector,
        "regime_series": regime_series,
        "proba_df":      proba_df,
        "macro_raw":     macro_raw,
    }


if __name__ == "__main__":
    main()
