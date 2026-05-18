# Macroeconomic Regime Allocation

Dynamic portfolio allocation using Hidden Markov Models to detect macroeconomic regimes and generate adaptive equity weights.

## Overview

This system classifies the economy into three regimes — **Stress**, **Expansion**, and **Baseline** — using a Gaussian HMM trained on FRED macroeconomic indicators. Portfolio weights across six US equity ETFs are then optimised for each regime via mean-variance optimisation, with two allocation modes: hard regime labels and probabilistic blending.

The pipeline supports full historical backtesting, live signal generation, and an interactive dashboard.

## Project Structure

```
macro_regime_allocation/
├── main.py                  # Full backtest pipeline entry point
├── live_signal.py           # Live regime signal & portfolio weights
├── dashboard.py             # Interactive visualisation dashboard
├── config.py                # All parameters (tickers, FRED series, HMM settings)
├── data/
│   ├── fred_loader.py       # FRED API data fetching
│   └── equity_loader.py     # yfinance ETF return fetching
├── models/
│   ├── hmm_regime.py        # GaussianHMM regime detector
│   └── mean_variance.py     # Mean-variance portfolio optimiser
├── backtest/
│   └── engine.py            # Walk-forward backtesting engine
├── analysis/
│   └── metrics.py           # Performance metrics (Sharpe, drawdown, etc.)
├── visualization/
│   └── plots.py             # Regime timelines, returns, drawdowns, weight charts
└── regime_allocation_results/
    ├── latest_signal.json   # Most recent live signal output
    └── signal_history.csv   # Historical signal log
```

## Methodology

### Regime Detection

A 3-state `GaussianHMM` (from `hmmlearn`) is fitted on nine FRED macroeconomic features after `StandardScaler` normalisation. States are relabelled consistently by descending VIX mean:

| Label | Description | Colour |
|-------|-------------|--------|
| 0 | Stress | Red |
| 1 | Expansion | Green |
| 2 | Baseline | Blue |

**FRED features used:** 10Y Treasury yield (GS10), 2Y Treasury yield (GS2), VIX (VIXCLS), unemployment rate (UNRATE), CPI (CPIAUCSL), industrial production (INDPRO), Baa corporate spread (BAA), Fed Funds rate (FEDFUNDS), UMich consumer sentiment (UMCSENT).

### Portfolio Optimisation

Mean-variance optimisation is run separately for each regime using a 36-month lookback window. Two strategies are compared:

- **Hard-label**: allocate 100% according to the MAP regime estimate.
- **Probabilistic blend**: weight regime-optimal portfolios by posterior probabilities, smoothing across regime boundaries.

**Universe:** SPY, IWM, XLU, XLY, XLK, XLF  
**Risk-aversion:** 1.0 | **Risk-free rate:** 2% | **Regularisation:** ridge (λ = 0.0001)

### Backtesting

Walk-forward backtesting begins January 2003 (after 36 months of burn-in). The engine evaluates both strategies against two benchmarks: S&P 500 (SPY buy-and-hold) and equal-weight portfolio.

## Installation

```bash
git clone https://github.com/maanitmehta/Macroeconomic_regime_allocation.git
cd Macroeconomic_regime_allocation
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `yfinance`, `fredapi`, `hmmlearn`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

## Configuration

Edit `config.py` before running:

```python
FRED_API_KEY = "your_fred_api_key"   # Get free key at fred.stlouisfed.org
```

All other parameters (tickers, FRED series, HMM iterations, lookback window, risk aversion) are set in `config.py`.

## Usage

### Full Backtest

Runs the complete pipeline: data loading → regime detection → backtesting → performance analysis → chart generation.

```bash
python main.py
```

Outputs are saved to `regime_allocation_results/`.

### Live Signal

Fetches the latest macro data, fits the HMM, and prints today's recommended portfolio weights. Also saves to `regime_allocation_results/latest_signal.json` and appends to `signal_history.csv`. Suitable for cron-job scheduling.

```bash
python live_signal.py
```

Example output:

```
Current Regime: Expansion (prob: 0.84)
Stress: 0.05 | Expansion: 0.84 | Baseline: 0.11

Probabilistic Weights:  SPY=0.31  IWM=0.18  XLU=0.07  XLY=0.22  XLK=0.17  XLF=0.05
Hard-Label Weights:     SPY=0.35  IWM=0.20  XLU=0.05  XLY=0.25  XLK=0.12  XLF=0.03
```

### Dashboard

```bash
python dashboard.py
```

## Results

The backtest compares cumulative returns, Sharpe ratios, and maximum drawdowns for both regime-allocation strategies against SPY and equal-weight benchmarks. Charts are saved as PNGs in `regime_allocation_results/`.
