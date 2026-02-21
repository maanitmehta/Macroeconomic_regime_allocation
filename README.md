# Macroeconomic Regime Allocation

A quantitative portfolio strategy that detects economic regimes using Hidden Markov Models and dynamically adjusts asset weights based on current macro state.

**Historical data:** 2001–2026 (300 months)  
**Assets tracked:** SPY, IWM, XLU, XLY, XLK, XLF  
**Rebalancing:** Monthly

---

## What this project does

- Detects 3 latent macroeconomic regimes from macro indicators
- Learns regime-specific asset behavior (returns, volatility, correlations)
- Allocates portfolio weights based on current detected regime
- Backtests strategy out-of-sample with rolling windows
- Compares performance vs S&P 500 buy-and-hold

---

## Modelling approach

### 1. Regime detection
**Hidden Markov Model** (3 states) trained on:
- Yield curve slope (10Y-3M Treasury spread)
- VIX (market volatility)
- Unemployment rate
- CPI inflation (YoY)
- Industrial production growth (YoY)

**Output:** Time series of regime classifications (0, 1, 2)

### 2. Feature engineering
For each regime, calculate:
- Mean returns (regime-conditional)
- Covariance matrix (regime-conditional)
- Sharpe ratios
- Asset correlations

### 3. Portfolio optimization
**Mean-variance optimization** with regime-specific parameters:
```
max μ'w - (λ/2)·w'Σw
subject to: Σw = 1, w ≥ 0
```

Where μ and Σ come from the detected regime.

### 4. Backtesting
- **Lookback window:** 36 months
- **Rebalancing:** Monthly
- **Method:** Rolling window, no look-ahead bias
- **Period:** 2003–2026 (264 out-of-sample periods)

---

## Model performance

**Backtest period:** Jan 2003 – Jan 2026

| Metric | Regime-Aware | S&P 500 | Difference |
|--------|--------------|---------|------------|
| **CAGR** | 9.13% | 10.57% | -1.44pp |
| **Sharpe** | 0.55 | 0.59 | -0.04 |
| **Max DD** | -50.14% | -50.78% | +0.64pp |
| **Volatility** | 14.2% | 16.8% | -2.6pp |

**Regime distribution:**
- Regime 0 (Stress): 17% of time
- Regime 1 (Expansion): 20% of time  
- Regime 2 (Baseline): 63% of time

---

## Regime characteristics

**Regime 0 (Stress) – 50 months**
- Financials: -2.3% annualized
- Utilities: -1.7% annualized
- VIX: Elevated (>25)
- Historical: 2008 crisis, COVID crash

**Regime 1 (Expansion) – 60 months**
- Technology: 19.6% annualized
- Consumer Discretionary: 21.8% annualized
- VIX: Low (<15)
- Historical: 2009-2011 recovery, 2017-2018 bull

**Regime 2 (Baseline) – 190 months**
- All assets positive (7-11% range)
- VIX: Moderate (15-20)
- Historical: 2003-2007, 2012-2019, 2021-2024

---

## Key findings

**Regime detection works** – HMM identified economically meaningful states  
**Volatility reduction** – 2.6pp lower annualized vol vs benchmark  
**Crisis detection** – 2008 and COVID correctly classified as stress regimes  
**Return tradeoff** – Strategy underperformed by 1.44pp, prioritized risk management  

**Main insight:** Hard regime classification may be suboptimal. Regime probabilities (soft assignment) could improve allocation decisions.

---

## Data sources

**Macroeconomic data:** FRED API (Federal Reserve Economic Data)
- Yield curve, VIX, unemployment, CPI, industrial production
- Monthly frequency, 2000–2026

**Asset data:** Yahoo Finance
- Adjusted close prices for 6 ETFs
- Monthly returns calculated

---

## Tech stack

```
Python 3.8+
pandas, numpy          # Data manipulation
yfinance, fredapi      # Data sources  
hmmlearn               # Hidden Markov Models
scipy                  # Optimization
matplotlib, seaborn    # Visualization
```

---

## Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Get FRED API key:**
- Sign up: https://fred.stlouisfed.org/
- Get free API key
- Create `.env` file:
```bash
FRED_API_KEY=your_key_here
```

**3. Run:**
```bash
python regime_allocation_complete.py
```

**Runtime:** ~3-5 minutes  
**Output:** Console metrics + 4 plots saved to `regime_allocation_results/`

---

## Project structure

```
Macroeconomic_regime_allocation/
├── regime_allocation_complete.py    # Full pipeline
├── results/                          # Output plots
│   ├── cumulative_returns.png
│   ├── regime_timeline.png
│   ├── drawdown_comparison.png
│   └── returns_by_regime.png
└── README.md
```

---

## Limitations

- **Monthly frequency** – May miss intra-month regime shifts
- **Small crisis sample** – Only 2-3 major recessions in 25 years
- **Hard classification** – Binary regime assignment vs probabilistic weighting
- **Transaction costs** – Not explicitly modeled
- **US-centric** – Limited to US macro and equities

---

## Future work

- [ ] Use regime probabilities for weighted allocation
- [ ] Add multi-asset classes (bonds, commodities)
- [ ] Test alternative regime methods (K-Means, thresholds)
- [ ] Model transaction costs explicitly
- [ ] Extend to international markets

---

## References

**HMM foundations:**
- Hamilton (1989) – "A New Approach to the Economic Analysis of Nonstationary Time Series"

**Portfolio applications:**
- Ang & Bekaert (2002) – "Regime Switches in Interest Rates"
- Guidolin & Timmermann (2007) – "Asset Allocation under Multivariate Regime Switching"

---

## About

**Author:** Maanit Mehta  
**Education:** MSc Financial Modelling & Investment, University of Glasgow  
**Contact:** maanitkeyhem@gmail.com | [LinkedIn](https://linkedin.com/in/maanit-mehta/)

Built to demonstrate:
- Econometric modeling (HMM)
- Portfolio optimization
- Backtesting discipline
- Python data pipelines

---



**Note:** Educational project. Not financial advice. 
