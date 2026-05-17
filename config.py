import os

FRED_API_KEY = os.environ.get("FRED_API_KEY", "e0b05c797576a3db6d0a186713a70f4d")

START_DATE = "2000-01-01"
END_DATE = None  # None = through today

N_REGIMES = 3
LOOKBACK_WINDOW = 36        # months of history used for optimization
RISK_AVERSION = 1.0         # lambda in max mu'w - (lambda/2)*w'Sigma*w
RISK_FREE_RATE = 0.02       # annual

TICKERS = ["SPY", "IWM", "XLU", "XLY", "XLK", "XLF"]
TICKER_NAMES = {
    "SPY": "S&P 500",
    "IWM": "Russell 2000",
    "XLU": "Utilities",
    "XLY": "Cons. Discretionary",
    "XLK": "Technology",
    "XLF": "Financials",
}

FRED_SERIES = ["GS10", "GS2", "VIXCLS", "UNRATE", "CPIAUCSL", "INDPRO"]

REGIME_LABELS = {0: "Stress", 1: "Expansion", 2: "Baseline"}
REGIME_COLORS = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4"}

OUTPUT_DIR = "regime_allocation_results"
SAVE_FIGURES = True

BACKTEST_START = "2003-01-01"   # START_DATE + LOOKBACK_WINDOW months
HMM_RANDOM_STATE = 42
HMM_N_ITER = 1000
MIN_REGIME_OBS = 6              # min observations in regime before falling back to all-data
COVARIANCE_REG = 1e-4           # ridge regularisation added to covariance diagonal
