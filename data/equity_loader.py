import pandas as pd
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TICKERS


def load_equity_data(
    tickers: list = TICKERS,
    start_date: str = "2000-01-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for each ticker and compute
    monthly total returns (last price of each calendar month).

    Returns
    -------
    DataFrame with tickers as columns and monthly return index
    """
    print(f"  Downloading {', '.join(tickers)} from Yahoo Finance...")
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # Multi-ticker download nests under 'Close'; single ticker is a Series
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Last trading day of each calendar month → approximate month-end NAV
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change()
    returns = returns.dropna(how="all")

    print(f"    Equity returns: {returns.index[0].date()} to {returns.index[-1].date()}")
    return returns[tickers]
