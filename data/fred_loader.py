import pandas as pd
import numpy as np
from fredapi import Fred

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FRED_API_KEY, FRED_SERIES


def load_fred_data(
    api_key: str = FRED_API_KEY,
    start_date: str = "2000-01-01",
    end_date: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    features : DataFrame  — standardised macro feature set for HMM
                            columns: yield_curve, vix, unemployment,
                                     inflation_yoy, indpro_yoy
    raw      : DataFrame  — all downloaded series + derived columns
    """
    print("  Connecting to FRED API...")
    fred = Fred(api_key=api_key)

    raw_series = {}
    for code in FRED_SERIES:
        s = fred.get_series(code, observation_start=start_date, observation_end=end_date)
        # Resample to calendar month-end mean; handles daily series like VIX
        raw_series[code] = s.resample("ME").mean()
        print(f"    {code}: {len(raw_series[code])} monthly observations")

    raw = pd.DataFrame(raw_series)

    # Forward-fill tiny gaps (e.g. UNRATE is monthly already; VIX has no weekends issue at ME)
    raw = raw.ffill(limit=3)

    # Derived macro features
    raw["yield_curve"]    = raw["GS10"] - raw["GS2"]       # slope of the yield curve
    raw["vix"]            = raw["VIXCLS"]
    raw["unemployment"]   = raw["UNRATE"]
    raw["inflation_yoy"]  = raw["CPIAUCSL"].pct_change(12) * 100
    raw["indpro_yoy"]     = raw["INDPRO"].pct_change(12) * 100

    feature_cols = ["yield_curve", "vix", "unemployment", "inflation_yoy", "indpro_yoy"]
    features = raw[feature_cols].dropna()

    return features, raw
