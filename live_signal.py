"""
Live Regime Signal
==================
Fetches the latest macro and equity data, detects the current regime,
and outputs today's recommended portfolio weights.

Run standalone:
    python live_signal.py

Or scheduled monthly — the last section prints a machine-readable JSON
block that a cron / scheduler can capture and forward.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import json
import os
from datetime import date

import pandas as pd
import numpy as np

from config import (
    FRED_API_KEY, START_DATE, N_REGIMES, LOOKBACK_WINDOW,
    RISK_AVERSION, TICKERS, TICKER_NAMES, REGIME_LABELS, OUTPUT_DIR,
)
from data.fred_loader import load_fred_data
from data.equity_loader import load_equity_data
from models.hmm_regime import RegimeDetector
from models.mean_variance import regime_optimal_weights, blended_regime_weights


def generate_signal() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("Fetching latest data...")
    macro_features, _ = load_fred_data(FRED_API_KEY, START_DATE)
    equity_returns = load_equity_data(TICKERS, START_DATE)

    # ── Fit HMM on full history ────────────────────────────────────────────
    print("Fitting HMM...")
    detector = RegimeDetector(n_regimes=N_REGIMES)
    detector.fit(macro_features)

    regime_series = detector.get_regime_series(macro_features)
    proba_df = pd.DataFrame(
        detector.predict_proba(macro_features),
        index=macro_features.index,
    )

    # ── Current state (latest macro observation) ───────────────────────────
    last_macro_date = macro_features.index[-1]
    current_proba   = proba_df.iloc[-1].values          # (n_regimes,)
    # Use MAP from posteriors (argmax) for display — consistent with the probability bars.
    # Viterbi (regime_series) can disagree because it optimises the whole sequence jointly;
    # argmax of local posteriors reflects what the evidence says right now.
    current_regime  = int(np.argmax(current_proba))

    # ── Regime-conditional weights using last LOOKBACK_WINDOW months ───────
    common = equity_returns.index.intersection(regime_series.index)
    ret_window = equity_returns.loc[common].iloc[-LOOKBACK_WINDOW:].values
    reg_window = regime_series.loc[common].iloc[-LOOKBACK_WINDOW:].values

    valid = ~np.any(np.isnan(ret_window), axis=1)
    ret_window = ret_window[valid]
    reg_window = reg_window[valid]

    w_prob = blended_regime_weights(ret_window, reg_window, current_proba, RISK_AVERSION)
    w_hard = regime_optimal_weights(ret_window, reg_window, current_regime, RISK_AVERSION)

    # ── Package signal ─────────────────────────────────────────────────────
    signal = {
        "as_of":          str(last_macro_date.date()),
        "run_date":       str(date.today()),
        "current_regime": {
            "id":    current_regime,
            "label": REGIME_LABELS[current_regime],
        },
        "regime_probabilities": {
            REGIME_LABELS[r]: round(float(current_proba[r]), 4)
            for r in range(N_REGIMES)
        },
        "weights_probabilistic": {t: round(float(w), 4) for t, w in zip(TICKERS, w_prob)},
        "weights_hard_label":    {t: round(float(w), 4) for t, w in zip(TICKERS, w_hard)},
    }

    # ── Print readable summary ─────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  LIVE REGIME SIGNAL  —  as of {signal['as_of']}")
    print(sep)
    print(f"  Current regime : {current_regime} — {REGIME_LABELS[current_regime]}")
    print()
    print("  Regime posterior probabilities:")
    for label, prob in signal["regime_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {label:12s}  {prob:5.1%}  {bar}")
    print()
    print("  Recommended weights (probabilistic blend):")
    for ticker, w in signal["weights_probabilistic"].items():
        name = TICKER_NAMES.get(ticker, ticker)
        bar = "█" * int(w * 40)
        print(f"    {ticker} ({name:20s})  {w:5.1%}  {bar}")
    print()
    print("  Weights (hard-label, for reference):")
    for ticker, w in signal["weights_hard_label"].items():
        print(f"    {ticker}  {w:5.1%}")
    print(sep)

    # ── Persist ───────────────────────────────────────────────────────────
    signal_path = os.path.join(OUTPUT_DIR, "latest_signal.json")
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)
    print(f"\n  Signal saved → {signal_path}")

    # Append to signal history CSV
    history_path = os.path.join(OUTPUT_DIR, "signal_history.csv")
    row = {"run_date": signal["run_date"], "as_of": signal["as_of"],
           "regime_id": current_regime, "regime_label": REGIME_LABELS[current_regime]}
    row.update({f"p_{REGIME_LABELS[r]}": round(float(current_proba[r]), 4) for r in range(N_REGIMES)})
    row.update({f"w_prob_{t}": round(float(w), 4) for t, w in zip(TICKERS, w_prob)})
    row.update({f"w_hard_{t}": round(float(w), 4) for t, w in zip(TICKERS, w_hard)})

    hist_df = pd.DataFrame([row])
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        # Drop any row with the same run_date before appending
        existing = existing[existing["run_date"] != signal["run_date"]]
        hist_df = pd.concat([existing, hist_df], ignore_index=True)
    hist_df.to_csv(history_path, index=False)
    print(f"  History  saved → {history_path}\n")

    # Machine-readable JSON block (easy to capture in shell scripts)
    print("JSON_SIGNAL_START")
    print(json.dumps(signal, indent=2))
    print("JSON_SIGNAL_END")

    return signal


if __name__ == "__main__":
    generate_signal()
