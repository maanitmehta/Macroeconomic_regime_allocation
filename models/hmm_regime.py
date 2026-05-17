import numpy as np
import pandas as pd
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import N_REGIMES, HMM_RANDOM_STATE, HMM_N_ITER


class RegimeDetector:
    """
    Gaussian HMM trained on macro features.

    Regimes are relabelled so that:
      0 = Stress    (highest average VIX)
      1 = Expansion (middle VIX)
      2 = Baseline  (lowest VIX)
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        n_iter: int = HMM_N_ITER,
        random_state: int = HMM_RANDOM_STATE,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self.scaler = StandardScaler()
        self.regime_map: dict[int, int] = {}
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        self._feature_names = list(features.columns)
        X = self.scaler.fit_transform(features.values)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self._build_regime_map()
        return self

    def _build_regime_map(self) -> None:
        means = self.model.means_   # shape: (n_regimes, n_features)  — standardised space
        vix_idx = (
            self._feature_names.index("vix")
            if "vix" in self._feature_names
            else 1
        )
        # Sort HMM states by their VIX mean, descending: highest VIX → Stress
        order = np.argsort(means[:, vix_idx])[::-1]
        self.regime_map = {int(hmm_state): label for label, hmm_state in enumerate(order)}

    # ------------------------------------------------------------------
    def _raw_predict(self, features: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(features.values)
        return self.model.predict(X)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        raw = self._raw_predict(features)
        return np.array([self.regime_map[s] for s in raw], dtype=int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Returns posterior probabilities re-ordered to match the relabelled regimes.
        Shape: (T, n_regimes)
        """
        X = self.scaler.transform(features.values)
        proba_raw = self.model.predict_proba(X)   # (T, n_regimes) in HMM-state order
        # Re-order columns to match our regime labels
        new_order = [
            hmm_state
            for _, hmm_state in sorted(self.regime_map.items(), key=lambda kv: kv[1])
        ]
        return proba_raw[:, new_order]

    def get_regime_series(self, features: pd.DataFrame) -> pd.Series:
        labels = self.predict(features)
        return pd.Series(labels, index=features.index, name="regime")

    # ------------------------------------------------------------------
    def regime_summary(self, features: pd.DataFrame) -> pd.DataFrame:
        """Descriptive statistics of each regime in original feature space."""
        from config import REGIME_LABELS
        regime_series = self.get_regime_series(features)
        rows = []
        for r in range(self.n_regimes):
            mask = regime_series == r
            count = mask.sum()
            freq = count / len(regime_series)
            means_orig = features[mask].mean()
            row = {"Regime": f"{r} ({REGIME_LABELS[r]})", "Months": count, "Frequency": f"{freq:.1%}"}
            for col in features.columns:
                row[col] = round(means_orig[col], 2)
            rows.append(row)
        return pd.DataFrame(rows).set_index("Regime")
