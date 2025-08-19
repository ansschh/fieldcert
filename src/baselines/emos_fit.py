# src/baselines/emos_fit.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


def _ensemble_stats(ensemble: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample ensemble mean and standard deviation.

    Parameters
    ----------
    ensemble : array (T,M,H,W) or (N,M) flattened

    Returns
    -------
    mean : array shape (T,H,W) or (N,)
    std  : array shape (T,H,W) or (N,)
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    if ens.ndim == 4:
        mu = np.nanmean(ens, axis=1)
        sd = np.nanstd(ens, axis=1, ddof=1)
    elif ens.ndim == 2:
        mu = np.nanmean(ens, axis=1)
        sd = np.nanstd(ens, axis=1, ddof=1)
    else:
        raise ValueError("ensemble must be 4D (T,M,H,W) or 2D (N,M)")
    # avoid zero std (can break logistic features with constant term)
    sd = np.maximum(sd, 1e-6)
    return mu, sd


@dataclass
class EMOSLogistic:
    """
    EMOS (event-probability) via logistic regression:

        P(event at threshold T) = sigmoid( a + b * ens_mean + c * ens_std )

    - Fits on flattened samples (time x grid).
    - Supports downsampling to limit memory.
    - Saves/loads with joblib (if available).

    This baseline is appropriate for Brier/reliability evaluation and
    for converting to sets by thresholding probability to meet an FPA target.
    """
    C: float = 1.0
    max_iter: int = 200
    class_weight: Optional[str] = "balanced"
    random_state: Optional[int] = 42

    def __post_init__(self):
        self._pipe: Optional[Pipeline] = None

    def fit(
        self,
        ens_cal: np.ndarray,          # (T,M,H,W)
        truth_cal: np.ndarray,        # (T,H,W)
        threshold: float,
        max_samples: Optional[int] = 2_000_000,
        sample_strategy: str = "stratified",  # 'uniform' or 'stratified'
    ) -> "EMOSLogistic":
        """
        Fit logistic EMOS on calibration data.

        - Extract features X = [mean, std].
        - Label y = 1{ truth >= T }.
        - Optionally subsample to max_samples.

        Returns self.
        """
        mu, sd = _ensemble_stats(ens_cal)  # (T,H,W)
        y = (np.asarray(truth_cal, dtype=np.float64) >= float(threshold)).astype(np.int8)

        if mu.shape != y.shape:
            raise ValueError(f"Shapes mismatch: features {mu.shape} vs labels {y.shape}")

        # Flatten
        X = np.stack([mu.ravel(), sd.ravel()], axis=1)  # (N,2)
        Y = y.ravel().astype(np.int8)

        # Optional subsampling
        if max_samples is not None and X.shape[0] > max_samples:
            rng = check_random_state(self.random_state)
            if sample_strategy == "stratified":
                idx_pos = np.where(Y == 1)[0]
                idx_neg = np.where(Y == 0)[0]
                n_pos = min(len(idx_pos), max_samples // 2)
                n_neg = max_samples - n_pos
                sel_pos = rng.choice(idx_pos, size=n_pos, replace=False) if len(idx_pos) > 0 else np.array([], dtype=int)
                sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
                idx = np.concatenate([sel_pos, sel_neg])
            else:
                idx = rng.choice(np.arange(X.shape[0]), size=max_samples, replace=False)
            X = X[idx]
            Y = Y[idx]

        # Build pipeline: standardize -> logistic
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("logit", LogisticRegression(
                    C=self.C,
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    solver="lbfgs",
                    n_jobs=None,
                    random_state=self.random_state,
                )),
            ]
        )
        pipe.fit(X, Y)
        self._pipe = pipe
        return self

    def predict_proba(
        self,
        ens: np.ndarray,  # (T,M,H,W)
    ) -> np.ndarray:
        """
        Predict event probabilities P(y >= T) on (T,H,W).

        Returns
        -------
        prob : (T,H,W) in [0,1]
        """
        if self._pipe is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        mu, sd = _ensemble_stats(ens)
        X = np.stack([mu.ravel(), sd.ravel()], axis=1)
        p = self._pipe.predict_proba(X)[:, 1]
        return p.reshape(mu.shape)

    def save(self, path: str) -> None:
        """
        Save model to disk (requires joblib).
        """
        if self._pipe is None:
            raise RuntimeError("Nothing to save; fit the model first.")
        if joblib is None:
            raise ImportError("joblib is required to save/load models.")
        joblib.dump({"pipe": self._pipe}, path)

    def load(self, path: str) -> "EMOSLogistic":
        """
        Load model from disk (requires joblib).
        """
        if joblib is None:
            raise ImportError("joblib is required to save/load models.")
        obj = joblib.load(path)
        self._pipe = obj["pipe"]
        return self
