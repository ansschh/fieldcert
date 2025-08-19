# src/calibration/regimes.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Sequence

try:
    # Optional dependency; only used if you want KNN selection
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None  # type: ignore


def assign_season(times: np.ndarray) -> np.ndarray:
    """
    Map array of datetime64[ns]/datetime64[D]/np.datetime64 to season labels {"DJF","MAM","JJA","SON"}.
    """
    t = np.asarray(times)
    if not np.issubdtype(t.dtype, np.datetime64):
        raise ValueError("times must be a numpy datetime64 array")

    months = np.array([int(str(x)[5:7]) for x in t])  # works for datetime64
    season = np.empty_like(months, dtype=object)
    # DJF = 12,1,2; MAM = 3,4,5; JJA = 6,7,8; SON = 9,10,11
    season[(months == 12) | (months <= 2)] = "DJF"
    season[(months >= 3) & (months <= 5)] = "MAM"
    season[(months >= 6) & (months <= 8)] = "JJA"
    season[(months >= 9) & (months <= 11)] = "SON"
    return season


def assign_blocks(times: np.ndarray, block: str = "week") -> np.ndarray:
    """
    Assign block IDs for exchangeable calibration units (e.g., weekly).

    Returns
    -------
    block_ids : int array of same length as times, starting at 0.
    """
    t = np.asarray(times)
    if not np.issubdtype(t.dtype, np.datetime64):
        raise ValueError("times must be datetime64")
    if block not in ("week", "5day", "month"):
        raise ValueError("block must be one of {'week','5day','month'}")

    # Convert to days since epoch
    days = (t - t.astype("datetime64[D]")).astype("timedelta64[D]")  # zero offset
    day_counts = t.astype("datetime64[D]").astype(int)

    if block == "week":
        # ISO week number using datetime64 is messy; approximate by floor(day_count/7)
        ids = (day_counts - day_counts.min()) // 7
    elif block == "5day":
        ids = (day_counts - day_counts.min()) // 5
    else:  # month
        months = np.array([int(str(x)[:7].replace("-", "")) for x in t])  # yyyymm
        unique, inv = np.unique(months, return_inverse=True)
        ids = inv
    # normalize to start from 0
    ids = ids - ids.min()
    return ids.astype(int)


def make_regime_keys(
    seasons: Sequence[str],
    leads_hours: Sequence[int],
    regions: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Build regime keys as tuples (season, lead_hours, region) -> dtype=object array of keys.
    """
    s = np.asarray(seasons)
    l = np.asarray(leads_hours)
    if s.shape != l.shape:
        raise ValueError("seasons and leads_hours must have the same shape")
    if regions is None:
        regions = ["global"] * s.shape[0]
    r = np.asarray(regions)
    if r.shape != s.shape:
        raise ValueError("regions must match seasons shape")
    keys = np.array(list(zip(s.tolist(), l.tolist(), r.tolist())), dtype=object)
    return keys


def group_indices_by_regime(keys: Sequence[object]) -> Dict[object, np.ndarray]:
    """
    Group integer indices by regime key. Returns dict: key -> np.array of indices (sorted).
    """
    keys = np.asarray(keys, dtype=object)
    groups: Dict[object, List[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in groups.items()}


@dataclass
class KNNSelector:
    """
    K-Nearest Neighbors helper for selecting calibration blocks similar to a query.

    Attributes
    ----------
    n_neighbors : number of neighbors
    metric : distance metric (passed to sklearn NearestNeighbors)
    """
    n_neighbors: int = 50
    metric: str = "euclidean"

    def __post_init__(self):
        if NearestNeighbors is None:
            raise ImportError(
                "scikit-learn is required for KNNSelector. Install with `conda install -c conda-forge scikit-learn`."
            )
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)

    def fit(self, X_blocks: np.ndarray) -> "KNNSelector":
        """
        Fit on calibration block-level feature matrix X_blocks with shape (n_blocks, n_features).
        """
        X = np.asarray(X_blocks, dtype=np.float64)
        self._nn.fit(X)
        return self

    def query(self, x_query: np.ndarray) -> np.ndarray:
        """
        Return indices of K nearest calibration blocks to x_query (shape (n_features,)).
        """
        x = np.asarray(x_query, dtype=np.float64).reshape(1, -1)
        dists, idx = self._nn.kneighbors(x, return_distance=True)
        return idx[0]

    def weights_for_query(self, X_blocks: np.ndarray, x_query: np.ndarray, kernel: str = "uniform") -> np.ndarray:
        """
        Return a weight vector (length n_blocks) with non-zero mass on the KNN set.

        kernel:
          - "uniform": 1/K on neighbors, 0 elsewhere
          - "distance": inverse-distance normalized on neighbors (adds small epsilon)
        """
        X = np.asarray(X_blocks, dtype=np.float64)
        idx = self.query(x_query)
        w = np.zeros(X.shape[0], dtype=np.float64)
        if len(idx) == 0:
            return w
        if kernel == "uniform":
            w[idx] = 1.0 / len(idx)
        elif kernel == "distance":
            x = x_query.reshape(1, -1)
            # compute distances only to neighbors
            d = np.linalg.norm(X[idx] - x, axis=1)
            d = np.maximum(d, 1e-8)
            inv = 1.0 / d
            w[idx] = inv / inv.sum()
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        return w
