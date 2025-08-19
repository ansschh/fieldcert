# src/calibration/morphology.py
from __future__ import annotations

import numpy as np
from typing import Literal, Tuple, Optional

try:
    from skimage.morphology import disk, square, binary_opening, binary_closing, binary_erosion, binary_dilation
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-image is required for morphology. Install with `conda install -c conda-forge scikit-image`."
    ) from e


def _structuring_element(kind: Literal["disk", "square"], radius: int) -> np.ndarray:
    if radius <= 0:
        # radius=0 means no-op; return 1x1 element
        return np.ones((1, 1), dtype=bool)
    if kind == "disk":
        return disk(radius)
    if kind == "square":
        return square(2 * radius + 1)
    raise ValueError(f"Unknown structuring element kind: {kind}")


def morphological_filter(
    mask: np.ndarray,
    operation: Literal["none", "open", "close", "erode", "dilate"] = "none",
    radius: int = 1,
    element: Literal["disk", "square"] = "disk",
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply a binary morphological operation to a mask.

    Parameters
    ----------
    mask : boolean array
    operation : one of {"none","open","close","erode","dilate"}
    radius : structuring element radius (pixels)
    element : {"disk","square"}
    iterations : repeat operation this many times

    Returns
    -------
    filtered mask (boolean)
    """
    m = np.asarray(mask).astype(bool)
    if operation == "none" or radius <= 0 or iterations <= 0:
        return m

    selem = _structuring_element(element, radius)
    out = m.copy()
    for _ in range(iterations):
        if operation == "open":
            out = binary_opening(out, footprint=selem)
        elif operation == "close":
            out = binary_closing(out, footprint=selem)
        elif operation == "erode":
            out = binary_erosion(out, footprint=selem)
        elif operation == "dilate":
            out = binary_dilation(out, footprint=selem)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    return out


def symmetric_filter(
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    operation: Literal["none", "open", "close", "erode", "dilate"] = "none",
    radius: int = 1,
    element: Literal["disk", "square"] = "disk",
    iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the *same* morphological filter to prediction and truth masks (symmetry preserves CRC validity).

    Returns
    -------
    pred_filt, truth_filt
    """
    pm = morphological_filter(pred_mask, operation, radius, element, iterations)
    gm = morphological_filter(truth_mask, operation, radius, element, iterations)
    return pm, gm


def threshold_bump_mask(
    pred_field: np.ndarray,
    threshold: float,
    margin_field: Optional[np.ndarray],
    lam: float,
) -> np.ndarray:
    """
    Build a set mask via threshold-bump transform:
        S(λ) = { x : pred_field(x) >= threshold + λ * margin_field(x) }

    If margin_field is None, uses a constant 1 field (global bump).
    """
    yhat = np.asarray(pred_field, dtype=np.float64)
    if margin_field is None:
        bumped = threshold + lam
    else:
        m = np.asarray(margin_field, dtype=np.float64)
        if m.shape != yhat.shape:
            try:
                m = np.broadcast_to(m, yhat.shape)
            except Exception as e:
                raise ValueError(f"margin_field shape {m.shape} not broadcastable to {yhat.shape}") from e
        bumped = threshold + lam * m
    return (yhat >= bumped)
