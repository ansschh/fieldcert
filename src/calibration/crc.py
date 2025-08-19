# src/calibration/crc.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Callable, Any, Literal

from .morphology import threshold_bump_mask, symmetric_filter
from .scores import fpa_loss, fna_loss, combined_set_loss


LossFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
Transform = Literal["threshold_bump"]  # placeholder for future transform families


@dataclass(frozen=True)
class CRCSettings:
    """
    Settings for Conformal Risk Control (CRC).

    Attributes
    ----------
    alpha : target risk level (e.g., 0.10 for 10% mean FPA)
    lambda_grid : array of non-negative λ values to search (must be sorted ascending)
    slack_B : finite-sample correction term B in [0,1]; conservative choice B=1 for bounded losses
    loss_type : "fpa" | "fna" | "combined"
    alpha_fp, beta_fn : weights for combined loss (ignored unless loss_type="combined")
    morph_operation : symmetric morphology op applied to both pred/true masks
    morph_radius : structuring element radius
    morph_element : "disk" | "square"
    morph_iterations : repetition count for morphology
    """
    alpha: float = 0.10
    lambda_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 2.0, 41))  # 0..2 in steps of 0.05
    slack_B: float = 1.0
    loss_type: Literal["fpa", "fna", "combined"] = "fpa"
    alpha_fp: float = 1.0
    beta_fn: float = 0.0
    morph_operation: Literal["none", "open", "close", "erode", "dilate"] = "none"
    morph_radius: int = 0
    morph_element: Literal["disk", "square"] = "disk"
    morph_iterations: int = 1


@dataclass
class CRCResult:
    """
    Holds calibration outcomes for a single regime.

    Attributes
    ----------
    lambda_star : selected λ for the regime
    risk_curve : array of empirical (corrected) risks per λ
    lambda_grid : the λ grid used
    block_losses : per-λ per-block losses (shape (n_lambda, n_blocks)), *uncorrected* empirical losses
    block_ids : array of block ids aligned with columns of block_losses
    """
    lambda_star: float
    risk_curve: np.ndarray
    lambda_grid: np.ndarray
    block_losses: np.ndarray
    block_ids: np.ndarray


class CRCCalibrator:
    """
    Conformal Risk Control calibrator for set-valued forecasts using threshold-bump transforms.

    This class calibrates λ per regime so that the *expected* risk (e.g., FPA) is ≤ α,
    using block-wise empirical risk and a finite-sample correction.

    Usage
    -----
    crc = CRCCalibrator(settings)
    result = crc.fit_for_regime(
        preds=pred_fields, truths=truth_fields, threshold=T, margins=m_fields,
        block_ids=block_ids, weights=weights, loss_weights=loss_weights
    )
    """

    def __init__(self, settings: Optional[CRCSettings] = None):
        self.settings = settings or CRCSettings()

        # Bind loss function
        if self.settings.loss_type == "fpa":
            self._loss_fn: LossFn = fpa_loss
        elif self.settings.loss_type == "fna":
            self._loss_fn = fna_loss
        else:
            def loss_fn_combined(p: np.ndarray, g: np.ndarray, w: Optional[np.ndarray]) -> float:
                return combined_set_loss(p, g, w, alpha_fp=self.settings.alpha_fp, beta_fn=self.settings.beta_fn)
            self._loss_fn = loss_fn_combined

    @staticmethod
    def _validate_3d(arr: np.ndarray, name: str) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim != 3:
            raise ValueError(f"{name} must be 3D (T,H,W), got shape {a.shape}")
        return a

    def _build_masks_for_lambda(
        self,
        yhat: np.ndarray,  # (T,H,W)
        y: np.ndarray,     # (T,H,W)
        T: float,
        m: Optional[np.ndarray],  # (T,H,W) or None
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binary masks for prediction and truth at a given λ. Apply symmetric morphology.
        """
        T3 = float(T)
        pred_masks = threshold_bump_mask(yhat, T3, m, lam)
        truth_masks = (y >= T3)

        if self.settings.morph_operation != "none" and self.settings.morph_radius > 0 and self.settings.morph_iterations > 0:
            # Apply slice-wise to reduce memory peaks
            pm_out = np.empty_like(pred_masks, dtype=bool)
            gm_out = np.empty_like(truth_masks, dtype=bool)
            for t in range(pred_masks.shape[0]):
                pm, gm = symmetric_filter(
                    pred_masks[t],
                    truth_masks[t],
                    operation=self.settings.morph_operation,
                    radius=self.settings.morph_radius,
                    element=self.settings.morph_element,
                    iterations=self.settings.morph_iterations,
                )
                pm_out[t] = pm
                gm_out[t] = gm
            pred_masks, truth_masks = pm_out, gm_out

        return pred_masks, truth_masks

    def _compute_block_losses(
        self,
        pred_masks: np.ndarray,  # (T,H,W)
        truth_masks: np.ndarray,  # (T,H,W)
        block_ids: np.ndarray,   # (T,)
        spatial_weights: Optional[np.ndarray] = None,  # (H,W)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-block losses for each time slice and average by block id.
        Returns:
            block_ids_unique, block_mean_losses
        """
        T = pred_masks.shape[0]
        if truth_masks.shape != pred_masks.shape:
            raise ValueError("pred_masks and truth_masks shape mismatch")
        if block_ids.shape[0] != T:
            raise ValueError("block_ids must have length T")

        w = spatial_weights if spatial_weights is not None else None

        # Compute per-time loss then aggregate
        per_time_losses = np.empty(T, dtype=np.float64)
        for t in range(T):
            per_time_losses[t] = self._loss_fn(pred_masks[t], truth_masks[t], w)

        # Aggregate by block id
        bid_unique, inv = np.unique(block_ids, return_inverse=True)
        block_means = np.zeros(bid_unique.shape[0], dtype=np.float64)
        counts = np.zeros_like(block_means)
        np.add.at(block_means, inv, per_time_losses)
        np.add.at(counts, inv, 1.0)
        counts = np.maximum(counts, 1.0)
        block_means /= counts
        return bid_unique, block_means

    def _empirical_risk(
        self,
        block_losses: np.ndarray,  # shape (n_blocks,)
        weights_per_block: Optional[np.ndarray] = None,
    ) -> float:
        """
        Weighted/unweighted empirical mean of block losses.
        """
        if weights_per_block is None:
            return float(np.mean(block_losses))
        w = np.asarray(weights_per_block, dtype=np.float64)
        if w.shape != block_losses.shape:
            raise ValueError(f"weights_per_block shape {w.shape} != block_losses {block_losses.shape}")
        s = np.sum(w)
        if s <= 0:
            return float(np.mean(block_losses))
        return float(np.sum(w * block_losses) / s)

    def fit_for_regime(
        self,
        preds: np.ndarray,                # (T,H,W) forecast fields
        truths: np.ndarray,               # (T,H,W) ERA5 fields
        threshold: float,
        margins: Optional[np.ndarray],    # (T,H,W) or None
        block_ids: np.ndarray,            # (T,) block assignment
        spatial_weights: Optional[np.ndarray] = None,  # (H,W) weights in [0,∞)
        weights_per_block: Optional[np.ndarray] = None,  # (n_blocks,) optional CRC weighting
    ) -> CRCResult:
        """
        Calibrate λ for a single regime using CRC.

        Returns a CRCResult with:
          - lambda_star (selected λ),
          - risk_curve (corrected empirical risk per λ),
          - block_losses per λ (uncorrected per-block losses),
          - block_ids (ordering for block_losses columns).
        """
        yhat = self._validate_3d(preds, "preds")
        y = self._validate_3d(truths, "truths")
        if margins is not None:
            m = self._validate_3d(margins, "margins")
        else:
            m = None

        if yhat.shape != y.shape:
            raise ValueError(f"preds and truths must match; got {yhat.shape} vs {y.shape}")

        lam_grid = np.asarray(self.settings.lambda_grid, dtype=np.float64)
        if lam_grid.ndim != 1 or lam_grid.size == 0 or np.any(lam_grid < 0):
            raise ValueError("lambda_grid must be a non-empty 1D array of nonnegative values")
        if not np.all(np.diff(lam_grid) >= 0):
            raise ValueError("lambda_grid must be sorted ascending")

        # Compute block losses for each λ
        block_ids_unique: Optional[np.ndarray] = None
        all_block_losses = np.empty((lam_grid.size, 0), dtype=np.float64)  # will set width after first λ

        for i, lam in enumerate(lam_grid):
            pred_masks, truth_masks = self._build_masks_for_lambda(yhat, y, threshold, m, float(lam))
            bid, bl = self._compute_block_losses(pred_masks, truth_masks, block_ids, spatial_weights)
            if block_ids_unique is None:
                block_ids_unique = bid
                all_block_losses = np.empty((lam_grid.size, bid.size), dtype=np.float64)
            else:
                # ensure block id consistency
                if not np.array_equal(bid, block_ids_unique):
                    # align by merging on block id
                    raise RuntimeError("Inconsistent block id set across λ. Ensure block_ids are identical.")
            all_block_losses[i] = bl

        assert block_ids_unique is not None
        n_blocks = block_ids_unique.size

        # Compute corrected empirical risk curve: (n/(n+1))*R + B/(n+1)
        risks = np.empty(lam_grid.size, dtype=np.float64)
        n = float(n_blocks)
        for i in range(lam_grid.size):
            R = self._empirical_risk(all_block_losses[i], weights_per_block)
            risks[i] = (n / (n + 1.0)) * R + (self.settings.slack_B / (n + 1.0))

        # Select smallest λ with corrected risk ≤ α. If none, pick max λ (most conservative).
        mask_ok = risks <= self.settings.alpha + 1e-12
        if np.any(mask_ok):
            idx = int(np.argmax(mask_ok))  # first True (since lam_grid asc)
        else:
            idx = lam_grid.size - 1
        lam_star = float(lam_grid[idx])

        return CRCResult(
            lambda_star=lam_star,
            risk_curve=risks,
            lambda_grid=lam_grid,
            block_losses=all_block_losses,
            block_ids=block_ids_unique,
        )
