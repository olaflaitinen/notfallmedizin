# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation metrics for emergency medicine AI models.

Provides classification, regression, and clinically oriented metric
suites designed for EM prediction tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class ClassificationReport:
    """Classification evaluation report."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    kappa: float
    auc_roc: float
    brier_score: float
    log_loss_value: float


@dataclass(frozen=True)
class RegressionReport:
    """Regression evaluation report."""

    mae: float
    mse: float
    rmse: float
    r_squared: float
    median_ae: float
    mape: float
    explained_variance: float


class ClassificationMetrics:
    """Classification evaluation suite."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        y_proba : np.ndarray, optional
            Predicted probabilities (for AUC, Brier, log-loss).

        Returns
        -------
        dict
        """
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        avg = "binary" if len(np.unique(yt)) <= 2 else "weighted"

        result: Dict[str, float] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, average=avg, zero_division=0)),
            "recall": float(recall_score(yt, yp, average=avg, zero_division=0)),
            "f1": float(f1_score(yt, yp, average=avg, zero_division=0)),
            "mcc": float(matthews_corrcoef(yt, yp)),
            "kappa": float(cohen_kappa_score(yt, yp)),
        }

        if y_proba is not None:
            ypr = np.asarray(y_proba)
            try:
                if ypr.ndim == 1 or ypr.shape[1] == 2:
                    p1 = ypr[:, 1] if ypr.ndim == 2 else ypr
                    result["auc_roc"] = float(roc_auc_score(yt, p1))
                    result["brier_score"] = float(brier_score_loss(yt, p1))
                else:
                    result["auc_roc"] = float(
                        roc_auc_score(yt, ypr, multi_class="ovr", average="weighted")
                    )
                result["log_loss"] = float(log_loss(yt, ypr))
            except ValueError:
                pass

        return result

    @staticmethod
    def bootstrap_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: str = "accuracy",
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Bootstrap confidence interval for a given metric.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        metric_fn : str
        n_bootstrap : int
        alpha : float

        Returns
        -------
        dict with keys: point_estimate, ci_lower, ci_upper
        """
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        n = len(yt)

        fn_map = {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "f1": lambda a, b: f1_score(a, b, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef,
            "kappa": cohen_kappa_score,
        }

        if metric_fn not in fn_map:
            raise ValidationError(f"Unsupported metric: {metric_fn}")

        fn = fn_map[metric_fn]
        rng = np.random.default_rng(42)

        scores = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            scores.append(fn(yt[idx], yp[idx]))

        scores_arr = np.array(scores)
        return {
            "point_estimate": round(float(fn(yt, yp)), 6),
            "ci_lower": round(float(np.percentile(scores_arr, 100 * alpha / 2)), 6),
            "ci_upper": round(float(np.percentile(scores_arr, 100 * (1 - alpha / 2))), 6),
        }


class RegressionMetrics:
    """Regression evaluation suite."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression metrics.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray

        Returns
        -------
        dict
        """
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)

        residuals = yt - yp
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        median_ae = float(np.median(np.abs(residuals)))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        nonzero = yt != 0
        if nonzero.any():
            mape = float(np.mean(np.abs(residuals[nonzero] / yt[nonzero]))) * 100
        else:
            mape = float("inf")

        ev = 1 - np.var(residuals) / np.var(yt) if np.var(yt) > 0 else 0.0

        return {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "r_squared": round(r2, 6),
            "median_ae": round(median_ae, 6),
            "mape_percent": round(mape, 4),
            "explained_variance": round(float(ev), 6),
        }


class ClinicalMetrics:
    """Clinically oriented evaluation metrics for EM models.

    Includes net reclassification improvement (NRI) and integrated
    discrimination improvement (IDI).
    """

    @staticmethod
    def net_reclassification_improvement(
        y_true: np.ndarray,
        proba_old: np.ndarray,
        proba_new: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute the Net Reclassification Improvement (NRI).

        Parameters
        ----------
        y_true : np.ndarray
        proba_old : np.ndarray
        proba_new : np.ndarray
        threshold : float

        Returns
        -------
        dict
        """
        yt = np.asarray(y_true, dtype=int)
        po = np.asarray(proba_old, dtype=np.float64)
        pn = np.asarray(proba_new, dtype=np.float64)

        old_class = (po >= threshold).astype(int)
        new_class = (pn >= threshold).astype(int)

        events = yt == 1
        non_events = yt == 0

        up_events = np.sum((new_class[events] > old_class[events]))
        down_events = np.sum((new_class[events] < old_class[events]))
        n_events = np.sum(events)

        up_non = np.sum((new_class[non_events] > old_class[non_events]))
        down_non = np.sum((new_class[non_events] < old_class[non_events]))
        n_non = np.sum(non_events)

        nri_events = (up_events - down_events) / n_events if n_events > 0 else 0.0
        nri_non = (down_non - up_non) / n_non if n_non > 0 else 0.0
        nri = nri_events + nri_non

        return {
            "nri": round(float(nri), 6),
            "nri_events": round(float(nri_events), 6),
            "nri_non_events": round(float(nri_non), 6),
        }

    @staticmethod
    def integrated_discrimination_improvement(
        y_true: np.ndarray,
        proba_old: np.ndarray,
        proba_new: np.ndarray,
    ) -> Dict[str, float]:
        """Compute the Integrated Discrimination Improvement (IDI).

        Parameters
        ----------
        y_true : np.ndarray
        proba_old : np.ndarray
        proba_new : np.ndarray

        Returns
        -------
        dict
        """
        yt = np.asarray(y_true, dtype=int)
        po = np.asarray(proba_old, dtype=np.float64)
        pn = np.asarray(proba_new, dtype=np.float64)

        events = yt == 1
        non_events = yt == 0

        is_events = float(np.mean(pn[events]) - np.mean(po[events]))
        is_non = float(np.mean(pn[non_events]) - np.mean(po[non_events]))
        idi = is_events - is_non

        return {
            "idi": round(idi, 6),
            "is_events": round(is_events, 6),
            "is_non_events": round(is_non, 6),
        }
