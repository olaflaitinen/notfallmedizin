# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic test evaluation and ROC analysis.

Provides comprehensive 2x2 contingency table metrics, ROC curve
analysis with AUC, and the DeLong test for comparing ROC curves.

References
----------
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing
    the areas under two or more correlated receiver operating
    characteristic curves. Biometrics, 44(3), 837-845.
Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. JASA, 22(158), 209-212.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class DiagnosticMetrics:
    """Complete diagnostic test metrics from a 2x2 table.

    Attributes
    ----------
    tp, fp, tn, fn : int
    sensitivity, specificity : float
    ppv, npv : float
    accuracy : float
    prevalence : float
    lr_positive, lr_negative : float
    diagnostic_odds_ratio : float
    f1_score : float
    youden_index : float
    """

    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    accuracy: float
    prevalence: float
    lr_positive: float
    lr_negative: float
    diagnostic_odds_ratio: float
    f1_score: float
    youden_index: float


@dataclass(frozen=True)
class DeLongResult:
    """DeLong test for comparing two AUCs.

    Attributes
    ----------
    z_statistic : float
    p_value : float
    auc_a : float
    auc_b : float
    auc_difference : float
    ci_difference : tuple
    """

    z_statistic: float
    p_value: float
    auc_a: float
    auc_b: float
    auc_difference: float
    ci_difference: Tuple[float, float] = (0.0, 0.0)


class DiagnosticTestEvaluator:
    """Evaluates a binary diagnostic test against ground truth."""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> DiagnosticMetrics:
        """Compute full diagnostic test metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels (0 or 1).
        y_pred : np.ndarray
            Predicted binary labels (0 or 1).

        Returns
        -------
        DiagnosticMetrics
        """
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        if len(yt) != len(yp):
            raise ValidationError("y_true and y_pred must have equal length.")

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        n = tp + fp + tn + fn
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        acc = (tp + tn) / n if n > 0 else 0.0
        prev = (tp + fn) / n if n > 0 else 0.0

        lr_pos = sens / (1 - spec) if spec < 1 else float("inf")
        lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float("inf")

        precision = ppv
        recall = sens
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        youden = sens + spec - 1

        return DiagnosticMetrics(
            tp=tp, fp=fp, tn=tn, fn=fn,
            sensitivity=round(sens, 6),
            specificity=round(spec, 6),
            ppv=round(ppv, 6),
            npv=round(npv, 6),
            accuracy=round(acc, 6),
            prevalence=round(prev, 6),
            lr_positive=round(lr_pos, 4),
            lr_negative=round(lr_neg, 4),
            diagnostic_odds_ratio=round(dor, 4) if dor != float("inf") else float("inf"),
            f1_score=round(f1, 6),
            youden_index=round(youden, 6),
        )

    def confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Tuple[float, float]]:
        """Wilson score confidence intervals for key metrics.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        alpha : float

        Returns
        -------
        dict
            Mapping metric name to (lower, upper) CI.
        """
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        cis = {}
        cis["sensitivity"] = self._wilson_ci(tp, tp + fn, alpha)
        cis["specificity"] = self._wilson_ci(tn, tn + fp, alpha)
        cis["ppv"] = self._wilson_ci(tp, tp + fp, alpha)
        cis["npv"] = self._wilson_ci(tn, tn + fn, alpha)
        cis["accuracy"] = self._wilson_ci(tp + tn, tp + fp + tn + fn, alpha)
        return cis

    @staticmethod
    def _wilson_ci(
        x: int, n: int, alpha: float
    ) -> Tuple[float, float]:
        """Wilson score interval for a binomial proportion."""
        if n == 0:
            return (0.0, 0.0)
        p_hat = x / n
        z = sp_stats.norm.ppf(1 - alpha / 2)
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
        return (
            round(max(center - margin, 0.0), 6),
            round(min(center + margin, 1.0), 6),
        )


class ROCAnalyzer:
    """ROC curve analysis with AUC computation and DeLong comparison."""

    def __init__(self) -> None:
        self._fpr: Optional[np.ndarray] = None
        self._tpr: Optional[np.ndarray] = None
        self._thresholds: Optional[np.ndarray] = None
        self._auc: Optional[float] = None
        self.is_fitted: bool = False

    def fit(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> "ROCAnalyzer":
        """Compute the ROC curve.

        Parameters
        ----------
        y_true : np.ndarray
        y_scores : np.ndarray

        Returns
        -------
        self
        """
        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_scores, dtype=np.float64)
        if len(yt) != len(ys):
            raise ValidationError("y_true and y_scores must have equal length.")

        thresholds = np.unique(ys)
        thresholds = np.sort(thresholds)[::-1]

        tpr_list = [0.0]
        fpr_list = [0.0]
        thresh_list = [thresholds[0] + 1.0]

        total_pos = int(np.sum(yt == 1))
        total_neg = int(np.sum(yt == 0))

        for t in thresholds:
            pred = (ys >= t).astype(int)
            tp = int(np.sum((yt == 1) & (pred == 1)))
            fp = int(np.sum((yt == 0) & (pred == 1)))
            tpr_list.append(tp / total_pos if total_pos > 0 else 0.0)
            fpr_list.append(fp / total_neg if total_neg > 0 else 0.0)
            thresh_list.append(t)

        self._fpr = np.array(fpr_list)
        self._tpr = np.array(tpr_list)
        self._thresholds = np.array(thresh_list)
        self._auc = float(np.trapz(self._tpr, self._fpr))
        if self._auc < 0:
            self._auc = -self._auc
        self.is_fitted = True
        return self

    def auc(self) -> float:
        """Return the area under the ROC curve.

        Returns
        -------
        float
        """
        self._check()
        return round(self._auc, 6)  # type: ignore[arg-type]

    def optimal_threshold(self, method: str = "youden") -> float:
        """Find the optimal classification threshold.

        Parameters
        ----------
        method : str
            'youden' maximises Youden's J = TPR - FPR.

        Returns
        -------
        float
        """
        self._check()
        if method == "youden":
            j = self._tpr - self._fpr
            idx = int(np.argmax(j))
            return float(self._thresholds[idx])
        raise ValidationError(f"Unknown method: {method}")

    def partial_auc(self, fpr_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Compute partial AUC over a restricted FPR range.

        Parameters
        ----------
        fpr_range : tuple of (fpr_min, fpr_max)

        Returns
        -------
        float
        """
        self._check()
        mask = (self._fpr >= fpr_range[0]) & (self._fpr <= fpr_range[1])
        if np.sum(mask) < 2:
            return 0.0
        return round(float(np.abs(np.trapz(self._tpr[mask], self._fpr[mask]))), 6)

    @staticmethod
    def compare_auc(
        y_true: np.ndarray,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alpha: float = 0.05,
    ) -> DeLongResult:
        """DeLong test for comparing two correlated AUCs.

        Parameters
        ----------
        y_true : np.ndarray
        scores_a : np.ndarray
        scores_b : np.ndarray
        alpha : float

        Returns
        -------
        DeLongResult
        """
        yt = np.asarray(y_true, dtype=int)
        sa = np.asarray(scores_a, dtype=np.float64)
        sb = np.asarray(scores_b, dtype=np.float64)

        pos = np.where(yt == 1)[0]
        neg = np.where(yt == 0)[0]

        m = len(pos)
        n = len(neg)

        def _placement(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            v10 = np.zeros(m)
            v01 = np.zeros(n)
            for i, pi in enumerate(pos):
                v10[i] = np.mean(scores[pi] > scores[neg]) + 0.5 * np.mean(
                    scores[pi] == scores[neg]
                )
            for j, nj in enumerate(neg):
                v01[j] = np.mean(scores[pos] > scores[nj]) + 0.5 * np.mean(
                    scores[pos] == scores[nj]
                )
            return v10, v01

        v10_a, v01_a = _placement(sa)
        v10_b, v01_b = _placement(sb)

        auc_a = float(np.mean(v10_a))
        auc_b = float(np.mean(v10_b))

        s10 = np.cov(np.stack([v10_a, v10_b]))
        s01 = np.cov(np.stack([v01_a, v01_b]))

        s = s10 / m + s01 / n

        diff = auc_a - auc_b
        var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
        se_diff = math.sqrt(max(var_diff, 1e-15))

        z = diff / se_diff if se_diff > 0 else 0.0
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (
            round(diff - z_crit * se_diff, 6),
            round(diff + z_crit * se_diff, 6),
        )

        return DeLongResult(
            z_statistic=round(z, 4),
            p_value=round(p, 6),
            auc_a=round(auc_a, 6),
            auc_b=round(auc_b, 6),
            auc_difference=round(diff, 6),
            ci_difference=ci,
        )

    def _check(self) -> None:
        if not self.is_fitted:
            raise ValidationError("ROCAnalyzer has not been fitted.")
