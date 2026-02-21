# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Arrhythmia detection and rhythm analysis.

Provides a trainable arrhythmia classifier using RR-interval features
and a rule-based rhythm analyzer for bedside monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from notfallmedizin.core.base import ClinicalModel
from notfallmedizin.core.exceptions import ModelNotFittedError, ValidationError
from notfallmedizin.core.config import get_config


class ArrhythmiaType(Enum):
    """ECG rhythm classifications."""

    NORMAL = "normal_sinus_rhythm"
    AFIB = "atrial_fibrillation"
    AFLUTTER = "atrial_flutter"
    SVT = "supraventricular_tachycardia"
    VTACH = "ventricular_tachycardia"
    VFIB = "ventricular_fibrillation"
    PVC = "premature_ventricular_complex"
    PAC = "premature_atrial_complex"
    BRADYCARDIA = "sinus_bradycardia"
    HEART_BLOCK_1 = "first_degree_heart_block"
    HEART_BLOCK_2 = "second_degree_heart_block"
    HEART_BLOCK_3 = "third_degree_heart_block"
    ASYSTOLE = "asystole"


@dataclass(frozen=True)
class RhythmAnalysis:
    """Result of a rhythm analysis.

    Attributes
    ----------
    is_regular : bool
    mean_rate_bpm : float
    rate_classification : str
        bradycardia / normal / tachycardia
    variability_score : float
    suspected_rhythm : str
    """

    is_regular: bool
    mean_rate_bpm: float
    rate_classification: str
    variability_score: float
    suspected_rhythm: str


class ArrhythmiaDetector(ClinicalModel):
    """Machine-learning arrhythmia classifier using RR-interval features.

    Uses a random forest on hand-crafted features extracted from each
    ECG segment (RR statistics, regularity, morphology proxies).

    Parameters
    ----------
    n_estimators : int, default=200
    """

    def __init__(self, n_estimators: int = 200) -> None:
        super().__init__()
        self._name = "ArrhythmiaDetector"
        self._description = "Random-forest arrhythmia classifier"
        self.n_estimators = n_estimators
        self.is_fitted_: bool = False
        self._model: Optional[RandomForestClassifier] = None

    def fit(self, X: Any, y: Any) -> "ArrhythmiaDetector":
        """Train the arrhythmia classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (e.g., from ``extract_features``).
        y : array-like of shape (n_samples,)
            Arrhythmia type labels.

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=np.float64)
        cfg = get_config()
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        self._model.fit(X_arr, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict arrhythmia type for each sample."""
        self._check_fitted()
        return self._model.predict(np.asarray(X, dtype=np.float64))  # type: ignore[union-attr]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return probability estimates per arrhythmia type."""
        self._check_fitted()
        return self._model.predict_proba(np.asarray(X, dtype=np.float64))  # type: ignore[union-attr]

    @staticmethod
    def extract_features(rr_intervals_ms: np.ndarray) -> np.ndarray:
        """Extract features from RR intervals for a single segment.

        Features (8):
            mean_rr, std_rr, cv_rr, rmssd, median_rr, iqr_rr,
            mean_successive_diff, max_rr_ratio

        Parameters
        ----------
        rr_intervals_ms : np.ndarray

        Returns
        -------
        np.ndarray of shape (8,)
        """
        rr = np.asarray(rr_intervals_ms, dtype=np.float64)
        if len(rr) < 3:
            raise ValidationError("At least 3 RR intervals required.")

        mean_rr = np.mean(rr)
        std_rr = np.std(rr, ddof=1)
        cv_rr = std_rr / mean_rr if mean_rr > 0 else 0.0
        diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(diffs ** 2))
        median_rr = np.median(rr)
        q75, q25 = np.percentile(rr, [75, 25])
        iqr_rr = q75 - q25
        mean_succ = np.mean(np.abs(diffs))
        max_ratio = np.max(rr) / np.min(rr) if np.min(rr) > 0 else 0.0

        return np.array([
            mean_rr, std_rr, cv_rr, rmssd,
            median_rr, iqr_rr, mean_succ, max_ratio,
        ])

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ModelNotFittedError("ArrhythmiaDetector has not been fitted.")


class RhythmAnalyzer:
    """Rule-based rhythm analysis from RR intervals.

    Uses coefficient of variation and successive-difference entropy
    to assess regularity and identify common rhythms.
    """

    @staticmethod
    def analyze_rhythm(
        rr_intervals_ms: np.ndarray,
        regularity_threshold: float = 0.10,
    ) -> RhythmAnalysis:
        """Analyze cardiac rhythm from RR intervals.

        Parameters
        ----------
        rr_intervals_ms : np.ndarray
        regularity_threshold : float
            CV below this value is considered regular.

        Returns
        -------
        RhythmAnalysis
        """
        rr = np.asarray(rr_intervals_ms, dtype=np.float64)
        if len(rr) < 3:
            raise ValidationError("At least 3 RR intervals required.")

        mean_rr = float(np.mean(rr))
        mean_rate = 60000.0 / mean_rr if mean_rr > 0 else 0.0

        cv = float(np.std(rr, ddof=1) / mean_rr) if mean_rr > 0 else 0.0
        is_regular = cv < regularity_threshold

        if mean_rate < 60:
            rate_class = "bradycardia"
        elif mean_rate > 100:
            rate_class = "tachycardia"
        else:
            rate_class = "normal"

        if is_regular and rate_class == "normal":
            suspected = "normal_sinus_rhythm"
        elif is_regular and rate_class == "bradycardia":
            suspected = "sinus_bradycardia"
        elif is_regular and rate_class == "tachycardia":
            if mean_rate > 140:
                suspected = "supraventricular_tachycardia"
            else:
                suspected = "sinus_tachycardia"
        elif not is_regular and cv > 0.20:
            suspected = "atrial_fibrillation"
        else:
            suspected = "irregular_rhythm"

        return RhythmAnalysis(
            is_regular=is_regular,
            mean_rate_bpm=round(mean_rate, 1),
            rate_classification=rate_class,
            variability_score=round(cv, 4),
            suspected_rhythm=suspected,
        )
