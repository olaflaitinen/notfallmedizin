# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Patient disposition prediction and bed availability estimation.

Provides multi-class classification for patient disposition and a
birth-death process model for bed capacity forecasting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from notfallmedizin.core.base import ClinicalModel, ClassifierMixin
from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    ValidationError,
)
from notfallmedizin.core.config import get_config


class DispositionCategory(Enum):
    """Patient disposition categories."""

    DISCHARGE = "discharge"
    OBSERVATION = "observation"
    FLOOR_ADMISSION = "floor_admission"
    ICU_ADMISSION = "icu_admission"
    TRANSFER = "transfer"
    AMA = "against_medical_advice"


@dataclass(frozen=True)
class BedEstimate:
    """Bed availability forecast result.

    Attributes
    ----------
    expected_available : float
    probability_full : float
    recommended_diversion : bool
    arrival_rate : float
    discharge_rate : float
    """

    expected_available: float
    probability_full: float
    recommended_diversion: bool
    arrival_rate: float
    discharge_rate: float


class DispositionPredictor(ClinicalModel, ClassifierMixin):
    """Multi-class gradient-boosting model for ED disposition prediction.

    Predicts one of six disposition categories for each patient.

    Parameters
    ----------
    n_estimators : int, default=300
    max_depth : int, default=5
    """

    CATEGORIES: List[str] = [c.value for c in DispositionCategory]

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
    ) -> None:
        super().__init__()
        self._name = "DispositionPredictor"
        self._description = "Multi-class ED disposition predictor"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.is_fitted_: bool = False
        self.feature_names_: Optional[List[str]] = None
        self._scaler = StandardScaler()
        self._model: Optional[GradientBoostingClassifier] = None

    def fit(self, X: Any, y: Any) -> "DispositionPredictor":
        """Fit the disposition model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Disposition category labels (strings matching
            ``DispositionCategory`` values).

        Returns
        -------
        self
        """
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.fit_transform(X_arr)

        cfg = get_config()
        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=cfg.random_state,
        )
        self._model.fit(X_scaled, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict disposition categories."""
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict(X_scaled)  # type: ignore[union-attr]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict disposition probabilities."""
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict_proba(X_scaled)  # type: ignore[union-attr]

    def get_classes(self) -> np.ndarray:
        """Return the class labels learned during fit."""
        self._check_fitted()
        return self._model.classes_  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ModelNotFittedError("DispositionPredictor has not been fitted.")


class BedAvailabilityEstimator:
    """Bed capacity forecasting using a simple birth-death process.

    Models the ED bed census as a continuous-time Markov chain where
    arrivals follow a Poisson process and service times are exponential.
    """

    @staticmethod
    def estimate(
        current_census: int,
        total_beds: int,
        arrival_rate: float,
        discharge_rate: float,
        time_horizon_hours: float = 4.0,
    ) -> BedEstimate:
        """Estimate bed availability over a time horizon.

        Parameters
        ----------
        current_census : int
            Number of beds currently occupied.
        total_beds : int
            Total bed capacity.
        arrival_rate : float
            Expected arrivals per hour.
        discharge_rate : float
            Expected discharges per hour.
        time_horizon_hours : float
            Forecast horizon in hours.

        Returns
        -------
        BedEstimate
        """
        if current_census < 0 or total_beds <= 0:
            raise ValidationError("Census and total_beds must be positive.")

        net_rate = arrival_rate - discharge_rate
        expected_census = current_census + net_rate * time_horizon_hours
        expected_available = max(0.0, total_beds - expected_census)

        variance = (arrival_rate + discharge_rate) * time_horizon_hours
        if variance > 0:
            std = math.sqrt(variance)
            z = (total_beds - expected_census) / std
            from scipy.stats import norm

            prob_full = 1.0 - norm.cdf(z)
        else:
            prob_full = 1.0 if expected_census >= total_beds else 0.0

        return BedEstimate(
            expected_available=round(expected_available, 1),
            probability_full=round(min(max(prob_full, 0.0), 1.0), 4),
            recommended_diversion=prob_full > 0.80,
            arrival_rate=arrival_rate,
            discharge_rate=discharge_rate,
        )
