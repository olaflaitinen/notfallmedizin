# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Length of stay prediction and ED throughput analysis.

Provides regression models for predicting ED and in-hospital length of stay,
plus queuing-theory utilities for throughput estimation.

References
----------
Erlang, A. K. (1917). Solution of some problems in the theory of
    probabilities of significance in automatic telephone exchanges.
    Elektroteknikeren, 13, 5-13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from notfallmedizin.core.base import ClinicalModel, RegressorMixin
from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    ValidationError,
)
from notfallmedizin.core.config import get_config


@dataclass(frozen=True)
class ThroughputMetrics:
    """Summary statistics for ED throughput.

    Attributes
    ----------
    median_los_hours : float
    mean_los_hours : float
    p90_los_hours : float
    door_to_provider_minutes : float
    left_without_being_seen_rate : float
    """

    median_los_hours: float
    mean_los_hours: float
    p90_los_hours: float
    door_to_provider_minutes: float
    left_without_being_seen_rate: float = 0.0


class LOSPredictor(ClinicalModel, RegressorMixin):
    """Gradient-boosting regression model for length-of-stay prediction.

    Parameters
    ----------
    n_estimators : int, default=300
    max_depth : int, default=5
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
    ) -> None:
        super().__init__()
        self._name = "LOSPredictor"
        self._description = "Length of stay regression predictor"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.is_fitted_: bool = False
        self.feature_names_: Optional[List[str]] = None
        self._scaler = StandardScaler()
        self._model: Optional[GradientBoostingRegressor] = None
        self._model_lower: Optional[GradientBoostingRegressor] = None
        self._model_upper: Optional[GradientBoostingRegressor] = None

    def fit(self, X: Any, y: Any) -> "LOSPredictor":
        """Fit the LOS model.

        Also fits quantile regressors at alpha=0.025 and alpha=0.975
        for prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Length of stay in hours.

        Returns
        -------
        self
        """
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_scaled = self._scaler.fit_transform(X_arr)

        cfg = get_config()
        common = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=cfg.random_state,
        )
        self._model = GradientBoostingRegressor(loss="squared_error", **common)
        self._model.fit(X_scaled, y_arr)

        self._model_lower = GradientBoostingRegressor(
            loss="quantile", alpha=0.025, **common
        )
        self._model_lower.fit(X_scaled, y_arr)

        self._model_upper = GradientBoostingRegressor(
            loss="quantile", alpha=0.975, **common
        )
        self._model_upper.fit(X_scaled, y_arr)

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict length of stay in hours."""
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        preds = self._model.predict(X_scaled)  # type: ignore[union-attr]
        return np.maximum(preds, 0.0)

    def predict_interval(
        self, X: Any, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals.

        Parameters
        ----------
        X : array-like
        confidence : float
            Nominal coverage (the quantile models are already at 95 %).

        Returns
        -------
        tuple of (lower, upper) np.ndarray
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        lower = np.maximum(
            self._model_lower.predict(X_scaled), 0.0  # type: ignore[union-attr]
        )
        upper = np.maximum(
            self._model_upper.predict(X_scaled), 0.0  # type: ignore[union-attr]
        )
        return lower, upper

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance mapping."""
        self._check_fitted()
        imp = self._model.feature_importances_  # type: ignore[union-attr]
        names = self.feature_names_ or [
            f"feature_{i}" for i in range(len(imp))
        ]
        return dict(zip(names, imp.tolist()))

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ModelNotFittedError("LOSPredictor has not been fitted.")


class EDThroughputAnalyzer:
    """Emergency department throughput analysis and queuing theory utilities."""

    @staticmethod
    def calculate_metrics(
        arrival_times: np.ndarray,
        departure_times: np.ndarray,
        provider_contact_times: Optional[np.ndarray] = None,
        lwbs_count: int = 0,
        total_arrivals: Optional[int] = None,
    ) -> ThroughputMetrics:
        """Compute throughput summary from timestamps.

        Parameters
        ----------
        arrival_times : np.ndarray
            Arrival epoch times (seconds or datetime64).
        departure_times : np.ndarray
            Departure epoch times (seconds or datetime64).
        provider_contact_times : np.ndarray, optional
            Time of first provider contact.
        lwbs_count : int
            Number of patients who left without being seen.
        total_arrivals : int, optional
            Total arrivals (defaults to len(arrival_times) + lwbs_count).

        Returns
        -------
        ThroughputMetrics
        """
        arr = np.asarray(arrival_times, dtype=np.float64)
        dep = np.asarray(departure_times, dtype=np.float64)
        los_seconds = dep - arr
        los_hours = los_seconds / 3600.0

        d2p = 0.0
        if provider_contact_times is not None:
            pct = np.asarray(provider_contact_times, dtype=np.float64)
            d2p = float(np.median(pct - arr)) / 60.0

        total = total_arrivals if total_arrivals is not None else len(arr) + lwbs_count
        lwbs_rate = lwbs_count / total if total > 0 else 0.0

        return ThroughputMetrics(
            median_los_hours=float(np.median(los_hours)),
            mean_los_hours=float(np.mean(los_hours)),
            p90_los_hours=float(np.percentile(los_hours, 90)),
            door_to_provider_minutes=d2p,
            left_without_being_seen_rate=lwbs_rate,
        )

    @staticmethod
    def estimate_wait_time(
        arrival_rate: float,
        service_rate: float,
        num_servers: int,
    ) -> float:
        """Estimate expected wait time using the M/M/c queuing model.

        Parameters
        ----------
        arrival_rate : float
            Mean arrivals per hour (lambda).
        service_rate : float
            Mean service completions per server per hour (mu).
        num_servers : int
            Number of parallel servers (c).

        Returns
        -------
        float
            Expected wait time in hours.

        Notes
        -----
        Uses the Erlang C formula:

        .. math::

            C(c, \\rho) = \\frac{(c \\rho)^c / c!}{
                \\sum_{k=0}^{c-1} (c \\rho)^k / k!  +  (c \\rho)^c / c!
                \\cdot 1/(1 - \\rho)}

        where :math:`\\rho = \\lambda / (c \\mu)`.
        """
        c = num_servers
        lam = arrival_rate
        mu = service_rate
        rho = lam / (c * mu)

        if rho >= 1.0:
            return float("inf")

        a = lam / mu  # offered load

        sum_terms = sum(a ** k / math.factorial(k) for k in range(c))
        last_term = (a ** c / math.factorial(c)) * (1.0 / (1.0 - rho))
        erlang_c = last_term / (sum_terms + last_term)

        wq = erlang_c / (c * mu - lam)
        return wq
