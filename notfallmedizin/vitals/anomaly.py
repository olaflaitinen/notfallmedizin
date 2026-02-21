# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Anomaly detection in vital sign time series.

This module provides unsupervised anomaly detection tailored to
multivariate vital sign data. Two complementary approaches are offered:

1. **Model-based detection** via :class:`VitalSignsAnomalyDetector`,
   which wraps scikit-learn ensemble methods (Isolation Forest, Local
   Outlier Factor) behind the :class:`BaseEstimator` interface.

2. **Statistical detection** via :class:`StatisticalAnomalyDetector`,
   which applies z-score and modified z-score (MAD-based) thresholding,
   optionally over a rolling window.

Classes
-------
VitalSignsAnomalyDetector
    Model-based anomaly detector inheriting from ``BaseEstimator``.
StatisticalAnomalyDetector
    Parametric detector using z-score and modified z-score methods.

References
----------
.. [1] Liu FT, Ting KM, Zhou ZH. Isolation forest. In: Proc 8th IEEE
       ICDM. 2008:413-422.
.. [2] Breunig MM et al. LOF: identifying density-based local outliers.
       In: Proc ACM SIGMOD. 2000:93-104.
.. [3] Iglewicz B, Hoaglin DC. Volume 16: How to Detect and Handle
       Outliers. ASQC Quality Press; 1993.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from notfallmedizin.core.base import BaseEstimator
from notfallmedizin.core.exceptions import (
    ComputationError,
    ValidationError,
)


# ======================================================================
# Model-based Anomaly Detector
# ======================================================================


class VitalSignsAnomalyDetector(BaseEstimator):
    """Detect anomalous vital sign patterns using ensemble methods.

    This estimator learns the distribution of "normal" multivariate
    vital sign observations during :meth:`fit` and flags deviations
    during :meth:`predict`. It delegates to scikit-learn's
    ``IsolationForest`` or ``LocalOutlierFactor`` internally.

    Parameters
    ----------
    method : {"isolation_forest", "local_outlier_factor", "statistical"}
        Detection algorithm to use. Default is ``"isolation_forest"``.
    contamination : float or "auto", optional
        Expected proportion of anomalies in the training data. Passed
        to the underlying scikit-learn estimator. Default is ``"auto"``.
    n_estimators : int, optional
        Number of base estimators in the ensemble (Isolation Forest
        only). Default is ``100``.
    random_state : int or None, optional
        Seed for reproducibility. Default is ``None``.

    Attributes
    ----------
    detector_ : object
        Fitted scikit-learn detector (available after :meth:`fit`).
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X_train = rng.normal(loc=[80, 120, 80, 16, 97, 37],
    ...                      scale=[5, 10, 5, 2, 1, 0.3],
    ...                      size=(200, 6))
    >>> detector = VitalSignsAnomalyDetector(method="isolation_forest",
    ...                                      random_state=42)
    >>> detector.fit(X_train)
    VitalSignsAnomalyDetector(...)
    >>> labels = detector.predict(X_train[:5])
    >>> labels.shape
    (5,)
    """

    _SUPPORTED_METHODS = ("isolation_forest", "local_outlier_factor", "statistical")

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: Union[float, str] = "auto",
        n_estimators: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        if method not in self._SUPPORTED_METHODS:
            raise ValidationError(
                message=(
                    f"'method' must be one of {self._SUPPORTED_METHODS}, "
                    f"got {method!r}."
                ),
                parameter="method",
            )
        self.method: str = method
        self.contamination: Union[float, str] = contamination
        self.n_estimators: int = n_estimators
        self.random_state: Optional[int] = random_state

        self.detector_: Any = None
        self.n_features_in_: int = 0
        self._statistical_detector: Optional[StatisticalAnomalyDetector] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "VitalSignsAnomalyDetector":
        """Learn the normal distribution from historical vital sign data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training observations (rows = patients/time-steps, columns =
            vital sign channels).
        y : ignored
            Present for API compatibility.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        self
            The fitted detector.

        Raises
        ------
        ValidationError
            If *X* has fewer than 2 samples or is not 2-D.
        """
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValidationError(
                message=f"X must be 2-D, got {X_arr.ndim}-D array.",
                parameter="X",
            )
        if X_arr.shape[0] < 2:
            raise ValidationError(
                message=(
                    f"At least 2 samples are required for fitting, "
                    f"got {X_arr.shape[0]}."
                ),
                parameter="X",
            )

        self.n_features_in_ = X_arr.shape[1]

        if self.method == "isolation_forest":
            self._fit_isolation_forest(X_arr)
        elif self.method == "local_outlier_factor":
            self._fit_lof(X_arr)
        elif self.method == "statistical":
            self._fit_statistical(X_arr)

        self.fit_time_ = time.perf_counter() - t0
        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Classify observations as normal (1) or anomalous (-1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations to classify.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            ``1`` for inliers, ``-1`` for outliers.
        """
        self._check_is_fitted()
        X_arr = self._validate_predict_input(X)

        if self.method == "statistical":
            mask = self._statistical_detector.detect(X_arr)  # type: ignore[union-attr]
            return np.where(mask.any(axis=1), -1, 1)

        return self.detector_.predict(X_arr)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Compute per-sample anomaly scores.

        Lower (more negative) scores indicate stronger anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations to score.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Anomaly scores. The sign convention matches scikit-learn:
            more negative values are more anomalous.

        Raises
        ------
        ComputationError
            If the underlying method does not support sample scoring
            (e.g. LOF in predict-only mode).
        """
        self._check_is_fitted()
        X_arr = self._validate_predict_input(X)

        if self.method == "statistical":
            scores = self._statistical_detector._compute_zscores(X_arr)  # type: ignore[union-attr]
            return -np.max(np.abs(scores), axis=1)

        if not hasattr(self.detector_, "score_samples"):
            raise ComputationError(
                f"Method '{self.method}' does not support score_samples."
            )
        return self.detector_.score_samples(X_arr)

    # ------------------------------------------------------------------
    # Internal fitting methods
    # ------------------------------------------------------------------

    def _fit_isolation_forest(self, X: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest

        contamination = self.contamination
        if isinstance(contamination, str) and contamination == "auto":
            contamination = "auto"

        self.detector_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=contamination,
            random_state=self.random_state,
        )
        self.detector_.fit(X)

    def _fit_lof(self, X: np.ndarray) -> None:
        from sklearn.neighbors import LocalOutlierFactor

        contamination = self.contamination
        if isinstance(contamination, str) and contamination == "auto":
            contamination = 0.1

        self.detector_ = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,
        )
        self.detector_.fit(X)

    def _fit_statistical(self, X: np.ndarray) -> None:
        self._statistical_detector = StatisticalAnomalyDetector()
        self._statistical_detector.fit(X)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_predict_input(self, X: ArrayLike) -> np.ndarray:
        """Coerce and validate input for predict / score_samples.

        Parameters
        ----------
        X : array-like
            Raw input.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features_in\\_)

        Raises
        ------
        ValidationError
            If the feature count does not match training data.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValidationError(
                message=(
                    f"Expected {self.n_features_in_} features, "
                    f"got {X_arr.shape[1]}."
                ),
                parameter="X",
            )
        return X_arr


# ======================================================================
# Statistical Anomaly Detector
# ======================================================================


class StatisticalAnomalyDetector:
    """Detect anomalies using z-score and modified z-score (MAD) methods.

    The standard z-score is sensitive to outliers in the reference data
    because it relies on the sample mean and standard deviation. The
    modified z-score replaces these with the median and the Median
    Absolute Deviation (MAD), which are robust to up to 50% contamination.

    Standard z-score:

    .. math::

        z_i = \\frac{x_i - \\bar{x}}{s}

    Modified z-score (MAD-based) [3]_:

    .. math::

        M_i = \\frac{0.6745 \\, (x_i - \\tilde{x})}{\\text{MAD}}

    where :math:`\\text{MAD} = \\text{median}(|x_i - \\tilde{x}|)` and
    :math:`0.6745 \\approx \\Phi^{-1}(0.75)` is the consistency constant
    for the normal distribution.

    Parameters
    ----------
    method : {"zscore", "modified_zscore"}, optional
        Scoring method. Default is ``"modified_zscore"``.
    rolling_window : int or None, optional
        If not ``None``, statistics are computed over a rolling window
        of this many recent samples instead of the full training set.
        Default is ``None`` (use the full training set).

    Attributes
    ----------
    mean_ : numpy.ndarray or None
        Per-feature mean from training data (z-score method).
    std_ : numpy.ndarray or None
        Per-feature standard deviation from training data.
    median_ : numpy.ndarray or None
        Per-feature median from training data (modified z-score method).
    mad_ : numpy.ndarray or None
        Per-feature MAD from training data.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 3))
    >>> detector = StatisticalAnomalyDetector(method="modified_zscore")
    >>> detector.fit(X)
    StatisticalAnomalyDetector(method='modified_zscore', rolling_window=None)
    >>> mask = detector.detect(X, threshold=3.0)
    >>> mask.shape
    (100, 3)
    """

    _CONSISTENCY_CONSTANT: float = 0.6745
    _SUPPORTED_METHODS = ("zscore", "modified_zscore")

    def __init__(
        self,
        method: str = "modified_zscore",
        rolling_window: Optional[int] = None,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise ValidationError(
                message=(
                    f"'method' must be one of {self._SUPPORTED_METHODS}, "
                    f"got {method!r}."
                ),
                parameter="method",
            )
        if rolling_window is not None:
            if not isinstance(rolling_window, int) or rolling_window < 2:
                raise ValidationError(
                    message=(
                        "'rolling_window' must be an integer >= 2, "
                        f"got {rolling_window!r}."
                    ),
                    parameter="rolling_window",
                )

        self.method: str = method
        self.rolling_window: Optional[int] = rolling_window

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.mad_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self._is_fitted: bool = False
        self._X_train: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike) -> "StatisticalAnomalyDetector":
        """Compute reference statistics from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reference (assumed normal) observations.

        Returns
        -------
        self
            The fitted detector.

        Raises
        ------
        ValidationError
            If *X* has fewer than 2 samples.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] < 2:
            raise ValidationError(
                message=(
                    f"At least 2 samples are required, got {X_arr.shape[0]}."
                ),
                parameter="X",
            )

        self.n_features_in_ = X_arr.shape[1]

        if self.rolling_window is not None:
            self._X_train = X_arr.copy()

        self._compute_reference_stats(X_arr)
        self._is_fitted = True
        return self

    def detect(
        self,
        X: ArrayLike,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """Flag anomalous values in each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations to test.
        threshold : float, optional
            Number of standard deviations (or modified z-score units)
            beyond which a value is classified as anomalous. Default
            is ``3.0``.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            Boolean mask where ``True`` indicates an anomaly.

        Raises
        ------
        ComputationError
            If the detector has not been fitted.
        """
        if not self._is_fitted:
            raise ComputationError(
                "StatisticalAnomalyDetector has not been fitted. "
                "Call 'fit' before 'detect'."
            )

        scores = self._compute_zscores(X)
        return np.abs(scores) > threshold

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_reference_stats(self, X: np.ndarray) -> None:
        """Compute mean/std or median/MAD from the reference data.

        Parameters
        ----------
        X : numpy.ndarray
            Reference array.
        """
        if self.rolling_window is not None:
            X = X[-self.rolling_window :]

        if self.method == "zscore":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0, ddof=1)
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        else:
            self.median_ = np.median(X, axis=0)
            deviations = np.abs(X - self.median_)
            self.mad_ = np.median(deviations, axis=0)
            self.mad_ = np.where(self.mad_ == 0, 1.0, self.mad_)

    def _compute_zscores(self, X: ArrayLike) -> np.ndarray:
        """Return z-scores (or modified z-scores) for *X*.

        Parameters
        ----------
        X : array-like
            Observations.

        Returns
        -------
        numpy.ndarray
            Score array with same shape as *X*.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self.method == "zscore":
            return (X_arr - self.mean_) / self.std_  # type: ignore[operator]
        else:
            return (
                self._CONSISTENCY_CONSTANT
                * (X_arr - self.median_)  # type: ignore[operator]
                / self.mad_  # type: ignore[operator]
            )

    def __repr__(self) -> str:
        return (
            f"StatisticalAnomalyDetector("
            f"method={self.method!r}, "
            f"rolling_window={self.rolling_window!r})"
        )
