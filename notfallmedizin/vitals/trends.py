# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Vital signs trend analysis.

This module provides temporal trend detection, changepoint identification,
and variability quantification for univariate vital sign time series.

The primary interface is :class:`VitalSignsTrendAnalyzer`, which operates
on a (timestamps, values) pair and exposes non-parametric trend tests,
CUSUM changepoint detection, and several smoothing operators.

Classes
-------
TrendDirection
    Enumeration of trend directions.
TrendResult
    Immutable result container for trend detection output.
VitalSignsTrendAnalyzer
    Stateful analyzer for a single vital sign channel.

References
----------
.. [1] Mann HB. Non-parametric tests against trend. Econometrica.
       1945;13(3):245-259.
.. [2] Kendall MG. Rank Correlation Methods. 4th ed. Charles Griffin;
       1975.
.. [3] Page ES. Continuous inspection schemes. Biometrika.
       1954;41(1-2):100-115.
.. [4] Richman JS, Moorman JR. Physiological time-series analysis using
       approximate entropy and sample entropy. Am J Physiol Heart Circ
       Physiol. 2000;278(6):H2039-H2049.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sp_stats

from notfallmedizin.core.exceptions import (
    ComputationError,
    ValidationError,
)


# ======================================================================
# Data Structures
# ======================================================================


class TrendDirection(enum.Enum):
    """Direction of a monotonic trend."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass(frozen=True)
class TrendResult:
    """Immutable container for trend detection output.

    Parameters
    ----------
    direction : TrendDirection
        Detected direction of the trend.
    slope : float
        Estimated slope (Theil-Sen estimator or OLS, depending on the
        method used internally).
    p_value : float
        Two-sided p-value for the null hypothesis of no trend
        (Mann-Kendall test).
    r_squared : float
        Coefficient of determination from a linear fit of the values
        against a numeric time index.
    confidence_interval : tuple of (float, float)
        95 % confidence interval for the slope.
    tau : float
        Kendall tau correlation coefficient.
    """

    direction: TrendDirection
    slope: float
    p_value: float
    r_squared: float
    confidence_interval: Tuple[float, float]
    tau: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation.

        Returns
        -------
        dict
            All fields serialized to basic Python types.
        """
        return {
            "direction": self.direction.value,
            "slope": self.slope,
            "p_value": self.p_value,
            "r_squared": self.r_squared,
            "confidence_interval": self.confidence_interval,
            "tau": self.tau,
        }


# ======================================================================
# Trend Analyzer
# ======================================================================


class VitalSignsTrendAnalyzer:
    """Analyze temporal trends in a single vital sign channel.

    After calling :meth:`fit` with timestamps and values, the analyzer
    offers:

    * **Trend detection** via the Mann-Kendall test.
    * **Changepoint detection** via the CUSUM algorithm.
    * **Variability metrics** (standard deviation, coefficient of
      variation, interquartile range, sample entropy).
    * **Smoothing** via simple moving average or exponential smoothing.

    Parameters
    ----------
    alpha : float, optional
        Significance level for the Mann-Kendall test used in
        :meth:`detect_trend`. Default is ``0.05``.

    Attributes
    ----------
    timestamps_ : numpy.ndarray or None
        Numeric representation of the fitted timestamps.
    values_ : numpy.ndarray or None
        Fitted values array.
    n_samples_ : int
        Number of observations in the fitted series.

    Examples
    --------
    >>> import numpy as np
    >>> analyzer = VitalSignsTrendAnalyzer()
    >>> t = np.arange(50, dtype=float)
    >>> v = 80.0 + 0.3 * t + np.random.default_rng(0).normal(0, 1, 50)
    >>> analyzer.fit(t, v)
    VitalSignsTrendAnalyzer(alpha=0.05)
    >>> result = analyzer.detect_trend()
    >>> result.direction
    <TrendDirection.INCREASING: 'increasing'>
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValidationError(
                message=f"'alpha' must be in (0, 1), got {alpha}.",
                parameter="alpha",
            )
        self.alpha: float = alpha
        self.timestamps_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.n_samples_: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
    ) -> "VitalSignsTrendAnalyzer":
        """Store the time series for subsequent analysis.

        Parameters
        ----------
        timestamps : array-like of shape (n,)
            Observation times. Can be numeric or ``datetime64``. If
            datetime-like, they are converted to seconds since the
            first observation.
        values : array-like of shape (n,)
            Observed values corresponding to each timestamp.

        Returns
        -------
        self
            The fitted analyzer.

        Raises
        ------
        ValidationError
            If inputs have mismatched lengths or fewer than 3 elements.
        """
        t_arr = np.asarray(timestamps)
        v_arr = np.asarray(values, dtype=np.float64)

        if t_arr.ndim != 1 or v_arr.ndim != 1:
            raise ValidationError(
                message="'timestamps' and 'values' must be 1-D arrays.",
                parameter="timestamps",
            )
        if t_arr.shape[0] != v_arr.shape[0]:
            raise ValidationError(
                message=(
                    f"Length mismatch: timestamps has {t_arr.shape[0]} "
                    f"elements, values has {v_arr.shape[0]}."
                ),
                parameter="values",
            )
        if t_arr.shape[0] < 3:
            raise ValidationError(
                message=(
                    f"At least 3 observations are required, "
                    f"got {t_arr.shape[0]}."
                ),
                parameter="timestamps",
            )

        if np.issubdtype(t_arr.dtype, np.datetime64):
            t_numeric = (t_arr - t_arr[0]) / np.timedelta64(1, "s")
            self.timestamps_ = t_numeric.astype(np.float64)
        else:
            self.timestamps_ = t_arr.astype(np.float64)

        self.values_ = v_arr
        self.n_samples_ = v_arr.shape[0]
        return self

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trend(self) -> TrendResult:
        """Detect monotonic trend using the Mann-Kendall test.

        The Mann-Kendall test [1]_ [2]_ is a non-parametric hypothesis
        test for the presence of a monotonic trend in a time series. The
        test statistic S is:

        .. math::

            S = \\sum_{k=1}^{n-1} \\sum_{j=k+1}^{n}
                \\operatorname{sgn}(x_j - x_k)

        Under the null hypothesis of no trend, S is approximately
        normally distributed for n >= 10 with:

        .. math::

            \\operatorname{Var}(S) = \\frac{n(n-1)(2n+5)}{18}

        The slope is estimated using the Theil-Sen estimator (median of
        pairwise slopes).

        Returns
        -------
        TrendResult
            Detection result including direction, slope, p-value,
            R-squared, confidence interval, and Kendall tau.

        Raises
        ------
        ComputationError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        t = self.timestamps_
        v = self.values_

        tau, mk_p = self._mann_kendall(v)  # type: ignore[arg-type]

        slope, intercept, lo_slope, hi_slope = sp_stats.theilslopes(
            v, t  # type: ignore[arg-type]
        )

        v_pred = intercept + slope * t  # type: ignore[operator]
        ss_res = float(np.sum((v - v_pred) ** 2))  # type: ignore[operator]
        ss_tot = float(np.sum((v - np.mean(v)) ** 2))  # type: ignore[arg-type]
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

        if mk_p < self.alpha:
            direction = (
                TrendDirection.INCREASING if tau > 0 else TrendDirection.DECREASING
            )
        else:
            direction = TrendDirection.STABLE

        return TrendResult(
            direction=direction,
            slope=float(slope),
            p_value=float(mk_p),
            r_squared=r_squared,
            confidence_interval=(float(lo_slope), float(hi_slope)),
            tau=float(tau),
        )

    # ------------------------------------------------------------------
    # Changepoint detection
    # ------------------------------------------------------------------

    def detect_changepoints(
        self,
        method: str = "cusum",
        threshold: Optional[float] = None,
        drift: float = 0.0,
    ) -> List[int]:
        """Identify changepoints in the time series.

        Parameters
        ----------
        method : {"cusum"}, optional
            Changepoint detection algorithm. Default is ``"cusum"``.
        threshold : float or None, optional
            Decision threshold for the CUSUM statistic. If ``None``,
            it is set to ``4 * std(values)``, a commonly used heuristic.
        drift : float, optional
            Allowance (slack) parameter for the CUSUM algorithm. Shifts
            below this magnitude are ignored. Default is ``0.0``.

        Returns
        -------
        list of int
            Indices at which changepoints were detected, sorted in
            ascending order.

        Raises
        ------
        ComputationError
            If :meth:`fit` has not been called.
        ValidationError
            If *method* is not supported.

        Notes
        -----
        The tabular CUSUM [3]_ maintains two running sums, one for
        detecting upward shifts and one for downward shifts:

        .. math::

            S_i^+ = \\max(0,\\; S_{i-1}^+ + (x_i - \\bar{x}) - k)

            S_i^- = \\max(0,\\; S_{i-1}^- - (x_i - \\bar{x}) - k)

        A changepoint is signalled when either sum exceeds the
        threshold *h*.
        """
        self._check_fitted()

        if method != "cusum":
            raise ValidationError(
                message=f"Unsupported changepoint method: {method!r}.",
                parameter="method",
            )

        v = self.values_
        v_mean = float(np.mean(v))  # type: ignore[arg-type]
        v_std = float(np.std(v, ddof=1))  # type: ignore[arg-type]

        if threshold is None:
            threshold = 4.0 * v_std if v_std > 0 else 1.0

        s_pos = 0.0
        s_neg = 0.0
        changepoints: List[int] = []

        for i in range(len(v)):  # type: ignore[arg-type]
            deviation = v[i] - v_mean  # type: ignore[index]
            s_pos = max(0.0, s_pos + deviation - drift)
            s_neg = max(0.0, s_neg - deviation - drift)

            if s_pos > threshold or s_neg > threshold:
                changepoints.append(i)
                s_pos = 0.0
                s_neg = 0.0

        return changepoints

    # ------------------------------------------------------------------
    # Variability metrics
    # ------------------------------------------------------------------

    def calculate_variability(self) -> Dict[str, float]:
        """Compute variability metrics for the fitted time series.

        Returns
        -------
        dict of str to float
            Keys:

            * ``"std"`` -- sample standard deviation.
            * ``"cv"`` -- coefficient of variation (std / mean).
            * ``"iqr"`` -- interquartile range (Q3 - Q1).
            * ``"entropy"`` -- sample entropy (m=2, r=0.2*std).

        Raises
        ------
        ComputationError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        v = self.values_

        std_val = float(np.std(v, ddof=1))  # type: ignore[arg-type]
        mean_val = float(np.mean(v))  # type: ignore[arg-type]
        cv_val = std_val / abs(mean_val) if mean_val != 0.0 else float("inf")
        q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))  # type: ignore[arg-type]
        iqr_val = q3 - q1

        try:
            entropy_val = self._sample_entropy(v, m=2, r=0.2 * std_val)  # type: ignore[arg-type]
        except ComputationError:
            entropy_val = float("nan")

        return {
            "std": std_val,
            "cv": cv_val,
            "iqr": iqr_val,
            "entropy": entropy_val,
        }

    # ------------------------------------------------------------------
    # Smoothing operators
    # ------------------------------------------------------------------

    def moving_average(self, window: int) -> np.ndarray:
        """Compute a simple (unweighted) moving average.

        Parameters
        ----------
        window : int
            Width of the sliding window. Must satisfy
            ``2 <= window <= n_samples``.

        Returns
        -------
        numpy.ndarray of shape (n_samples - window + 1,)
            Smoothed values.

        Raises
        ------
        ValidationError
            If *window* is out of range.
        ComputationError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        self._validate_window(window)

        kernel = np.ones(window) / window
        return np.convolve(self.values_, kernel, mode="valid")  # type: ignore[arg-type]

    def exponential_smoothing(self, alpha: float) -> np.ndarray:
        """Apply single exponential smoothing (SES).

        .. math::

            s_0 = x_0, \\qquad
            s_t = \\alpha \\, x_t + (1 - \\alpha) \\, s_{t-1}

        Parameters
        ----------
        alpha : float
            Smoothing factor in (0, 1]. Values closer to 1 give more
            weight to recent observations.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Exponentially smoothed values.

        Raises
        ------
        ValidationError
            If *alpha* is not in (0, 1].
        ComputationError
            If :meth:`fit` has not been called.
        """
        self._check_fitted()
        if not 0.0 < alpha <= 1.0:
            raise ValidationError(
                message=f"'alpha' must be in (0, 1], got {alpha}.",
                parameter="alpha",
            )

        v = self.values_
        out = np.empty_like(v)
        out[0] = v[0]  # type: ignore[index]
        for i in range(1, len(v)):  # type: ignore[arg-type]
            out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]  # type: ignore[index]
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called."""
        if self.values_ is None or self.timestamps_ is None:
            raise ComputationError(
                "VitalSignsTrendAnalyzer has not been fitted. "
                "Call 'fit' before using this method."
            )

    def _validate_window(self, window: int) -> None:
        """Validate a window size parameter.

        Parameters
        ----------
        window : int
            The window size.

        Raises
        ------
        ValidationError
            If *window* is not in [2, n_samples].
        """
        if not isinstance(window, int) or window < 2:
            raise ValidationError(
                message=f"'window' must be an integer >= 2, got {window!r}.",
                parameter="window",
            )
        if window > self.n_samples_:
            raise ValidationError(
                message=(
                    f"'window' ({window}) exceeds the number of "
                    f"observations ({self.n_samples_})."
                ),
                parameter="window",
            )

    @staticmethod
    def _mann_kendall(x: np.ndarray) -> Tuple[float, float]:
        """Perform the Mann-Kendall trend test.

        Parameters
        ----------
        x : numpy.ndarray of shape (n,)
            Univariate time series.

        Returns
        -------
        tuple of (float, float)
            ``(tau, p_value)`` where tau is the Kendall rank correlation
            coefficient and p_value is the two-sided significance.
        """
        n = len(x)
        s = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                diff = x[j] - x[k]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1

        n_pairs = n * (n - 1) / 2.0
        tau = s / n_pairs if n_pairs > 0 else 0.0

        var_s = n * (n - 1) * (2 * n + 5) / 18.0

        if var_s == 0.0:
            return tau, 1.0

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        p_value = 2.0 * sp_stats.norm.sf(abs(z))
        return tau, float(p_value)

    @staticmethod
    def _sample_entropy(
        x: np.ndarray,
        m: int = 2,
        r: float = 0.2,
    ) -> float:
        """Compute sample entropy (SampEn) of a time series.

        Sample entropy [4]_ quantifies the regularity (predictability)
        of a time series. Lower values indicate more regularity.

        .. math::

            \\text{SampEn}(m, r, N) = -\\ln \\frac{A}{B}

        where *A* is the number of template matches of length *m+1*
        and *B* is the number of template matches of length *m*, both
        within tolerance *r*.

        Parameters
        ----------
        x : numpy.ndarray of shape (n,)
            Input time series.
        m : int
            Embedding dimension.
        r : float
            Tolerance (typically 0.1--0.25 times the standard deviation
            of *x*).

        Returns
        -------
        float
            Sample entropy value.

        Raises
        ------
        ComputationError
            If no template matches are found (B = 0 or A = 0).
        """
        n = len(x)
        if n < m + 2:
            raise ComputationError(
                f"Time series too short for SampEn: need >= {m + 2} "
                f"samples, got {n}."
            )

        def _count_matches(dim: int) -> int:
            templates = np.array(
                [x[i : i + dim] for i in range(n - dim + 1)]
            )
            count = 0
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) <= r:
                        count += 1
            return count

        b = _count_matches(m)
        a = _count_matches(m + 1)

        if b == 0:
            raise ComputationError(
                "No template matches found at embedding dimension m; "
                "cannot compute sample entropy."
            )
        if a == 0:
            return float("inf")

        return -np.log(a / b)

    def __repr__(self) -> str:
        return f"VitalSignsTrendAnalyzer(alpha={self.alpha})"
