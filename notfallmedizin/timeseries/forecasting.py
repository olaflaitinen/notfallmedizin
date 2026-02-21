# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Time series forecasting for clinical data.

This module provides forecasting models designed for clinical vital sign
and laboratory time series. Three forecaster classes are offered:

1. :class:`ExponentialSmoothingForecaster` -- single, double (Holt),
   and triple (Holt-Winters) exponential smoothing with additive or
   multiplicative seasonality.

2. :class:`ARIMAForecaster` -- ARIMA wrapper that optionally performs
   automatic model selection via AIC/BIC grid search.  Requires
   ``statsmodels`` as an optional dependency.

3. :class:`VitalSignsForecaster` -- a multi-variate forecaster that
   clips predictions to physiological ranges and supports a simple
   Vector Autoregression (VAR) approach.

References
----------
.. [1] Hyndman RJ, Athanasopoulos G. Forecasting: Principles and
       Practice. 3rd ed. OTexts; 2021.
.. [2] Gardner ES Jr. Exponential smoothing: The state of the art.
       J Forecasting. 1985;4(1):1-28.
.. [3] Hamilton JD. Time Series Analysis. Princeton University Press;
       1994.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from notfallmedizin.core.base import BaseEstimator
from notfallmedizin.core.exceptions import (
    ComputationError,
    InsufficientDataError,
    ValidationError,
)


# ======================================================================
# Exponential Smoothing Forecaster
# ======================================================================


class ExponentialSmoothingForecaster(BaseEstimator):
    """Exponential smoothing forecaster with single, double, and triple modes.

    The smoothing variant is selected automatically based on the
    ``trend`` and ``seasonal`` constructor arguments:

    * ``trend=None, seasonal=None`` -- simple (single) exponential
      smoothing.
    * ``trend="add"|"mul", seasonal=None`` -- Holt (double) exponential
      smoothing.
    * ``trend="add"|"mul", seasonal="add"|"mul"`` -- Holt-Winters
      (triple) exponential smoothing.

    Parameters for ``alpha``, ``beta``, and ``gamma`` are estimated by
    minimising the sum of squared one-step-ahead errors using
    ``scipy.optimize.minimize``.

    Parameters
    ----------
    trend : {None, "add", "mul"}, optional
        Type of trend component.  ``None`` disables the trend.
        Default is ``None``.
    seasonal : {None, "add", "mul"}, optional
        Type of seasonal component.  ``None`` disables seasonality.
        Default is ``None``.
    seasonal_periods : int, optional
        Number of observations per seasonal cycle.  Required when
        ``seasonal`` is not ``None``.  Default is ``1``.
    damped : bool, optional
        If ``True``, apply a damping factor ``phi`` to the trend
        component.  Default is ``False``.

    Attributes
    ----------
    alpha_ : float
        Fitted level smoothing parameter.
    beta_ : float or None
        Fitted trend smoothing parameter (``None`` when trend is
        disabled).
    gamma_ : float or None
        Fitted seasonal smoothing parameter (``None`` when seasonality
        is disabled).
    phi_ : float or None
        Fitted damping parameter (``None`` when ``damped=False``).
    level_ : numpy.ndarray
        Fitted level component at each observation.
    trend_ : numpy.ndarray or None
        Fitted trend component at each observation.
    season_ : numpy.ndarray or None
        Fitted seasonal component at each observation.
    residuals_ : numpy.ndarray
        One-step-ahead forecast errors (observed - fitted).

    Notes
    -----
    The level update equation for additive trend and additive
    seasonality is:

    .. math::

        l_t = \\alpha (y_t - s_{t-m}) + (1 - \\alpha)(l_{t-1} + b_{t-1})

        b_t = \\beta (l_t - l_{t-1}) + (1 - \\beta) b_{t-1}

        s_t = \\gamma (y_t - l_t) + (1 - \\gamma) s_{t-m}

    where *m* is the seasonal period.

    For multiplicative seasonality the equations become:

    .. math::

        l_t = \\alpha (y_t / s_{t-m}) + (1 - \\alpha)(l_{t-1} + b_{t-1})

        b_t = \\beta (l_t - l_{t-1}) + (1 - \\beta) b_{t-1}

        s_t = \\gamma (y_t / l_t) + (1 - \\gamma) s_{t-m}
    """

    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: int = 1,
        damped: bool = False,
    ) -> None:
        super().__init__()
        if trend is not None and trend not in ("add", "mul"):
            raise ValidationError(
                f"trend must be None, 'add', or 'mul', got '{trend}'.",
                parameter="trend",
            )
        if seasonal is not None and seasonal not in ("add", "mul"):
            raise ValidationError(
                f"seasonal must be None, 'add', or 'mul', got '{seasonal}'.",
                parameter="seasonal",
            )
        if seasonal is not None and seasonal_periods < 2:
            raise ValidationError(
                "seasonal_periods must be >= 2 when seasonality is enabled.",
                parameter="seasonal_periods",
            )
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped = damped

        self.alpha_: float = 0.0
        self.beta_: Optional[float] = None
        self.gamma_: Optional[float] = None
        self.phi_: Optional[float] = None
        self.level_: np.ndarray = np.array([])
        self.trend_: Optional[np.ndarray] = None
        self.season_: Optional[np.ndarray] = None
        self.residuals_: np.ndarray = np.array([])
        self._values: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_values(
        self, y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Compute initial level, trend, and seasonal components.

        Parameters
        ----------
        y : numpy.ndarray
            Observed time series.

        Returns
        -------
        l0 : float
            Initial level.
        b0 : float
            Initial trend (0.0 when trend is disabled).
        s0 : numpy.ndarray
            Initial seasonal indices (empty when seasonality is disabled).
        """
        m = self.seasonal_periods
        if self.seasonal is not None:
            n_cycles = len(y) // m
            if n_cycles < 1:
                raise InsufficientDataError(
                    "Need at least one full seasonal cycle.",
                    n_samples=len(y),
                    n_required=m,
                )
            l0 = float(np.mean(y[:m]))
            if self.trend is not None and n_cycles >= 2:
                second_cycle_mean = float(np.mean(y[m : 2 * m]))
                b0 = (second_cycle_mean - l0) / m
            elif self.trend is not None:
                b0 = 0.0
            else:
                b0 = 0.0
            if self.seasonal == "add":
                s0 = np.array([y[j] - l0 for j in range(m)], dtype=np.float64)
            else:
                s0 = np.array(
                    [y[j] / l0 if l0 != 0.0 else 1.0 for j in range(m)],
                    dtype=np.float64,
                )
        else:
            l0 = float(y[0])
            b0 = float(y[1] - y[0]) if self.trend is not None and len(y) > 1 else 0.0
            s0 = np.array([], dtype=np.float64)
        return l0, b0, s0

    def _smooth(
        self,
        y: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
        l0: float,
        b0: float,
        s0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run exponential smoothing forward pass.

        Parameters
        ----------
        y : numpy.ndarray
            Observed values.
        alpha, beta, gamma, phi : float
            Smoothing and damping parameters.
        l0, b0 : float
            Initial level and trend.
        s0 : numpy.ndarray
            Initial seasonal indices.

        Returns
        -------
        level : numpy.ndarray
        trend : numpy.ndarray
        season : numpy.ndarray
        fitted : numpy.ndarray
        residuals : numpy.ndarray
        """
        n = len(y)
        m = self.seasonal_periods
        level = np.empty(n, dtype=np.float64)
        trend = np.empty(n, dtype=np.float64)
        season = np.empty(n + m, dtype=np.float64)
        fitted = np.empty(n, dtype=np.float64)

        if self.seasonal is not None:
            season[:m] = s0
        trend_prev = b0
        level_prev = l0

        for t in range(n):
            # one-step-ahead forecast
            if self.seasonal == "add":
                f_t = level_prev + phi * trend_prev + season[t]
            elif self.seasonal == "mul":
                f_t = (level_prev + phi * trend_prev) * season[t]
            else:
                f_t = level_prev + phi * trend_prev
            fitted[t] = f_t

            # level update
            if self.seasonal == "add":
                level[t] = alpha * (y[t] - season[t]) + (1.0 - alpha) * (
                    level_prev + phi * trend_prev
                )
            elif self.seasonal == "mul":
                s_t = season[t] if season[t] != 0.0 else 1.0
                level[t] = alpha * (y[t] / s_t) + (1.0 - alpha) * (
                    level_prev + phi * trend_prev
                )
            else:
                level[t] = alpha * y[t] + (1.0 - alpha) * (
                    level_prev + phi * trend_prev
                )

            # trend update
            if self.trend is not None:
                if self.trend == "add":
                    trend[t] = beta * (level[t] - level_prev) + (1.0 - beta) * phi * trend_prev
                else:
                    if level_prev != 0.0:
                        trend[t] = beta * (level[t] / level_prev) + (1.0 - beta) * phi * trend_prev
                    else:
                        trend[t] = (1.0 - beta) * phi * trend_prev
            else:
                trend[t] = 0.0

            # seasonal update
            if self.seasonal == "add":
                season[t + m] = gamma * (y[t] - level[t]) + (1.0 - gamma) * season[t]
            elif self.seasonal == "mul":
                l_t = level[t] if level[t] != 0.0 else 1.0
                season[t + m] = gamma * (y[t] / l_t) + (1.0 - gamma) * season[t]

            level_prev = level[t]
            trend_prev = trend[t]

        residuals = y - fitted
        return level, trend, season, fitted, residuals

    def _objective(
        self,
        params: np.ndarray,
        y: np.ndarray,
        l0: float,
        b0: float,
        s0: np.ndarray,
    ) -> float:
        """Sum of squared one-step-ahead errors for parameter optimisation.

        Parameters
        ----------
        params : numpy.ndarray
            Parameter vector [alpha, beta, gamma, phi] where unused
            entries are ignored.
        y : numpy.ndarray
            Observed values.
        l0, b0 : float
            Initial level and trend.
        s0 : numpy.ndarray
            Initial seasonal indices.

        Returns
        -------
        float
            Sum of squared residuals.
        """
        alpha = params[0]
        beta = params[1] if self.trend is not None else 0.0
        gamma = params[2] if self.seasonal is not None else 0.0
        phi = params[3] if self.damped else 1.0

        try:
            _, _, _, _, residuals = self._smooth(
                y, alpha, beta, gamma, phi, l0, b0, s0
            )
            sse = float(np.sum(residuals**2))
        except (FloatingPointError, ZeroDivisionError):
            sse = 1e20
        if not np.isfinite(sse):
            sse = 1e20
        return sse

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        values: ArrayLike,
        timestamps: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "ExponentialSmoothingForecaster":
        """Fit the exponential smoothing model to the observed data.

        Parameters
        ----------
        values : array-like of shape (n_samples,)
            Observed time series values.
        timestamps : array-like of shape (n_samples,) or None, optional
            Timestamps corresponding to each value. Currently stored
            but not used for computation.  Default is ``None``.
        **kwargs
            Unused. Present for API compatibility.

        Returns
        -------
        self
            The fitted forecaster.

        Raises
        ------
        InsufficientDataError
            If the series is too short for the selected model.
        ComputationError
            If parameter optimisation fails to converge.
        """
        y = np.asarray(values, dtype=np.float64).ravel()
        if len(y) < 2:
            raise InsufficientDataError(
                "At least 2 observations are required.",
                n_samples=len(y),
                n_required=2,
            )
        self._values = y.copy()

        l0, b0, s0 = self._initial_values(y)

        x0 = [0.3]
        bounds = [(1e-4, 1.0 - 1e-4)]
        if self.trend is not None:
            x0.append(0.1)
            bounds.append((1e-4, 1.0 - 1e-4))
        else:
            x0.append(0.0)
            bounds.append((0.0, 0.0 + 1e-12))
        if self.seasonal is not None:
            x0.append(0.1)
            bounds.append((1e-4, 1.0 - 1e-4))
        else:
            x0.append(0.0)
            bounds.append((0.0, 0.0 + 1e-12))
        if self.damped:
            x0.append(0.98)
            bounds.append((0.8, 1.0))
        else:
            x0.append(1.0)
            bounds.append((1.0 - 1e-12, 1.0))

        result = optimize.minimize(
            self._objective,
            x0=np.array(x0),
            args=(y, l0, b0, s0),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success and result.fun > 1e18:
            raise ComputationError(
                f"Exponential smoothing optimisation failed: {result.message}"
            )

        self.alpha_ = float(result.x[0])
        self.beta_ = float(result.x[1]) if self.trend is not None else None
        self.gamma_ = float(result.x[2]) if self.seasonal is not None else None
        self.phi_ = float(result.x[3]) if self.damped else None

        beta_val = self.beta_ if self.beta_ is not None else 0.0
        gamma_val = self.gamma_ if self.gamma_ is not None else 0.0
        phi_val = self.phi_ if self.phi_ is not None else 1.0

        lev, tr, sea, fitted, resid = self._smooth(
            y, self.alpha_, beta_val, gamma_val, phi_val, l0, b0, s0
        )
        self.level_ = lev
        self.trend_ = tr if self.trend is not None else None
        self.season_ = sea if self.seasonal is not None else None
        self.residuals_ = resid
        self._fitted_values = fitted
        self._l0 = l0
        self._b0 = b0
        self._s0 = s0

        self._set_fitted()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast future values.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast into the future.

        Returns
        -------
        numpy.ndarray of shape (horizon,)
            Point forecasts.

        Raises
        ------
        ValidationError
            If ``horizon`` is not a positive integer.
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValidationError(
                "horizon must be a positive integer.",
                parameter="horizon",
            )

        n = len(self._values)
        m = self.seasonal_periods
        last_level = self.level_[-1]
        last_trend = self.trend_[-1] if self.trend_ is not None else 0.0
        phi = self.phi_ if self.phi_ is not None else 1.0

        forecasts = np.empty(horizon, dtype=np.float64)
        for h in range(1, horizon + 1):
            if self.damped:
                phi_sum = np.sum([phi**i for i in range(1, h + 1)])
            else:
                phi_sum = float(h)

            if self.trend == "add" or self.trend is None:
                trend_component = last_trend * phi_sum
            else:
                trend_component = last_trend**phi_sum if last_trend > 0 else 0.0

            if self.seasonal is not None:
                season_arr = self.season_
                idx = n + h - m * (1 + (h - 1) // m)
                if idx < 0:
                    idx = idx % m
                if idx >= len(season_arr):
                    idx = idx % m
                s_h = season_arr[idx]
            else:
                s_h = 0.0

            if self.seasonal == "add":
                forecasts[h - 1] = last_level + trend_component + s_h
            elif self.seasonal == "mul":
                if self.trend == "mul":
                    forecasts[h - 1] = last_level * trend_component * s_h
                else:
                    forecasts[h - 1] = (last_level + trend_component) * s_h
            else:
                if self.trend == "mul":
                    forecasts[h - 1] = last_level * trend_component
                else:
                    forecasts[h - 1] = last_level + trend_component

        return forecasts

    def predict_interval(
        self,
        horizon: int,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast with prediction intervals.

        The prediction intervals are based on a Gaussian assumption
        with variance estimated from the in-sample residuals. For
        h-step-ahead forecasts the variance is scaled by h (random
        walk variance growth).

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.
        confidence : float, optional
            Confidence level for the prediction interval (e.g. 0.95
            for 95 % intervals). Default is ``0.95``.

        Returns
        -------
        lower : numpy.ndarray of shape (horizon,)
            Lower bounds of the prediction intervals.
        upper : numpy.ndarray of shape (horizon,)
            Upper bounds of the prediction intervals.
        """
        self._check_is_fitted()
        if not 0.0 < confidence < 1.0:
            raise ValidationError(
                "confidence must be between 0 and 1 (exclusive).",
                parameter="confidence",
            )

        point = self.predict(horizon)
        sigma = float(np.std(self.residuals_, ddof=1))
        z = stats.norm.ppf((1.0 + confidence) / 2.0)

        steps = np.arange(1, horizon + 1, dtype=np.float64)
        widths = z * sigma * np.sqrt(steps)

        lower = point - widths
        upper = point + widths
        return lower, upper


# ======================================================================
# ARIMA Forecaster
# ======================================================================


class ARIMAForecaster(BaseEstimator):
    """ARIMA forecaster with optional automatic model selection.

    This class wraps ``statsmodels.tsa.arima.model.ARIMA``.  Because
    ``statsmodels`` is an optional dependency, the constructor checks
    for its availability and raises :class:`ImportError` if it is not
    installed.

    When ``auto=True``, :meth:`fit` performs a grid search over
    ``p`` in ``[0, max_p]``, ``d`` in ``[0, max_d]``, and ``q`` in
    ``[0, max_q]``, selecting the order that minimises AIC (or BIC,
    depending on ``information_criterion``).

    Parameters
    ----------
    order : tuple of (int, int, int), optional
        ``(p, d, q)`` order of the ARIMA model. Default is ``(1, 1, 1)``.
    auto : bool, optional
        If ``True``, perform automatic model selection. Default is
        ``False``.
    max_p : int, optional
        Maximum AR order for auto selection. Default is ``5``.
    max_d : int, optional
        Maximum differencing order for auto selection. Default is ``2``.
    max_q : int, optional
        Maximum MA order for auto selection. Default is ``5``.
    information_criterion : {"aic", "bic"}, optional
        Criterion for model selection. Default is ``"aic"``.

    Attributes
    ----------
    model_ : object
        Fitted ``statsmodels`` ARIMA result (available after
        :meth:`fit`).
    order_ : tuple of (int, int, int)
        The ARIMA order of the fitted model.
    aic_ : float
        Akaike Information Criterion of the fitted model.
    bic_ : float
        Bayesian Information Criterion of the fitted model.

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        auto: bool = False,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        information_criterion: str = "aic",
    ) -> None:
        super().__init__()
        try:
            import statsmodels.tsa.arima.model  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ARIMAForecaster requires 'statsmodels'. "
                "Install it with: pip install statsmodels"
            ) from exc

        if information_criterion not in ("aic", "bic"):
            raise ValidationError(
                "information_criterion must be 'aic' or 'bic'.",
                parameter="information_criterion",
            )

        self.order = order
        self.auto = auto
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.information_criterion = information_criterion

        self.model_: Any = None
        self.order_: Tuple[int, int, int] = order
        self.aic_: float = np.inf
        self.bic_: float = np.inf
        self._values: np.ndarray = np.array([])

    def _fit_single(
        self, y: np.ndarray, order: Tuple[int, int, int]
    ) -> Any:
        """Fit a single ARIMA model and return the result object.

        Parameters
        ----------
        y : numpy.ndarray
            Observed values.
        order : tuple of (int, int, int)
            ``(p, d, q)`` order.

        Returns
        -------
        result
            Fitted ARIMA result from statsmodels.
        """
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(y, order=order)
        return model.fit()

    def fit(
        self,
        values: ArrayLike,
        order: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ) -> "ARIMAForecaster":
        """Fit the ARIMA model to the observed data.

        Parameters
        ----------
        values : array-like of shape (n_samples,)
            Observed time series values.
        order : tuple of (int, int, int) or None, optional
            Override for the ``(p, d, q)`` order. If ``None``, uses
            the instance attribute ``self.order``. Default is ``None``.
        **kwargs
            Unused. Present for API compatibility.

        Returns
        -------
        self
            The fitted forecaster.

        Raises
        ------
        InsufficientDataError
            If the series is too short.
        ComputationError
            If model fitting fails.
        """
        y = np.asarray(values, dtype=np.float64).ravel()
        if len(y) < 3:
            raise InsufficientDataError(
                "ARIMA requires at least 3 observations.",
                n_samples=len(y),
                n_required=3,
            )
        self._values = y.copy()

        if order is not None:
            self.order = order

        if self.auto:
            best_ic = np.inf
            best_result = None
            best_order: Tuple[int, int, int] = self.order

            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        if p == 0 and q == 0:
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                result = self._fit_single(y, (p, d, q))
                            ic = (
                                result.aic
                                if self.information_criterion == "aic"
                                else result.bic
                            )
                            if ic < best_ic:
                                best_ic = ic
                                best_result = result
                                best_order = (p, d, q)
                        except Exception:
                            continue

            if best_result is None:
                raise ComputationError(
                    "Auto ARIMA failed: no valid model found in the search grid."
                )
            self.model_ = best_result
            self.order_ = best_order
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model_ = self._fit_single(y, self.order)
                self.order_ = self.order
            except Exception as exc:
                raise ComputationError(
                    f"ARIMA fitting failed for order {self.order}: {exc}"
                ) from exc

        self.aic_ = float(self.model_.aic)
        self.bic_ = float(self.model_.bic)
        self._set_fitted()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast future values using the fitted ARIMA model.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        numpy.ndarray of shape (horizon,)
            Point forecasts.
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValidationError(
                "horizon must be a positive integer.",
                parameter="horizon",
            )
        forecast = self.model_.forecast(steps=horizon)
        return np.asarray(forecast, dtype=np.float64)

    def predict_interval(
        self,
        horizon: int,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast with prediction intervals via statsmodels.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.
        confidence : float, optional
            Confidence level. Default is ``0.95``.

        Returns
        -------
        lower : numpy.ndarray of shape (horizon,)
            Lower prediction bounds.
        upper : numpy.ndarray of shape (horizon,)
            Upper prediction bounds.
        """
        self._check_is_fitted()
        if not 0.0 < confidence < 1.0:
            raise ValidationError(
                "confidence must be between 0 and 1 (exclusive).",
                parameter="confidence",
            )
        fc = self.model_.get_forecast(steps=horizon)
        ci = fc.conf_int(alpha=1.0 - confidence)
        lower = np.asarray(ci.iloc[:, 0], dtype=np.float64)
        upper = np.asarray(ci.iloc[:, 1], dtype=np.float64)
        return lower, upper

    def summary(self) -> str:
        """Return a text summary of the fitted model.

        Returns
        -------
        str
            Statsmodels summary output.
        """
        self._check_is_fitted()
        return str(self.model_.summary())


# ======================================================================
# Vital Signs Forecaster (multivariate VAR)
# ======================================================================

_DEFAULT_VITAL_RANGES: Dict[str, Tuple[float, float]] = {
    "heart_rate": (20.0, 300.0),
    "systolic_bp": (40.0, 300.0),
    "diastolic_bp": (20.0, 200.0),
    "respiratory_rate": (2.0, 60.0),
    "spo2": (50.0, 100.0),
    "temperature": (25.0, 45.0),
    "map": (30.0, 250.0),
}


class VitalSignsForecaster:
    """Multi-variate vital signs forecaster with clinical constraints.

    Uses a Vector Autoregression (VAR) approach where each variable at
    time *t* is regressed on lagged values of all variables. Predictions
    are clipped to physiologically plausible ranges.

    The VAR(p) model is:

    .. math::

        \\mathbf{y}_t = \\mathbf{c} + A_1 \\mathbf{y}_{t-1}
                       + \\cdots + A_p \\mathbf{y}_{t-p} + \\mathbf{u}_t

    where :math:`A_i` are coefficient matrices estimated by ordinary
    least squares.

    Parameters
    ----------
    lag_order : int, optional
        Number of lags (p) for the VAR model. Default is ``2``.
    vital_ranges : dict of str to tuple of (float, float) or None, optional
        Mapping of variable name to ``(min, max)`` physiological range
        for output clipping.  If ``None``, the default clinical ranges
        are used.

    Attributes
    ----------
    coefficients_ : numpy.ndarray
        Fitted VAR coefficient matrix of shape ``(n_vars, n_vars * lag_order + 1)``.
    variable_names_ : list of str
        Names of the variables (columns).
    residuals_ : numpy.ndarray
        In-sample residuals.
    """

    def __init__(
        self,
        lag_order: int = 2,
        vital_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        if lag_order < 1:
            raise ValidationError(
                "lag_order must be >= 1.",
                parameter="lag_order",
            )
        self.lag_order = lag_order
        self.vital_ranges = (
            vital_ranges if vital_ranges is not None else dict(_DEFAULT_VITAL_RANGES)
        )
        self.is_fitted_: bool = False
        self.coefficients_: np.ndarray = np.array([])
        self.variable_names_: List[str] = []
        self.residuals_: np.ndarray = np.array([])
        self._last_observations: np.ndarray = np.array([])

    def _build_design_matrix(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct the lagged design matrix and response matrix.

        Parameters
        ----------
        data : numpy.ndarray of shape (T, n_vars)
            Multivariate time series.

        Returns
        -------
        X : numpy.ndarray of shape (T - lag_order, n_vars * lag_order + 1)
            Design matrix with intercept column appended.
        Y : numpy.ndarray of shape (T - lag_order, n_vars)
            Response matrix.
        """
        T, n_vars = data.shape
        p = self.lag_order
        n_rows = T - p
        X = np.ones((n_rows, n_vars * p + 1), dtype=np.float64)

        for lag in range(1, p + 1):
            col_start = (lag - 1) * n_vars
            col_end = lag * n_vars
            X[:, col_start:col_end] = data[p - lag : T - lag, :]

        X[:, -1] = 1.0  # intercept
        Y = data[p:, :]
        return X, Y

    def fit(
        self,
        data: ArrayLike,
        variable_names: Optional[List[str]] = None,
    ) -> "VitalSignsForecaster":
        """Fit the VAR model via ordinary least squares.

        Parameters
        ----------
        data : array-like of shape (T, n_vars)
            Multivariate time series where each column is a vital sign.
        variable_names : list of str or None, optional
            Column names. If ``None``, defaults to
            ``["var_0", "var_1", ...]``.

        Returns
        -------
        self
            The fitted forecaster.

        Raises
        ------
        InsufficientDataError
            If the number of observations is not sufficient for the
            chosen lag order.
        ComputationError
            If the OLS system is singular.
        """
        data_arr = np.asarray(data, dtype=np.float64)
        if data_arr.ndim == 1:
            data_arr = data_arr.reshape(-1, 1)

        T, n_vars = data_arr.shape
        if T <= self.lag_order:
            raise InsufficientDataError(
                f"Need more than {self.lag_order} observations for VAR({self.lag_order}).",
                n_samples=T,
                n_required=self.lag_order + 1,
            )

        if variable_names is not None:
            if len(variable_names) != n_vars:
                raise ValidationError(
                    f"variable_names length ({len(variable_names)}) does not match "
                    f"number of variables ({n_vars}).",
                    parameter="variable_names",
                )
            self.variable_names_ = list(variable_names)
        else:
            self.variable_names_ = [f"var_{i}" for i in range(n_vars)]

        X, Y = self._build_design_matrix(data_arr)

        try:
            # OLS: B = (X^T X)^{-1} X^T Y
            XtX = X.T @ X
            XtY = X.T @ Y
            self.coefficients_ = np.linalg.solve(XtX, XtY)  # shape: (n_vars*p+1, n_vars)
        except np.linalg.LinAlgError as exc:
            raise ComputationError(
                f"VAR coefficient estimation failed (singular matrix): {exc}"
            ) from exc

        fitted = X @ self.coefficients_
        self.residuals_ = Y - fitted
        self._last_observations = data_arr[-self.lag_order :, :].copy()
        self.is_fitted_ = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast future multivariate values.

        Predictions are clipped to the physiological ranges defined in
        ``self.vital_ranges``.

        Parameters
        ----------
        horizon : int
            Number of time steps to forecast.

        Returns
        -------
        numpy.ndarray of shape (horizon, n_vars)
            Forecasted values, clipped to physiological ranges.
        """
        if not self.is_fitted_:
            raise ComputationError(
                "VitalSignsForecaster has not been fitted. Call 'fit' first."
            )
        if horizon < 1:
            raise ValidationError(
                "horizon must be a positive integer.",
                parameter="horizon",
            )

        n_vars = self._last_observations.shape[1]
        p = self.lag_order
        history = self._last_observations.copy()  # shape: (p, n_vars)
        forecasts = np.empty((horizon, n_vars), dtype=np.float64)

        for h in range(horizon):
            x_row = np.ones(n_vars * p + 1, dtype=np.float64)
            for lag in range(1, p + 1):
                col_start = (lag - 1) * n_vars
                col_end = lag * n_vars
                idx = len(history) - lag
                x_row[col_start:col_end] = history[idx, :]
            x_row[-1] = 1.0

            y_hat = x_row @ self.coefficients_
            forecasts[h, :] = y_hat
            history = np.vstack([history, y_hat.reshape(1, -1)])

        forecasts = self._clip_to_ranges(forecasts)
        return forecasts

    def _clip_to_ranges(self, forecasts: np.ndarray) -> np.ndarray:
        """Clip each variable column to its physiological range.

        Parameters
        ----------
        forecasts : numpy.ndarray of shape (horizon, n_vars)
            Raw forecasted values.

        Returns
        -------
        numpy.ndarray
            Clipped forecasted values.
        """
        clipped = forecasts.copy()
        for i, name in enumerate(self.variable_names_):
            if name in self.vital_ranges:
                lo, hi = self.vital_ranges[name]
                clipped[:, i] = np.clip(clipped[:, i], lo, hi)
        return clipped

    def __repr__(self) -> str:
        return (
            f"VitalSignsForecaster(lag_order={self.lag_order}, "
            f"n_variables={len(self.variable_names_)})"
        )
