# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Survival analysis methods.

Implements Kaplan-Meier estimation with Greenwood's variance formula,
the log-rank test for comparing survival curves, and a Cox proportional
hazards model fitted by Newton-Raphson.

References
----------
Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from
    incomplete observations. JASA, 53(282), 457-481.
Cox, D. R. (1972). Regression models and life-tables. JRSS-B, 34(2),
    187-220.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from notfallmedizin.core.exceptions import InsufficientDataError, ValidationError


@dataclass(frozen=True)
class LogRankResult:
    """Log-rank test result.

    Attributes
    ----------
    test_statistic : float
    p_value : float
    degrees_of_freedom : int
    significant : bool
    """

    test_statistic: float
    p_value: float
    degrees_of_freedom: int = 1
    significant: bool = False


class KaplanMeierEstimator:
    """Kaplan-Meier product-limit survival estimator.

    Variance is computed via Greenwood's formula. Confidence intervals
    use the log-log transformation for bounded coverage.
    """

    def __init__(self) -> None:
        self._times: Optional[np.ndarray] = None
        self._survival: Optional[np.ndarray] = None
        self._variance: Optional[np.ndarray] = None
        self._n_events: Optional[np.ndarray] = None
        self._n_at_risk: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self,
        durations: np.ndarray,
        event_observed: np.ndarray,
    ) -> "KaplanMeierEstimator":
        """Fit the estimator.

        Parameters
        ----------
        durations : np.ndarray
            Time to event or censoring.
        event_observed : np.ndarray
            1 if event occurred, 0 if censored.

        Returns
        -------
        self
        """
        durations = np.asarray(durations, dtype=np.float64)
        event_observed = np.asarray(event_observed, dtype=np.int64)
        if len(durations) < 2:
            raise InsufficientDataError("At least 2 observations required.")
        if len(durations) != len(event_observed):
            raise ValidationError("durations and event_observed must have equal length.")

        order = np.argsort(durations)
        t_sorted = durations[order]
        e_sorted = event_observed[order]

        unique_times = np.unique(t_sorted[e_sorted == 1])
        n = len(durations)

        times_list = [0.0]
        survival_list = [1.0]
        greenwood_sum = 0.0
        var_list = [0.0]
        n_events_list = [0]
        n_risk_list = [n]

        current_s = 1.0
        for t in unique_times:
            n_i = int(np.sum(t_sorted >= t))
            d_i = int(np.sum((t_sorted == t) & (e_sorted == 1)))
            if n_i == 0:
                continue
            current_s *= (n_i - d_i) / n_i
            if n_i > d_i:
                greenwood_sum += d_i / (n_i * (n_i - d_i))
            var_i = current_s ** 2 * greenwood_sum

            times_list.append(t)
            survival_list.append(current_s)
            var_list.append(var_i)
            n_events_list.append(d_i)
            n_risk_list.append(n_i)

        self._times = np.array(times_list)
        self._survival = np.array(survival_list)
        self._variance = np.array(var_list)
        self._n_events = np.array(n_events_list)
        self._n_at_risk = np.array(n_risk_list)
        self.is_fitted = True
        return self

    def survival_function(self, alpha: float = 0.05) -> pd.DataFrame:
        """Return the survival function as a DataFrame.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()
        z = sp_stats.norm.ppf(1 - alpha / 2)

        s = self._survival
        v = self._variance
        se = np.sqrt(v)

        log_s = np.log(np.clip(s, 1e-15, None))
        ci_lower = np.exp(log_s - z * se / np.clip(s, 1e-15, None))
        ci_upper = np.exp(log_s + z * se / np.clip(s, 1e-15, None))
        ci_lower = np.clip(ci_lower, 0, 1)
        ci_upper = np.clip(ci_upper, 0, 1)

        return pd.DataFrame({
            "time": self._times,
            "survival_probability": s,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_at_risk": self._n_at_risk,
            "n_events": self._n_events,
        })

    def median_survival_time(self) -> float:
        """Return the median survival time.

        Returns
        -------
        float
            Time at which S(t) <= 0.5 for the first time, or NaN.
        """
        self._check_fitted()
        below = np.where(self._survival <= 0.5)[0]
        if len(below) == 0:
            return float("nan")
        return float(self._times[below[0]])

    def plot_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays suitable for step-function plotting.

        Returns
        -------
        tuple of (times, survival)
        """
        self._check_fitted()
        return self._times.copy(), self._survival.copy()

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise InsufficientDataError("Estimator has not been fitted.")


class LogRankTest:
    """Log-rank (Mantel-Cox) test for comparing two survival curves."""

    @staticmethod
    def test(
        durations_a: np.ndarray,
        events_a: np.ndarray,
        durations_b: np.ndarray,
        events_b: np.ndarray,
        alpha: float = 0.05,
    ) -> LogRankResult:
        """Perform the log-rank test.

        Parameters
        ----------
        durations_a, events_a : np.ndarray
            Group A survival data.
        durations_b, events_b : np.ndarray
            Group B survival data.
        alpha : float

        Returns
        -------
        LogRankResult
        """
        da = np.asarray(durations_a, dtype=np.float64)
        ea = np.asarray(events_a, dtype=np.int64)
        db = np.asarray(durations_b, dtype=np.float64)
        eb = np.asarray(events_b, dtype=np.int64)

        all_times = np.unique(np.concatenate([
            da[ea == 1], db[eb == 1]
        ]))

        o_a = 0.0
        e_a = 0.0
        var_sum = 0.0

        for t in all_times:
            n_a = np.sum(da >= t)
            n_b = np.sum(db >= t)
            n = n_a + n_b
            d_a = np.sum((da == t) & (ea == 1))
            d_b = np.sum((db == t) & (eb == 1))
            d = d_a + d_b

            if n == 0:
                continue

            e_ai = n_a * d / n
            o_a += d_a
            e_a += e_ai

            if n > 1:
                var_sum += (n_a * n_b * d * (n - d)) / (n ** 2 * (n - 1))

        if var_sum <= 0:
            return LogRankResult(
                test_statistic=0.0, p_value=1.0, significant=False
            )

        chi2 = (o_a - e_a) ** 2 / var_sum
        p_value = 1.0 - sp_stats.chi2.cdf(chi2, df=1)

        return LogRankResult(
            test_statistic=round(float(chi2), 4),
            p_value=round(float(p_value), 6),
            degrees_of_freedom=1,
            significant=p_value < alpha,
        )


class CoxPHModel:
    """Cox proportional hazards regression.

    Fits coefficients by maximising the partial log-likelihood via
    Newton-Raphson.  Baseline hazard is estimated with the Breslow
    estimator.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-9) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._beta: Optional[np.ndarray] = None
        self._se: Optional[np.ndarray] = None
        self._baseline_hazard: Optional[np.ndarray] = None
        self._baseline_times: Optional[np.ndarray] = None
        self._feature_names: Optional[list] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> "CoxPHModel":
        """Fit the Cox PH model using Newton-Raphson.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
        durations : np.ndarray of shape (n,)
        events : np.ndarray of shape (n,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.int64)

        n, p = X.shape
        beta = np.zeros(p)

        order = np.argsort(-durations)
        X_sorted = X[order]
        t_sorted = durations[order]
        e_sorted = events[order]

        for iteration in range(self.max_iter):
            risk_scores = np.exp(X_sorted @ beta)

            cumsum_risk = np.cumsum(risk_scores)
            cumsum_risk_x = np.cumsum(risk_scores[:, None] * X_sorted, axis=0)
            cumsum_risk_xx = np.zeros((n, p, p))
            for i in range(n):
                outer = np.outer(X_sorted[i], X_sorted[i])
                prev = cumsum_risk_xx[i - 1] if i > 0 else np.zeros((p, p))
                cumsum_risk_xx[i] = prev + risk_scores[i] * outer

            gradient = np.zeros(p)
            hessian = np.zeros((p, p))

            for i in range(n):
                if e_sorted[i] == 0:
                    continue
                denom = cumsum_risk[i]
                if denom <= 0:
                    continue
                mean_x = cumsum_risk_x[i] / denom
                gradient += X_sorted[i] - mean_x

                mean_xx = cumsum_risk_xx[i] / denom
                hessian -= mean_xx - np.outer(mean_x, mean_x)

            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break

            beta -= step

            if np.max(np.abs(step)) < self.tol:
                break

        self._beta = beta
        try:
            info_matrix = -hessian
            inv_info = np.linalg.inv(info_matrix)
            self._se = np.sqrt(np.diag(inv_info))
        except np.linalg.LinAlgError:
            self._se = np.full(p, np.nan)

        self._breslow_baseline(X_sorted, t_sorted, e_sorted, beta)
        self.is_fitted = True
        return self

    def _breslow_baseline(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        beta: np.ndarray,
    ) -> None:
        event_times = np.unique(durations[events == 1])
        risk_scores = np.exp(X @ beta)

        h0 = []
        times = []
        for t in np.sort(event_times):
            d_i = np.sum((durations == t) & (events == 1))
            at_risk = durations >= t
            denom = np.sum(risk_scores[at_risk])
            if denom > 0:
                h0.append(d_i / denom)
                times.append(t)

        self._baseline_hazard = np.array(h0)
        self._baseline_times = np.array(times)

    def predict_hazard_ratio(self, X: np.ndarray) -> np.ndarray:
        """Return hazard ratios relative to baseline.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        self._check_fitted()
        return np.exp(np.asarray(X, dtype=np.float64) @ self._beta)

    def predict_survival_function(
        self, X: np.ndarray, times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict survival probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
        times : np.ndarray, optional

        Returns
        -------
        np.ndarray of shape (n, len(times))
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        hr = np.exp(X @ self._beta)

        cum_h0 = np.cumsum(self._baseline_hazard)
        eval_times = times if times is not None else self._baseline_times

        result = np.zeros((X.shape[0], len(eval_times)))
        for j, t in enumerate(eval_times):
            idx = np.searchsorted(self._baseline_times, t, side="right") - 1
            ch = cum_h0[max(idx, 0)] if idx >= 0 else 0.0
            result[:, j] = np.exp(-ch * hr)

        return result

    def summary(self) -> pd.DataFrame:
        """Return a summary table of coefficients.

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()
        p = len(self._beta)
        z = self._beta / self._se
        p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(z)))
        hr = np.exp(self._beta)
        ci_lo = np.exp(self._beta - 1.96 * self._se)
        ci_hi = np.exp(self._beta + 1.96 * self._se)

        names = self._feature_names or [f"x{i}" for i in range(p)]
        return pd.DataFrame({
            "coef": self._beta,
            "se": self._se,
            "z": z,
            "p_value": p_vals,
            "hazard_ratio": hr,
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
        }, index=names)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise InsufficientDataError("CoxPHModel has not been fitted.")
