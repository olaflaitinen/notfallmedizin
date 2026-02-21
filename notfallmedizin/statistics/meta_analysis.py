# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Meta-analysis methods.

Implements fixed-effects (inverse-variance) and random-effects
(DerSimonian-Laird) meta-analysis, with heterogeneity assessment
and funnel plot / Egger's test for publication bias.

References
----------
DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
    Controlled Clinical Trials, 7(3), 177-188.
Egger, M., et al. (1997). Bias in meta-analysis detected by a simple,
    graphical test. BMJ, 315(7109), 629-634.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from notfallmedizin.core.exceptions import InsufficientDataError, ValidationError


@dataclass(frozen=True)
class MetaAnalysisResult:
    """Meta-analysis pooled result.

    Attributes
    ----------
    pooled_effect : float
    pooled_se : float
    confidence_interval : tuple
    z_statistic : float
    p_value : float
    tau_squared : float
        Between-study variance (random effects only; 0 for fixed).
    i_squared : float
        Percentage of variability due to heterogeneity.
    q_statistic : float
        Cochran's Q.
    q_p_value : float
    """

    pooled_effect: float
    pooled_se: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    z_statistic: float = 0.0
    p_value: float = 1.0
    tau_squared: float = 0.0
    i_squared: float = 0.0
    q_statistic: float = 0.0
    q_p_value: float = 1.0


class FixedEffectsMetaAnalysis:
    """Fixed-effects (inverse-variance weighted) meta-analysis."""

    def fit(
        self,
        effect_sizes: np.ndarray,
        standard_errors: np.ndarray,
        alpha: float = 0.05,
    ) -> MetaAnalysisResult:
        """Pool effect sizes using inverse-variance weighting.

        Parameters
        ----------
        effect_sizes : np.ndarray
        standard_errors : np.ndarray
        alpha : float

        Returns
        -------
        MetaAnalysisResult
        """
        y = np.asarray(effect_sizes, dtype=np.float64)
        se = np.asarray(standard_errors, dtype=np.float64)
        self._validate(y, se)

        w = 1.0 / (se ** 2)
        pooled = float(np.sum(w * y) / np.sum(w))
        pooled_se = float(1.0 / np.sqrt(np.sum(w)))

        z = pooled / pooled_se if pooled_se > 0 else 0.0
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (
            round(pooled - z_alpha * pooled_se, 6),
            round(pooled + z_alpha * pooled_se, 6),
        )

        q, q_p, i_sq = self._heterogeneity(y, se, w, pooled)

        return MetaAnalysisResult(
            pooled_effect=round(pooled, 6),
            pooled_se=round(pooled_se, 6),
            confidence_interval=ci,
            z_statistic=round(z, 4),
            p_value=round(p, 6),
            tau_squared=0.0,
            i_squared=round(i_sq, 2),
            q_statistic=round(q, 4),
            q_p_value=round(q_p, 6),
        )

    @staticmethod
    def _heterogeneity(
        y: np.ndarray,
        se: np.ndarray,
        w: np.ndarray,
        pooled: float,
    ) -> Tuple[float, float, float]:
        k = len(y)
        q = float(np.sum(w * (y - pooled) ** 2))
        q_p = 1 - sp_stats.chi2.cdf(q, df=max(k - 1, 1))
        i_sq = max(0.0, (q - (k - 1)) / q * 100) if q > 0 else 0.0
        return q, q_p, i_sq

    @staticmethod
    def _validate(y: np.ndarray, se: np.ndarray) -> None:
        if len(y) < 2:
            raise InsufficientDataError("At least 2 studies required.")
        if len(y) != len(se):
            raise ValidationError("effect_sizes and standard_errors must match.")
        if np.any(se <= 0):
            raise ValidationError("All standard errors must be positive.")


class RandomEffectsMetaAnalysis:
    """DerSimonian-Laird random-effects meta-analysis."""

    def fit(
        self,
        effect_sizes: np.ndarray,
        standard_errors: np.ndarray,
        alpha: float = 0.05,
    ) -> MetaAnalysisResult:
        """Pool effect sizes with random-effects weights.

        Parameters
        ----------
        effect_sizes : np.ndarray
        standard_errors : np.ndarray
        alpha : float

        Returns
        -------
        MetaAnalysisResult
        """
        y = np.asarray(effect_sizes, dtype=np.float64)
        se = np.asarray(standard_errors, dtype=np.float64)
        if len(y) < 2:
            raise InsufficientDataError("At least 2 studies required.")
        if len(y) != len(se):
            raise ValidationError("effect_sizes and standard_errors must match.")
        if np.any(se <= 0):
            raise ValidationError("All standard errors must be positive.")

        w = 1.0 / (se ** 2)
        pooled_fixed = float(np.sum(w * y) / np.sum(w))

        k = len(y)
        q = float(np.sum(w * (y - pooled_fixed) ** 2))
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0

        w_star = 1.0 / (se ** 2 + tau2)
        pooled = float(np.sum(w_star * y) / np.sum(w_star))
        pooled_se = float(1.0 / np.sqrt(np.sum(w_star)))

        z = pooled / pooled_se if pooled_se > 0 else 0.0
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (
            round(pooled - z_alpha * pooled_se, 6),
            round(pooled + z_alpha * pooled_se, 6),
        )

        q_p = 1 - sp_stats.chi2.cdf(q, df=max(k - 1, 1))
        i_sq = max(0.0, (q - (k - 1)) / q * 100) if q > 0 else 0.0

        return MetaAnalysisResult(
            pooled_effect=round(pooled, 6),
            pooled_se=round(pooled_se, 6),
            confidence_interval=ci,
            z_statistic=round(z, 4),
            p_value=round(p, 6),
            tau_squared=round(tau2, 6),
            i_squared=round(i_sq, 2),
            q_statistic=round(q, 4),
            q_p_value=round(q_p, 6),
        )


class FunnelPlotData:
    """Funnel plot generation and Egger's regression test."""

    @staticmethod
    def generate(
        effect_sizes: np.ndarray,
        standard_errors: np.ndarray,
        pooled_effect: Optional[float] = None,
    ) -> dict:
        """Generate data for a funnel plot.

        Parameters
        ----------
        effect_sizes : np.ndarray
        standard_errors : np.ndarray
        pooled_effect : float, optional

        Returns
        -------
        dict
            Keys: effects, se, pooled, pseudo_ci_lower_95, pseudo_ci_upper_95.
        """
        y = np.asarray(effect_sizes, dtype=np.float64)
        se = np.asarray(standard_errors, dtype=np.float64)

        if pooled_effect is None:
            w = 1.0 / (se ** 2)
            pooled_effect = float(np.sum(w * y) / np.sum(w))

        se_range = np.linspace(0.001, float(np.max(se)) * 1.1, 100)
        ci_lo = pooled_effect - 1.96 * se_range
        ci_hi = pooled_effect + 1.96 * se_range

        return {
            "effects": y.tolist(),
            "standard_errors": se.tolist(),
            "pooled": pooled_effect,
            "pseudo_ci_lower_95": ci_lo.tolist(),
            "pseudo_ci_upper_95": ci_hi.tolist(),
            "pseudo_ci_se": se_range.tolist(),
        }

    @staticmethod
    def eggers_test(
        effect_sizes: np.ndarray,
        standard_errors: np.ndarray,
    ) -> dict:
        """Egger's regression test for funnel plot asymmetry.

        Regresses standardised effect (y/se) on precision (1/se).
        The intercept tests for asymmetry.

        Parameters
        ----------
        effect_sizes : np.ndarray
        standard_errors : np.ndarray

        Returns
        -------
        dict
            Keys: intercept, intercept_se, t_statistic, p_value, significant.
        """
        y = np.asarray(effect_sizes, dtype=np.float64)
        se = np.asarray(standard_errors, dtype=np.float64)
        if len(y) < 3:
            raise InsufficientDataError("At least 3 studies for Egger's test.")

        precision = 1.0 / se
        standardised = y / se

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            precision, standardised
        )

        t_stat = intercept / std_err if std_err > 0 else 0.0

        return {
            "intercept": round(intercept, 4),
            "intercept_se": round(std_err, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        }
