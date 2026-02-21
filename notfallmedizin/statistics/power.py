# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Statistical power analysis and multiplicity correction.

Provides sample size, power, and effect size calculations for common
test types, plus Bonferroni, Holm, and Benjamini-Hochberg procedures.

References
----------
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
    (2nd ed.). Lawrence Erlbaum Associates.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate. JRSS-B, 57(1), 289-300.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy import stats as sp_stats

from notfallmedizin.core.exceptions import ValidationError


class PowerAnalyzer:
    """Power analysis for common hypothesis tests.

    Supported test types: ``two_sample_t``, ``paired_t``,
    ``one_sample_t``, ``chi_squared``, ``proportion``.
    """

    _VALID_TESTS = {
        "two_sample_t",
        "paired_t",
        "one_sample_t",
        "chi_squared",
        "proportion",
    }

    def calculate_sample_size(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: str = "two_sample_t",
        ratio: float = 1.0,
    ) -> int:
        """Calculate the required sample size (per group for two-sample tests).

        Parameters
        ----------
        effect_size : float
            Cohen's d for t-tests, w for chi-squared, h for proportions.
        alpha : float
        power : float
        test_type : str
        ratio : float
            Allocation ratio n2/n1 for two-sample tests.

        Returns
        -------
        int
            Sample size per group.
        """
        self._validate_inputs(effect_size, alpha, power, test_type)

        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)

        if test_type in ("two_sample_t", "paired_t"):
            n = ((z_alpha + z_beta) ** 2 * (1 + 1.0 / ratio)) / effect_size ** 2
        elif test_type == "one_sample_t":
            n = ((z_alpha + z_beta) / effect_size) ** 2
        elif test_type == "chi_squared":
            n = ((z_alpha + z_beta) / effect_size) ** 2
        elif test_type == "proportion":
            n = ((z_alpha + z_beta) / effect_size) ** 2
        else:
            raise ValidationError(f"Unknown test_type: {test_type}")

        return int(math.ceil(n))

    def calculate_power(
        self,
        n: int,
        effect_size: float,
        alpha: float = 0.05,
        test_type: str = "two_sample_t",
    ) -> float:
        """Calculate statistical power for a given sample size.

        Parameters
        ----------
        n : int
        effect_size : float
        alpha : float
        test_type : str

        Returns
        -------
        float
        """
        self._validate_inputs(effect_size, alpha, 0.5, test_type)
        if n < 2:
            raise ValidationError("Sample size must be at least 2.")

        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)

        if test_type in ("two_sample_t", "paired_t"):
            se_factor = math.sqrt(2.0 / n)
        elif test_type in ("one_sample_t", "chi_squared", "proportion"):
            se_factor = 1.0 / math.sqrt(n)
        else:
            raise ValidationError(f"Unknown test_type: {test_type}")

        ncp = effect_size / se_factor
        power = 1.0 - sp_stats.norm.cdf(z_alpha - ncp)

        return round(min(max(power, 0.0), 1.0), 4)

    def calculate_effect_size(
        self,
        n: int,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: str = "two_sample_t",
    ) -> float:
        """Calculate the minimum detectable effect size.

        Parameters
        ----------
        n : int
        alpha : float
        power : float
        test_type : str

        Returns
        -------
        float
        """
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)

        if test_type in ("two_sample_t", "paired_t"):
            se_factor = math.sqrt(2.0 / n)
        elif test_type in ("one_sample_t", "chi_squared", "proportion"):
            se_factor = 1.0 / math.sqrt(n)
        else:
            raise ValidationError(f"Unknown test_type: {test_type}")

        return round((z_alpha + z_beta) * se_factor, 4)

    @staticmethod
    def _validate_inputs(
        effect_size: float, alpha: float, power: float, test_type: str
    ) -> None:
        if effect_size <= 0:
            raise ValidationError("effect_size must be positive.")
        if not 0 < alpha < 1:
            raise ValidationError("alpha must be in (0, 1).")
        if not 0 < power < 1:
            raise ValidationError("power must be in (0, 1).")
        if test_type not in PowerAnalyzer._VALID_TESTS:
            raise ValidationError(
                f"test_type must be one of {PowerAnalyzer._VALID_TESTS}."
            )


class MultiplicityCorrectionMethods:
    """Multiple testing correction procedures."""

    @staticmethod
    def bonferroni(
        p_values: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bonferroni correction.

        Parameters
        ----------
        p_values : np.ndarray
        alpha : float

        Returns
        -------
        tuple of (corrected_p_values, rejected)
            corrected : np.ndarray of adjusted p-values.
            rejected : np.ndarray of bool.
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        corrected = np.minimum(p * m, 1.0)
        rejected = corrected < alpha
        return corrected, rejected

    @staticmethod
    def holm(
        p_values: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Holm step-down procedure.

        Parameters
        ----------
        p_values : np.ndarray
        alpha : float

        Returns
        -------
        tuple of (corrected_p_values, rejected)
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        order = np.argsort(p)
        sorted_p = p[order]

        corrected_sorted = np.zeros(m)
        for i in range(m):
            corrected_sorted[i] = sorted_p[i] * (m - i)

        for i in range(1, m):
            corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i - 1])
        corrected_sorted = np.minimum(corrected_sorted, 1.0)

        corrected = np.empty(m)
        corrected[order] = corrected_sorted
        rejected = corrected < alpha
        return corrected, rejected

    @staticmethod
    def benjamini_hochberg(
        p_values: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Benjamini-Hochberg FDR control procedure.

        Parameters
        ----------
        p_values : np.ndarray
        alpha : float

        Returns
        -------
        tuple of (corrected_p_values, rejected)
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        order = np.argsort(p)
        sorted_p = p[order]

        corrected_sorted = np.zeros(m)
        for i in range(m):
            corrected_sorted[i] = sorted_p[i] * m / (i + 1)

        for i in range(m - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        corrected_sorted = np.minimum(corrected_sorted, 1.0)

        corrected = np.empty(m)
        corrected[order] = corrected_sorted
        rejected = corrected < alpha
        return corrected, rejected
