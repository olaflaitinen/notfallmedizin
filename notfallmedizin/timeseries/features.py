# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Time series feature extraction for clinical data.

This module provides a comprehensive feature extraction pipeline for
univariate clinical time series. Extracted features span five
categories:

1. **Statistical** -- mean, standard deviation, minimum, maximum,
   median, skewness, kurtosis, interquartile range, coefficient of
   variation.
2. **Temporal** -- trend slope (OLS), lag-1 autocorrelation, lag-1
   partial autocorrelation.
3. **Complexity** -- sample entropy, approximate entropy, permutation
   entropy.
4. **Frequency domain** -- dominant frequency (FFT), spectral entropy,
   power spectral density summary.
5. **Clinical** -- percentage of time above/below a threshold, number
   of threshold crossings, maximum consecutive observations above a
   threshold.

Classes
-------
ClinicalTimeSeriesFeatureExtractor
    Transformer that converts a set of time series into a feature
    matrix.

References
----------
.. [1] Richman JS, Moorman JR. Physiological time-series analysis
       using approximate entropy and sample entropy. Am J Physiol Heart
       Circ Physiol. 2000;278(6):H2039-H2049.
.. [2] Bandt C, Pompe B. Permutation entropy: a natural complexity
       measure for time series. Phys Rev Lett. 2002;88(17):174102.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from notfallmedizin.core.base import BaseTransformer
from notfallmedizin.core.exceptions import (
    InsufficientDataError,
    ValidationError,
)


# ======================================================================
# Entropy helpers (module-private)
# ======================================================================


def _max_norm_distance(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """Chebyshev (L-infinity) distance between two vectors.

    Parameters
    ----------
    x_i, x_j : numpy.ndarray
        Vectors of equal length.

    Returns
    -------
    float
        Maximum absolute element-wise difference.
    """
    return float(np.max(np.abs(x_i - x_j)))


def _count_matches(
    templates: np.ndarray, tolerance: float
) -> int:
    """Count template matches within a tolerance (Chebyshev distance).

    Parameters
    ----------
    templates : numpy.ndarray of shape (n_templates, m)
        Template vectors.
    tolerance : float
        Maximum Chebyshev distance for a match.

    Returns
    -------
    int
        Number of matched pairs (excluding self-matches).
    """
    n = templates.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _max_norm_distance(templates[i], templates[j]) <= tolerance:
                count += 1
    return count


def _approximate_entropy(
    x: np.ndarray, m: int = 2, r: Optional[float] = None
) -> float:
    """Compute the approximate entropy (ApEn) of a time series.

    ApEn quantifies the regularity and unpredictability of a signal.
    Lower values indicate more regularity.

    .. math::

        ApEn(m, r, N) = \\Phi^m(r) - \\Phi^{m+1}(r)

    where :math:`\\Phi^m(r)` is the average log-frequency of template
    matches of length *m* within tolerance *r*.

    Parameters
    ----------
    x : numpy.ndarray
        Time series of length *N*.
    m : int
        Embedding dimension. Default is ``2``.
    r : float or None
        Tolerance. Defaults to ``0.2 * std(x)``.

    Returns
    -------
    float
        Approximate entropy value.
    """
    n = len(x)
    if r is None:
        r = 0.2 * float(np.std(x, ddof=0))
    if r == 0.0:
        return 0.0

    def phi(m_val: int) -> float:
        templates = np.array(
            [x[i : i + m_val] for i in range(n - m_val + 1)],
            dtype=np.float64,
        )
        n_t = len(templates)
        counts = np.zeros(n_t)
        for i in range(n_t):
            for j in range(n_t):
                if _max_norm_distance(templates[i], templates[j]) <= r:
                    counts[i] += 1
            counts[i] /= n_t
        return float(np.mean(np.log(counts + 1e-100)))

    return phi(m) - phi(m + 1)


def _sample_entropy(
    x: np.ndarray, m: int = 2, r: Optional[float] = None
) -> float:
    """Compute sample entropy (SampEn) of a time series.

    SampEn is a modification of approximate entropy that does not
    count self-matches, reducing bias.

    .. math::

        SampEn(m, r, N) = -\\ln \\frac{A}{B}

    where *A* is the number of template matches of length ``m+1`` and
    *B* is the number of matches of length ``m``, both within tolerance
    *r*, excluding self-matches.

    Parameters
    ----------
    x : numpy.ndarray
        Time series of length *N*.
    m : int
        Embedding dimension. Default is ``2``.
    r : float or None
        Tolerance. Defaults to ``0.2 * std(x)``.

    Returns
    -------
    float
        Sample entropy value. Returns ``inf`` if no matches found.
    """
    n = len(x)
    if r is None:
        r = 0.2 * float(np.std(x, ddof=0))
    if r == 0.0:
        return 0.0

    templates_m = np.array(
        [x[i : i + m] for i in range(n - m)], dtype=np.float64
    )
    templates_m1 = np.array(
        [x[i : i + m + 1] for i in range(n - m)], dtype=np.float64
    )

    B = _count_matches(templates_m, r)
    A = _count_matches(templates_m1, r)

    if B == 0:
        return float("inf")
    if A == 0:
        return float("inf")
    return -np.log(A / B)


def _permutation_entropy(
    x: np.ndarray, order: int = 3, delay: int = 1
) -> float:
    """Compute the permutation entropy of a time series.

    Permutation entropy measures the complexity of a signal by
    analysing the frequency distribution of ordinal patterns.

    .. math::

        H_p = -\\sum_{\\pi} p(\\pi) \\ln p(\\pi)

    normalized by :math:`\\ln(m!)` where *m* is the embedding
    ``order``.

    Parameters
    ----------
    x : numpy.ndarray
        Time series.
    order : int
        Embedding dimension (order of the permutation). Default is ``3``.
    delay : int
        Embedding delay. Default is ``1``.

    Returns
    -------
    float
        Normalised permutation entropy in ``[0, 1]``.
    """
    n = len(x)
    n_patterns = n - (order - 1) * delay
    if n_patterns < 1:
        return 0.0

    pattern_counts: Dict[Tuple[int, ...], int] = {}
    for i in range(n_patterns):
        indices = list(range(i, i + order * delay, delay))
        window = x[indices]
        pattern = tuple(int(v) for v in np.argsort(window))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    total = sum(pattern_counts.values())
    probs = np.array(list(pattern_counts.values()), dtype=np.float64) / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-100)))

    max_entropy = np.log(math.factorial(order))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


# ======================================================================
# Clinical Time Series Feature Extractor
# ======================================================================


class ClinicalTimeSeriesFeatureExtractor(BaseTransformer):
    """Extract a comprehensive feature vector from clinical time series.

    Given a collection of univariate time series (rows of a 2-D array,
    where each row is a zero-padded series), this transformer produces
    a feature matrix where each row contains the features of the
    corresponding input series.

    Parameters
    ----------
    threshold_upper : float, optional
        Upper threshold for clinical threshold features. Default is
        ``100.0``.
    threshold_lower : float, optional
        Lower threshold for clinical threshold features. Default is
        ``60.0``.
    entropy_m : int, optional
        Embedding dimension for approximate and sample entropy.
        Default is ``2``.
    entropy_r_factor : float, optional
        Factor multiplied by std(x) to get the entropy tolerance *r*.
        Default is ``0.2``.
    permutation_order : int, optional
        Order parameter for permutation entropy. Default is ``3``.

    Attributes
    ----------
    feature_names_ : list of str
        Names of the extracted features.
    n_features_ : int
        Number of features per time series.
    """

    _STAT_FEATURES = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skewness",
        "kurtosis",
        "iqr",
        "cv",
    ]
    _TEMPORAL_FEATURES = [
        "trend_slope",
        "autocorrelation_lag1",
        "partial_autocorrelation_lag1",
    ]
    _COMPLEXITY_FEATURES = [
        "sample_entropy",
        "approximate_entropy",
        "permutation_entropy",
    ]
    _FREQ_FEATURES = [
        "dominant_frequency",
        "spectral_entropy",
        "psd_mean",
        "psd_max",
        "psd_total",
    ]
    _CLINICAL_FEATURES = [
        "pct_time_above_threshold",
        "pct_time_below_threshold",
        "n_threshold_crossings_upper",
        "max_consecutive_above_threshold",
    ]

    def __init__(
        self,
        threshold_upper: float = 100.0,
        threshold_lower: float = 60.0,
        entropy_m: int = 2,
        entropy_r_factor: float = 0.2,
        permutation_order: int = 3,
    ) -> None:
        super().__init__()
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.entropy_m = entropy_m
        self.entropy_r_factor = entropy_r_factor
        self.permutation_order = permutation_order

        self.feature_names_: List[str] = (
            self._STAT_FEATURES
            + self._TEMPORAL_FEATURES
            + self._COMPLEXITY_FEATURES
            + self._FREQ_FEATURES
            + self._CLINICAL_FEATURES
        )
        self.n_features_: int = len(self.feature_names_)

    # ------------------------------------------------------------------
    # Single-series feature computation
    # ------------------------------------------------------------------

    def _statistical_features(self, x: np.ndarray) -> np.ndarray:
        """Compute statistical features for a single series.

        Parameters
        ----------
        x : numpy.ndarray
            Univariate time series.

        Returns
        -------
        numpy.ndarray of shape (9,)
            [mean, std, min, max, median, skewness, kurtosis, iqr, cv]
        """
        n = len(x)
        mean_val = float(np.mean(x))
        std_val = float(np.std(x, ddof=1)) if n > 1 else 0.0
        min_val = float(np.min(x))
        max_val = float(np.max(x))
        median_val = float(np.median(x))

        # Skewness: Fisher definition
        if n > 2 and std_val > 0:
            skew = float(np.mean(((x - mean_val) / std_val) ** 3))
        else:
            skew = 0.0

        # Kurtosis: excess kurtosis (Fisher)
        if n > 3 and std_val > 0:
            kurt = float(np.mean(((x - mean_val) / std_val) ** 4)) - 3.0
        else:
            kurt = 0.0

        q75, q25 = float(np.percentile(x, 75)), float(np.percentile(x, 25))
        iqr = q75 - q25

        cv = std_val / abs(mean_val) if mean_val != 0.0 else 0.0

        return np.array(
            [mean_val, std_val, min_val, max_val, median_val, skew, kurt, iqr, cv],
            dtype=np.float64,
        )

    def _temporal_features(self, x: np.ndarray) -> np.ndarray:
        """Compute temporal features for a single series.

        Parameters
        ----------
        x : numpy.ndarray
            Univariate time series.

        Returns
        -------
        numpy.ndarray of shape (3,)
            [trend_slope, autocorrelation_lag1, partial_autocorrelation_lag1]
        """
        n = len(x)

        # Trend slope via OLS: y = a + b*t
        t = np.arange(n, dtype=np.float64)
        if n > 1:
            t_mean = np.mean(t)
            x_mean = np.mean(x)
            denom = float(np.sum((t - t_mean) ** 2))
            if denom != 0.0:
                slope = float(np.sum((t - t_mean) * (x - x_mean))) / denom
            else:
                slope = 0.0
        else:
            slope = 0.0

        # Lag-1 autocorrelation
        if n > 1:
            x_centered = x - np.mean(x)
            var = float(np.sum(x_centered**2))
            if var > 0:
                acf1 = float(np.sum(x_centered[:-1] * x_centered[1:])) / var
            else:
                acf1 = 0.0
        else:
            acf1 = 0.0

        # Partial autocorrelation at lag 1 is the same as acf at lag 1
        # for a univariate series. For lag-1, PACF = ACF(1).
        pacf1 = acf1

        return np.array([slope, acf1, pacf1], dtype=np.float64)

    def _complexity_features(self, x: np.ndarray) -> np.ndarray:
        """Compute complexity/entropy features for a single series.

        Parameters
        ----------
        x : numpy.ndarray
            Univariate time series.

        Returns
        -------
        numpy.ndarray of shape (3,)
            [sample_entropy, approximate_entropy, permutation_entropy]
        """
        r = self.entropy_r_factor * float(np.std(x, ddof=0))
        m = self.entropy_m

        if len(x) < m + 2:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        sampen = _sample_entropy(x, m=m, r=r)
        apen = _approximate_entropy(x, m=m, r=r)
        pe = _permutation_entropy(x, order=self.permutation_order)

        if not np.isfinite(sampen):
            sampen = 0.0
        if not np.isfinite(apen):
            apen = 0.0

        return np.array([sampen, apen, pe], dtype=np.float64)

    def _frequency_features(self, x: np.ndarray) -> np.ndarray:
        """Compute frequency-domain features for a single series.

        Parameters
        ----------
        x : numpy.ndarray
            Univariate time series.

        Returns
        -------
        numpy.ndarray of shape (5,)
            [dominant_frequency, spectral_entropy, psd_mean, psd_max, psd_total]
        """
        n = len(x)
        if n < 4:
            return np.zeros(5, dtype=np.float64)

        x_centered = x - np.mean(x)
        fft_vals = np.fft.rfft(x_centered)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # Exclude DC
        psd_no_dc = psd[1:]
        freqs_no_dc = freqs[1:]

        if len(psd_no_dc) == 0 or float(np.sum(psd_no_dc)) == 0.0:
            return np.zeros(5, dtype=np.float64)

        # Dominant frequency
        peak_idx = int(np.argmax(psd_no_dc))
        dominant_freq = float(freqs_no_dc[peak_idx])

        # Spectral entropy (normalised Shannon entropy of the PSD)
        psd_norm = psd_no_dc / np.sum(psd_no_dc)
        spectral_ent = -float(np.sum(psd_norm * np.log(psd_norm + 1e-100)))
        max_ent = np.log(len(psd_no_dc))
        if max_ent > 0:
            spectral_ent /= max_ent

        psd_mean = float(np.mean(psd_no_dc))
        psd_max = float(np.max(psd_no_dc))
        psd_total = float(np.sum(psd_no_dc))

        return np.array(
            [dominant_freq, spectral_ent, psd_mean, psd_max, psd_total],
            dtype=np.float64,
        )

    def _clinical_features(self, x: np.ndarray) -> np.ndarray:
        """Compute clinical threshold features for a single series.

        Parameters
        ----------
        x : numpy.ndarray
            Univariate time series.

        Returns
        -------
        numpy.ndarray of shape (4,)
            [pct_above, pct_below, n_crossings, max_consecutive_above]
        """
        n = len(x)
        if n == 0:
            return np.zeros(4, dtype=np.float64)

        above = x > self.threshold_upper
        below = x < self.threshold_lower

        pct_above = float(np.sum(above)) / n * 100.0
        pct_below = float(np.sum(below)) / n * 100.0

        # Count upward crossings of the upper threshold
        crossings = 0
        for i in range(1, n):
            if x[i] > self.threshold_upper and x[i - 1] <= self.threshold_upper:
                crossings += 1
            elif x[i] <= self.threshold_upper and x[i - 1] > self.threshold_upper:
                crossings += 1

        # Maximum consecutive above threshold
        max_consec = 0
        current_consec = 0
        for i in range(n):
            if above[i]:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        return np.array(
            [pct_above, pct_below, float(crossings), float(max_consec)],
            dtype=np.float64,
        )

    def _extract_single(self, x: np.ndarray) -> np.ndarray:
        """Extract all features for a single time series.

        Parameters
        ----------
        x : numpy.ndarray
            Univariate time series.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
        """
        stat = self._statistical_features(x)
        temp = self._temporal_features(x)
        comp = self._complexity_features(x)
        freq = self._frequency_features(x)
        clin = self._clinical_features(x)
        return np.concatenate([stat, temp, comp, freq, clin])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "ClinicalTimeSeriesFeatureExtractor":
        """No-op fit for API compatibility.

        Parameters
        ----------
        X : array-like
            Ignored.
        y : array-like or None
            Ignored.

        Returns
        -------
        self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Extract features from one or more time series.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timesteps) or (n_timesteps,)
            Input time series. If 1-D, treated as a single series.

        Returns
        -------
        numpy.ndarray of shape (n_series, n_features)
            Feature matrix.

        Raises
        ------
        InsufficientDataError
            If any series has fewer than 4 observations.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValidationError(
                "X must be 1-D or 2-D.",
                parameter="X",
            )

        n_series, n_timesteps = arr.shape
        if n_timesteps < 4:
            raise InsufficientDataError(
                "Each time series must have at least 4 observations.",
                n_samples=n_timesteps,
                n_required=4,
            )

        result = np.empty((n_series, self.n_features_), dtype=np.float64)
        for i in range(n_series):
            result[i, :] = self._extract_single(arr[i, :])
        return result

    def fit_transform(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Fit and extract features in one step.

        Parameters
        ----------
        X : array-like
            Input time series.
        y : array-like or None
            Ignored.

        Returns
        -------
        numpy.ndarray
            Feature matrix.
        """
        return self.fit(X, y, **kwargs).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return the names of the extracted features.

        Returns
        -------
        list of str
            Feature names in the same order as the columns of the
            output feature matrix.
        """
        return list(self.feature_names_)
