# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Signal decomposition for clinical time series.

This module provides decomposition methods for separating time series
data into constituent components (trend, seasonal, residual) and for
multi-resolution analysis via discrete wavelet transforms.

Classes
-------
DecompositionResult
    Dataclass holding the decomposed components.
SeasonalDecomposer
    Classical additive and multiplicative seasonal decomposition with
    automatic period detection.
WaveletDecomposer
    Haar wavelet decomposition, reconstruction, and denoising.

References
----------
.. [1] Cleveland RB, Cleveland WS, McRae JE, Terpenning I. STL: A
       seasonal-trend decomposition procedure based on loess. J Official
       Statistics. 1990;6(1):3-73.
.. [2] Mallat S. A Wavelet Tour of Signal Processing. 3rd ed. Academic
       Press; 2009.
.. [3] Donoho DL. De-noising by soft-thresholding. IEEE Trans Inform
       Theory. 1995;41(3):613-627.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sp_signal

from notfallmedizin.core.base import BaseTransformer
from notfallmedizin.core.exceptions import (
    ComputationError,
    InsufficientDataError,
    ValidationError,
)


# ======================================================================
# Decomposition Result
# ======================================================================


@dataclass
class DecompositionResult:
    """Container for seasonal decomposition output.

    Attributes
    ----------
    observed : numpy.ndarray
        Original input series.
    trend : numpy.ndarray
        Estimated trend component.
    seasonal : numpy.ndarray
        Estimated seasonal component.
    residual : numpy.ndarray
        Residual component (observed minus trend and seasonal).
    period : int
        Seasonal period used for the decomposition.
    method : str
        Decomposition method: ``"additive"`` or ``"multiplicative"``.
    """

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    period: int
    method: str


# ======================================================================
# Seasonal Decomposer
# ======================================================================


class SeasonalDecomposer(BaseTransformer):
    """Classical seasonal decomposition of a time series.

    Decomposes an observed series *y* into three components:

    * **Trend** (*T*) -- extracted using a centred moving average of
      width equal to the seasonal period.
    * **Seasonal** (*S*) -- average deviation from the trend within
      each seasonal position.
    * **Residual** (*R*) -- whatever remains after removing trend and
      seasonal effects.

    For the **additive** model:

    .. math::

        y_t = T_t + S_t + R_t

    For the **multiplicative** model:

    .. math::

        y_t = T_t \\times S_t \\times R_t

    If no ``period`` is supplied to :meth:`transform`, the period is
    estimated automatically using autocorrelation peak detection or FFT
    peak finding.

    Parameters
    ----------
    model : {"additive", "multiplicative"}, optional
        Decomposition model. Default is ``"additive"``.

    Attributes
    ----------
    result_ : DecompositionResult or None
        The most recent decomposition result (available after
        :meth:`transform`).
    """

    def __init__(self, model: str = "additive") -> None:
        super().__init__()
        if model not in ("additive", "multiplicative"):
            raise ValidationError(
                "model must be 'additive' or 'multiplicative'.",
                parameter="model",
            )
        self.model = model
        self.result_: Optional[DecompositionResult] = None

    # ------------------------------------------------------------------
    # Period detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_period_autocorrelation(y: np.ndarray, max_lag: Optional[int] = None) -> int:
        """Estimate the dominant period via autocorrelation peak finding.

        Computes the full autocorrelation of the de-meaned series and
        returns the lag of the first prominent peak after lag 0.

        Parameters
        ----------
        y : numpy.ndarray
            Input series.
        max_lag : int or None, optional
            Maximum lag to consider. Defaults to ``len(y) // 2``.

        Returns
        -------
        int
            Estimated period (>= 2).
        """
        n = len(y)
        if max_lag is None:
            max_lag = n // 2
        max_lag = min(max_lag, n - 1)

        y_centered = y - np.mean(y)
        acf = np.correlate(y_centered, y_centered, mode="full")
        acf = acf[n - 1 :]  # keep non-negative lags only
        if acf[0] != 0.0:
            acf = acf / acf[0]

        acf = acf[: max_lag + 1]

        # Find first local maximum after the initial decay
        for i in range(2, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1]:
                return max(i, 2)

        return 2

    @staticmethod
    def detect_period_fft(y: np.ndarray) -> int:
        """Estimate the dominant period via FFT peak finding.

        Computes the power spectral density and returns the period
        corresponding to the dominant non-DC frequency.

        Parameters
        ----------
        y : numpy.ndarray
            Input series.

        Returns
        -------
        int
            Estimated period (>= 2).
        """
        n = len(y)
        y_centered = y - np.mean(y)
        fft_vals = np.fft.rfft(y_centered)
        psd = np.abs(fft_vals) ** 2

        # Exclude DC component
        psd[0] = 0.0
        if len(psd) < 2:
            return 2

        peak_idx = int(np.argmax(psd))
        if peak_idx == 0:
            return 2

        period = int(round(n / peak_idx))
        return max(period, 2)

    # ------------------------------------------------------------------
    # Moving average trend extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        """Centred moving average.

        For even window sizes, a 2xm centred moving average is used:
        first a simple moving average of width *window* is computed,
        then a second pass of width 2 centres the result.

        Parameters
        ----------
        y : numpy.ndarray
            Input series.
        window : int
            Window width.

        Returns
        -------
        numpy.ndarray
            Trend estimate with ``NaN`` at boundaries where the full
            window is not available.
        """
        n = len(y)
        trend = np.full(n, np.nan, dtype=np.float64)

        if window % 2 == 1:
            half = window // 2
            for t in range(half, n - half):
                trend[t] = np.mean(y[t - half : t + half + 1])
        else:
            half = window // 2
            ma1 = np.full(n, np.nan, dtype=np.float64)
            for t in range(half, n - half + 1):
                if t - half >= 0 and t - half + window <= n:
                    ma1[t] = np.mean(y[t - half : t - half + window])
            for t in range(1, n):
                if np.isfinite(ma1[t]) and np.isfinite(ma1[t - 1]):
                    trend[t] = (ma1[t] + ma1[t - 1]) / 2.0
        return trend

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "SeasonalDecomposer":
        """No-op fit for API compatibility.

        The decomposer is stateless; all work happens in
        :meth:`transform`.

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

    def transform(
        self,
        values: ArrayLike,
        period: Optional[int] = None,
    ) -> DecompositionResult:
        """Decompose the time series into trend, seasonal, and residual.

        Parameters
        ----------
        values : array-like of shape (n_samples,)
            Observed time series.
        period : int or None, optional
            Seasonal period. If ``None``, the period is detected
            automatically using FFT and autocorrelation. Default is
            ``None``.

        Returns
        -------
        DecompositionResult
            Decomposed components.

        Raises
        ------
        InsufficientDataError
            If the series is shorter than two full periods.
        ComputationError
            If the multiplicative model encounters zero or negative
            trend values.
        """
        y = np.asarray(values, dtype=np.float64).ravel()
        if len(y) < 4:
            raise InsufficientDataError(
                "At least 4 observations required for decomposition.",
                n_samples=len(y),
                n_required=4,
            )

        if period is None:
            period_fft = self.detect_period_fft(y)
            period_acf = self.detect_period_autocorrelation(y)
            period = period_fft if period_fft >= 2 else period_acf
            period = max(period, 2)

        if len(y) < 2 * period:
            raise InsufficientDataError(
                f"Need at least 2 * period = {2 * period} observations, "
                f"got {len(y)}.",
                n_samples=len(y),
                n_required=2 * period,
            )

        # Step 1: Trend via centred moving average
        trend = self._moving_average(y, period)

        # Step 2: De-trended series
        if self.model == "additive":
            detrended = y - trend
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                detrended = np.where(
                    (trend != 0.0) & np.isfinite(trend),
                    y / trend,
                    np.nan,
                )

        # Step 3: Average seasonal component per position
        seasonal = np.zeros_like(y)
        for pos in range(period):
            indices = np.arange(pos, len(y), period)
            vals = detrended[indices]
            valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                avg = float(np.mean(valid))
            else:
                avg = 0.0 if self.model == "additive" else 1.0
            seasonal[indices] = avg

        # Normalise seasonal so it sums to zero (additive) or averages
        # to 1 (multiplicative) over one full period.
        season_block = seasonal[:period]
        if self.model == "additive":
            seasonal -= np.mean(season_block)
            season_block = seasonal[:period]
            for pos in range(period):
                seasonal[np.arange(pos, len(y), period)] = season_block[pos]
        else:
            s_mean = np.mean(season_block)
            if s_mean != 0.0:
                seasonal /= s_mean
                season_block = seasonal[:period]
                for pos in range(period):
                    seasonal[np.arange(pos, len(y), period)] = season_block[pos]

        # Step 4: Residual
        if self.model == "additive":
            residual = y - trend - seasonal
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = trend * seasonal
                residual = np.where(
                    (denom != 0.0) & np.isfinite(denom),
                    y / denom,
                    np.nan,
                )

        result = DecompositionResult(
            observed=y,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            period=period,
            method=self.model,
        )
        self.result_ = result
        self.is_fitted_ = True
        return result


# ======================================================================
# Wavelet Decomposer
# ======================================================================


class WaveletDecomposer(BaseTransformer):
    """Discrete wavelet transform using the Haar wavelet.

    The Haar wavelet is the simplest wavelet, defined by the scaling
    function and wavelet function:

    .. math::

        \\phi(t) = \\begin{cases} 1 & 0 \\le t < 1 \\\\ 0 & \\text{otherwise} \\end{cases}

        \\psi(t) = \\begin{cases} 1 & 0 \\le t < 1/2 \\\\ -1 & 1/2 \\le t < 1 \\\\ 0 & \\text{otherwise} \\end{cases}

    The forward transform recursively splits the signal into
    approximation (low-frequency) and detail (high-frequency)
    coefficients:

    .. math::

        a_k = \\frac{x_{2k} + x_{2k+1}}{\\sqrt{2}}, \\quad
        d_k = \\frac{x_{2k} - x_{2k+1}}{\\sqrt{2}}

    Parameters
    ----------
    levels : int, optional
        Number of decomposition levels. Default is ``3``.

    Attributes
    ----------
    coefficients_ : list of tuple of (numpy.ndarray, numpy.ndarray)
        Coefficients from the most recent transform, as
        ``(approximation, detail)`` pairs from coarsest to finest.
    original_length_ : int
        Length of the original signal (before any padding).
    """

    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        if levels < 1:
            raise ValidationError(
                "levels must be >= 1.",
                parameter="levels",
            )
        self.levels = levels
        self.coefficients_: List[Tuple[np.ndarray, np.ndarray]] = []
        self.original_length_: int = 0
        self._padded_length: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _haar_forward_step(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-level Haar forward transform.

        Parameters
        ----------
        x : numpy.ndarray
            Input signal of even length.

        Returns
        -------
        approx : numpy.ndarray
            Approximation coefficients (length ``len(x) // 2``).
        detail : numpy.ndarray
            Detail coefficients (length ``len(x) // 2``).
        """
        n = len(x)
        if n % 2 != 0:
            x = np.append(x, x[-1])
            n = len(x)
        half = n // 2
        sqrt2 = np.sqrt(2.0)
        approx = (x[0::2] + x[1::2]) / sqrt2
        detail = (x[0::2] - x[1::2]) / sqrt2
        return approx, detail

    @staticmethod
    def _haar_inverse_step(
        approx: np.ndarray, detail: np.ndarray
    ) -> np.ndarray:
        """Single-level Haar inverse transform.

        Parameters
        ----------
        approx : numpy.ndarray
            Approximation coefficients.
        detail : numpy.ndarray
            Detail coefficients.

        Returns
        -------
        numpy.ndarray
            Reconstructed signal (length ``2 * len(approx)``).
        """
        sqrt2 = np.sqrt(2.0)
        n = len(approx)
        result = np.empty(2 * n, dtype=np.float64)
        result[0::2] = (approx + detail) / sqrt2
        result[1::2] = (approx - detail) / sqrt2
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "WaveletDecomposer":
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

    def transform(
        self,
        signal: ArrayLike,
        levels: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Perform multi-level Haar wavelet decomposition.

        Parameters
        ----------
        signal : array-like of shape (n_samples,)
            Input signal.
        levels : int or None, optional
            Override the number of decomposition levels. If ``None``,
            uses ``self.levels``. Default is ``None``.

        Returns
        -------
        list of (numpy.ndarray, numpy.ndarray)
            List of ``(approximation, detail)`` coefficient pairs,
            ordered from coarsest level to finest level.

        Raises
        ------
        InsufficientDataError
            If the signal is too short for the requested number of
            levels.
        """
        x = np.asarray(signal, dtype=np.float64).ravel()
        self.original_length_ = len(x)
        n_levels = levels if levels is not None else self.levels

        min_length = 2**n_levels
        if len(x) < min_length:
            raise InsufficientDataError(
                f"Signal length ({len(x)}) too short for {n_levels} levels. "
                f"Minimum length is {min_length}.",
                n_samples=len(x),
                n_required=min_length,
            )

        # Pad to next power of 2 if necessary
        padded_len = 1
        while padded_len < len(x):
            padded_len *= 2
        if padded_len != len(x):
            x = np.pad(x, (0, padded_len - len(x)), mode="edge")
        self._padded_length = len(x)

        coefficients: List[Tuple[np.ndarray, np.ndarray]] = []
        current = x.copy()
        for _ in range(n_levels):
            approx, detail = self._haar_forward_step(current)
            coefficients.append((approx, detail))
            current = approx

        self.coefficients_ = coefficients
        self.is_fitted_ = True
        return coefficients

    def reconstruct(
        self,
        coefficients: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> np.ndarray:
        """Reconstruct a signal from wavelet coefficients.

        Parameters
        ----------
        coefficients : list of (numpy.ndarray, numpy.ndarray) or None
            Coefficient pairs from coarsest to finest. If ``None``,
            uses the coefficients from the most recent
            :meth:`transform` call.

        Returns
        -------
        numpy.ndarray
            Reconstructed signal, trimmed to the original length.

        Raises
        ------
        ValidationError
            If no coefficients are available.
        """
        if coefficients is None:
            coefficients = self.coefficients_
        if not coefficients:
            raise ValidationError("No coefficients available for reconstruction.")

        # Start from the coarsest-level approximation and iteratively
        # apply the inverse Haar step with the detail coefficients at
        # each level, working from coarsest back to finest.
        current = coefficients[-1][0]
        for i in range(len(coefficients) - 1, -1, -1):
            current = self._haar_inverse_step(current, coefficients[i][1])

        if self.original_length_ > 0:
            current = current[: self.original_length_]
        return current

    def denoise(
        self,
        signal: ArrayLike,
        threshold: str = "soft",
        threshold_value: Optional[float] = None,
        levels: Optional[int] = None,
    ) -> np.ndarray:
        """Denoise a signal using wavelet thresholding.

        Applies the universal threshold (VisuShrink) by default:

        .. math::

            \\lambda = \\sigma \\sqrt{2 \\ln n}

        where :math:`\\sigma` is estimated from the median absolute
        deviation of the finest-level detail coefficients and *n* is
        the signal length.

        Parameters
        ----------
        signal : array-like of shape (n_samples,)
            Noisy input signal.
        threshold : {"soft", "hard"}, optional
            Thresholding method.  Default is ``"soft"``.

            - ``"soft"``: coefficients are shrunk towards zero by
              ``lambda`` (sign(d) * max(|d| - lambda, 0)).
            - ``"hard"``: coefficients below ``lambda`` in absolute
              value are set to zero.
        threshold_value : float or None, optional
            Explicit threshold value. If ``None``, the universal
            threshold is computed. Default is ``None``.
        levels : int or None, optional
            Number of decomposition levels. If ``None``, uses
            ``self.levels``. Default is ``None``.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Denoised signal.

        Raises
        ------
        ValidationError
            If ``threshold`` is not ``"soft"`` or ``"hard"``.
        """
        if threshold not in ("soft", "hard"):
            raise ValidationError(
                "threshold must be 'soft' or 'hard'.",
                parameter="threshold",
            )

        x = np.asarray(signal, dtype=np.float64).ravel()
        orig_len = len(x)

        coefficients = self.transform(x, levels=levels)

        # Estimate noise sigma from finest detail coefficients
        finest_detail = coefficients[0][1]
        mad = float(np.median(np.abs(finest_detail - np.median(finest_detail))))
        sigma = mad / 0.6745

        if threshold_value is None:
            n = orig_len
            lam = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
        else:
            lam = threshold_value

        # Threshold detail coefficients at each level
        thresholded: List[Tuple[np.ndarray, np.ndarray]] = []
        for approx, detail in coefficients:
            if threshold == "soft":
                detail_new = np.sign(detail) * np.maximum(np.abs(detail) - lam, 0.0)
            else:
                detail_new = np.where(np.abs(detail) >= lam, detail, 0.0)
            thresholded.append((approx, detail_new))

        # Reconstruct from the thresholded coefficients
        current = thresholded[-1][0]
        for i in range(len(thresholded) - 1, -1, -1):
            current = self._haar_inverse_step(current, thresholded[i][1])

        return current[:orig_len]
