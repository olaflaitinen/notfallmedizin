# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Real-time streaming processors for clinical time series.

This module provides components for online (streaming) analysis of
clinical time series data:

1. :class:`StreamingProcessor` -- sliding window management with
   configurable window and step sizes, callback registration, and
   rolling statistics computation.

2. :class:`OnlineChangePointDetector` -- Bayesian Online Changepoint
   Detection (BOCD) using a conjugate Normal-Inverse-Gamma prior for
   detecting abrupt distributional shifts in a data stream.

References
----------
.. [1] Adams RP, MacKay DJC. Bayesian online changepoint detection.
       arXiv:0710.3742. 2007.
.. [2] Murphy KP. Conjugate Bayesian analysis of the Gaussian
       distribution. Technical report. 2007.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from notfallmedizin.core.exceptions import (
    InsufficientDataError,
    ValidationError,
)


# ======================================================================
# Streaming Processor
# ======================================================================


class StreamingProcessor:
    """Sliding window processor for streaming clinical data.

    Maintains a FIFO buffer of ``(timestamp, value)`` pairs. When the
    buffer reaches ``window_size`` elements, it is considered "full"
    and registered callbacks are invoked. The window then advances by
    ``step_size`` elements.

    Parameters
    ----------
    window_size : int
        Number of observations in the sliding window.
    step_size : int, optional
        Number of observations the window advances after becoming
        full. Default is ``1``.

    Attributes
    ----------
    buffer_timestamps : list of float
        Timestamps of buffered observations.
    buffer_values : list of float
        Values of buffered observations.
    windows_emitted : int
        Number of full windows emitted so far.

    Examples
    --------
    >>> sp = StreamingProcessor(window_size=5, step_size=2)
    >>> results = []
    >>> sp.on_window_full(lambda ts, vals: results.append(vals.copy()))
    >>> for i in range(10):
    ...     sp.add_value(float(i), float(i) * 1.5)
    """

    def __init__(
        self,
        window_size: int,
        step_size: int = 1,
    ) -> None:
        if window_size < 1:
            raise ValidationError(
                "window_size must be >= 1.",
                parameter="window_size",
            )
        if step_size < 1:
            raise ValidationError(
                "step_size must be >= 1.",
                parameter="step_size",
            )
        self.window_size = window_size
        self.step_size = step_size

        self.buffer_timestamps: List[float] = []
        self.buffer_values: List[float] = []
        self.windows_emitted: int = 0

        self._callbacks: List[Callable[[np.ndarray, np.ndarray], None]] = []
        self._pending_steps: int = 0

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def on_window_full(
        self,
        callback: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        """Register a callback invoked when the window is full.

        The callback receives two positional arguments:

        1. ``timestamps`` -- ``numpy.ndarray`` of shape ``(window_size,)``
        2. ``values`` -- ``numpy.ndarray`` of shape ``(window_size,)``

        Parameters
        ----------
        callback : callable
            Function or method to invoke.
        """
        if not callable(callback):
            raise ValidationError("callback must be callable.", parameter="callback")
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_value(self, timestamp: float, value: float) -> None:
        """Feed a single observation into the stream.

        Parameters
        ----------
        timestamp : float
            Observation timestamp (e.g. epoch seconds).
        value : float
            Observed value.
        """
        self.buffer_timestamps.append(float(timestamp))
        self.buffer_values.append(float(value))

        if len(self.buffer_values) >= self.window_size:
            self._emit_window()

    def add_batch(
        self,
        timestamps: ArrayLike,
        values: ArrayLike,
    ) -> None:
        """Feed multiple observations into the stream.

        Parameters
        ----------
        timestamps : array-like of shape (n,)
            Observation timestamps.
        values : array-like of shape (n,)
            Observed values.
        """
        ts = np.asarray(timestamps, dtype=np.float64).ravel()
        vs = np.asarray(values, dtype=np.float64).ravel()
        if len(ts) != len(vs):
            raise ValidationError(
                "timestamps and values must have the same length.",
            )
        for t, v in zip(ts, vs):
            self.add_value(float(t), float(v))

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _emit_window(self) -> None:
        """Fire callbacks with the current full window, then advance."""
        ts_arr = np.array(
            self.buffer_timestamps[-self.window_size :], dtype=np.float64
        )
        val_arr = np.array(
            self.buffer_values[-self.window_size :], dtype=np.float64
        )

        for cb in self._callbacks:
            cb(ts_arr, val_arr)

        self.windows_emitted += 1

        # Advance by step_size
        if self.step_size >= self.window_size:
            self.buffer_timestamps.clear()
            self.buffer_values.clear()
        else:
            self.buffer_timestamps = self.buffer_timestamps[self.step_size :]
            self.buffer_values = self.buffer_values[self.step_size :]

    def get_window(self) -> np.ndarray:
        """Return the current window contents as an array.

        If the buffer contains fewer than ``window_size`` observations,
        the entire buffer is returned.

        Returns
        -------
        numpy.ndarray
            Current window values.
        """
        if len(self.buffer_values) >= self.window_size:
            return np.array(
                self.buffer_values[-self.window_size :], dtype=np.float64
            )
        return np.array(self.buffer_values, dtype=np.float64)

    def get_timestamps(self) -> np.ndarray:
        """Return timestamps corresponding to the current window.

        Returns
        -------
        numpy.ndarray
            Current window timestamps.
        """
        if len(self.buffer_timestamps) >= self.window_size:
            return np.array(
                self.buffer_timestamps[-self.window_size :], dtype=np.float64
            )
        return np.array(self.buffer_timestamps, dtype=np.float64)

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------

    def compute_rolling_statistics(self) -> Dict[str, float]:
        """Compute rolling statistics over the current window.

        Returns
        -------
        dict of str to float
            Dictionary with keys ``"mean"``, ``"std"``, ``"min"``,
            ``"max"``.

        Raises
        ------
        InsufficientDataError
            If the buffer is empty.
        """
        if len(self.buffer_values) == 0:
            raise InsufficientDataError(
                "Cannot compute statistics on an empty buffer.",
                n_samples=0,
                n_required=1,
            )
        window = self.get_window()
        return {
            "mean": float(np.mean(window)),
            "std": float(np.std(window, ddof=1)) if len(window) > 1 else 0.0,
            "min": float(np.min(window)),
            "max": float(np.max(window)),
        }

    def reset(self) -> None:
        """Clear the buffer and reset state."""
        self.buffer_timestamps.clear()
        self.buffer_values.clear()
        self.windows_emitted = 0

    def __repr__(self) -> str:
        return (
            f"StreamingProcessor(window_size={self.window_size}, "
            f"step_size={self.step_size}, "
            f"buffer_length={len(self.buffer_values)})"
        )


# ======================================================================
# Bayesian Online Changepoint Detection
# ======================================================================


class OnlineChangePointDetector:
    """Bayesian online changepoint detection (BOCD).

    Implements the algorithm of Adams and MacKay (2007) with a
    conjugate Normal-Inverse-Gamma prior for Gaussian observations.

    At each time step the detector maintains a probability distribution
    over "run lengths" (the number of observations since the last
    changepoint). A changepoint is signalled when the probability mass
    at run length zero exceeds a configurable threshold.

    The conjugate prior is parameterised by:

    * ``mu0`` -- prior mean
    * ``kappa0`` -- prior precision weight (number of pseudo-observations)
    * ``alpha0`` -- shape of the inverse-gamma on the variance
    * ``beta0`` -- scale of the inverse-gamma on the variance

    The posterior predictive distribution at each run length is
    Student-t:

    .. math::

        p(x_{t+1} \\mid r_t) = \\text{St}\\left(
            2\\alpha_t^{(r)},\\;
            \\mu_t^{(r)},\\;
            \\frac{\\beta_t^{(r)}(\\kappa_t^{(r)}+1)}
                 {\\alpha_t^{(r)}\\kappa_t^{(r)}}
        \\right)

    Parameters
    ----------
    hazard_rate : float, optional
        Constant hazard rate *H* (reciprocal of the expected run
        length). Default is ``1/250`` (one changepoint per 250
        observations on average).
    mu0 : float, optional
        Prior mean. Default is ``0.0``.
    kappa0 : float, optional
        Prior precision weight. Default is ``1.0``.
    alpha0 : float, optional
        Shape parameter of the inverse-gamma prior. Default is ``1.0``.
    beta0 : float, optional
        Scale parameter of the inverse-gamma prior. Default is ``1.0``.

    Attributes
    ----------
    run_length_probs : numpy.ndarray
        Current run-length probability distribution.
    t : int
        Number of observations processed so far.
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 250.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        if hazard_rate <= 0.0 or hazard_rate >= 1.0:
            raise ValidationError(
                "hazard_rate must be in (0, 1).",
                parameter="hazard_rate",
            )
        if kappa0 <= 0.0:
            raise ValidationError(
                "kappa0 must be positive.",
                parameter="kappa0",
            )
        if alpha0 <= 0.0:
            raise ValidationError(
                "alpha0 must be positive.",
                parameter="alpha0",
            )
        if beta0 <= 0.0:
            raise ValidationError(
                "beta0 must be positive.",
                parameter="beta0",
            )

        self.hazard_rate = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Run-length probability starts with P(r=0) = 1
        self.run_length_probs: np.ndarray = np.array([1.0], dtype=np.float64)
        self.t: int = 0

        # Sufficient statistics vectors, one entry per possible run length
        self._mu: np.ndarray = np.array([mu0], dtype=np.float64)
        self._kappa: np.ndarray = np.array([kappa0], dtype=np.float64)
        self._alpha: np.ndarray = np.array([alpha0], dtype=np.float64)
        self._beta: np.ndarray = np.array([beta0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _student_t_pdf(
        self,
        x: float,
        df: np.ndarray,
        loc: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Evaluate Student-t pdf for vectorised parameters.

        Parameters
        ----------
        x : float
            Observation.
        df : numpy.ndarray
            Degrees of freedom for each run length.
        loc : numpy.ndarray
            Location (mean) for each run length.
        scale : numpy.ndarray
            Scale for each run length.

        Returns
        -------
        numpy.ndarray
            Predictive probability densities.
        """
        from scipy.special import gammaln

        z = (x - loc) / scale
        log_pdf = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(np.pi * df)
            - np.log(scale)
            - ((df + 1.0) / 2.0) * np.log(1.0 + z**2 / df)
        )
        return np.exp(log_pdf)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, value: float) -> np.ndarray:
        """Process a new observation and update the run-length distribution.

        Parameters
        ----------
        value : float
            New observation.

        Returns
        -------
        numpy.ndarray
            Updated run-length probability distribution.
        """
        x = float(value)
        self.t += 1

        # 1. Predictive probabilities under each run length
        df = 2.0 * self._alpha
        loc = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa))
        pred_probs = self._student_t_pdf(x, df, loc, scale)

        # 2. Growth probabilities: P(r_t = r_{t-1}+1, x_{1:t})
        H = self.hazard_rate
        growth = self.run_length_probs * pred_probs * (1.0 - H)

        # 3. Changepoint probability: P(r_t = 0, x_{1:t})
        cp = float(np.sum(self.run_length_probs * pred_probs * H))

        # 4. Concatenate new run-length distribution
        new_probs = np.empty(len(growth) + 1, dtype=np.float64)
        new_probs[0] = cp
        new_probs[1:] = growth

        # 5. Normalise
        evidence = float(np.sum(new_probs))
        if evidence > 0:
            new_probs /= evidence
        self.run_length_probs = new_probs

        # 6. Update sufficient statistics
        new_mu = np.empty(len(self._mu) + 1, dtype=np.float64)
        new_kappa = np.empty(len(self._kappa) + 1, dtype=np.float64)
        new_alpha = np.empty(len(self._alpha) + 1, dtype=np.float64)
        new_beta = np.empty(len(self._beta) + 1, dtype=np.float64)

        # New changepoint resets to prior
        new_mu[0] = self.mu0
        new_kappa[0] = self.kappa0
        new_alpha[0] = self.alpha0
        new_beta[0] = self.beta0

        # Existing run lengths get updated
        old_mu = self._mu
        old_kappa = self._kappa
        old_alpha = self._alpha
        old_beta = self._beta

        new_kappa[1:] = old_kappa + 1.0
        new_mu[1:] = (old_kappa * old_mu + x) / new_kappa[1:]
        new_alpha[1:] = old_alpha + 0.5
        new_beta[1:] = (
            old_beta
            + 0.5 * old_kappa * (x - old_mu) ** 2 / new_kappa[1:]
        )

        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        return self.run_length_probs.copy()

    def detect(self, threshold: float = 0.5) -> bool:
        """Check whether a changepoint has been detected.

        A changepoint is signalled when the probability of run length
        zero exceeds the given threshold.

        Parameters
        ----------
        threshold : float, optional
            Detection threshold for ``P(r_t = 0)``. Default is
            ``0.5``.

        Returns
        -------
        bool
            ``True`` if a changepoint is detected.
        """
        if not 0.0 < threshold < 1.0:
            raise ValidationError(
                "threshold must be in (0, 1).",
                parameter="threshold",
            )
        if self.t == 0:
            return False
        return bool(self.run_length_probs[0] > threshold)

    def get_run_length_probabilities(self) -> np.ndarray:
        """Return the current run-length probability distribution.

        Returns
        -------
        numpy.ndarray of shape (t + 1,)
            Probability of each possible run length from 0 to *t*.
        """
        return self.run_length_probs.copy()

    def get_most_probable_run_length(self) -> int:
        """Return the run length with the highest probability.

        Returns
        -------
        int
            Most probable run length.
        """
        return int(np.argmax(self.run_length_probs))

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self.run_length_probs = np.array([1.0], dtype=np.float64)
        self.t = 0
        self._mu = np.array([self.mu0], dtype=np.float64)
        self._kappa = np.array([self.kappa0], dtype=np.float64)
        self._alpha = np.array([self.alpha0], dtype=np.float64)
        self._beta = np.array([self.beta0], dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"OnlineChangePointDetector("
            f"hazard_rate={self.hazard_rate}, "
            f"t={self.t})"
        )
