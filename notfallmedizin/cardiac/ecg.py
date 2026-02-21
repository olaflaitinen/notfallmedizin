# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""ECG signal processing and heart rate variability analysis.

Implements the Pan-Tompkins algorithm for QRS detection and computes
both time-domain and frequency-domain HRV metrics.

References
----------
Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
    IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.
Task Force of ESC and NASPE. (1996). Heart rate variability: standards of
    measurement. Circulation, 93(5), 1043-1065.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, welch

from notfallmedizin.core.base import BaseTransformer
from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class HRVMetrics:
    """Heart rate variability metrics.

    Attributes
    ----------
    mean_rr_ms : float
    sdnn_ms : float
    rmssd_ms : float
    pnn50_percent : float
    lf_power_ms2 : float
    hf_power_ms2 : float
    lf_hf_ratio : float
    """

    mean_rr_ms: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_percent: float
    lf_power_ms2: float
    hf_power_ms2: float
    lf_hf_ratio: float


class ECGProcessor(BaseTransformer):
    """ECG signal processor with Pan-Tompkins QRS detection and HRV analysis.

    Parameters
    ----------
    sampling_rate : int, default=500
        Sampling frequency of the ECG signal in Hz.
    """

    def __init__(self, sampling_rate: int = 500) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate

    def fit(self, X: np.ndarray, y: object = None) -> "ECGProcessor":
        """No-op; provided for API compatibility."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Preprocess a batch of ECG signals.

        Parameters
        ----------
        X : np.ndarray of shape (n_signals, n_samples)

        Returns
        -------
        np.ndarray
            Bandpass-filtered signals.
        """
        if X.ndim == 1:
            return self.preprocess(X)
        return np.array([self.preprocess(row) for row in X])

    def preprocess(
        self, signal: np.ndarray, sampling_rate: Optional[int] = None
    ) -> np.ndarray:
        """Apply bandpass filter (0.5-40 Hz) to an ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal (1-D).
        sampling_rate : int, optional
            Overrides instance sampling rate.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        fs = sampling_rate or self.sampling_rate
        nyq = fs / 2.0
        low = 0.5 / nyq
        high = min(40.0 / nyq, 0.99)
        b, a = butter(N=2, Wn=[low, high], btype="band")
        return filtfilt(b, a, signal).astype(np.float64)

    def detect_r_peaks(
        self, signal: np.ndarray, sampling_rate: Optional[int] = None
    ) -> np.ndarray:
        """Detect R-peaks using the Pan-Tompkins algorithm.

        Steps
        -----
        1. Bandpass filter (5-15 Hz)
        2. Differentiation
        3. Squaring
        4. Moving-window integration
        5. Adaptive thresholding

        Parameters
        ----------
        signal : np.ndarray
        sampling_rate : int, optional

        Returns
        -------
        np.ndarray
            Indices of detected R-peaks.
        """
        fs = sampling_rate or self.sampling_rate
        nyq = fs / 2.0

        low = 5.0 / nyq
        high = min(15.0 / nyq, 0.99)
        b, a = butter(N=1, Wn=[low, high], btype="band")
        filtered = filtfilt(b, a, signal)

        diff = np.diff(filtered)
        squared = diff ** 2

        win_size = int(0.150 * fs)
        if win_size < 1:
            win_size = 1
        integrated = np.convolve(squared, np.ones(win_size) / win_size, mode="same")

        threshold = 0.5 * np.max(integrated)
        candidates = np.where(integrated > threshold)[0]

        if len(candidates) == 0:
            return np.array([], dtype=int)

        refractory = int(0.2 * fs)
        peaks = [candidates[0]]
        for idx in candidates[1:]:
            if idx - peaks[-1] >= refractory:
                peaks.append(idx)

        refined = []
        search_radius = int(0.075 * fs)
        for pk in peaks:
            lo = max(0, pk - search_radius)
            hi = min(len(signal), pk + search_radius + 1)
            refined.append(lo + int(np.argmax(signal[lo:hi])))

        return np.array(refined, dtype=int)

    @staticmethod
    def calculate_heart_rate(r_peaks: np.ndarray, sampling_rate: int = 500) -> float:
        """Calculate heart rate from R-peak indices.

        Parameters
        ----------
        r_peaks : np.ndarray
        sampling_rate : int

        Returns
        -------
        float
            Heart rate in beats per minute.
        """
        if len(r_peaks) < 2:
            raise ValidationError("At least 2 R-peaks required.")
        rr_seconds = np.diff(r_peaks) / sampling_rate
        mean_rr = float(np.mean(rr_seconds))
        if mean_rr <= 0:
            raise ValidationError("Invalid RR intervals.")
        return 60.0 / mean_rr

    @staticmethod
    def calculate_rr_intervals(
        r_peaks: np.ndarray, sampling_rate: int = 500
    ) -> np.ndarray:
        """Calculate RR intervals in milliseconds.

        Parameters
        ----------
        r_peaks : np.ndarray
        sampling_rate : int

        Returns
        -------
        np.ndarray
        """
        if len(r_peaks) < 2:
            raise ValidationError("At least 2 R-peaks required.")
        return (np.diff(r_peaks) / sampling_rate) * 1000.0

    @staticmethod
    def calculate_hrv_metrics(rr_intervals_ms: np.ndarray) -> HRVMetrics:
        """Compute time-domain and frequency-domain HRV metrics.

        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            RR intervals in milliseconds.

        Returns
        -------
        HRVMetrics
        """
        rr = np.asarray(rr_intervals_ms, dtype=np.float64)
        if len(rr) < 5:
            raise ValidationError("At least 5 RR intervals required for HRV.")

        mean_rr = float(np.mean(rr))
        sdnn = float(np.std(rr, ddof=1))
        successive_diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)

        # Frequency-domain via Welch on interpolated RR series
        rr_seconds = rr / 1000.0
        cumulative = np.cumsum(rr_seconds)
        cumulative = cumulative - cumulative[0]

        fs_interp = 4.0  # 4 Hz resampling
        t_interp = np.arange(0, cumulative[-1], 1.0 / fs_interp)
        rr_interp = np.interp(t_interp, cumulative[:-1], rr_seconds[:-1])
        rr_interp = rr_interp - np.mean(rr_interp)

        freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=min(256, len(rr_interp)))

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)

        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
        lf_hf = lf_power / hf_power if hf_power > 0 else 0.0

        return HRVMetrics(
            mean_rr_ms=round(mean_rr, 2),
            sdnn_ms=round(sdnn, 2),
            rmssd_ms=round(rmssd, 2),
            pnn50_percent=round(pnn50, 2),
            lf_power_ms2=round(lf_power * 1e6, 2),
            hf_power_ms2=round(hf_power * 1e6, 2),
            lf_hf_ratio=round(lf_hf, 3),
        )
