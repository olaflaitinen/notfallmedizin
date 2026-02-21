# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Real-time vital signs monitoring engine.

This module provides a stateful monitor that tracks patient vital signs
over time, computes derived hemodynamic indices, and exposes a sliding
window view of observation history.

Classes
-------
VitalSignsState
    Immutable snapshot of a patient's current vital signs and derived
    metrics.
VitalSignsMonitor
    Accumulates observations and computes derived indices such as MAP,
    Shock Index, Modified Shock Index, and Age-Adjusted Shock Index.

References
----------
.. [1] Allgower M, Burri C. Schockindex. Dtsch Med Wochenschr.
       1967;92(43):1947-1950.
.. [2] Singh A et al. Systolic blood pressure, diastolic blood pressure,
       and mean arterial pressure. Chest. 2002;122(5):1633-1639.
.. [3] Zarzaur BL et al. Age-adjusted shock index. J Trauma Acute Care
       Surg. 2015;78(2):352-359.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from notfallmedizin.core.exceptions import (
    ClinicalRangeError,
    ComputationError,
    ValidationError,
)
from notfallmedizin.core.validators import validate_vital_signs


# ======================================================================
# Data Structures
# ======================================================================


@dataclass(frozen=True)
class VitalSignsState:
    """Immutable snapshot of a patient's vital signs and derived metrics.

    Parameters
    ----------
    timestamp : datetime
        Time at which the observation was recorded.
    heart_rate : float
        Heart rate in beats per minute (bpm).
    systolic_bp : float
        Systolic blood pressure in mmHg.
    diastolic_bp : float
        Diastolic blood pressure in mmHg.
    respiratory_rate : float
        Respiratory rate in breaths per minute.
    spo2 : float
        Peripheral oxygen saturation as a percentage (0--100).
    temperature : float
        Body temperature in degrees Celsius.
    mean_arterial_pressure : float
        MAP computed as DBP + (1/3) * (SBP - DBP).
    shock_index : float
        Classic Shock Index (HR / SBP).
    modified_shock_index : float
        Modified Shock Index (HR / MAP).
    """

    timestamp: datetime
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    respiratory_rate: float
    spo2: float
    temperature: float
    mean_arterial_pressure: float
    shock_index: float
    modified_shock_index: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dictionary representation of the state.

        Returns
        -------
        dict of str to Any
            All fields as key-value pairs.
        """
        return {
            "timestamp": self.timestamp,
            "heart_rate": self.heart_rate,
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "respiratory_rate": self.respiratory_rate,
            "spo2": self.spo2,
            "temperature": self.temperature,
            "mean_arterial_pressure": self.mean_arterial_pressure,
            "shock_index": self.shock_index,
            "modified_shock_index": self.modified_shock_index,
        }


@dataclass
class _Observation:
    """Internal mutable record for a single set of vital sign readings."""

    timestamp: datetime
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    respiratory_rate: float
    spo2: float
    temperature: float


# ======================================================================
# Vital Signs Monitor
# ======================================================================


class VitalSignsMonitor:
    """Real-time vital signs tracking with derived hemodynamic indices.

    The monitor accumulates time-stamped observations, validates each
    measurement against physiologically plausible ranges, and provides
    methods to compute clinically relevant derived values.

    Parameters
    ----------
    max_history : int, optional
        Maximum number of observations retained in the internal buffer.
        Oldest observations are discarded when the limit is reached.
        Default is ``10_000``.

    Attributes
    ----------
    n_observations : int
        Total number of observations added since instantiation.

    Examples
    --------
    >>> from datetime import datetime
    >>> monitor = VitalSignsMonitor()
    >>> monitor.add_observation(
    ...     timestamp=datetime(2025, 1, 1, 12, 0),
    ...     heart_rate=72.0,
    ...     systolic_bp=120.0,
    ...     diastolic_bp=80.0,
    ...     respiratory_rate=16.0,
    ...     spo2=98.0,
    ...     temperature=36.8,
    ... )
    >>> state = monitor.get_current_state()
    >>> round(state.mean_arterial_pressure, 2)
    93.33
    """

    def __init__(self, max_history: int = 10_000) -> None:
        if not isinstance(max_history, int) or max_history < 1:
            raise ValidationError(
                message=(
                    "'max_history' must be a positive integer, "
                    f"got {max_history!r}."
                ),
                parameter="max_history",
            )
        self.max_history: int = max_history
        self._buffer: Deque[_Observation] = deque(maxlen=max_history)
        self._lock: threading.Lock = threading.Lock()
        self.n_observations: int = 0

    # ------------------------------------------------------------------
    # Observation management
    # ------------------------------------------------------------------

    def add_observation(
        self,
        timestamp: datetime,
        heart_rate: float,
        systolic_bp: float,
        diastolic_bp: float,
        respiratory_rate: float,
        spo2: float,
        temperature: float,
    ) -> VitalSignsState:
        """Record a new set of vital sign measurements.

        All six vital parameters are validated for type and
        physiologically plausible range before storage. The derived
        metrics (MAP, SI, MSI) are computed and returned.

        Parameters
        ----------
        timestamp : datetime
            Observation time.
        heart_rate : float
            Heart rate in bpm.
        systolic_bp : float
            Systolic blood pressure in mmHg.
        diastolic_bp : float
            Diastolic blood pressure in mmHg.
        respiratory_rate : float
            Respiratory rate in breaths per minute.
        spo2 : float
            SpO2 as a percentage (0--100).
        temperature : float
            Body temperature in degrees Celsius.

        Returns
        -------
        VitalSignsState
            Snapshot including derived metrics for the new observation.

        Raises
        ------
        ValidationError
            If *timestamp* is not a ``datetime`` instance or any vital
            parameter is non-numeric.
        ClinicalRangeError
            If any vital parameter is outside its acceptable range.
        """
        if not isinstance(timestamp, datetime):
            raise ValidationError(
                message=(
                    "'timestamp' must be a datetime instance, "
                    f"got {type(timestamp).__name__}."
                ),
                parameter="timestamp",
            )

        validated = validate_vital_signs(
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            respiratory_rate=respiratory_rate,
            spo2=spo2,
            temperature=temperature,
        )

        obs = _Observation(
            timestamp=timestamp,
            heart_rate=validated["heart_rate"],
            systolic_bp=validated["systolic_bp"],
            diastolic_bp=validated["diastolic_bp"],
            respiratory_rate=validated["respiratory_rate"],
            spo2=validated["spo2"],
            temperature=validated["temperature"],
        )

        with self._lock:
            self._buffer.append(obs)
            self.n_observations += 1

        return self._build_state(obs)

    def get_current_state(self) -> VitalSignsState:
        """Return a snapshot of the most recent observation.

        Returns
        -------
        VitalSignsState
            Current vital signs and derived metrics.

        Raises
        ------
        ComputationError
            If no observations have been recorded.
        """
        with self._lock:
            if not self._buffer:
                raise ComputationError(
                    "No observations have been recorded yet."
                )
            obs = self._buffer[-1]
        return self._build_state(obs)

    def get_history(
        self,
        window_minutes: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of recent observations.

        Parameters
        ----------
        window_minutes : float or None, optional
            If provided, only observations within the last
            *window_minutes* minutes (relative to the most recent
            observation) are included. If ``None``, the entire buffer
            is returned.

        Returns
        -------
        pandas.DataFrame
            Columns: ``timestamp``, ``heart_rate``, ``systolic_bp``,
            ``diastolic_bp``, ``respiratory_rate``, ``spo2``,
            ``temperature``, ``mean_arterial_pressure``,
            ``shock_index``, ``modified_shock_index``.

        Raises
        ------
        ComputationError
            If no observations have been recorded.
        ValidationError
            If *window_minutes* is not a positive number.
        """
        with self._lock:
            if not self._buffer:
                raise ComputationError(
                    "No observations have been recorded yet."
                )
            snapshot: List[_Observation] = list(self._buffer)

        if window_minutes is not None:
            if not isinstance(window_minutes, (int, float)) or window_minutes <= 0:
                raise ValidationError(
                    message=(
                        "'window_minutes' must be a positive number, "
                        f"got {window_minutes!r}."
                    ),
                    parameter="window_minutes",
                )
            cutoff = snapshot[-1].timestamp - timedelta(minutes=window_minutes)
            snapshot = [obs for obs in snapshot if obs.timestamp >= cutoff]

        rows: List[Dict[str, Any]] = []
        for obs in snapshot:
            state = self._build_state(obs)
            rows.append(state.to_dict())

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Derived hemodynamic indices
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_map(systolic_bp: float, diastolic_bp: float) -> float:
        """Compute mean arterial pressure (MAP).

        .. math::

            \\text{MAP} = \\text{DBP} + \\frac{1}{3}(\\text{SBP} - \\text{DBP})

        Parameters
        ----------
        systolic_bp : float
            Systolic blood pressure in mmHg.
        diastolic_bp : float
            Diastolic blood pressure in mmHg.

        Returns
        -------
        float
            Mean arterial pressure in mmHg.
        """
        return diastolic_bp + (systolic_bp - diastolic_bp) / 3.0

    @staticmethod
    def calculate_shock_index(heart_rate: float, systolic_bp: float) -> float:
        """Compute the classic Shock Index (SI).

        .. math::

            \\text{SI} = \\frac{\\text{HR}}{\\text{SBP}}

        A value > 0.7 suggests hemodynamic instability; values > 1.0
        are associated with increased mortality risk [1]_.

        Parameters
        ----------
        heart_rate : float
            Heart rate in bpm.
        systolic_bp : float
            Systolic blood pressure in mmHg.

        Returns
        -------
        float
            Shock Index (dimensionless).

        Raises
        ------
        ComputationError
            If *systolic_bp* is zero.
        """
        if systolic_bp == 0.0:
            raise ComputationError(
                "Cannot compute Shock Index: systolic_bp is zero."
            )
        return heart_rate / systolic_bp

    @staticmethod
    def calculate_modified_shock_index(
        heart_rate: float,
        mean_arterial_pressure: float,
    ) -> float:
        """Compute the Modified Shock Index (MSI).

        .. math::

            \\text{MSI} = \\frac{\\text{HR}}{\\text{MAP}}

        Parameters
        ----------
        heart_rate : float
            Heart rate in bpm.
        mean_arterial_pressure : float
            Mean arterial pressure in mmHg.

        Returns
        -------
        float
            Modified Shock Index (dimensionless).

        Raises
        ------
        ComputationError
            If *mean_arterial_pressure* is zero.
        """
        if mean_arterial_pressure == 0.0:
            raise ComputationError(
                "Cannot compute Modified Shock Index: MAP is zero."
            )
        return heart_rate / mean_arterial_pressure

    @staticmethod
    def calculate_age_adjusted_shock_index(
        heart_rate: float,
        systolic_bp: float,
        age: float,
    ) -> float:
        """Compute the Age-Adjusted Shock Index (AASI).

        .. math::

            \\text{AASI} = \\text{HR} \\times \\frac{\\text{age}}{\\text{SBP}}

        Higher values indicate greater hemodynamic compromise,
        normalized for age-related differences in baseline heart rate
        and blood pressure [3]_.

        Parameters
        ----------
        heart_rate : float
            Heart rate in bpm.
        systolic_bp : float
            Systolic blood pressure in mmHg.
        age : float
            Patient age in years.

        Returns
        -------
        float
            Age-Adjusted Shock Index (dimensionless).

        Raises
        ------
        ComputationError
            If *systolic_bp* is zero.
        ValidationError
            If *age* is negative.
        """
        if systolic_bp == 0.0:
            raise ComputationError(
                "Cannot compute Age-Adjusted Shock Index: systolic_bp is zero."
            )
        if age < 0.0:
            raise ValidationError(
                message=f"'age' must be non-negative, got {age}.",
                parameter="age",
            )
        return heart_rate * age / systolic_bp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(self, obs: _Observation) -> VitalSignsState:
        """Construct a ``VitalSignsState`` from an observation record.

        Parameters
        ----------
        obs : _Observation
            Raw observation data.

        Returns
        -------
        VitalSignsState
            Snapshot with derived metrics populated.
        """
        map_val = self.calculate_map(obs.systolic_bp, obs.diastolic_bp)

        si = (
            self.calculate_shock_index(obs.heart_rate, obs.systolic_bp)
            if obs.systolic_bp > 0.0
            else float("inf")
        )

        msi = (
            self.calculate_modified_shock_index(obs.heart_rate, map_val)
            if map_val > 0.0
            else float("inf")
        )

        return VitalSignsState(
            timestamp=obs.timestamp,
            heart_rate=obs.heart_rate,
            systolic_bp=obs.systolic_bp,
            diastolic_bp=obs.diastolic_bp,
            respiratory_rate=obs.respiratory_rate,
            spo2=obs.spo2,
            temperature=obs.temperature,
            mean_arterial_pressure=map_val,
            shock_index=si,
            modified_shock_index=msi,
        )

    def __repr__(self) -> str:
        return (
            f"VitalSignsMonitor("
            f"max_history={self.max_history}, "
            f"n_observations={self.n_observations})"
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
