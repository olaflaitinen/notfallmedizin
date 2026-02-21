# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Clinical alert generation for vital sign monitoring.

This module implements a rule-based (and extensible) alert engine that
evaluates a dictionary of vital sign measurements against configurable
clinical rules, assigns severity levels, computes the National Early
Warning Score 2 (NEWS2), and applies alert suppression logic to reduce
alarm fatigue.

Classes
-------
AlertSeverity
    Enumeration of alert severity levels.
ClinicalAlert
    Immutable record for a single alert event.
ClinicalAlertEngine
    Configurable rule engine with NEWS2 integration and cooldown logic.

References
----------
.. [1] Royal College of Physicians. National Early Warning Score (NEWS) 2:
       Standardising the assessment of acute-illness severity in the NHS.
       London: RCP; 2017.
"""

from __future__ import annotations

import enum
import operator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from notfallmedizin.core.exceptions import (
    ComputationError,
    ValidationError,
)


# ======================================================================
# Enumerations and Data Structures
# ======================================================================


class AlertSeverity(enum.IntEnum):
    """Alert severity levels, ordered from most to least severe.

    The integer values allow direct comparison (``CRITICAL > HIGH``).
    """

    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass(frozen=True)
class ClinicalAlert:
    """Immutable record describing a single clinical alert.

    Parameters
    ----------
    parameter : str
        Vital sign parameter that triggered the alert (e.g.
        ``"heart_rate"``).
    value : float
        Observed value of the parameter.
    threshold : float
        Threshold value used in the triggering rule.
    severity : AlertSeverity
        Severity classification of the alert.
    message : str
        Human-readable description of the alert condition.
    timestamp : datetime
        Time at which the alert was generated.
    alert_type : str
        Short label for the clinical condition (e.g. ``"tachycardia"``).
    """

    parameter: str
    value: float
    threshold: float
    severity: AlertSeverity
    message: str
    timestamp: datetime
    alert_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dictionary representation of the alert.

        Returns
        -------
        dict of str to Any
            All fields serialized to basic Python types.
        """
        return {
            "parameter": self.parameter,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
        }


# ======================================================================
# Internal rule representation
# ======================================================================


_OPERATOR_MAP: Dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class _AlertRule:
    """Internal representation of a single alert rule."""

    parameter: str
    condition: str
    threshold: float
    severity: AlertSeverity
    message: str
    alert_type: str
    comparator: Callable[[float, float], bool]


# ======================================================================
# Clinical Alert Engine
# ======================================================================


class ClinicalAlertEngine:
    """Rule-based alert engine with NEWS2 integration and suppression.

    The engine evaluates a set of configurable rules against a
    dictionary of vital sign measurements. It supports:

    * Custom and default clinical rules.
    * NEWS2 score calculation and automatic escalation.
    * Aggregate severity scoring.
    * Alert suppression via per-rule cooldown timers to prevent
      alarm fatigue.

    Parameters
    ----------
    cooldown_seconds : float, optional
        Minimum number of seconds between two alerts of the same type.
        Default is ``0.0`` (no suppression).
    include_default_rules : bool, optional
        If ``True``, populate the rule set with standard clinical
        thresholds on construction. Default is ``True``.

    Attributes
    ----------
    rules : list of _AlertRule
        Currently registered alert rules.

    Examples
    --------
    >>> from datetime import datetime
    >>> engine = ClinicalAlertEngine(cooldown_seconds=300)
    >>> vitals = {
    ...     "heart_rate": 130.0,
    ...     "systolic_bp": 85.0,
    ...     "diastolic_bp": 55.0,
    ...     "respiratory_rate": 24.0,
    ...     "spo2": 91.0,
    ...     "temperature": 38.8,
    ... }
    >>> alerts = engine.evaluate(vitals, timestamp=datetime.now())
    >>> [a.alert_type for a in alerts]  # doctest: +SKIP
    ['tachycardia', 'hypotension', 'hypoxia', 'tachypnea']
    """

    def __init__(
        self,
        cooldown_seconds: float = 0.0,
        include_default_rules: bool = True,
    ) -> None:
        if cooldown_seconds < 0.0:
            raise ValidationError(
                message=(
                    "'cooldown_seconds' must be non-negative, "
                    f"got {cooldown_seconds}."
                ),
                parameter="cooldown_seconds",
            )
        self.cooldown_seconds: float = cooldown_seconds
        self.rules: List[_AlertRule] = []
        self._last_fired: Dict[str, datetime] = {}

        if include_default_rules:
            self._register_default_rules()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        parameter: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        message: str,
        alert_type: str = "",
    ) -> None:
        """Register a new alert rule.

        Parameters
        ----------
        parameter : str
            Vital sign parameter name the rule applies to.
        condition : str
            Comparison operator as a string. One of ``"<"``, ``"<="``,
            ``">"``, ``">="``, ``"=="``, ``"!="``.
        threshold : float
            Threshold value for the comparison.
        severity : AlertSeverity
            Severity assigned when the rule fires.
        message : str
            Human-readable message template for the alert.
        alert_type : str, optional
            Short label identifying the clinical condition. If empty,
            defaults to ``parameter + "_alert"``.

        Raises
        ------
        ValidationError
            If *condition* is not a recognized operator string.
        """
        if condition not in _OPERATOR_MAP:
            raise ValidationError(
                message=(
                    f"'condition' must be one of {list(_OPERATOR_MAP.keys())}, "
                    f"got {condition!r}."
                ),
                parameter="condition",
            )
        if not alert_type:
            alert_type = f"{parameter}_alert"

        rule = _AlertRule(
            parameter=parameter,
            condition=condition,
            threshold=threshold,
            severity=severity,
            message=message,
            alert_type=alert_type,
            comparator=_OPERATOR_MAP[condition],
        )
        self.rules.append(rule)

    def clear_rules(self) -> None:
        """Remove all registered rules."""
        self.rules.clear()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        vital_signs: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> List[ClinicalAlert]:
        """Evaluate all rules against the provided vital signs.

        Parameters
        ----------
        vital_signs : dict of str to float
            Current vital sign measurements. Keys should match the
            parameter names used in the registered rules.
        timestamp : datetime or None, optional
            Time of the evaluation. If ``None``, the current UTC time
            is used.

        Returns
        -------
        list of ClinicalAlert
            Alerts that fired, sorted by severity (most severe first).
            Alerts suppressed by the cooldown mechanism are excluded.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        fired: List[ClinicalAlert] = []

        for rule in self.rules:
            value = vital_signs.get(rule.parameter)
            if value is None:
                continue

            if rule.comparator(value, rule.threshold):
                if self._is_suppressed(rule.alert_type, timestamp):
                    continue

                alert = ClinicalAlert(
                    parameter=rule.parameter,
                    value=value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=rule.message,
                    timestamp=timestamp,
                    alert_type=rule.alert_type,
                )
                fired.append(alert)
                self._last_fired[rule.alert_type] = timestamp

        fired.sort(key=lambda a: a.severity, reverse=True)
        return fired

    # ------------------------------------------------------------------
    # NEWS2 scoring
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_news2(
        respiratory_rate: float,
        spo2: float,
        systolic_bp: float,
        heart_rate: float,
        temperature: float,
        consciousness: str = "alert",
        supplemental_o2: bool = False,
    ) -> int:
        """Compute the National Early Warning Score 2 (NEWS2).

        Each physiological parameter is mapped to a sub-score of 0--3
        according to the thresholds in the RCP NEWS2 specification [1]_.
        The sub-scores are summed to produce a total between 0 and 20.

        Parameters
        ----------
        respiratory_rate : float
            Breaths per minute.
        spo2 : float
            Peripheral oxygen saturation (%).
        systolic_bp : float
            Systolic blood pressure (mmHg).
        heart_rate : float
            Heart rate (bpm).
        temperature : float
            Body temperature (degrees Celsius).
        consciousness : {"alert", "cvpu"}, optional
            Level of consciousness. ``"alert"`` = fully alert,
            ``"cvpu"`` = new confusion, voice, pain, or unresponsive.
            Default is ``"alert"``.
        supplemental_o2 : bool, optional
            Whether the patient is receiving supplemental oxygen.
            Default is ``False``.

        Returns
        -------
        int
            Total NEWS2 score (0--20).

        Raises
        ------
        ValidationError
            If *consciousness* is not a recognized value.

        Notes
        -----
        The scoring tables are reproduced from the Royal College of
        Physicians NEWS2 chart [1]_.
        """
        if consciousness not in ("alert", "cvpu"):
            raise ValidationError(
                message=(
                    "'consciousness' must be 'alert' or 'cvpu', "
                    f"got {consciousness!r}."
                ),
                parameter="consciousness",
            )

        score = 0

        # Respiratory rate
        if respiratory_rate <= 8:
            score += 3
        elif respiratory_rate <= 11:
            score += 1
        elif respiratory_rate <= 20:
            score += 0
        elif respiratory_rate <= 24:
            score += 2
        else:
            score += 3

        # SpO2 (Scale 1, no supplemental O2 target)
        if not supplemental_o2:
            if spo2 <= 91:
                score += 3
            elif spo2 <= 93:
                score += 2
            elif spo2 <= 95:
                score += 1
            else:
                score += 0
        else:
            # Scale 2 for patients on supplemental O2
            if spo2 <= 83:
                score += 3
            elif spo2 <= 85:
                score += 2
            elif spo2 <= 87:
                score += 1
            elif spo2 <= 92:
                score += 0
            elif spo2 <= 94:
                score += 1
            elif spo2 <= 96:
                score += 2
            else:
                score += 3

        # Supplemental oxygen
        if supplemental_o2:
            score += 2

        # Systolic blood pressure
        if systolic_bp <= 90:
            score += 3
        elif systolic_bp <= 100:
            score += 2
        elif systolic_bp <= 110:
            score += 1
        elif systolic_bp <= 219:
            score += 0
        else:
            score += 3

        # Heart rate
        if heart_rate <= 40:
            score += 3
        elif heart_rate <= 50:
            score += 1
        elif heart_rate <= 90:
            score += 0
        elif heart_rate <= 110:
            score += 1
        elif heart_rate <= 130:
            score += 2
        else:
            score += 3

        # Temperature
        if temperature <= 35.0:
            score += 3
        elif temperature <= 36.0:
            score += 1
        elif temperature <= 38.0:
            score += 0
        elif temperature <= 39.0:
            score += 1
        else:
            score += 2

        # Consciousness
        if consciousness == "cvpu":
            score += 3

        return score

    @staticmethod
    def interpret_news2(score: int) -> str:
        """Return a clinical risk category for a NEWS2 score.

        Parameters
        ----------
        score : int
            Total NEWS2 score (0--20).

        Returns
        -------
        str
            One of ``"low"``, ``"low-medium"``, ``"medium"``, or
            ``"high"`` clinical risk.
        """
        if score <= 0:
            return "low"
        if score <= 4:
            return "low-medium"
        if score <= 6:
            return "medium"
        return "high"

    # ------------------------------------------------------------------
    # Aggregate scoring
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_severity(alerts: Sequence[ClinicalAlert]) -> AlertSeverity:
        """Determine the overall severity from a collection of alerts.

        The aggregate severity is the maximum individual severity. If
        three or more alerts of ``MEDIUM`` severity or above are present,
        the aggregate is promoted to at least ``HIGH``.

        Parameters
        ----------
        alerts : sequence of ClinicalAlert
            Alerts to aggregate.

        Returns
        -------
        AlertSeverity
            Overall severity level.
        """
        if not alerts:
            return AlertSeverity.INFO

        max_severity = max(a.severity for a in alerts)

        n_medium_plus = sum(
            1 for a in alerts if a.severity >= AlertSeverity.MEDIUM
        )
        if n_medium_plus >= 3 and max_severity < AlertSeverity.HIGH:
            return AlertSeverity.HIGH

        return AlertSeverity(max_severity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_suppressed(self, alert_type: str, now: datetime) -> bool:
        """Check whether an alert type is in its cooldown period.

        Parameters
        ----------
        alert_type : str
            Alert type identifier.
        now : datetime
            Current timestamp.

        Returns
        -------
        bool
            ``True`` if the alert should be suppressed.
        """
        if self.cooldown_seconds <= 0.0:
            return False

        last = self._last_fired.get(alert_type)
        if last is None:
            return False

        elapsed = (now - last).total_seconds()
        return elapsed < self.cooldown_seconds

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers, allowing suppressed alerts to fire."""
        self._last_fired.clear()

    # ------------------------------------------------------------------
    # Default clinical rules
    # ------------------------------------------------------------------

    def _register_default_rules(self) -> None:
        """Populate the rule set with standard vital sign thresholds.

        The default rules cover the most common emergency medicine alert
        conditions with thresholds based on widely accepted clinical
        practice.
        """
        defaults: List[Tuple[str, str, float, AlertSeverity, str, str]] = [
            # Heart rate
            (
                "heart_rate", "<", 50.0, AlertSeverity.HIGH,
                "Bradycardia: heart rate below 50 bpm.",
                "bradycardia",
            ),
            (
                "heart_rate", ">", 100.0, AlertSeverity.MEDIUM,
                "Tachycardia: heart rate above 100 bpm.",
                "tachycardia",
            ),
            (
                "heart_rate", ">", 150.0, AlertSeverity.CRITICAL,
                "Severe tachycardia: heart rate above 150 bpm.",
                "severe_tachycardia",
            ),
            # Blood pressure
            (
                "systolic_bp", "<", 90.0, AlertSeverity.CRITICAL,
                "Hypotension: systolic BP below 90 mmHg.",
                "hypotension",
            ),
            (
                "systolic_bp", ">", 180.0, AlertSeverity.HIGH,
                "Hypertension: systolic BP above 180 mmHg.",
                "hypertension",
            ),
            # SpO2
            (
                "spo2", "<", 94.0, AlertSeverity.MEDIUM,
                "Mild hypoxia: SpO2 below 94%.",
                "hypoxia",
            ),
            (
                "spo2", "<", 90.0, AlertSeverity.CRITICAL,
                "Severe hypoxia: SpO2 below 90%.",
                "severe_hypoxia",
            ),
            # Respiratory rate
            (
                "respiratory_rate", ">", 20.0, AlertSeverity.MEDIUM,
                "Tachypnea: respiratory rate above 20 breaths/min.",
                "tachypnea",
            ),
            (
                "respiratory_rate", "<", 8.0, AlertSeverity.CRITICAL,
                "Bradypnea: respiratory rate below 8 breaths/min.",
                "bradypnea",
            ),
            # Temperature
            (
                "temperature", "<", 35.0, AlertSeverity.HIGH,
                "Hypothermia: temperature below 35.0 C.",
                "hypothermia",
            ),
            (
                "temperature", ">", 38.5, AlertSeverity.MEDIUM,
                "Hyperthermia: temperature above 38.5 C.",
                "hyperthermia",
            ),
            (
                "temperature", ">", 40.0, AlertSeverity.CRITICAL,
                "Severe hyperthermia: temperature above 40.0 C.",
                "severe_hyperthermia",
            ),
        ]

        for param, cond, thresh, sev, msg, atype in defaults:
            self.add_rule(
                parameter=param,
                condition=cond,
                threshold=thresh,
                severity=sev,
                message=msg,
                alert_type=atype,
            )

    def __repr__(self) -> str:
        return (
            f"ClinicalAlertEngine("
            f"n_rules={len(self.rules)}, "
            f"cooldown_seconds={self.cooldown_seconds})"
        )
