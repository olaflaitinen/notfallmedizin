# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Manchester Triage System (MTS) implementation.

The Manchester Triage System is a clinical risk-management tool used
worldwide to prioritise patients presenting to emergency departments.
It assigns one of five clinical priority categories based on the
presenting complaint (presentograph) and a hierarchy of discriminators
evaluated from most urgent to least urgent.

Categories
----------
- **Red** (Immediate): Life-threatening conditions requiring immediate
  treatment. Target time to treatment: 0 minutes.
- **Orange** (Very Urgent): Serious conditions with potential for rapid
  deterioration. Target: 10 minutes.
- **Yellow** (Urgent): Conditions requiring prompt attention but not
  immediately life-threatening. Target: 60 minutes.
- **Green** (Standard): Conditions that are not urgent but require
  assessment. Target: 120 minutes.
- **Blue** (Non-Urgent): Minor conditions. Target: 240 minutes.

References
----------
.. [1] Mackway-Jones K, Marsden J, Windle J, eds. Emergency Triage:
       Manchester Triage Group. 3rd ed. Chichester, UK: John Wiley &
       Sons; 2014. ISBN 978-1-118-29906-1.
.. [2] Parenti N, Reggiani ML, Iannone P, Percudani D, Dowding D.
       A systematic review on the validity and reliability of an
       emergency department triage scale, the Manchester Triage System.
       Int J Nurs Stud. 2014;51(7):1062-1069.
       doi:10.1016/j.ijnurstu.2014.01.013
.. [3] Azeredo TR, Guedes HM, Rebelo de Almeida RA, et al. Efficacy of
       the Manchester Triage System: a systematic review. Int Emerg
       Nurs. 2015;23(2):47-52. doi:10.1016/j.ienj.2014.06.001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from notfallmedizin.core.base import BaseScorer
from notfallmedizin.core.exceptions import ValidationError


# ======================================================================
# Enumerations and constants
# ======================================================================


class MTSCategory(Enum):
    """MTS priority categories.

    Each member carries a ``(color, priority_level, max_wait_minutes)``
    tuple as its value.  ``priority_level`` 1 is the most urgent.
    """

    RED = ("red", 1, 0)
    ORANGE = ("orange", 2, 10)
    YELLOW = ("yellow", 3, 60)
    GREEN = ("green", 4, 120)
    BLUE = ("blue", 5, 240)

    def __init__(self, color: str, priority: int, max_wait: int) -> None:
        self.color = color
        self.priority_level = priority
        self.max_wait_minutes = max_wait


class MTSDiscriminatorType(Enum):
    """Recognised general discriminator categories.

    These are high-level discriminator families. Each family may
    contain multiple specific discriminators that map to different
    MTS categories.
    """

    LIFE_THREAT = "life_threat"
    PAIN = "pain"
    HEMORRHAGE = "hemorrhage"
    CONSCIOUSNESS = "consciousness"
    TEMPERATURE = "temperature"
    ACUITY_OF_ONSET = "acuity_of_onset"
    BREATHING = "breathing"
    SHOCK = "shock"
    NEUROLOGICAL_DEFICIT = "neurological_deficit"
    SEIZURE = "seizure"


_PAIN_SCALE_MAP: Dict[str, MTSCategory] = {
    "severe": MTSCategory.ORANGE,
    "moderate": MTSCategory.YELLOW,
    "mild": MTSCategory.GREEN,
    "none": MTSCategory.BLUE,
}
"""Mapping from pain-severity descriptors to MTS categories.

Pain is scored on a 0--10 numeric scale and mapped to descriptors:
severe (8--10), moderate (5--7), mild (1--4), none (0).
"""

_HEMORRHAGE_MAP: Dict[str, MTSCategory] = {
    "massive": MTSCategory.RED,
    "uncontrollable_major": MTSCategory.ORANGE,
    "moderate": MTSCategory.YELLOW,
    "minor": MTSCategory.GREEN,
    "none": MTSCategory.BLUE,
}
"""Mapping from hemorrhage severity to MTS categories."""

_CONSCIOUSNESS_MAP: Dict[str, MTSCategory] = {
    "unresponsive": MTSCategory.RED,
    "responds_to_pain": MTSCategory.ORANGE,
    "responds_to_voice": MTSCategory.YELLOW,
    "alert": MTSCategory.GREEN,
}
"""Mapping from consciousness level (AVPU variant) to MTS categories."""

_TEMPERATURE_MAP: Dict[str, MTSCategory] = {
    "very_hot": MTSCategory.ORANGE,
    "hot": MTSCategory.YELLOW,
    "warm": MTSCategory.GREEN,
    "normal": MTSCategory.BLUE,
    "cold": MTSCategory.YELLOW,
    "very_cold": MTSCategory.ORANGE,
}
"""Mapping from temperature descriptors to MTS categories.

Temperature ranges (degrees Celsius):
- very_hot:  > 41.0
- hot:       38.5 -- 41.0
- warm:      37.5 -- 38.5 (exclusive)
- normal:    35.0 -- 37.5
- cold:      32.0 -- 35.0 (exclusive)
- very_cold: < 32.0
"""

_TEMPERATURE_THRESHOLDS: List[Tuple[float, str]] = [
    (41.0, "very_hot"),
    (38.5, "hot"),
    (37.5, "warm"),
    (35.0, "normal"),
    (32.0, "cold"),
]
"""Ordered thresholds for classifying temperature into descriptors.

Evaluated top-down: the first threshold that the value exceeds (or
equals) determines the descriptor.  Values below 32.0 are classified
as ``very_cold``.
"""

_BREATHING_MAP: Dict[str, MTSCategory] = {
    "airway_compromise": MTSCategory.RED,
    "inadequate_breathing": MTSCategory.RED,
    "stridor": MTSCategory.ORANGE,
    "severe_respiratory_distress": MTSCategory.ORANGE,
    "moderate_respiratory_distress": MTSCategory.YELLOW,
    "mild_respiratory_distress": MTSCategory.GREEN,
    "normal": MTSCategory.BLUE,
}
"""Mapping from breathing assessment descriptors to MTS categories."""

_GENERAL_DISCRIMINATOR_MAPS: Dict[str, Dict[str, MTSCategory]] = {
    "pain_scale": _PAIN_SCALE_MAP,
    "hemorrhage_severity": _HEMORRHAGE_MAP,
    "consciousness_level": _CONSCIOUSNESS_MAP,
    "temperature_range": _TEMPERATURE_MAP,
    "breathing": _BREATHING_MAP,
}
"""Master registry of discriminator-type to severity-category maps."""

_PRESENTOGRAPH_DEFAULTS: Dict[str, MTSCategory] = {
    "cardiac_chest_pain": MTSCategory.ORANGE,
    "respiratory_distress": MTSCategory.ORANGE,
    "major_trauma": MTSCategory.RED,
    "head_injury": MTSCategory.ORANGE,
    "abdominal_pain": MTSCategory.YELLOW,
    "limb_injury": MTSCategory.YELLOW,
    "allergic_reaction": MTSCategory.ORANGE,
    "poisoning": MTSCategory.ORANGE,
    "mental_health": MTSCategory.YELLOW,
    "unwell_child": MTSCategory.YELLOW,
    "ear_problems": MTSCategory.GREEN,
    "dental_problems": MTSCategory.GREEN,
    "minor_wound": MTSCategory.GREEN,
    "rash": MTSCategory.GREEN,
    "general_unwell": MTSCategory.YELLOW,
    "other": MTSCategory.YELLOW,
}
"""Default (baseline) category for selected presentographs.

The baseline is used when no specific discriminator escalates the
category.  If a discriminator evaluates to a more urgent category,
the more urgent category takes precedence.
"""


# ======================================================================
# Result dataclass
# ======================================================================


@dataclass(frozen=True)
class MTSResult:
    """Structured result of an MTS triage assessment.

    Parameters
    ----------
    category : str
        Priority category name: ``"Red"``, ``"Orange"``, ``"Yellow"``,
        ``"Green"``, or ``"Blue"``.
    color : str
        Lower-case colour string matching the category.
    max_wait_minutes : int
        Maximum recommended time (in minutes) before the patient
        should be seen by a clinician.
    discriminator_matched : str
        The specific discriminator that determined the assigned
        category, or ``"presentograph_default"`` if no discriminator
        escalated the category above the baseline.
    priority_level : int
        Numeric priority (1 = most urgent, 5 = least urgent).
    reasoning : list of str
        Ordered list of reasoning steps.
    """

    category: str
    color: str
    max_wait_minutes: int
    discriminator_matched: str
    priority_level: int
    reasoning: List[str] = field(default_factory=list)

    def __int__(self) -> int:
        return self.priority_level

    def __float__(self) -> float:
        return float(self.priority_level)


# ======================================================================
# Calculator
# ======================================================================


class MTSTriageCalculator(BaseScorer):
    """Manchester Triage System (MTS) calculator.

    Implements a discriminator-based triage system that evaluates a
    patient's presenting complaint (presentograph) and a set of clinical
    discriminators to assign one of five priority categories.

    The algorithm evaluates discriminators in descending order of
    urgency.  The most urgent discriminator match determines the
    final category, subject to the constraint that the result is
    never less urgent than the presentograph's baseline category.

    Parameters
    ----------
    allow_blue : bool, optional
        If ``False`` (default ``True``), the calculator never assigns
        the Blue (non-urgent) category and uses Green as the minimum.
        Some institutions disable Blue for liability reasons.

    Examples
    --------
    >>> calc = MTSTriageCalculator()
    >>> result = calc.calculate(
    ...     presentograph="abdominal_pain",
    ...     discriminators={
    ...         "pain_scale": "severe",
    ...         "hemorrhage_severity": "none",
    ...         "consciousness_level": "alert",
    ...         "temperature_range": "normal",
    ...     },
    ... )
    >>> result.category
    'Orange'
    >>> result.max_wait_minutes
    10

    References
    ----------
    .. [1] Mackway-Jones K, et al. Emergency Triage: Manchester Triage
           Group. 3rd ed. Wiley; 2014.
    """

    def __init__(self, allow_blue: bool = True) -> None:
        super().__init__(
            name="Manchester Triage System",
            version="3.0",
            references=[
                "Mackway-Jones K, Marsden J, Windle J, eds. Emergency "
                "Triage: Manchester Triage Group. 3rd ed. Wiley; 2014.",
                "Parenti N, et al. A systematic review on the validity "
                "and reliability of the Manchester Triage System. Int J "
                "Nurs Stud. 2014;51(7):1062-1069.",
            ],
        )
        self.allow_blue = allow_blue

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    def validate_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate and normalize MTS triage inputs.

        Parameters
        ----------
        **kwargs
            Expected keys:

            - ``presentograph`` (str): Presenting complaint category.
            - ``discriminators`` (dict of str to str): Discriminator
              evaluations.  Keys are discriminator types (e.g.
              ``"pain_scale"``); values are severity descriptors (e.g.
              ``"severe"``).

        Returns
        -------
        dict of str to Any
            Validated inputs.

        Raises
        ------
        ValidationError
            If required keys are missing, presentograph is unrecognised,
            or discriminator values are invalid.
        """
        if "presentograph" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'presentograph'.",
                parameter="presentograph",
            )
        if "discriminators" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'discriminators'.",
                parameter="discriminators",
            )

        presentograph = str(kwargs["presentograph"]).strip().lower()

        disc_raw = kwargs["discriminators"]
        if not isinstance(disc_raw, dict):
            raise ValidationError(
                message=(
                    "Parameter 'discriminators' must be a dictionary, "
                    f"got {type(disc_raw).__name__}."
                ),
                parameter="discriminators",
            )

        discriminators: Dict[str, str] = {}
        for disc_type, severity in disc_raw.items():
            disc_type_lower = str(disc_type).strip().lower()
            severity_lower = str(severity).strip().lower()

            if disc_type_lower in _GENERAL_DISCRIMINATOR_MAPS:
                valid_severities = set(
                    _GENERAL_DISCRIMINATOR_MAPS[disc_type_lower].keys()
                )
                if severity_lower not in valid_severities:
                    raise ValidationError(
                        message=(
                            f"Invalid severity '{severity_lower}' for "
                            f"discriminator '{disc_type_lower}'. "
                            f"Valid values: {sorted(valid_severities)}."
                        ),
                        parameter=disc_type_lower,
                    )
            discriminators[disc_type_lower] = severity_lower

        return {
            "presentograph": presentograph,
            "discriminators": discriminators,
        }

    def calculate(self, **kwargs: Any) -> MTSResult:
        """Compute the MTS triage category.

        Parameters
        ----------
        **kwargs
            Validated MTS parameters. See :meth:`validate_inputs` for
            expected keys.

        Returns
        -------
        MTSResult
            Structured triage result including category, colour,
            maximum wait time, matched discriminator, and reasoning
            trace.
        """
        presentograph: str = kwargs["presentograph"]
        discriminators: Dict[str, str] = kwargs["discriminators"]
        reasoning: List[str] = []

        # Determine the presentograph baseline
        if presentograph in _PRESENTOGRAPH_DEFAULTS:
            baseline = _PRESENTOGRAPH_DEFAULTS[presentograph]
            reasoning.append(
                f"Presentograph '{presentograph}' has baseline category "
                f"{baseline.name} (priority {baseline.priority_level})."
            )
        else:
            baseline = MTSCategory.YELLOW
            reasoning.append(
                f"Presentograph '{presentograph}' is not in the standard "
                f"set; defaulting to YELLOW baseline."
            )

        # Evaluate each discriminator and find the most urgent match
        best_category = baseline
        best_discriminator = "presentograph_default"

        for disc_type, severity in discriminators.items():
            category_map = _GENERAL_DISCRIMINATOR_MAPS.get(disc_type)
            if category_map is None:
                reasoning.append(
                    f"Discriminator '{disc_type}' is not in the standard "
                    f"set; skipping."
                )
                continue

            matched_category = category_map.get(severity)
            if matched_category is None:
                continue

            reasoning.append(
                f"Discriminator '{disc_type}' = '{severity}' maps to "
                f"{matched_category.name} (priority "
                f"{matched_category.priority_level})."
            )

            if matched_category.priority_level < best_category.priority_level:
                best_category = matched_category
                best_discriminator = f"{disc_type}:{severity}"

        # Apply minimum floor
        if not self.allow_blue and best_category == MTSCategory.BLUE:
            best_category = MTSCategory.GREEN
            reasoning.append(
                "Blue category is disabled; escalated to GREEN."
            )

        reasoning.append(
            f"Final assignment: {best_category.name} "
            f"(matched discriminator: {best_discriminator})."
        )

        return MTSResult(
            category=best_category.name.capitalize(),
            color=best_category.color,
            max_wait_minutes=best_category.max_wait_minutes,
            discriminator_matched=best_discriminator,
            priority_level=best_category.priority_level,
            reasoning=reasoning,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the theoretical range of MTS priority levels.

        Returns
        -------
        tuple of (float, float)
            ``(1.0, 5.0)`` representing priority levels Red (1) through
            Blue (5).
        """
        return (1.0, 5.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Return a clinical interpretation for a given MTS priority level.

        Parameters
        ----------
        score : int or float
            MTS priority level (1--5).

        Returns
        -------
        str
            Textual interpretation.

        Raises
        ------
        ValidationError
            If the score is not in {1, 2, 3, 4, 5}.
        """
        level = int(score)
        interpretations = {
            1: "Red (Immediate): Life-threatening. Treatment required immediately.",
            2: "Orange (Very Urgent): Serious condition. Target: 10 minutes.",
            3: "Yellow (Urgent): Requires prompt attention. Target: 60 minutes.",
            4: "Green (Standard): Not urgent. Target: 120 minutes.",
            5: "Blue (Non-Urgent): Minor condition. Target: 240 minutes.",
        }
        if level not in interpretations:
            raise ValidationError(
                message=f"MTS priority level must be in {{1,2,3,4,5}}, got {level}.",
                parameter="score",
            )
        return interpretations[level]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def classify_temperature(temperature_celsius: float) -> str:
        """Classify a body temperature into an MTS descriptor.

        Parameters
        ----------
        temperature_celsius : float
            Body temperature in degrees Celsius.

        Returns
        -------
        str
            One of ``"very_hot"``, ``"hot"``, ``"warm"``, ``"normal"``,
            ``"cold"``, or ``"very_cold"``.
        """
        for threshold, descriptor in _TEMPERATURE_THRESHOLDS:
            if temperature_celsius >= threshold:
                return descriptor
        return "very_cold"

    @staticmethod
    def classify_pain(pain_score: int) -> str:
        """Classify a numeric pain score (0--10) into an MTS descriptor.

        Parameters
        ----------
        pain_score : int
            Pain intensity on a 0--10 numeric rating scale.

        Returns
        -------
        str
            One of ``"severe"``, ``"moderate"``, ``"mild"``, or
            ``"none"``.

        Raises
        ------
        ValidationError
            If the pain score is outside [0, 10].
        """
        if not (0 <= pain_score <= 10):
            raise ValidationError(
                message=f"Pain score must be in [0, 10], got {pain_score}.",
                parameter="pain_score",
            )
        if pain_score >= 8:
            return "severe"
        if pain_score >= 5:
            return "moderate"
        if pain_score >= 1:
            return "mild"
        return "none"

    @staticmethod
    def get_available_presentographs() -> FrozenSet[str]:
        """Return the set of recognised presentograph identifiers.

        Returns
        -------
        frozenset of str
            Presentograph names that have explicit baseline categories.
        """
        return frozenset(_PRESENTOGRAPH_DEFAULTS.keys())
