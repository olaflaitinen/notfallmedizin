# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Pediatric clinical scoring systems.

This module implements the Pediatric Early Warning Score (PEWS) and
the APGAR score for neonatal assessment.

Classes
-------
PEWSScore
    Pediatric Early Warning Score with 3 components (0--9).
APGARScore
    Neonatal assessment with 5 components (0--10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from notfallmedizin.core.base import BaseScorer
from notfallmedizin.core.exceptions import ClinicalRangeError, ValidationError


@dataclass(frozen=True)
class ScoringResult:
    """Immutable container for a clinical score calculation result.

    Attributes
    ----------
    total_score : float
        The aggregate numeric score.
    component_scores : dict of str to float
        Individual sub-score contributions keyed by component name.
    interpretation : str
        Human-readable interpretation of the total score.
    risk_category : str
        Risk stratification label (e.g. ``"low"``, ``"high"``).
    mortality_estimate : float or None
        Estimated mortality proportion, if applicable.
    """

    total_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    risk_category: str = ""
    mortality_estimate: Optional[float] = None


# ======================================================================
# Pediatric Early Warning Score (PEWS)
# ======================================================================


class PEWSScore(BaseScorer):
    """Pediatric Early Warning Score (PEWS).

    Three components, each scored 0--3, yielding a total of 0--9.
    Higher scores indicate increased risk of clinical deterioration.

    Parameters (passed to ``calculate``)
    -------------------------------------
    behavior : int
        Behavioral assessment (0--3).
            0 = Playing / appropriate.
            1 = Sleeping.
            2 = Irritable.
            3 = Lethargic / reduced response to pain.
    cardiovascular : int
        Cardiovascular assessment (0--3).
            0 = Pink, capillary refill 1-2 s.
            1 = Pale or capillary refill 3 s.
            2 = Grey or capillary refill 4 s, tachycardia.
            3 = Grey and mottled, capillary refill >=5 s,
                tachycardia or bradycardia.
    respiratory : int
        Respiratory assessment (0--3).
            0 = Normal parameters, no retractions.
            1 = >10 above normal, using accessory muscles,
                FiO2 >=30% or 3+ L/min.
            2 = >20 above normal, retractions, FiO2 >=40% or
                6+ L/min.
            3 = Five below normal with retractions and/or
                grunting, FiO2 >=50% or 8+ L/min.

    References
    ----------
    Monaghan A. "Detecting and managing deterioration in children."
    Paediatr Nurs. 2005;17(1):32-35.
    """

    _COMPONENTS = ("behavior", "cardiovascular", "respiratory")

    def __init__(self) -> None:
        super().__init__()
        self._name = "Pediatric Early Warning Score"
        self._description = (
            "Bedside tool with 3 components (behavior, cardiovascular, "
            "respiratory) each 0-3 for a total of 0-9."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate PEWS component scores.

        Parameters
        ----------
        **kwargs
            Must include ``behavior``, ``cardiovascular``, and
            ``respiratory``, each an int 0--3.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a component is outside [0, 3].
        """
        for comp in self._COMPONENTS:
            if comp not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{comp}' for PEWS.",
                    parameter=comp,
                )
            val = kwargs[comp]
            if not isinstance(val, int) or val < 0 or val > 3:
                raise ClinicalRangeError(comp, float(val), 0, 3)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the Pediatric Early Warning Score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with PEWS total (0--9) and component breakdown.
        """
        self.validate_inputs(**kwargs)

        components: Dict[str, float] = {
            comp: float(kwargs[comp]) for comp in self._COMPONENTS
        }
        total = sum(components.values())

        interpretation = self.interpret(total)
        risk = self._categorize(int(total))

        return ScoringResult(
            total_score=total,
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum PEWS values.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 9.0)``.
        """
        return (0.0, 9.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the PEWS total.

        Parameters
        ----------
        score : int or float
            PEWS total (0--9).

        Returns
        -------
        str
            Clinical interpretation with escalation guidance.
        """
        s = int(score)
        if s <= 2:
            return f"PEWS {s}: Low risk. Continue routine monitoring."
        if s <= 4:
            return (
                f"PEWS {s}: Moderate risk. Increase monitoring frequency "
                "and notify medical team."
            )
        if s <= 6:
            return (
                f"PEWS {s}: High risk. Immediate medical review required."
            )
        return (
            f"PEWS {s}: Critical. Activate rapid response or "
            "emergency team immediately."
        )

    @staticmethod
    def _categorize(score: int) -> str:
        """Map PEWS total to a risk category.

        Parameters
        ----------
        score : int
            PEWS total.

        Returns
        -------
        str
            ``"low"``, ``"moderate"``, ``"high"``, or ``"critical"``.
        """
        if score <= 2:
            return "low"
        if score <= 4:
            return "moderate"
        if score <= 6:
            return "high"
        return "critical"


# ======================================================================
# APGAR Score
# ======================================================================


class APGARScore(BaseScorer):
    """APGAR neonatal assessment score.

    Five components, each scored 0--2, yielding a total of 0--10.
    Typically assessed at 1 and 5 minutes after birth.

    Parameters (passed to ``calculate``)
    -------------------------------------
    appearance : int
        Skin color (0--2).
            0 = Blue/pale all over.
            1 = Blue at extremities (acrocyanosis).
            2 = No cyanosis, body and extremities pink.
    pulse : int
        Heart rate (0--2).
            0 = Absent.
            1 = <100 bpm.
            2 = >=100 bpm.
    grimace : int
        Reflex irritability (0--2).
            0 = No response to stimulation.
            1 = Grimace on suction or aggressive stimulation.
            2 = Cry on stimulation.
    activity : int
        Muscle tone (0--2).
            0 = None (floppy).
            1 = Some flexion.
            2 = Active motion, flexed arms and legs.
    respiration : int
        Respiratory effort (0--2).
            0 = Absent.
            1 = Weak, irregular, gasping.
            2 = Strong cry, good respiratory effort.

    References
    ----------
    Apgar V. "A proposal for a new method of evaluation of the
    newborn infant." Anesth Analg. 1953;32:260-267.
    """

    _COMPONENTS = ("appearance", "pulse", "grimace", "activity", "respiration")

    def __init__(self) -> None:
        super().__init__()
        self._name = "APGAR Score"
        self._description = (
            "Neonatal assessment: Appearance, Pulse, Grimace, Activity, "
            "Respiration, each 0-2 for a total of 0-10."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate APGAR component scores.

        Parameters
        ----------
        **kwargs
            Must include ``appearance``, ``pulse``, ``grimace``,
            ``activity``, and ``respiration``, each an int 0--2.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a component is outside [0, 2].
        """
        for comp in self._COMPONENTS:
            if comp not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{comp}' for APGAR.",
                    parameter=comp,
                )
            val = kwargs[comp]
            if not isinstance(val, int) or val < 0 or val > 2:
                raise ClinicalRangeError(comp, float(val), 0, 2)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the APGAR score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with APGAR total (0--10) and component breakdown.
        """
        self.validate_inputs(**kwargs)

        components: Dict[str, float] = {
            comp: float(kwargs[comp]) for comp in self._COMPONENTS
        }
        total = sum(components.values())

        interpretation = self.interpret(total)
        risk = self._categorize(int(total))

        return ScoringResult(
            total_score=total,
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum APGAR values.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 10.0)``.
        """
        return (0.0, 10.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the APGAR score.

        Parameters
        ----------
        score : int or float
            APGAR total (0--10).

        Returns
        -------
        str
            Clinical interpretation.
        """
        s = int(score)
        if s >= 7:
            return (
                f"APGAR {s}: Normal. Infant is in good condition."
            )
        if s >= 4:
            return (
                f"APGAR {s}: Moderately depressed. "
                "May require some resuscitative measures."
            )
        return (
            f"APGAR {s}: Severely depressed. "
            "Immediate resuscitation required."
        )

    @staticmethod
    def _categorize(score: int) -> str:
        """Map APGAR total to a clinical category.

        Parameters
        ----------
        score : int
            APGAR total.

        Returns
        -------
        str
            ``"normal"``, ``"moderate_depression"``, or
            ``"severe_depression"``.
        """
        if score >= 7:
            return "normal"
        if score >= 4:
            return "moderate_depression"
        return "severe_depression"
