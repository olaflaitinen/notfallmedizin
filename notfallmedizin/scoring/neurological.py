# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Neurological clinical scoring systems.

This module implements the Glasgow Coma Scale (GCS) calculator and the
National Institutes of Health Stroke Scale (NIHSS) calculator.

Classes
-------
GCSCalculator
    Glasgow Coma Scale scoring with severity classification.
NIHSSCalculator
    National Institutes of Health Stroke Scale (0--42).
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
# Glasgow Coma Scale
# ======================================================================


class GCSCalculator(BaseScorer):
    """Glasgow Coma Scale (GCS) calculator.

    Sums three components: Eye opening (1--4), Verbal response (1--5),
    and Motor response (1--6) for a total of 3--15.

    Severity classification:
        - 13--15: Mild
        - 9--12: Moderate
        - 3--8: Severe

    Parameters (passed to ``calculate``)
    -------------------------------------
    eye : int
        Eye opening response (1--4).
            1 = No eye opening.
            2 = Eye opening to pain.
            3 = Eye opening to voice.
            4 = Eyes open spontaneously.
    verbal : int
        Verbal response (1--5).
            1 = No verbal response.
            2 = Incomprehensible sounds.
            3 = Inappropriate words.
            4 = Confused.
            5 = Oriented.
    motor : int
        Motor response (1--6).
            1 = No motor response.
            2 = Extension to pain (decerebrate).
            3 = Abnormal flexion to pain (decorticate).
            4 = Withdrawal from pain.
            5 = Localizing pain.
            6 = Obeys commands.

    References
    ----------
    Teasdale G, Jennett B. "Assessment of coma and impaired
    consciousness. A practical scale." Lancet. 1974;2(7872):81-84.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "Glasgow Coma Scale"
        self._description = (
            "Eye (1-4) + Verbal (1-5) + Motor (1-6) = 3-15. "
            "Classifies brain injury as mild, moderate, or severe."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate GCS component scores.

        Parameters
        ----------
        **kwargs
            Must include ``eye`` (1--4), ``verbal`` (1--5), and
            ``motor`` (1--6).

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a component is outside its valid range.
        """
        ranges = {"eye": (1, 4), "verbal": (1, 5), "motor": (1, 6)}
        for param, (lo, hi) in ranges.items():
            if param not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{param}' for GCS.",
                    parameter=param,
                )
            val = kwargs[param]
            if not isinstance(val, int) or val < lo or val > hi:
                raise ClinicalRangeError(param, float(val), lo, hi)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the Glasgow Coma Scale total.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with GCS total (3--15) and component breakdown.
        """
        self.validate_inputs(**kwargs)

        eye = kwargs["eye"]
        verbal = kwargs["verbal"]
        motor = kwargs["motor"]
        total = eye + verbal + motor

        components: Dict[str, float] = {
            "eye": float(eye),
            "verbal": float(verbal),
            "motor": float(motor),
        }

        interpretation = self.interpret(total)
        severity = self._severity(total)

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=severity,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum GCS values.

        Returns
        -------
        tuple of (float, float)
            ``(3.0, 15.0)``.
        """
        return (3.0, 15.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the GCS total.

        Parameters
        ----------
        score : int or float
            GCS total (3--15).

        Returns
        -------
        str
            Severity-based interpretation.
        """
        s = int(score)
        if s >= 13:
            return f"GCS {s}: Mild brain injury."
        if s >= 9:
            return f"GCS {s}: Moderate brain injury."
        return f"GCS {s}: Severe brain injury."

    @staticmethod
    def _severity(score: int) -> str:
        """Map GCS total to a severity category.

        Parameters
        ----------
        score : int
            GCS total.

        Returns
        -------
        str
            ``"mild"``, ``"moderate"``, or ``"severe"``.
        """
        if score >= 13:
            return "mild"
        if score >= 9:
            return "moderate"
        return "severe"


# ======================================================================
# NIHSS Calculator
# ======================================================================


_NIHSS_ITEMS: Dict[str, Tuple[int, int]] = {
    "loc": (0, 3),
    "loc_questions": (0, 2),
    "loc_commands": (0, 2),
    "best_gaze": (0, 2),
    "visual_fields": (0, 3),
    "facial_palsy": (0, 3),
    "motor_arm_left": (0, 4),
    "motor_arm_right": (0, 4),
    "motor_leg_left": (0, 4),
    "motor_leg_right": (0, 4),
    "limb_ataxia": (0, 2),
    "sensory": (0, 2),
    "best_language": (0, 3),
    "dysarthria": (0, 2),
    "extinction_inattention": (0, 2),
}


class NIHSSCalculator(BaseScorer):
    """National Institutes of Health Stroke Scale (NIHSS).

    Evaluates 15 sub-items spanning 11 categories, yielding a total
    of 0--42. Used to quantify the severity of stroke symptoms.

    Severity classification:
        - 0: No stroke symptoms
        - 1--4: Minor stroke
        - 5--15: Moderate stroke
        - 16--20: Moderate-to-severe stroke
        - 21--42: Severe stroke

    Parameters (passed to ``calculate``)
    -------------------------------------
    loc : int
        Level of consciousness (0--3).
    loc_questions : int
        LOC questions (0--2).
    loc_commands : int
        LOC commands (0--2).
    best_gaze : int
        Best gaze (0--2).
    visual_fields : int
        Visual fields (0--3).
    facial_palsy : int
        Facial palsy (0--3).
    motor_arm_left : int
        Left arm motor (0--4).
    motor_arm_right : int
        Right arm motor (0--4).
    motor_leg_left : int
        Left leg motor (0--4).
    motor_leg_right : int
        Right leg motor (0--4).
    limb_ataxia : int
        Limb ataxia (0--2).
    sensory : int
        Sensory (0--2).
    best_language : int
        Best language / aphasia (0--3).
    dysarthria : int
        Dysarthria (0--2).
    extinction_inattention : int
        Extinction and inattention (0--2).

    References
    ----------
    Brott T, et al. "Measurements of acute cerebral infarction:
    a clinical examination scale." Stroke. 1989;20(7):864-870.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "NIH Stroke Scale"
        self._description = (
            "Quantitative measure of stroke severity across 15 "
            "sub-items in 11 categories, total 0-42."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate NIHSS item scores.

        Parameters
        ----------
        **kwargs
            Must include all 15 sub-item scores.

        Raises
        ------
        ValidationError
            If a required item is missing.
        ClinicalRangeError
            If an item score is outside its valid range.
        """
        for item, (lo, hi) in _NIHSS_ITEMS.items():
            if item not in kwargs:
                raise ValidationError(
                    message=(
                        f"Missing required NIHSS item '{item}'."
                    ),
                    parameter=item,
                )
            val = kwargs[item]
            if not isinstance(val, int) or val < lo or val > hi:
                raise ClinicalRangeError(item, float(val), lo, hi)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the NIHSS total.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with NIHSS total (0--42) and per-item scores.
        """
        self.validate_inputs(**kwargs)

        components: Dict[str, float] = {
            item: float(kwargs[item]) for item in _NIHSS_ITEMS
        }
        total = sum(components.values())

        interpretation = self.interpret(total)
        severity = self._severity(int(total))

        return ScoringResult(
            total_score=total,
            component_scores=components,
            interpretation=interpretation,
            risk_category=severity,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum NIHSS values.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 42.0)``.
        """
        return (0.0, 42.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the NIHSS total.

        Parameters
        ----------
        score : int or float
            NIHSS total (0--42).

        Returns
        -------
        str
            Stroke severity interpretation.
        """
        s = int(score)
        if s == 0:
            return "NIHSS 0: No stroke symptoms."
        if s <= 4:
            return f"NIHSS {s}: Minor stroke."
        if s <= 15:
            return f"NIHSS {s}: Moderate stroke."
        if s <= 20:
            return f"NIHSS {s}: Moderate-to-severe stroke."
        return f"NIHSS {s}: Severe stroke."

    @staticmethod
    def _severity(score: int) -> str:
        """Map NIHSS total to a severity category.

        Parameters
        ----------
        score : int
            NIHSS total.

        Returns
        -------
        str
            Severity category string.
        """
        if score == 0:
            return "no_symptoms"
        if score <= 4:
            return "minor"
        if score <= 15:
            return "moderate"
        if score <= 20:
            return "moderate_to_severe"
        return "severe"
