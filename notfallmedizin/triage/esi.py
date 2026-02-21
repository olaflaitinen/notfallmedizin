# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Emergency Severity Index (ESI) v4 triage implementation.

The Emergency Severity Index is a five-level emergency department (ED)
triage algorithm that stratifies patients from level 1 (most urgent) to
level 5 (least urgent) based on acuity and expected resource needs.

The ESI algorithm uses a sequential decision-tree approach:

1. Does the patient require an immediate life-saving intervention?
   If yes, assign ESI Level 1.
2. Is this a high-risk situation, or is the patient confused, lethargic,
   disoriented, or in severe pain/distress?
   If yes, assign ESI Level 2.
3. How many different resources does the patient need?

   - Two or more: ESI Level 3 (with vital-sign danger-zone check that
     may escalate to Level 2).
   - Exactly one: ESI Level 4.
   - Zero: ESI Level 5.

References
----------
.. [1] Gilboy N, Tanabe T, Travers D, Rosenau AM. Emergency Severity
       Index (ESI): A Triage Tool for Emergency Department Care,
       Version 4. Implementation Handbook, 2012 Edition. AHRQ
       Publication No. 12-0014. Rockville, MD: Agency for Healthcare
       Research and Quality; November 2011.
.. [2] Tanabe P, Gimbel R, Yarnold PR, Kyriacou DN, Adams JG.
       Reliability and validity of scores on The Emergency Severity
       Index version 3. Acad Emerg Med. 2004;11(1):59-65.
       doi:10.1197/j.aem.2003.06.013
.. [3] Wuerz RC, Milne LW, Eitel DR, Travers D, Gilboy N. Reliability
       and validity of a new five-level triage instrument. Acad Emerg
       Med. 2000;7(3):236-242. doi:10.1111/j.1553-2712.2000.tb01066.x
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from notfallmedizin.core.base import BaseScorer
from notfallmedizin.core.exceptions import ClinicalRangeError, ValidationError
from notfallmedizin.core.validators import validate_vital_signs


# ======================================================================
# Constants
# ======================================================================


class ESIMentalStatus(Enum):
    """AVPU mental-status scale used during ESI triage.

    The AVPU scale is a rapid assessment of a patient's level of
    consciousness. It is used at ESI decision point B to identify
    patients who should not wait (ESI Level 2).

    Attributes
    ----------
    ALERT : str
        Patient is fully alert and oriented.
    VERBAL : str
        Patient responds to verbal stimuli only.
    PAIN : str
        Patient responds to painful stimuli only.
    UNRESPONSIVE : str
        Patient does not respond to any stimuli.
    """

    ALERT = "alert"
    VERBAL = "verbal"
    PAIN = "pain"
    UNRESPONSIVE = "unresponsive"


_LIFE_THREATENING_COMPLAINTS: FrozenSet[str] = frozenset({
    "cardiac_arrest",
    "respiratory_arrest",
    "major_trauma",
    "severe_respiratory_distress",
    "anaphylaxis_with_airway_compromise",
    "massive_hemorrhage",
    "status_epilepticus",
    "pulseless_electrical_activity",
})
"""Chief complaints that indicate the need for immediate life-saving
intervention (ESI Level 1)."""

_HIGH_RISK_COMPLAINTS: FrozenSet[str] = frozenset({
    "chest_pain",
    "stroke_symptoms",
    "seizure",
    "overdose",
    "poisoning",
    "anaphylaxis",
    "severe_allergic_reaction",
    "suicidal_ideation",
    "homicidal_ideation",
    "sexual_assault",
    "acute_psychosis",
    "ectopic_pregnancy",
    "severe_asthma",
    "diabetic_emergency",
    "sepsis",
    "meningitis",
    "aortic_dissection",
    "pulmonary_embolism",
    "acute_coronary_syndrome",
    "gi_hemorrhage",
})
"""Chief complaints indicating high-risk situations (ESI Level 2)."""

_DANGER_ZONE_THRESHOLDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "heart_rate": (50.0, 100.0),
    "respiratory_rate": (None, 20.0),
    "spo2": (92.0, None),
}
"""Adult vital-sign danger-zone thresholds.

For each parameter the tuple is ``(lower_safe, upper_safe)``.  A value
below ``lower_safe`` or above ``upper_safe`` triggers the danger-zone
flag.  ``None`` means there is no threshold on that side.

Source: ESI v4 Implementation Handbook, Table 2 [1].
"""

_REASSESSMENT_MINUTES: Dict[int, int] = {
    1: 0,
    2: 10,
    3: 30,
    4: 60,
    5: 120,
}
"""Recommended reassessment intervals (minutes) by ESI level."""


# ======================================================================
# Result dataclass
# ======================================================================


@dataclass(frozen=True)
class ESIResult:
    """Structured result of an ESI v4 triage assessment.

    Parameters
    ----------
    level : int
        ESI triage level in {1, 2, 3, 4, 5}. Level 1 is the most
        urgent; level 5 is the least urgent.
    confidence : float
        Algorithmic confidence in the assigned level, in [0.0, 1.0].
        A value of 1.0 indicates that the decision path was
        unambiguous.
    reasoning : list of str
        Ordered list of clinical reasoning steps that led to the
        assigned level. Each entry describes one decision point
        evaluated during the algorithm.
    recommended_reassessment_minutes : int
        Recommended interval in minutes before the next patient
        reassessment. A value of 0 indicates continuous monitoring.
    vital_sign_flags : dict of str to bool
        Boolean flags indicating which vital signs (if any) fall
        within the ESI danger zone. Keys correspond to vital-sign
        parameter names.
    """

    level: int
    confidence: float
    reasoning: List[str]
    recommended_reassessment_minutes: int
    vital_sign_flags: Dict[str, bool] = field(default_factory=dict)

    def __int__(self) -> int:
        return self.level

    def __float__(self) -> float:
        return float(self.level)


# ======================================================================
# Calculator
# ======================================================================


class ESITriageCalculator(BaseScorer):
    """Emergency Severity Index (ESI) v4 triage calculator.

    Implements the complete ESI v4 five-level decision-tree algorithm
    as described in the AHRQ Implementation Handbook [1]. The three
    sequential decision points (A, B, C/D) are evaluated in order,
    with a danger-zone vital-sign check applied when the initial
    assignment is Level 3.

    Parameters
    ----------
    danger_zone_escalation : bool, optional
        If ``True`` (default), patients initially classified as ESI
        Level 3 whose vital signs fall within the danger zone are
        escalated to Level 2.

    Examples
    --------
    >>> calc = ESITriageCalculator()
    >>> result = calc.calculate(
    ...     chief_complaint="chest_pain",
    ...     vital_signs={
    ...         "heart_rate": 110.0,
    ...         "systolic_bp": 90.0,
    ...         "diastolic_bp": 60.0,
    ...         "respiratory_rate": 22.0,
    ...         "spo2": 94.0,
    ...         "temperature": 37.0,
    ...     },
    ...     resource_estimate=3,
    ...     mental_status="alert",
    ...     severe_pain_distress=False,
    ...     requires_immediate_intervention=False,
    ... )
    >>> result.level
    2

    References
    ----------
    .. [1] Gilboy N, et al. ESI v4 Implementation Handbook, 2012.
    """

    def __init__(self, danger_zone_escalation: bool = True) -> None:
        super().__init__(
            name="Emergency Severity Index",
            version="4.0",
            references=[
                "Gilboy N, Tanabe T, Travers D, Rosenau AM. Emergency "
                "Severity Index (ESI): A Triage Tool for Emergency "
                "Department Care, Version 4. AHRQ Pub. No. 12-0014. 2012.",
                "Tanabe P, Gimbel R, Yarnold PR, et al. Reliability and "
                "validity of scores on The Emergency Severity Index "
                "version 3. Acad Emerg Med. 2004;11(1):59-65.",
            ],
        )
        self.danger_zone_escalation = danger_zone_escalation

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    def validate_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate and normalize ESI triage inputs.

        Parameters
        ----------
        **kwargs
            Expected keys:

            - ``chief_complaint`` (str): Presenting complaint identifier.
            - ``vital_signs`` (dict): Dictionary with keys
              ``heart_rate``, ``systolic_bp``, ``diastolic_bp``,
              ``respiratory_rate``, ``spo2``, ``temperature``.
            - ``resource_estimate`` (int): Estimated number of
              different resource types the patient will consume
              (0, 1, or 2+).
            - ``mental_status`` (str): One of ``"alert"``,
              ``"verbal"``, ``"pain"``, ``"unresponsive"``.
            - ``severe_pain_distress`` (bool): Whether the patient
              exhibits severe pain or distress.
            - ``requires_immediate_intervention`` (bool): Whether an
              immediate life-saving intervention is needed.

        Returns
        -------
        dict of str to Any
            Validated and normalized input dictionary.

        Raises
        ------
        ValidationError
            If required keys are missing or values are invalid.
        ClinicalRangeError
            If vital signs fall outside physiologically plausible
            ranges.
        """
        required_keys = {
            "chief_complaint",
            "vital_signs",
            "resource_estimate",
            "mental_status",
            "severe_pain_distress",
            "requires_immediate_intervention",
        }
        missing = required_keys - set(kwargs.keys())
        if missing:
            raise ValidationError(
                message=f"Missing required ESI parameters: {sorted(missing)}.",
                parameter=", ".join(sorted(missing)),
            )

        chief_complaint = str(kwargs["chief_complaint"]).strip().lower()

        vs = kwargs["vital_signs"]
        if not isinstance(vs, dict):
            raise ValidationError(
                message=(
                    "Parameter 'vital_signs' must be a dictionary, "
                    f"got {type(vs).__name__}."
                ),
                parameter="vital_signs",
            )
        validated_vs = validate_vital_signs(**vs)

        resource_estimate = kwargs["resource_estimate"]
        if not isinstance(resource_estimate, int) or resource_estimate < 0:
            raise ValidationError(
                message=(
                    "Parameter 'resource_estimate' must be a non-negative "
                    f"integer, got {resource_estimate!r}."
                ),
                parameter="resource_estimate",
            )

        ms_raw = str(kwargs["mental_status"]).strip().lower()
        try:
            mental_status = ESIMentalStatus(ms_raw)
        except ValueError:
            valid = [m.value for m in ESIMentalStatus]
            raise ValidationError(
                message=(
                    f"Parameter 'mental_status' must be one of {valid}, "
                    f"got {ms_raw!r}."
                ),
                parameter="mental_status",
            )

        severe_pain_distress = bool(kwargs["severe_pain_distress"])
        requires_immediate = bool(kwargs["requires_immediate_intervention"])

        return {
            "chief_complaint": chief_complaint,
            "vital_signs": validated_vs,
            "resource_estimate": resource_estimate,
            "mental_status": mental_status,
            "severe_pain_distress": severe_pain_distress,
            "requires_immediate_intervention": requires_immediate,
        }

    def calculate(self, **kwargs: Any) -> ESIResult:
        """Compute the ESI triage level using the v4 decision tree.

        This method expects pre-validated inputs (as returned by
        :meth:`validate_inputs`). To validate and calculate in one
        step, call the calculator instance directly (``__call__``).

        Parameters
        ----------
        **kwargs
            Validated ESI parameters. See :meth:`validate_inputs` for
            the expected keys and their types.

        Returns
        -------
        ESIResult
            Structured triage result including the assigned level,
            confidence, reasoning trace, reassessment interval, and
            vital-sign danger-zone flags.
        """
        chief_complaint: str = kwargs["chief_complaint"]
        vital_signs: Dict[str, float] = kwargs["vital_signs"]
        resource_estimate: int = kwargs["resource_estimate"]
        mental_status: ESIMentalStatus = kwargs["mental_status"]
        severe_pain_distress: bool = kwargs["severe_pain_distress"]
        requires_immediate: bool = kwargs["requires_immediate_intervention"]

        reasoning: List[str] = []
        vs_flags = self._evaluate_danger_zone(vital_signs)

        # ---- Decision Point A: Immediate life-saving intervention ----
        if requires_immediate or chief_complaint in _LIFE_THREATENING_COMPLAINTS:
            reason = (
                "Decision Point A: Patient requires immediate life-saving "
                "intervention"
            )
            if chief_complaint in _LIFE_THREATENING_COMPLAINTS:
                reason += f" (chief complaint: {chief_complaint})"
            reason += "."
            reasoning.append(reason)
            return ESIResult(
                level=1,
                confidence=1.0,
                reasoning=reasoning,
                recommended_reassessment_minutes=_REASSESSMENT_MINUTES[1],
                vital_sign_flags=vs_flags,
            )

        reasoning.append(
            "Decision Point A: No immediate life-saving intervention required."
        )

        # ---- Decision Point B: High-risk / altered mental status ----
        is_high_risk = chief_complaint in _HIGH_RISK_COMPLAINTS
        is_altered = mental_status in (
            ESIMentalStatus.VERBAL,
            ESIMentalStatus.PAIN,
            ESIMentalStatus.UNRESPONSIVE,
        )

        if is_high_risk or is_altered or severe_pain_distress:
            reasons_b: List[str] = []
            if is_high_risk:
                reasons_b.append(f"high-risk chief complaint ({chief_complaint})")
            if is_altered:
                reasons_b.append(f"altered mental status ({mental_status.value})")
            if severe_pain_distress:
                reasons_b.append("severe pain or distress")
            reasoning.append(
                "Decision Point B: Patient should not wait -- "
                + "; ".join(reasons_b) + "."
            )
            return ESIResult(
                level=2,
                confidence=0.95 if len(reasons_b) > 1 else 0.90,
                reasoning=reasoning,
                recommended_reassessment_minutes=_REASSESSMENT_MINUTES[2],
                vital_sign_flags=vs_flags,
            )

        reasoning.append(
            "Decision Point B: No high-risk features, mental status is alert, "
            "no severe pain/distress."
        )

        # ---- Decision Points C/D: Resource-based classification ----
        if resource_estimate == 0:
            reasoning.append(
                "Decision Point C: Zero resources expected. Assigned ESI-5."
            )
            return ESIResult(
                level=5,
                confidence=0.90,
                reasoning=reasoning,
                recommended_reassessment_minutes=_REASSESSMENT_MINUTES[5],
                vital_sign_flags=vs_flags,
            )

        if resource_estimate == 1:
            reasoning.append(
                "Decision Point C: One resource expected. Assigned ESI-4."
            )
            return ESIResult(
                level=4,
                confidence=0.90,
                reasoning=reasoning,
                recommended_reassessment_minutes=_REASSESSMENT_MINUTES[4],
                vital_sign_flags=vs_flags,
            )

        # resource_estimate >= 2
        reasoning.append(
            f"Decision Point C: {resource_estimate} resources expected "
            f"(>=2). Preliminary assignment ESI-3."
        )

        # ---- Decision Point D: Danger-zone vital-sign check ----
        any_danger = any(vs_flags.values())
        if any_danger and self.danger_zone_escalation:
            flagged = [k for k, v in vs_flags.items() if v]
            reasoning.append(
                "Decision Point D: Vital-sign danger zone triggered for "
                + ", ".join(flagged)
                + ". Escalated from ESI-3 to ESI-2."
            )
            return ESIResult(
                level=2,
                confidence=0.85,
                reasoning=reasoning,
                recommended_reassessment_minutes=_REASSESSMENT_MINUTES[2],
                vital_sign_flags=vs_flags,
            )

        if any_danger:
            reasoning.append(
                "Decision Point D: Vital-sign danger zone detected but "
                "escalation is disabled. Remaining at ESI-3."
            )
        else:
            reasoning.append(
                "Decision Point D: Vital signs within acceptable range. "
                "Confirmed ESI-3."
            )

        return ESIResult(
            level=3,
            confidence=0.90,
            reasoning=reasoning,
            recommended_reassessment_minutes=_REASSESSMENT_MINUTES[3],
            vital_sign_flags=vs_flags,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the theoretical range of the ESI level.

        Returns
        -------
        tuple of (float, float)
            ``(1.0, 5.0)`` representing ESI levels 1 through 5.
        """
        return (1.0, 5.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Return a clinical interpretation for a given ESI level.

        Parameters
        ----------
        score : int or float
            An ESI triage level (1--5).

        Returns
        -------
        str
            A textual interpretation of the ESI level.

        Raises
        ------
        ValidationError
            If the score is not in {1, 2, 3, 4, 5}.
        """
        level = int(score)
        interpretations = {
            1: (
                "ESI Level 1 (Resuscitation): Requires immediate "
                "life-saving intervention. Continuous monitoring indicated."
            ),
            2: (
                "ESI Level 2 (Emergent): High-risk situation or severely "
                "altered condition. Patient should not wait. Reassess "
                "within 10 minutes."
            ),
            3: (
                "ESI Level 3 (Urgent): Two or more resources expected. "
                "Reassess within 30 minutes."
            ),
            4: (
                "ESI Level 4 (Less Urgent): One resource expected. "
                "Reassess within 60 minutes."
            ),
            5: (
                "ESI Level 5 (Non-Urgent): No resources expected. "
                "Reassess within 120 minutes."
            ),
        }
        if level not in interpretations:
            raise ValidationError(
                message=f"ESI level must be in {{1,2,3,4,5}}, got {level}.",
                parameter="score",
            )
        return interpretations[level]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_danger_zone(
        vital_signs: Dict[str, float],
    ) -> Dict[str, bool]:
        """Evaluate vital signs against the ESI danger-zone thresholds.

        Parameters
        ----------
        vital_signs : dict of str to float
            Validated vital-sign dictionary.

        Returns
        -------
        dict of str to bool
            A flag for each danger-zone parameter. ``True`` means the
            value falls within the danger zone.
        """
        flags: Dict[str, bool] = {}

        for param, (lower_safe, upper_safe) in _DANGER_ZONE_THRESHOLDS.items():
            value = vital_signs.get(param)
            if value is None:
                flags[param] = False
                continue

            in_danger = False
            if lower_safe is not None and value < lower_safe:
                in_danger = True
            if upper_safe is not None and value > upper_safe:
                in_danger = True
            flags[param] = in_danger

        return flags

    @staticmethod
    def get_life_threatening_complaints() -> FrozenSet[str]:
        """Return the set of recognized life-threatening chief complaints.

        Returns
        -------
        frozenset of str
            Chief-complaint identifiers that trigger ESI Level 1.
        """
        return _LIFE_THREATENING_COMPLAINTS

    @staticmethod
    def get_high_risk_complaints() -> FrozenSet[str]:
        """Return the set of recognized high-risk chief complaints.

        Returns
        -------
        frozenset of str
            Chief-complaint identifiers that trigger ESI Level 2.
        """
        return _HIGH_RISK_COMPLAINTS
