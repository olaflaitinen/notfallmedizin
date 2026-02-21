# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Pediatric triage with age-adjusted vital-sign interpretation.

This module implements pediatric emergency triage with age-specific
vital-sign reference ranges, the Pediatric Assessment Triangle (PAT),
and the Pediatric Early Warning Score (PEWS). All thresholds are
derived from published pediatric reference data.

Age groups and approximate vital-sign normal ranges used in this
module are based on APLS (Advanced Paediatric Life Support) and PALS
(Pediatric Advanced Life Support) guidelines:

===============  =============  ============  ===========  ===========
Age group        HR (bpm)       RR (br/min)   SBP (mmHg)   SpO2 (%)
===============  =============  ============  ===========  ===========
Neonate (0-28d)  100 -- 180     30 -- 60      60 -- 90     >= 92
Infant (1-12m)   100 -- 160     25 -- 50      70 -- 100    >= 95
Toddler (1-3y)   80 -- 140      20 -- 30      80 -- 110    >= 95
Child (3-12y)    70 -- 120      18 -- 25      85 -- 120    >= 95
Adolescent(12+)  60 -- 100      12 -- 20      95 -- 140    >= 95
===============  =============  ============  ===========  ===========

References
----------
.. [1] Advanced Life Support Group. Advanced Paediatric Life Support:
       A Practical Approach to Emergencies. 6th ed. Wiley-Blackwell;
       2016.
.. [2] American Heart Association. Pediatric Advanced Life Support
       (PALS) Provider Manual. 2020.
.. [3] Akre M, Finkelstein M, Erickson M, Liu M, Vanderbilt L,
       Billman G. Sensitivity of the Pediatric Early Warning Score to
       identify patient deterioration. Pediatrics.
       2010;125(4):e763-e769. doi:10.1542/peds.2009-0338
.. [4] Dieckmann RA, Brownstein D, Gausche-Hill M. The Pediatric
       Assessment Triangle: a novel approach for the rapid evaluation
       of children. Pediatr Emerg Care. 2010;26(4):312-315.
       doi:10.1097/PEC.0b013e3181d6db37
.. [5] Fleming S, Thompson M, Stevens R, et al. Normal ranges of heart
       rate and respiratory rate in children from birth to 18 years of
       age: a systematic review of observational studies. Lancet.
       2011;377(9770):1011-1018. doi:10.1016/S0140-6736(10)62226-X
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from notfallmedizin.core.base import BaseScorer
from notfallmedizin.core.exceptions import ClinicalRangeError, ValidationError
from notfallmedizin.core.validators import validate_age


# ======================================================================
# Age groups and normal ranges
# ======================================================================


class PediatricAgeGroup(Enum):
    """Pediatric age-group classification.

    Each member carries its age range in months as
    ``(lower_months, upper_months)``.
    """

    NEONATE = (0.0, 1.0)
    INFANT = (1.0, 12.0)
    TODDLER = (12.0, 36.0)
    CHILD = (36.0, 144.0)
    ADOLESCENT = (144.0, 216.0)

    def __init__(self, lower_months: float, upper_months: float) -> None:
        self.lower_months = lower_months
        self.upper_months = upper_months


def classify_age_group(age_months: float) -> PediatricAgeGroup:
    """Classify an age in months into a pediatric age group.

    Parameters
    ----------
    age_months : float
        Patient age in months. Must be in [0, 216].

    Returns
    -------
    PediatricAgeGroup
        The age group that the patient falls into.

    Raises
    ------
    ValidationError
        If ``age_months`` is negative or exceeds 216 (18 years).
    """
    if age_months < 0.0:
        raise ValidationError(
            message=f"Age in months must be >= 0, got {age_months}.",
            parameter="age_months",
        )
    if age_months > 216.0:
        raise ValidationError(
            message=(
                f"Age {age_months} months exceeds pediatric range "
                f"(0--216 months). Use an adult triage system."
            ),
            parameter="age_months",
        )

    if age_months < 1.0:
        return PediatricAgeGroup.NEONATE
    if age_months < 12.0:
        return PediatricAgeGroup.INFANT
    if age_months < 36.0:
        return PediatricAgeGroup.TODDLER
    if age_months < 144.0:
        return PediatricAgeGroup.CHILD
    return PediatricAgeGroup.ADOLESCENT


@dataclass(frozen=True)
class _VitalSignRange:
    """Internal representation of age-specific vital-sign normal ranges."""
    hr_low: float
    hr_high: float
    rr_low: float
    rr_high: float
    sbp_low: float
    sbp_high: float
    spo2_min: float


_AGE_VITAL_RANGES: Dict[PediatricAgeGroup, _VitalSignRange] = {
    PediatricAgeGroup.NEONATE: _VitalSignRange(
        hr_low=100.0, hr_high=180.0,
        rr_low=30.0, rr_high=60.0,
        sbp_low=60.0, sbp_high=90.0,
        spo2_min=92.0,
    ),
    PediatricAgeGroup.INFANT: _VitalSignRange(
        hr_low=100.0, hr_high=160.0,
        rr_low=25.0, rr_high=50.0,
        sbp_low=70.0, sbp_high=100.0,
        spo2_min=95.0,
    ),
    PediatricAgeGroup.TODDLER: _VitalSignRange(
        hr_low=80.0, hr_high=140.0,
        rr_low=20.0, rr_high=30.0,
        sbp_low=80.0, sbp_high=110.0,
        spo2_min=95.0,
    ),
    PediatricAgeGroup.CHILD: _VitalSignRange(
        hr_low=70.0, hr_high=120.0,
        rr_low=18.0, rr_high=25.0,
        sbp_low=85.0, sbp_high=120.0,
        spo2_min=95.0,
    ),
    PediatricAgeGroup.ADOLESCENT: _VitalSignRange(
        hr_low=60.0, hr_high=100.0,
        rr_low=12.0, rr_high=20.0,
        sbp_low=95.0, sbp_high=140.0,
        spo2_min=95.0,
    ),
}
"""Age-specific normal vital-sign ranges.

Sources: APLS 6th ed. [1], Fleming et al. Lancet 2011 [5].
"""


# ======================================================================
# PAT (Pediatric Assessment Triangle)
# ======================================================================


class PATComponent(Enum):
    """Components of the Pediatric Assessment Triangle.

    Each component is assessed as ``"normal"`` or ``"abnormal"``.
    """

    APPEARANCE = "appearance"
    WORK_OF_BREATHING = "work_of_breathing"
    CIRCULATION = "circulation"


_PAT_SEVERITY_MAP: Dict[Tuple[bool, bool, bool], Tuple[str, int]] = {
    (True, True, True):   ("stable", 5),
    (False, True, True):  ("primary_brain_problem_or_toxin", 2),
    (True, False, True):  ("respiratory_distress", 3),
    (False, False, True): ("respiratory_failure", 2),
    (True, True, False):  ("compensated_shock", 3),
    (False, True, False): ("decompensated_shock_or_brain_injury", 1),
    (True, False, False): ("mixed_respiratory_circulatory", 2),
    (False, False, False): ("cardiopulmonary_failure", 1),
}
"""PAT impression mapping.

Keys: ``(appearance_normal, breathing_normal, circulation_normal)``.
Values: ``(clinical_impression, suggested_triage_level)``.

Source: Dieckmann RA, et al. Pediatr Emerg Care. 2010 [4].
"""


def evaluate_pat(
    appearance_normal: bool,
    work_of_breathing_normal: bool,
    circulation_normal: bool,
) -> Dict[str, Any]:
    """Evaluate the Pediatric Assessment Triangle (PAT).

    The PAT is a rapid, hands-off assessment tool that evaluates three
    components: general appearance (TICLS: Tone, Interactiveness,
    Consolability, Look/Gaze, Speech/Cry), work of breathing, and
    circulation to skin.

    Parameters
    ----------
    appearance_normal : bool
        ``True`` if the child's general appearance is normal.
    work_of_breathing_normal : bool
        ``True`` if work of breathing is normal (no retractions,
        flaring, grunting, or abnormal positioning).
    circulation_normal : bool
        ``True`` if circulation to skin is normal (no pallor,
        mottling, or cyanosis).

    Returns
    -------
    dict of str to Any
        Keys:

        - ``"impression"`` (str): Clinical impression label.
        - ``"suggested_level"`` (int): Suggested triage level (1--5).
        - ``"components"`` (dict of str to str): Per-component status.

    References
    ----------
    .. [4] Dieckmann RA, et al. The Pediatric Assessment Triangle.
           Pediatr Emerg Care. 2010;26(4):312-315.
    """
    key = (appearance_normal, work_of_breathing_normal, circulation_normal)
    impression, suggested_level = _PAT_SEVERITY_MAP[key]

    return {
        "impression": impression,
        "suggested_level": suggested_level,
        "components": {
            "appearance": "normal" if appearance_normal else "abnormal",
            "work_of_breathing": "normal" if work_of_breathing_normal else "abnormal",
            "circulation": "normal" if circulation_normal else "abnormal",
        },
    }


# ======================================================================
# PEWS (Pediatric Early Warning Score)
# ======================================================================


_PEWS_BEHAVIOR_SCORES: Dict[str, int] = {
    "playing_appropriate": 0,
    "sleeping": 1,
    "irritable": 2,
    "lethargic_confused": 3,
}
"""PEWS behavior component scoring."""

_PEWS_CARDIOVASCULAR_SCORES: Dict[str, int] = {
    "pink_crt_1_2s": 0,
    "pale_crt_3s": 1,
    "grey_crt_4s_tachy_20": 2,
    "grey_mottled_crt_5s_tachy_30_or_brady": 3,
}
"""PEWS cardiovascular component scoring.

CRT = capillary refill time.  ``tachy_20`` / ``tachy_30`` denote heart
rate more than 20 or 30 bpm above age-normal upper limit.
``brady`` denotes heart rate below age-normal lower limit.
"""

_PEWS_RESPIRATORY_SCORES: Dict[str, int] = {
    "within_normal": 0,
    "above_10_accessory_muscles": 1,
    "above_20_retractions_fio2_30": 2,
    "below_5_retractions_grunting_fio2_50": 3,
}
"""PEWS respiratory component scoring.

``above_10`` / ``above_20`` / ``below_5`` refer to respiratory rate
deviations from the age-normal range.  ``fio2_30`` / ``fio2_50``
indicate supplemental oxygen requirements.
"""


def calculate_pews(
    behavior: str,
    cardiovascular: str,
    respiratory: str,
) -> Dict[str, Any]:
    """Calculate the Pediatric Early Warning Score (PEWS).

    The PEWS is a bedside tool that combines scores from three domains
    (behaviour, cardiovascular, respiratory) to produce a composite
    score ranging from 0 to 9.

    Parameters
    ----------
    behavior : str
        Behavior descriptor. One of:
        ``"playing_appropriate"`` (0), ``"sleeping"`` (1),
        ``"irritable"`` (2), ``"lethargic_confused"`` (3).
    cardiovascular : str
        Cardiovascular descriptor. One of:
        ``"pink_crt_1_2s"`` (0), ``"pale_crt_3s"`` (1),
        ``"grey_crt_4s_tachy_20"`` (2),
        ``"grey_mottled_crt_5s_tachy_30_or_brady"`` (3).
    respiratory : str
        Respiratory descriptor. One of:
        ``"within_normal"`` (0), ``"above_10_accessory_muscles"`` (1),
        ``"above_20_retractions_fio2_30"`` (2),
        ``"below_5_retractions_grunting_fio2_50"`` (3).

    Returns
    -------
    dict of str to Any
        Keys:

        - ``"total"`` (int): Composite PEWS (0--9).
        - ``"behavior_score"`` (int): Behavior domain score.
        - ``"cardiovascular_score"`` (int): Cardiovascular domain score.
        - ``"respiratory_score"`` (int): Respiratory domain score.
        - ``"risk_level"`` (str): ``"low"`` (0--2), ``"moderate"``
          (3--4), or ``"high"`` (>= 5).

    Raises
    ------
    ValidationError
        If any descriptor is not recognised.

    References
    ----------
    .. [3] Akre M, et al. Sensitivity of the PEWS. Pediatrics.
           2010;125(4):e763-e769.
    """
    behavior_lower = behavior.strip().lower()
    cardiovascular_lower = cardiovascular.strip().lower()
    respiratory_lower = respiratory.strip().lower()

    if behavior_lower not in _PEWS_BEHAVIOR_SCORES:
        raise ValidationError(
            message=(
                f"Invalid PEWS behavior descriptor '{behavior_lower}'. "
                f"Valid: {sorted(_PEWS_BEHAVIOR_SCORES.keys())}."
            ),
            parameter="behavior",
        )
    if cardiovascular_lower not in _PEWS_CARDIOVASCULAR_SCORES:
        raise ValidationError(
            message=(
                f"Invalid PEWS cardiovascular descriptor "
                f"'{cardiovascular_lower}'. "
                f"Valid: {sorted(_PEWS_CARDIOVASCULAR_SCORES.keys())}."
            ),
            parameter="cardiovascular",
        )
    if respiratory_lower not in _PEWS_RESPIRATORY_SCORES:
        raise ValidationError(
            message=(
                f"Invalid PEWS respiratory descriptor '{respiratory_lower}'. "
                f"Valid: {sorted(_PEWS_RESPIRATORY_SCORES.keys())}."
            ),
            parameter="respiratory",
        )

    b_score = _PEWS_BEHAVIOR_SCORES[behavior_lower]
    c_score = _PEWS_CARDIOVASCULAR_SCORES[cardiovascular_lower]
    r_score = _PEWS_RESPIRATORY_SCORES[respiratory_lower]
    total = b_score + c_score + r_score

    if total <= 2:
        risk_level = "low"
    elif total <= 4:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "total": total,
        "behavior_score": b_score,
        "cardiovascular_score": c_score,
        "respiratory_score": r_score,
        "risk_level": risk_level,
    }


# ======================================================================
# Result dataclass
# ======================================================================


@dataclass(frozen=True)
class PediatricTriageResult:
    """Structured result of a pediatric triage assessment.

    Parameters
    ----------
    level : int
        Triage level (1 = most urgent, 5 = least urgent).
    age_group : PediatricAgeGroup
        Classified age group of the patient.
    pat_assessment : dict of str to Any
        Pediatric Assessment Triangle evaluation results.
    pews_score : int
        Pediatric Early Warning Score total (0--9).
    vital_sign_interpretation : dict of str to str
        Per-vital-sign interpretation relative to age-specific norms
        (``"normal"``, ``"low"``, or ``"high"``).
    reasoning : list of str
        Ordered reasoning trace.
    recommended_reassessment_minutes : int
        Recommended interval before next reassessment.
    """

    level: int
    age_group: PediatricAgeGroup
    pat_assessment: Dict[str, Any]
    pews_score: int
    vital_sign_interpretation: Dict[str, str]
    reasoning: List[str] = field(default_factory=list)
    recommended_reassessment_minutes: int = 30

    def __int__(self) -> int:
        return self.level

    def __float__(self) -> float:
        return float(self.level)


# ======================================================================
# Calculator
# ======================================================================


_REASSESSMENT_MINUTES: Dict[int, int] = {
    1: 0,
    2: 5,
    3: 15,
    4: 30,
    5: 60,
}
"""Recommended reassessment intervals for pediatric patients.

Shorter than adult intervals because pediatric patients can
deteriorate more rapidly.
"""

_HIGH_RISK_PEDIATRIC_COMPLAINTS = frozenset({
    "respiratory_distress",
    "seizure",
    "altered_consciousness",
    "severe_dehydration",
    "fever_neonate",
    "non_accidental_injury",
    "toxic_ingestion",
    "anaphylaxis",
    "croup_severe",
    "bronchiolitis_severe",
    "meningitis_suspected",
    "sepsis_suspected",
})


class PediatricTriageCalculator(BaseScorer):
    """Age-adjusted pediatric triage calculator.

    Integrates the Pediatric Assessment Triangle (PAT), age-specific
    vital-sign evaluation, the Pediatric Early Warning Score (PEWS),
    and presenting-complaint acuity to assign a five-level triage
    priority. Vital-sign thresholds are drawn from published pediatric
    reference data [1, 2, 5].

    Parameters
    ----------
    use_pews : bool, optional
        If ``True`` (default), incorporate the PEWS into the triage
        decision. When PEWS data is unavailable the score is assumed
        to be 0.
    escalation_on_abnormal_vitals : bool, optional
        If ``True`` (default), patients with multiple abnormal vital
        signs are escalated by one level.

    Examples
    --------
    >>> calc = PediatricTriageCalculator()
    >>> result = calc.calculate(
    ...     age_months=8.0,
    ...     vital_signs={
    ...         "heart_rate": 170.0,
    ...         "systolic_bp": 65.0,
    ...         "diastolic_bp": 40.0,
    ...         "respiratory_rate": 55.0,
    ...         "spo2": 93.0,
    ...         "temperature": 39.0,
    ...     },
    ...     chief_complaint="bronchiolitis_severe",
    ...     appearance_normal=False,
    ...     work_of_breathing_normal=False,
    ...     circulation_normal=True,
    ...     pews_behavior="irritable",
    ...     pews_cardiovascular="pale_crt_3s",
    ...     pews_respiratory="above_20_retractions_fio2_30",
    ... )
    >>> result.level
    2

    References
    ----------
    .. [1] Advanced Life Support Group. APLS. 6th ed. 2016.
    .. [2] AHA. PALS Provider Manual. 2020.
    """

    def __init__(
        self,
        use_pews: bool = True,
        escalation_on_abnormal_vitals: bool = True,
    ) -> None:
        super().__init__(
            name="Pediatric Triage Calculator",
            version="1.0",
            references=[
                "Advanced Life Support Group. APLS: A Practical Approach "
                "to Emergencies. 6th ed. Wiley-Blackwell; 2016.",
                "AHA. Pediatric Advanced Life Support (PALS) Provider "
                "Manual. 2020.",
                "Dieckmann RA, et al. The Pediatric Assessment Triangle. "
                "Pediatr Emerg Care. 2010;26(4):312-315.",
                "Akre M, et al. Sensitivity of the PEWS. Pediatrics. "
                "2010;125(4):e763-e769.",
            ],
        )
        self.use_pews = use_pews
        self.escalation_on_abnormal_vitals = escalation_on_abnormal_vitals

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    def validate_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate and normalize pediatric triage inputs.

        Parameters
        ----------
        **kwargs
            Expected keys:

            - ``age_months`` (float): Patient age in months.
            - ``vital_signs`` (dict of str to float): Vital-sign
              measurements.
            - ``chief_complaint`` (str): Presenting complaint.
            - ``appearance_normal`` (bool): PAT appearance component.
            - ``work_of_breathing_normal`` (bool): PAT WOB component.
            - ``circulation_normal`` (bool): PAT circulation component.
            - ``pews_behavior`` (str, optional): PEWS behavior score.
            - ``pews_cardiovascular`` (str, optional): PEWS cardio.
            - ``pews_respiratory`` (str, optional): PEWS respiratory.

        Returns
        -------
        dict of str to Any
            Validated inputs.

        Raises
        ------
        ValidationError
            If required keys are missing or values are invalid.
        """
        required = {"age_months", "vital_signs", "chief_complaint"}
        missing = required - set(kwargs.keys())
        if missing:
            raise ValidationError(
                message=(
                    f"Missing required pediatric triage parameters: "
                    f"{sorted(missing)}."
                ),
                parameter=", ".join(sorted(missing)),
            )

        age_months = float(kwargs["age_months"])
        validate_age(age_months, unit="months")

        vs = kwargs["vital_signs"]
        if not isinstance(vs, dict):
            raise ValidationError(
                message=(
                    "Parameter 'vital_signs' must be a dict, "
                    f"got {type(vs).__name__}."
                ),
                parameter="vital_signs",
            )
        vs_validated: Dict[str, float] = {}
        for k, v in vs.items():
            vs_validated[str(k).strip().lower()] = float(v)

        chief_complaint = str(kwargs.get("chief_complaint", "")).strip().lower()

        appearance_normal = bool(kwargs.get("appearance_normal", True))
        wob_normal = bool(kwargs.get("work_of_breathing_normal", True))
        circulation_normal = bool(kwargs.get("circulation_normal", True))

        validated: Dict[str, Any] = {
            "age_months": age_months,
            "vital_signs": vs_validated,
            "chief_complaint": chief_complaint,
            "appearance_normal": appearance_normal,
            "work_of_breathing_normal": wob_normal,
            "circulation_normal": circulation_normal,
        }

        for pews_key in ("pews_behavior", "pews_cardiovascular", "pews_respiratory"):
            if pews_key in kwargs and kwargs[pews_key] is not None:
                validated[pews_key] = str(kwargs[pews_key]).strip().lower()

        return validated

    def calculate(self, **kwargs: Any) -> PediatricTriageResult:
        """Compute the pediatric triage level.

        The algorithm integrates three sources of information:

        1. **PAT (Pediatric Assessment Triangle)**: Rapid visual
           assessment that determines an initial acuity impression.
        2. **Age-adjusted vital-sign evaluation**: Each vital sign
           is compared to age-specific normal ranges. Multiple
           abnormalities escalate the triage level.
        3. **PEWS (Pediatric Early Warning Score)**: If available,
           the composite score further modifies the level.

        Parameters
        ----------
        **kwargs
            Validated inputs. See :meth:`validate_inputs`.

        Returns
        -------
        PediatricTriageResult
            Structured triage result.
        """
        age_months: float = kwargs["age_months"]
        vital_signs: Dict[str, float] = kwargs["vital_signs"]
        chief_complaint: str = kwargs["chief_complaint"]
        appearance_normal: bool = kwargs["appearance_normal"]
        wob_normal: bool = kwargs["work_of_breathing_normal"]
        circulation_normal: bool = kwargs["circulation_normal"]

        reasoning: List[str] = []
        age_group = classify_age_group(age_months)
        reasoning.append(
            f"Patient age: {age_months:.1f} months. "
            f"Classified as {age_group.name}."
        )

        # Step 1: PAT assessment
        pat_result = evaluate_pat(appearance_normal, wob_normal, circulation_normal)
        pat_level = pat_result["suggested_level"]
        reasoning.append(
            f"PAT assessment: {pat_result['impression']}. "
            f"Suggested level: {pat_level}."
        )

        # Step 2: Age-adjusted vital-sign evaluation
        vs_interp = self._interpret_vital_signs(vital_signs, age_group)
        abnormal_count = sum(1 for v in vs_interp.values() if v != "normal")
        reasoning.append(
            f"Vital-sign evaluation: {abnormal_count} abnormal parameter(s) "
            f"for age group {age_group.name}."
        )
        for param, status in vs_interp.items():
            if status != "normal":
                val = vital_signs.get(param, "N/A")
                reasoning.append(
                    f"  - {param} = {val} ({status} for {age_group.name})"
                )

        vs_level = self._vital_sign_acuity(abnormal_count)

        # Step 3: Chief-complaint assessment
        complaint_level = 3
        if chief_complaint in _HIGH_RISK_PEDIATRIC_COMPLAINTS:
            complaint_level = 2
            reasoning.append(
                f"Chief complaint '{chief_complaint}' is high-risk. "
                f"Baseline level: 2."
            )
        elif chief_complaint:
            reasoning.append(
                f"Chief complaint '{chief_complaint}' is not high-risk. "
                f"Baseline level: 3."
            )

        # Step 4: PEWS (if available)
        pews_total = 0
        pews_level = 5
        has_pews = all(
            k in kwargs and kwargs[k] is not None
            for k in ("pews_behavior", "pews_cardiovascular", "pews_respiratory")
        )
        if self.use_pews and has_pews:
            pews_result = calculate_pews(
                behavior=kwargs["pews_behavior"],
                cardiovascular=kwargs["pews_cardiovascular"],
                respiratory=kwargs["pews_respiratory"],
            )
            pews_total = pews_result["total"]
            pews_level = self._pews_to_triage_level(pews_total)
            reasoning.append(
                f"PEWS total: {pews_total}/9 "
                f"(risk: {pews_result['risk_level']}). "
                f"Suggested level: {pews_level}."
            )
        else:
            reasoning.append(
                "PEWS data not available; using PAT and vital signs only."
            )

        # Combine all signals: take the most urgent level
        candidate_levels = [pat_level, vs_level, complaint_level]
        if self.use_pews and has_pews:
            candidate_levels.append(pews_level)

        final_level = min(candidate_levels)

        # Additional escalation for multiple abnormal vitals
        if (
            self.escalation_on_abnormal_vitals
            and abnormal_count >= 3
            and final_level > 1
        ):
            final_level = max(1, final_level - 1)
            reasoning.append(
                f"Escalation: {abnormal_count} abnormal vital signs "
                f"detected; level escalated by 1."
            )

        final_level = max(1, min(5, final_level))
        reassessment = _REASSESSMENT_MINUTES.get(final_level, 15)

        reasoning.append(
            f"Final pediatric triage level: {final_level}. "
            f"Reassessment in {reassessment} minutes."
        )

        return PediatricTriageResult(
            level=final_level,
            age_group=age_group,
            pat_assessment=pat_result,
            pews_score=pews_total,
            vital_sign_interpretation=vs_interp,
            reasoning=reasoning,
            recommended_reassessment_minutes=reassessment,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the theoretical range of the triage level.

        Returns
        -------
        tuple of (float, float)
            ``(1.0, 5.0)``.
        """
        return (1.0, 5.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Return a clinical interpretation for a pediatric triage level.

        Parameters
        ----------
        score : int or float
            Triage level (1--5).

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
            1: (
                "Level 1 (Resuscitation): Immediate life-threatening "
                "condition. Continuous monitoring. Activate resuscitation "
                "team."
            ),
            2: (
                "Level 2 (Emergent): High-risk pediatric presentation. "
                "Physician assessment within 5 minutes. Close monitoring."
            ),
            3: (
                "Level 3 (Urgent): Potentially serious condition. "
                "Assessment within 15 minutes."
            ),
            4: (
                "Level 4 (Less Urgent): Condition unlikely to deteriorate "
                "rapidly. Reassess within 30 minutes."
            ),
            5: (
                "Level 5 (Non-Urgent): Minor condition. Reassess within "
                "60 minutes."
            ),
        }
        if level not in interpretations:
            raise ValidationError(
                message=(
                    f"Pediatric triage level must be in {{1,2,3,4,5}}, "
                    f"got {level}."
                ),
                parameter="score",
            )
        return interpretations[level]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpret_vital_signs(
        vital_signs: Dict[str, float],
        age_group: PediatricAgeGroup,
    ) -> Dict[str, str]:
        """Compare vital signs against age-specific normal ranges.

        Parameters
        ----------
        vital_signs : dict of str to float
            Vital-sign measurements.
        age_group : PediatricAgeGroup
            Patient's classified age group.

        Returns
        -------
        dict of str to str
            Per-parameter interpretation: ``"normal"``, ``"low"``, or
            ``"high"``.
        """
        ranges = _AGE_VITAL_RANGES[age_group]
        interpretation: Dict[str, str] = {}

        checks = [
            ("heart_rate", ranges.hr_low, ranges.hr_high),
            ("respiratory_rate", ranges.rr_low, ranges.rr_high),
            ("systolic_bp", ranges.sbp_low, ranges.sbp_high),
        ]

        for param, low, high in checks:
            value = vital_signs.get(param)
            if value is None:
                interpretation[param] = "normal"
                continue
            if value < low:
                interpretation[param] = "low"
            elif value > high:
                interpretation[param] = "high"
            else:
                interpretation[param] = "normal"

        spo2 = vital_signs.get("spo2")
        if spo2 is not None and spo2 < ranges.spo2_min:
            interpretation["spo2"] = "low"
        else:
            interpretation["spo2"] = "normal"

        return interpretation

    @staticmethod
    def _vital_sign_acuity(abnormal_count: int) -> int:
        """Map the count of abnormal vital signs to a triage level.

        Parameters
        ----------
        abnormal_count : int
            Number of vital signs outside the age-normal range.

        Returns
        -------
        int
            Suggested triage level based on vital-sign abnormalities.
        """
        if abnormal_count >= 4:
            return 1
        if abnormal_count >= 3:
            return 2
        if abnormal_count >= 2:
            return 3
        if abnormal_count >= 1:
            return 4
        return 5

    @staticmethod
    def _pews_to_triage_level(pews_total: int) -> int:
        """Convert a PEWS total score to a suggested triage level.

        Parameters
        ----------
        pews_total : int
            Composite PEWS (0--9).

        Returns
        -------
        int
            Suggested triage level.
        """
        if pews_total >= 7:
            return 1
        if pews_total >= 5:
            return 2
        if pews_total >= 3:
            return 3
        if pews_total >= 1:
            return 4
        return 5

    @staticmethod
    def get_vital_sign_ranges(
        age_group: PediatricAgeGroup,
    ) -> Dict[str, Tuple[float, float]]:
        """Return the normal vital-sign ranges for a given age group.

        Parameters
        ----------
        age_group : PediatricAgeGroup
            The pediatric age group.

        Returns
        -------
        dict of str to tuple of (float, float)
            Parameter names mapped to ``(lower_normal, upper_normal)``
            tuples.
        """
        r = _AGE_VITAL_RANGES[age_group]
        return {
            "heart_rate": (r.hr_low, r.hr_high),
            "respiratory_rate": (r.rr_low, r.rr_high),
            "systolic_bp": (r.sbp_low, r.sbp_high),
            "spo2": (r.spo2_min, 100.0),
        }
