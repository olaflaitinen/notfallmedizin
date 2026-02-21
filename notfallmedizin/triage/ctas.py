# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Canadian Triage and Acuity Scale (CTAS) implementation.

The Canadian Triage and Acuity Scale is a five-level triage system used
in Canadian emergency departments to assign patients a level of acuity
based on their presenting complaint, first-order modifiers (vital signs,
consciousness, pain severity, mechanism of injury), and second-order
modifiers (hemorrhage, acute pain, time since symptom onset).

CTAS Levels
-----------
- **Level I** (Resuscitation): Conditions threatening life or limb
  requiring immediate aggressive intervention. Time to physician:
  immediate (0 min).
- **Level II** (Emergent): Potential threat to life, limb, or function
  requiring rapid intervention. Time to physician: 15 min.
- **Level III** (Urgent): Conditions that could progress to a serious
  problem. Time to physician: 30 min.
- **Level IV** (Less Urgent): Conditions related to patient age,
  distress, or potential for deterioration. Time to physician: 60 min.
- **Level V** (Non-Urgent): Acute but non-urgent, or chronic conditions.
  Time to physician: 120 min.

References
----------
.. [1] Beveridge R, Clarke B, Janes L, et al. Canadian Emergency
       Department Triage and Acuity Scale: Implementation Guidelines.
       Can J Emerg Med. 1999;1(3 suppl):S2-S28.
       doi:10.1017/S1481803500004765
.. [2] Bullard MJ, Unger B, Spence J, Grafstein E; CTAS National
       Working Group. Revisions to the Canadian Emergency Department
       Triage and Acuity Scale (CTAS) adult guidelines. Can J Emerg
       Med. 2008;10(2):136-151. doi:10.1017/S1481803500009854
.. [3] Grafstein E, Bullard MJ, Warren D, Unger B; CTAS National
       Working Group. Revision of the Canadian Emergency Department
       Information System (CEDIS) Presenting Complaint List version
       1.1. Can J Emerg Med. 2008;10(2):151-173.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from notfallmedizin.core.base import BaseScorer
from notfallmedizin.core.exceptions import ClinicalRangeError, ValidationError


# ======================================================================
# Constants and enumerations
# ======================================================================


class CTASLevel(Enum):
    """CTAS acuity levels with associated metadata.

    Each member carries ``(level, acuity_name, target_minutes)`` as
    its value.
    """

    I = (1, "Resuscitation", 0)
    II = (2, "Emergent", 15)
    III = (3, "Urgent", 30)
    IV = (4, "Less Urgent", 60)
    V = (5, "Non-Urgent", 120)

    def __init__(
        self,
        level: int,
        acuity_name: str,
        target_minutes: int,
    ) -> None:
        self.level = level
        self.acuity_name = acuity_name
        self.target_minutes = target_minutes


_COMPLAINT_GROUP_LEVELS: Dict[str, CTASLevel] = {
    # Level I complaints
    "cardiac_arrest": CTASLevel.I,
    "respiratory_arrest": CTASLevel.I,
    "major_trauma": CTASLevel.I,
    "shock": CTASLevel.I,
    "unconscious": CTASLevel.I,
    # Level II complaints
    "chest_pain_cardiac": CTASLevel.II,
    "overdose_altered_consciousness": CTASLevel.II,
    "severe_respiratory_distress": CTASLevel.II,
    "acute_stroke": CTASLevel.II,
    "severe_allergic_reaction": CTASLevel.II,
    "gi_hemorrhage": CTASLevel.II,
    "ectopic_pregnancy": CTASLevel.II,
    "status_epilepticus": CTASLevel.II,
    "sepsis_suspected": CTASLevel.II,
    "acute_psychosis": CTASLevel.II,
    # Level III complaints
    "moderate_asthma": CTASLevel.III,
    "moderate_abdominal_pain": CTASLevel.III,
    "minor_head_injury": CTASLevel.III,
    "chest_pain_non_cardiac": CTASLevel.III,
    "moderate_trauma": CTASLevel.III,
    "febrile_seizure": CTASLevel.III,
    "urinary_retention": CTASLevel.III,
    "vaginal_bleeding": CTASLevel.III,
    "renal_colic": CTASLevel.III,
    # Level IV complaints
    "chronic_pain_exacerbation": CTASLevel.IV,
    "sore_throat": CTASLevel.IV,
    "minor_musculoskeletal": CTASLevel.IV,
    "urinary_symptoms": CTASLevel.IV,
    "mild_allergic_reaction": CTASLevel.IV,
    "earache": CTASLevel.IV,
    "minor_laceration": CTASLevel.IV,
    "mild_abdominal_pain": CTASLevel.IV,
    # Level V complaints
    "prescription_refill": CTASLevel.V,
    "medication_request": CTASLevel.V,
    "chronic_stable": CTASLevel.V,
    "suture_removal": CTASLevel.V,
    "dressing_change": CTASLevel.V,
    "minor_rash": CTASLevel.V,
    "insect_bite": CTASLevel.V,
}
"""Mapping from CEDIS complaint groups to baseline CTAS levels.

Based on the Canadian Emergency Department Information System (CEDIS)
presenting-complaint list version 1.1 [3].
"""

_FIRST_ORDER_MODIFIERS = frozenset({
    "vital_signs",
    "consciousness",
    "pain_severity",
    "mechanism_of_injury",
})

_SECOND_ORDER_MODIFIERS = frozenset({
    "hemorrhage",
    "acute_pain",
    "time_since_onset",
})

_PAIN_SEVERITY_SHIFT: Dict[str, int] = {
    "severe": -1,
    "moderate": 0,
    "mild": 1,
    "none": 1,
}
"""Level shift applied by the pain-severity first-order modifier.

Negative values escalate (lower level number = more urgent); positive
values de-escalate.
"""

_CONSCIOUSNESS_SHIFT: Dict[str, int] = {
    "unresponsive": -3,
    "responds_to_pain": -2,
    "responds_to_voice": -1,
    "confused": -1,
    "alert": 0,
}
"""Level shift applied by the consciousness first-order modifier."""

_MECHANISM_SHIFT: Dict[str, int] = {
    "high_energy": -1,
    "moderate_energy": 0,
    "low_energy": 0,
    "trivial": 1,
}
"""Level shift applied by the mechanism-of-injury first-order modifier."""

_VITAL_SIGN_ABNORMALITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "heart_rate": (50.0, 150.0),
    "systolic_bp": (80.0, 200.0),
    "respiratory_rate": (10.0, 30.0),
    "spo2": (92.0, 100.0),
    "temperature": (35.0, 40.0),
}
"""Thresholds for vital-sign abnormality detection in CTAS.

Values outside ``(lower, upper)`` are considered abnormal and trigger
a level escalation.
"""

_HEMORRHAGE_SHIFT: Dict[str, int] = {
    "major": -1,
    "moderate": 0,
    "minor": 0,
    "none": 0,
}

_TIME_ONSET_SHIFT: Dict[str, int] = {
    "minutes": -1,
    "hours": 0,
    "days": 0,
    "weeks": 1,
}
"""Level shift applied by the time-since-onset second-order modifier."""


# ======================================================================
# Result dataclass
# ======================================================================


@dataclass(frozen=True)
class CTASResult:
    """Structured result of a CTAS triage assessment.

    Parameters
    ----------
    level : int
        CTAS acuity level in {1, 2, 3, 4, 5}. Level 1 is the most
        acute.
    acuity_name : str
        Human-readable acuity name (e.g. ``"Resuscitation"``).
    target_time_to_physician_minutes : int
        Maximum recommended time (in minutes) until physician
        assessment.
    modifiers_applied : list of str
        List of modifiers that influenced the final level assignment.
    reasoning : list of str
        Ordered list of reasoning steps.
    baseline_level : int
        Initial CTAS level derived from the complaint group before
        modifier adjustments.
    """

    level: int
    acuity_name: str
    target_time_to_physician_minutes: int
    modifiers_applied: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    baseline_level: int = 3

    def __int__(self) -> int:
        return self.level

    def __float__(self) -> float:
        return float(self.level)


# ======================================================================
# Calculator
# ======================================================================


class CTASTriageCalculator(BaseScorer):
    """Canadian Triage and Acuity Scale (CTAS) calculator.

    Implements the CTAS triage algorithm including baseline assignment
    from complaint groups and two tiers of modifiers that can escalate
    or de-escalate the acuity level.

    Parameters
    ----------
    max_escalation_steps : int, optional
        Maximum number of acuity levels a modifier can shift the
        assignment. Default is ``2``.
    allow_deescalation : bool, optional
        If ``True`` (default), second-order modifiers may de-escalate
        the level. If ``False``, modifiers can only escalate.

    Examples
    --------
    >>> calc = CTASTriageCalculator()
    >>> result = calc.calculate(
    ...     complaint_group="moderate_abdominal_pain",
    ...     first_order_modifiers={
    ...         "pain_severity": "severe",
    ...         "consciousness": "alert",
    ...     },
    ...     second_order_modifiers={
    ...         "hemorrhage": "none",
    ...         "time_since_onset": "hours",
    ...     },
    ... )
    >>> result.level
    2

    References
    ----------
    .. [1] Beveridge R, et al. CTAS Implementation Guidelines. CJEM.
           1999;1(3 suppl):S2-S28.
    .. [2] Bullard MJ, et al. Revisions to the CTAS adult guidelines.
           CJEM. 2008;10(2):136-151.
    """

    def __init__(
        self,
        max_escalation_steps: int = 2,
        allow_deescalation: bool = True,
    ) -> None:
        super().__init__(
            name="Canadian Triage and Acuity Scale",
            version="2008",
            references=[
                "Beveridge R, Clarke B, Janes L, et al. CTAS "
                "Implementation Guidelines. CJEM. 1999;1(3 suppl):S2-S28.",
                "Bullard MJ, Unger B, Spence J, Grafstein E. Revisions "
                "to the CTAS adult guidelines. CJEM. 2008;10(2):136-151.",
            ],
        )
        self.max_escalation_steps = max_escalation_steps
        self.allow_deescalation = allow_deescalation

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    def validate_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate and normalize CTAS triage inputs.

        Parameters
        ----------
        **kwargs
            Expected keys:

            - ``complaint_group`` (str): CEDIS complaint-group
              identifier.
            - ``first_order_modifiers`` (dict of str to str or dict):
              First-order modifiers. Keys are modifier types; values are
              severity descriptors. For ``"vital_signs"``, the value
              must be a dict of vital-sign measurements.
            - ``second_order_modifiers`` (dict of str to str, optional):
              Second-order modifiers.

        Returns
        -------
        dict of str to Any
            Validated inputs.

        Raises
        ------
        ValidationError
            If required keys are missing or modifier values are
            unrecognised.
        """
        if "complaint_group" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'complaint_group'.",
                parameter="complaint_group",
            )
        if "first_order_modifiers" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'first_order_modifiers'.",
                parameter="first_order_modifiers",
            )

        complaint_group = str(kwargs["complaint_group"]).strip().lower()

        fo_mods_raw = kwargs["first_order_modifiers"]
        if not isinstance(fo_mods_raw, dict):
            raise ValidationError(
                message=(
                    "Parameter 'first_order_modifiers' must be a dict, "
                    f"got {type(fo_mods_raw).__name__}."
                ),
                parameter="first_order_modifiers",
            )

        fo_mods: Dict[str, Any] = {}
        for key, value in fo_mods_raw.items():
            key_lower = str(key).strip().lower()
            if key_lower == "vital_signs":
                if not isinstance(value, dict):
                    raise ValidationError(
                        message=(
                            "First-order modifier 'vital_signs' must be "
                            f"a dict, got {type(value).__name__}."
                        ),
                        parameter="vital_signs",
                    )
                fo_mods[key_lower] = value
            elif key_lower == "pain_severity":
                sev = str(value).strip().lower()
                if sev not in _PAIN_SEVERITY_SHIFT:
                    raise ValidationError(
                        message=(
                            f"Invalid pain severity '{sev}'. "
                            f"Valid: {sorted(_PAIN_SEVERITY_SHIFT.keys())}."
                        ),
                        parameter="pain_severity",
                    )
                fo_mods[key_lower] = sev
            elif key_lower == "consciousness":
                con = str(value).strip().lower()
                if con not in _CONSCIOUSNESS_SHIFT:
                    raise ValidationError(
                        message=(
                            f"Invalid consciousness level '{con}'. "
                            f"Valid: {sorted(_CONSCIOUSNESS_SHIFT.keys())}."
                        ),
                        parameter="consciousness",
                    )
                fo_mods[key_lower] = con
            elif key_lower == "mechanism_of_injury":
                mech = str(value).strip().lower()
                if mech not in _MECHANISM_SHIFT:
                    raise ValidationError(
                        message=(
                            f"Invalid mechanism '{mech}'. "
                            f"Valid: {sorted(_MECHANISM_SHIFT.keys())}."
                        ),
                        parameter="mechanism_of_injury",
                    )
                fo_mods[key_lower] = mech
            else:
                fo_mods[key_lower] = str(value).strip().lower()

        so_mods_raw = kwargs.get("second_order_modifiers", {})
        if not isinstance(so_mods_raw, dict):
            raise ValidationError(
                message=(
                    "Parameter 'second_order_modifiers' must be a dict, "
                    f"got {type(so_mods_raw).__name__}."
                ),
                parameter="second_order_modifiers",
            )

        so_mods: Dict[str, str] = {}
        for key, value in so_mods_raw.items():
            so_mods[str(key).strip().lower()] = str(value).strip().lower()

        return {
            "complaint_group": complaint_group,
            "first_order_modifiers": fo_mods,
            "second_order_modifiers": so_mods,
        }

    def calculate(self, **kwargs: Any) -> CTASResult:
        """Compute the CTAS triage level.

        The algorithm proceeds in three phases:

        1. **Baseline**: Determine the initial CTAS level from the
           complaint group.
        2. **First-order modifiers**: Vital signs, consciousness,
           pain severity, and mechanism of injury may escalate (or
           de-escalate) the level.
        3. **Second-order modifiers**: Hemorrhage, acute pain, and
           time since onset provide additional adjustment.

        Parameters
        ----------
        **kwargs
            Validated CTAS parameters. See :meth:`validate_inputs`.

        Returns
        -------
        CTASResult
            Structured triage result.
        """
        complaint_group: str = kwargs["complaint_group"]
        fo_mods: Dict[str, Any] = kwargs["first_order_modifiers"]
        so_mods: Dict[str, str] = kwargs["second_order_modifiers"]

        reasoning: List[str] = []
        modifiers_applied: List[str] = []

        # Phase 1: Baseline from complaint group
        if complaint_group in _COMPLAINT_GROUP_LEVELS:
            baseline_enum = _COMPLAINT_GROUP_LEVELS[complaint_group]
        else:
            baseline_enum = CTASLevel.III
            reasoning.append(
                f"Complaint group '{complaint_group}' is not in the "
                f"standard CEDIS set; defaulting to Level III (Urgent)."
            )

        baseline_level = baseline_enum.level
        reasoning.append(
            f"Baseline: complaint group '{complaint_group}' maps to "
            f"CTAS Level {baseline_level} ({baseline_enum.acuity_name})."
        )
        current_level = baseline_level

        # Phase 2: First-order modifiers
        fo_shift = 0

        if "vital_signs" in fo_mods:
            vs_shift = self._evaluate_vital_sign_modifier(
                fo_mods["vital_signs"], reasoning,
            )
            if vs_shift != 0:
                fo_shift = min(fo_shift, vs_shift)
                modifiers_applied.append(f"vital_signs (shift {vs_shift:+d})")

        if "consciousness" in fo_mods:
            con_shift = _CONSCIOUSNESS_SHIFT.get(fo_mods["consciousness"], 0)
            if con_shift != 0:
                fo_shift = min(fo_shift, con_shift)
                modifiers_applied.append(
                    f"consciousness={fo_mods['consciousness']} "
                    f"(shift {con_shift:+d})"
                )
                reasoning.append(
                    f"First-order: consciousness='{fo_mods['consciousness']}' "
                    f"applies shift {con_shift:+d}."
                )

        if "pain_severity" in fo_mods:
            pain_shift = _PAIN_SEVERITY_SHIFT.get(fo_mods["pain_severity"], 0)
            if pain_shift != 0:
                if pain_shift < 0:
                    fo_shift = min(fo_shift, pain_shift)
                else:
                    fo_shift = max(fo_shift, pain_shift) if fo_shift >= 0 else fo_shift
                modifiers_applied.append(
                    f"pain_severity={fo_mods['pain_severity']} "
                    f"(shift {pain_shift:+d})"
                )
                reasoning.append(
                    f"First-order: pain_severity='{fo_mods['pain_severity']}' "
                    f"applies shift {pain_shift:+d}."
                )

        if "mechanism_of_injury" in fo_mods:
            mech_shift = _MECHANISM_SHIFT.get(fo_mods["mechanism_of_injury"], 0)
            if mech_shift != 0:
                if mech_shift < 0:
                    fo_shift = min(fo_shift, mech_shift)
                else:
                    fo_shift = max(fo_shift, mech_shift) if fo_shift >= 0 else fo_shift
                modifiers_applied.append(
                    f"mechanism_of_injury={fo_mods['mechanism_of_injury']} "
                    f"(shift {mech_shift:+d})"
                )
                reasoning.append(
                    f"First-order: mechanism='{fo_mods['mechanism_of_injury']}' "
                    f"applies shift {mech_shift:+d}."
                )

        clamped_fo = max(-self.max_escalation_steps, min(self.max_escalation_steps, fo_shift))
        current_level = self._clamp_level(current_level + clamped_fo)
        if clamped_fo != 0:
            reasoning.append(
                f"After first-order modifiers: net shift {clamped_fo:+d}, "
                f"level adjusted to {current_level}."
            )

        # Phase 3: Second-order modifiers
        so_shift = 0

        if "hemorrhage" in so_mods:
            h_shift = _HEMORRHAGE_SHIFT.get(so_mods["hemorrhage"], 0)
            if h_shift != 0:
                so_shift = min(so_shift, h_shift)
                modifiers_applied.append(
                    f"hemorrhage={so_mods['hemorrhage']} (shift {h_shift:+d})"
                )
                reasoning.append(
                    f"Second-order: hemorrhage='{so_mods['hemorrhage']}' "
                    f"applies shift {h_shift:+d}."
                )

        if "time_since_onset" in so_mods:
            t_shift = _TIME_ONSET_SHIFT.get(so_mods["time_since_onset"], 0)
            if t_shift != 0:
                if t_shift < 0:
                    so_shift = min(so_shift, t_shift)
                elif self.allow_deescalation:
                    so_shift = max(so_shift, t_shift) if so_shift >= 0 else so_shift
                modifiers_applied.append(
                    f"time_since_onset={so_mods['time_since_onset']} "
                    f"(shift {t_shift:+d})"
                )
                reasoning.append(
                    f"Second-order: time_since_onset='{so_mods['time_since_onset']}' "
                    f"applies shift {t_shift:+d}."
                )

        if "acute_pain" in so_mods and so_mods["acute_pain"] in ("yes", "true"):
            so_shift = min(so_shift, -1)
            modifiers_applied.append("acute_pain=yes (shift -1)")
            reasoning.append(
                "Second-order: acute_pain=yes applies shift -1."
            )

        if not self.allow_deescalation:
            so_shift = min(so_shift, 0)

        clamped_so = max(-1, min(1, so_shift))
        current_level = self._clamp_level(current_level + clamped_so)
        if clamped_so != 0:
            reasoning.append(
                f"After second-order modifiers: net shift {clamped_so:+d}, "
                f"level adjusted to {current_level}."
            )

        # Resolve the final CTAS level enum
        final_enum = self._level_to_enum(current_level)
        reasoning.append(
            f"Final CTAS assignment: Level {final_enum.level} "
            f"({final_enum.acuity_name}). "
            f"Target time to physician: {final_enum.target_minutes} min."
        )

        return CTASResult(
            level=final_enum.level,
            acuity_name=final_enum.acuity_name,
            target_time_to_physician_minutes=final_enum.target_minutes,
            modifiers_applied=modifiers_applied,
            reasoning=reasoning,
            baseline_level=baseline_level,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the theoretical range of the CTAS level.

        Returns
        -------
        tuple of (float, float)
            ``(1.0, 5.0)`` representing CTAS levels I through V.
        """
        return (1.0, 5.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Return a clinical interpretation of a CTAS level.

        Parameters
        ----------
        score : int or float
            A CTAS level (1--5).

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
        try:
            ctas = self._level_to_enum(level)
        except ValidationError:
            raise
        return (
            f"CTAS Level {ctas.level} ({ctas.acuity_name}): "
            f"Target time to physician assessment is "
            f"{ctas.target_minutes} minutes."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_level(level: int) -> int:
        """Clamp a CTAS level to the valid range [1, 5]."""
        return max(1, min(5, level))

    @staticmethod
    def _level_to_enum(level: int) -> CTASLevel:
        """Convert a numeric level to its ``CTASLevel`` enum member.

        Parameters
        ----------
        level : int
            CTAS level in {1, 2, 3, 4, 5}.

        Returns
        -------
        CTASLevel
            Corresponding enum member.

        Raises
        ------
        ValidationError
            If the level is not in {1, 2, 3, 4, 5}.
        """
        for member in CTASLevel:
            if member.level == level:
                return member
        raise ValidationError(
            message=f"CTAS level must be in {{1,2,3,4,5}}, got {level}.",
            parameter="level",
        )

    @staticmethod
    def _evaluate_vital_sign_modifier(
        vital_signs: Dict[str, float],
        reasoning: List[str],
    ) -> int:
        """Evaluate vital-sign abnormalities and return a level shift.

        Each vital sign is checked against the CTAS adult normal
        thresholds.  Any value outside the normal range triggers an
        escalation of -1.

        Parameters
        ----------
        vital_signs : dict of str to float
            Vital-sign measurements.
        reasoning : list of str
            Reasoning list to append explanations to (mutated in place).

        Returns
        -------
        int
            Cumulative level shift (negative = more urgent).
        """
        shift = 0
        for param, (lower, upper) in _VITAL_SIGN_ABNORMALITY_THRESHOLDS.items():
            value = vital_signs.get(param)
            if value is None:
                continue
            if value < lower or value > upper:
                shift = -1
                reasoning.append(
                    f"First-order: vital sign '{param}'={value} is outside "
                    f"normal range [{lower}, {upper}]; escalation indicated."
                )
        return shift

    @staticmethod
    def get_complaint_groups() -> Dict[str, int]:
        """Return the mapping of complaint groups to baseline levels.

        Returns
        -------
        dict of str to int
            Complaint-group identifiers mapped to their baseline CTAS
            level numbers.
        """
        return {k: v.level for k, v in _COMPLAINT_GROUP_LEVELS.items()}
