# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Cardiac clinical scoring systems.

This module implements the HEART score for acute coronary syndrome
risk, the TIMI risk score for UA/NSTEMI, and the CHA2DS2-VASc score
for stroke risk in atrial fibrillation.

Classes
-------
HEARTScore
    History, ECG, Age, Risk factors, Troponin score (0--10).
TIMIScore
    Thrombolysis In Myocardial Infarction risk score (0--7).
CHA2DS2VAScScore
    Stroke risk stratification in atrial fibrillation (0--9).
"""

from __future__ import annotations

import math
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
# HEART Score
# ======================================================================


class HEARTScore(BaseScorer):
    """HEART score for major adverse cardiac events (MACE) risk.

    Five components, each scored 0--2, yielding a total of 0--10.

    Parameters (passed to ``calculate``)
    -------------------------------------
    history : int
        0 = slightly suspicious, 1 = moderately suspicious,
        2 = highly suspicious.
    ecg : int
        0 = normal, 1 = non-specific repolarization disturbance,
        2 = significant ST deviation.
    age : int
        0 = <45 years, 1 = 45--64 years, 2 = >=65 years.
    risk_factors : int
        0 = no known risk factors, 1 = 1--2 risk factors,
        2 = >=3 risk factors or history of atherosclerotic disease.
    troponin : int
        0 = <=normal limit, 1 = 1--3x normal limit,
        2 = >3x normal limit.

    References
    ----------
    Six AJ, et al. "Chest pain in the emergency room: value of the
    HEART score." Neth Heart J. 2008;16(6):191-196.
    """

    _COMPONENTS = ("history", "ecg", "age", "risk_factors", "troponin")

    def __init__(self) -> None:
        super().__init__()
        self._name = "HEART Score"
        self._description = (
            "History, ECG, Age, Risk factors, and Troponin score "
            "for chest pain risk stratification (0-10)."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for HEART score calculation.

        Parameters
        ----------
        **kwargs
            Must include ``history``, ``ecg``, ``age``,
            ``risk_factors``, and ``troponin``, each an int 0--2.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a component value is not in [0, 2].
        """
        for comp in self._COMPONENTS:
            if comp not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{comp}' for HEART score.",
                    parameter=comp,
                )
            val = kwargs[comp]
            if not isinstance(val, int) or val < 0 or val > 2:
                raise ClinicalRangeError(comp, float(val), 0, 2)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the HEART score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with total (0--10), component breakdown, and risk.
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
        """Return the minimum and maximum possible HEART scores.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 10.0)``.
        """
        return (0.0, 10.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the HEART score.

        Parameters
        ----------
        score : int or float
            HEART score (0--10).

        Returns
        -------
        str
            Risk-stratified interpretation.
        """
        s = int(score)
        if s <= 3:
            return (
                f"HEART {s}: Low risk (1.7% MACE). "
                "Consider early discharge with outpatient follow-up."
            )
        if s <= 6:
            return (
                f"HEART {s}: Moderate risk (12-16.6% MACE). "
                "Admit for observation and further workup."
            )
        return (
            f"HEART {s}: High risk (50-65% MACE). "
            "Early invasive strategy recommended."
        )

    @staticmethod
    def _categorize(score: int) -> str:
        """Map total score to a risk category string.

        Parameters
        ----------
        score : int
            HEART score.

        Returns
        -------
        str
            ``"low"``, ``"moderate"``, or ``"high"``.
        """
        if score <= 3:
            return "low"
        if score <= 6:
            return "moderate"
        return "high"


# ======================================================================
# TIMI Score
# ======================================================================


class TIMIScore(BaseScorer):
    """TIMI risk score for unstable angina / NSTEMI.

    Seven binary criteria, each contributing one point (total 0--7).

    Parameters (passed to ``calculate``)
    -------------------------------------
    age_ge_65 : bool
        Age >= 65 years.
    cad_risk_factors_ge_3 : bool
        >= 3 coronary artery disease risk factors (family history,
        hypertension, hypercholesterolemia, diabetes, active smoker).
    known_cad : bool
        Known CAD with stenosis >= 50%.
    aspirin_use : bool
        Aspirin use in the past 7 days.
    severe_angina : bool
        >= 2 anginal episodes in the past 24 hours.
    st_deviation : bool
        ST deviation >= 0.5 mm on presenting ECG.
    positive_cardiac_marker : bool
        Elevated cardiac biomarkers (troponin or CK-MB).

    References
    ----------
    Antman EM, et al. "The TIMI risk score for unstable angina/
    non-ST elevation MI." JAMA. 2000;284(7):835-842.
    """

    _CRITERIA = (
        "age_ge_65",
        "cad_risk_factors_ge_3",
        "known_cad",
        "aspirin_use",
        "severe_angina",
        "st_deviation",
        "positive_cardiac_marker",
    )

    def __init__(self) -> None:
        super().__init__()
        self._name = "TIMI Risk Score"
        self._description = (
            "Thrombolysis In Myocardial Infarction risk score with "
            "7 binary criteria for UA/NSTEMI (0-7)."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for TIMI score calculation.

        Parameters
        ----------
        **kwargs
            Must include all 7 boolean criteria.

        Raises
        ------
        ValidationError
            If a required parameter is missing or not boolean.
        """
        for crit in self._CRITERIA:
            if crit not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{crit}' for TIMI score.",
                    parameter=crit,
                )
            if not isinstance(kwargs[crit], bool):
                raise ValidationError(
                    message=f"Parameter '{crit}' must be a boolean.",
                    parameter=crit,
                )

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the TIMI risk score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with total score (0--7) and per-criterion breakdown.
        """
        self.validate_inputs(**kwargs)

        components: Dict[str, float] = {
            crit: 1.0 if kwargs[crit] else 0.0 for crit in self._CRITERIA
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
        """Return the minimum and maximum TIMI scores.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 7.0)``.
        """
        return (0.0, 7.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the TIMI risk score.

        Parameters
        ----------
        score : int or float
            TIMI score (0--7).

        Returns
        -------
        str
            Risk interpretation with approximate event rates.
        """
        s = int(score)
        if s <= 1:
            return f"TIMI {s}: Low risk (~5% 14-day event rate)."
        if s <= 2:
            return f"TIMI {s}: Low-intermediate risk (~8% 14-day event rate)."
        if s <= 4:
            return f"TIMI {s}: Intermediate risk (~13-20% 14-day event rate)."
        return f"TIMI {s}: High risk (~25-41% 14-day event rate)."

    @staticmethod
    def _categorize(score: int) -> str:
        """Map total TIMI score to a risk category string.

        Parameters
        ----------
        score : int
            TIMI total score.

        Returns
        -------
        str
            ``"low"``, ``"moderate"``, or ``"high"``.
        """
        if score <= 2:
            return "low"
        if score <= 4:
            return "moderate"
        return "high"


# ======================================================================
# CHA2DS2-VASc Score
# ======================================================================


class CHA2DS2VAScScore(BaseScorer):
    """CHA2DS2-VASc score for stroke risk in atrial fibrillation.

    Weighted criteria yielding a total of 0--9.

    Parameters (passed to ``calculate``)
    -------------------------------------
    chf : bool
        Congestive heart failure / LV dysfunction.
    hypertension : bool
        Hypertension.
    age : int
        Patient age in years.
    diabetes : bool
        Diabetes mellitus.
    stroke_tia_te : bool
        Prior stroke, TIA, or thromboembolism.
    vascular_disease : bool
        Prior MI, peripheral artery disease, or aortic plaque.
    female : bool
        Female sex.

    References
    ----------
    Lip GYH, et al. "Refining clinical risk stratification for
    predicting stroke and thromboembolism in atrial fibrillation
    using a novel risk factor-based approach." Chest. 2010;137(2):263-272.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "CHA2DS2-VASc Score"
        self._description = (
            "Stroke risk stratification in atrial fibrillation "
            "using weighted criteria (0-9)."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for CHA2DS2-VASc calculation.

        Parameters
        ----------
        **kwargs
            Must include ``chf``, ``hypertension``, ``age``,
            ``diabetes``, ``stroke_tia_te``, ``vascular_disease``,
            and ``female``.

        Raises
        ------
        ValidationError
            If a required parameter is missing or has wrong type.
        ClinicalRangeError
            If age is out of plausible range.
        """
        bool_params = [
            "chf",
            "hypertension",
            "diabetes",
            "stroke_tia_te",
            "vascular_disease",
            "female",
        ]
        for param in bool_params:
            if param not in kwargs:
                raise ValidationError(
                    message=(
                        f"Missing required parameter '{param}' for "
                        "CHA2DS2-VASc score."
                    ),
                    parameter=param,
                )
            if not isinstance(kwargs[param], bool):
                raise ValidationError(
                    message=f"Parameter '{param}' must be a boolean.",
                    parameter=param,
                )

        if "age" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'age' for CHA2DS2-VASc score.",
                parameter="age",
            )
        age = kwargs["age"]
        if not isinstance(age, int) or age < 0 or age > 150:
            raise ClinicalRangeError("age", float(age), 0, 150)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the CHA2DS2-VASc score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with total score (0--9) and component breakdown.
        """
        self.validate_inputs(**kwargs)

        age = kwargs["age"]

        chf_pts = 1 if kwargs["chf"] else 0
        htn_pts = 1 if kwargs["hypertension"] else 0
        dm_pts = 1 if kwargs["diabetes"] else 0
        stroke_pts = 2 if kwargs["stroke_tia_te"] else 0
        vasc_pts = 1 if kwargs["vascular_disease"] else 0
        sex_pts = 1 if kwargs["female"] else 0

        age_pts = 0
        if age >= 75:
            age_pts = 2
        elif age >= 65:
            age_pts = 1

        total = chf_pts + htn_pts + age_pts + dm_pts + stroke_pts + vasc_pts + sex_pts

        components: Dict[str, float] = {
            "chf": float(chf_pts),
            "hypertension": float(htn_pts),
            "age": float(age_pts),
            "diabetes": float(dm_pts),
            "stroke_tia_te": float(stroke_pts),
            "vascular_disease": float(vasc_pts),
            "female_sex": float(sex_pts),
        }

        interpretation = self.interpret(total)
        risk = self._categorize(total)

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum CHA2DS2-VASc scores.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 9.0)``.
        """
        return (0.0, 9.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the CHA2DS2-VASc score.

        Parameters
        ----------
        score : int or float
            CHA2DS2-VASc score (0--9).

        Returns
        -------
        str
            Stroke risk interpretation with annual stroke rate.
        """
        rates = {
            0: 0.0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0,
            5: 6.7, 6: 9.8, 7: 9.6, 8: 6.7, 9: 15.2,
        }
        s = int(score)
        s_clamped = max(0, min(s, 9))
        rate = rates[s_clamped]

        if s == 0:
            return (
                f"CHA2DS2-VASc {s}: Low risk. Annual stroke rate ~{rate}%. "
                "No antithrombotic therapy may be considered."
            )
        if s == 1:
            return (
                f"CHA2DS2-VASc {s}: Low-moderate risk. Annual stroke rate "
                f"~{rate}%. Oral anticoagulation or antiplatelet therapy "
                "should be considered."
            )
        return (
            f"CHA2DS2-VASc {s}: Moderate-high risk. Annual stroke rate "
            f"~{rate}%. Oral anticoagulation is recommended."
        )

    @staticmethod
    def _categorize(score: int) -> str:
        """Map CHA2DS2-VASc score to a risk category.

        Parameters
        ----------
        score : int
            Total score.

        Returns
        -------
        str
            ``"low"``, ``"moderate"``, or ``"high"``.
        """
        if score == 0:
            return "low"
        if score == 1:
            return "moderate"
        return "high"
