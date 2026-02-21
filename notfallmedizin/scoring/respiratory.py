# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Respiratory clinical scoring systems.

This module implements the CURB-65 pneumonia severity score and the
ROX index for evaluating high-flow nasal cannula (HFNC) therapy.

Classes
-------
CURB65Score
    Community-acquired pneumonia severity (0--5).
ROXIndex
    SpO2/FiO2 to respiratory rate ratio for HFNC assessment.
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
# CURB-65 Score
# ======================================================================

_CURB65_MORTALITY = {
    0: 0.007,
    1: 0.021,
    2: 0.092,
    3: 0.145,
    4: 0.40,
    5: 0.57,
}


class CURB65Score(BaseScorer):
    """CURB-65 severity score for community-acquired pneumonia.

    Five binary criteria, each worth one point (total 0--5).

    Parameters (passed to ``calculate``)
    -------------------------------------
    confusion : bool
        New-onset mental confusion (AMT score <= 8 or new disorientation).
    bun : float
        Blood urea nitrogen in mg/dL, or urea in mmol/L (see
        ``bun_unit``).
    bun_unit : str
        ``"mg_dl"`` (default) or ``"mmol_l"``. When ``"mg_dl"``,
        the threshold is BUN > 19 mg/dL. When ``"mmol_l"``, the
        threshold is urea > 7 mmol/L.
    respiratory_rate : float
        Respiratory rate in breaths per minute.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    diastolic_bp : float
        Diastolic blood pressure in mmHg.
    age : int
        Patient age in years.

    References
    ----------
    Lim WS, et al. "Defining community acquired pneumonia severity
    on presentation to hospital: an international derivation and
    validation study." Thorax. 2003;58(5):377-382.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "CURB-65 Score"
        self._description = (
            "Community-acquired pneumonia severity with 5 binary "
            "criteria: Confusion, Urea, Respiratory rate, Blood "
            "pressure, and age >= 65."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for CURB-65 calculation.

        Parameters
        ----------
        **kwargs
            Must include ``confusion``, ``bun``, ``respiratory_rate``,
            ``systolic_bp``, ``diastolic_bp``, and ``age``.

        Raises
        ------
        ValidationError
            If a required parameter is missing or has wrong type.
        ClinicalRangeError
            If a numeric value is out of range.
        """
        if "confusion" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'confusion' for CURB-65.",
                parameter="confusion",
            )
        if not isinstance(kwargs["confusion"], bool):
            raise ValidationError(
                message="Parameter 'confusion' must be a boolean.",
                parameter="confusion",
            )

        numeric_ranges = {
            "bun": (0, 300),
            "respiratory_rate": (0, 80),
            "systolic_bp": (0, 400),
            "diastolic_bp": (0, 250),
        }
        for param, (lo, hi) in numeric_ranges.items():
            if param not in kwargs:
                raise ValidationError(
                    message=(
                        f"Missing required parameter '{param}' for CURB-65."
                    ),
                    parameter=param,
                )
            val = kwargs[param]
            if not isinstance(val, (int, float)) or val < lo or val > hi:
                raise ClinicalRangeError(param, float(val), lo, hi)

        if "age" not in kwargs:
            raise ValidationError(
                message="Missing required parameter 'age' for CURB-65.",
                parameter="age",
            )
        age = kwargs["age"]
        if not isinstance(age, int) or age < 0 or age > 150:
            raise ClinicalRangeError("age", float(age), 0, 150)

        bun_unit = kwargs.get("bun_unit", "mg_dl")
        if bun_unit not in ("mg_dl", "mmol_l"):
            raise ValidationError(
                message=(
                    f"Parameter 'bun_unit' must be 'mg_dl' or 'mmol_l', "
                    f"got '{bun_unit}'."
                ),
                parameter="bun_unit",
            )

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the CURB-65 score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with CURB-65 total (0--5), component breakdown,
            and 30-day mortality estimate.
        """
        self.validate_inputs(**kwargs)

        confusion_pt = 1 if kwargs["confusion"] else 0

        bun_unit = kwargs.get("bun_unit", "mg_dl")
        bun_val = kwargs["bun"]
        if bun_unit == "mg_dl":
            urea_pt = 1 if bun_val > 19 else 0
        else:
            urea_pt = 1 if bun_val > 7 else 0

        rr_pt = 1 if kwargs["respiratory_rate"] >= 30 else 0

        bp_pt = 0
        if kwargs["systolic_bp"] < 90 or kwargs["diastolic_bp"] <= 60:
            bp_pt = 1

        age_pt = 1 if kwargs["age"] >= 65 else 0

        total = confusion_pt + urea_pt + rr_pt + bp_pt + age_pt

        components: Dict[str, float] = {
            "confusion": float(confusion_pt),
            "urea": float(urea_pt),
            "respiratory_rate": float(rr_pt),
            "blood_pressure": float(bp_pt),
            "age_ge_65": float(age_pt),
        }

        interpretation = self.interpret(total)
        risk = self._categorize(total)
        mortality = _CURB65_MORTALITY.get(total, 0.57)

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
            mortality_estimate=mortality,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum CURB-65 values.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 5.0)``.
        """
        return (0.0, 5.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the CURB-65 score.

        Parameters
        ----------
        score : int or float
            CURB-65 score (0--5).

        Returns
        -------
        str
            Severity interpretation with disposition guidance.
        """
        s = int(score)
        mortality = _CURB65_MORTALITY.get(s, 0.57)
        pct = round(mortality * 100, 1)

        if s <= 1:
            return (
                f"CURB-65 {s}: Low severity (30-day mortality ~{pct}%). "
                "Consider outpatient treatment."
            )
        if s == 2:
            return (
                f"CURB-65 {s}: Moderate severity (30-day mortality ~{pct}%). "
                "Consider short inpatient stay or closely supervised "
                "outpatient treatment."
            )
        return (
            f"CURB-65 {s}: High severity (30-day mortality ~{pct}%). "
            "Hospitalization required. Consider ICU admission if score >= 4."
        )

    @staticmethod
    def _categorize(score: int) -> str:
        """Map CURB-65 score to a severity category.

        Parameters
        ----------
        score : int
            CURB-65 total.

        Returns
        -------
        str
            ``"low"``, ``"moderate"``, or ``"high"``.
        """
        if score <= 1:
            return "low"
        if score == 2:
            return "moderate"
        return "high"


# ======================================================================
# ROX Index
# ======================================================================


class ROXIndex(BaseScorer):
    """ROX index for predicting HFNC success in hypoxemic respiratory failure.

    Calculated as:

        ROX = (SpO2 / FiO2) / respiratory_rate

    A ROX index >= 4.88 at 2, 6, or 12 hours after HFNC initiation
    is associated with a lower risk of intubation.

    Parameters (passed to ``calculate``)
    -------------------------------------
    spo2 : float
        Peripheral oxygen saturation as a percentage (0--100).
    fio2 : float
        Fraction of inspired oxygen as a proportion (0.21--1.0).
    respiratory_rate : float
        Respiratory rate in breaths per minute.

    References
    ----------
    Roca O, et al. "Predicting success of high-flow nasal cannula in
    pneumonia patients with hypoxemic respiratory failure: The utility
    of the ROX index." J Crit Care. 2016;35:200-205.
    """

    HFNC_SUCCESS_THRESHOLD: float = 4.88

    def __init__(self) -> None:
        super().__init__()
        self._name = "ROX Index"
        self._description = (
            "Ratio of SpO2/FiO2 to respiratory rate for predicting "
            "HFNC success. Threshold >= 4.88 favors HFNC success."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for ROX index calculation.

        Parameters
        ----------
        **kwargs
            Must include ``spo2``, ``fio2``, and ``respiratory_rate``.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a value is out of range.
        """
        required = ["spo2", "fio2", "respiratory_rate"]
        for param in required:
            if param not in kwargs:
                raise ValidationError(
                    message=(
                        f"Missing required parameter '{param}' for ROX index."
                    ),
                    parameter=param,
                )

        spo2 = kwargs["spo2"]
        if not isinstance(spo2, (int, float)) or spo2 < 0 or spo2 > 100:
            raise ClinicalRangeError("spo2", float(spo2), 0, 100)

        fio2 = kwargs["fio2"]
        if not isinstance(fio2, (int, float)) or fio2 < 0.21 or fio2 > 1.0:
            raise ClinicalRangeError("fio2", float(fio2), 0.21, 1.0)

        rr = kwargs["respiratory_rate"]
        if not isinstance(rr, (int, float)) or rr <= 0 or rr > 80:
            raise ClinicalRangeError("respiratory_rate", float(rr), 0, 80)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the ROX index.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with ROX value and HFNC success prediction.
        """
        self.validate_inputs(**kwargs)

        spo2 = float(kwargs["spo2"])
        fio2 = float(kwargs["fio2"])
        rr = float(kwargs["respiratory_rate"])

        rox = (spo2 / fio2) / rr
        rox = round(rox, 4)

        components: Dict[str, float] = {
            "spo2": spo2,
            "fio2": fio2,
            "respiratory_rate": rr,
            "spo2_fio2_ratio": round(spo2 / fio2, 4),
        }

        interpretation = self.interpret(rox)
        risk = "hfnc_success" if rox >= self.HFNC_SUCCESS_THRESHOLD else "hfnc_failure_risk"

        return ScoringResult(
            total_score=rox,
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the theoretical ROX index bounds.

        The lower bound approaches 0 (high FiO2, high RR, low SpO2).
        The upper bound is determined by SpO2=100, FiO2=0.21, RR=1.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 476.19)`` approximate bounds.
        """
        return (0.0, round(100.0 / 0.21 / 1.0, 2))

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the ROX index value.

        Parameters
        ----------
        score : int or float
            ROX index value.

        Returns
        -------
        str
            Clinical interpretation regarding HFNC therapy.
        """
        s = float(score)
        if s >= self.HFNC_SUCCESS_THRESHOLD:
            return (
                f"ROX {s:.2f}: >= {self.HFNC_SUCCESS_THRESHOLD}. "
                "Low risk of HFNC failure. Continue current therapy."
            )
        return (
            f"ROX {s:.2f}: < {self.HFNC_SUCCESS_THRESHOLD}. "
            "Risk of HFNC failure. Consider escalation to "
            "non-invasive or invasive ventilation."
        )
