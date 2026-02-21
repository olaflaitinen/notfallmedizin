# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Trauma scoring systems.

This module implements the Injury Severity Score (ISS), the Revised
Trauma Score (RTS), and the Trauma and Injury Severity Score (TRISS)
as concrete subclasses of :class:`BaseScorer`.

Classes
-------
ISSScore
    Injury Severity Score from Abbreviated Injury Scale (AIS) ratings.
RTSScore
    Revised Trauma Score using coded GCS, SBP, and RR values.
TRISSScore
    Trauma and Injury Severity Score combining RTS, ISS, and age.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
# ISS Score
# ======================================================================


_ISS_BODY_REGIONS = (
    "head_neck",
    "face",
    "chest",
    "abdomen",
    "extremities",
    "external",
)


class ISSScore(BaseScorer):
    """Injury Severity Score (ISS).

    The ISS is the sum of squares of the three highest AIS scores from
    six body regions. If any single AIS equals 6 (unsurvivable), the
    ISS is automatically set to 75.

    Parameters (passed to ``calculate``)
    -------------------------------------
    head_neck : int
        AIS score for head/neck region (0--6).
    face : int
        AIS score for face region (0--6).
    chest : int
        AIS score for chest region (0--6).
    abdomen : int
        AIS score for abdominal/pelvic contents (0--6).
    extremities : int
        AIS score for extremities/pelvic girdle (0--6).
    external : int
        AIS score for external/skin (0--6).

    References
    ----------
    Baker SP, et al. "The Injury Severity Score: a method for
    describing patients with multiple injuries and evaluating
    emergency care." J Trauma. 1974;14(3):187-196.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "Injury Severity Score"
        self._description = (
            "Sum of squares of the 3 highest AIS region scores (0-75). "
            "Any AIS of 6 automatically yields ISS 75."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate AIS scores for each body region.

        Parameters
        ----------
        **kwargs
            Must include all six body region AIS scores.

        Raises
        ------
        ValidationError
            If a required region is missing.
        ClinicalRangeError
            If an AIS score is not in [0, 6].
        """
        for region in _ISS_BODY_REGIONS:
            if region not in kwargs:
                raise ValidationError(
                    message=(
                        f"Missing required AIS score for body region "
                        f"'{region}' in ISS calculation."
                    ),
                    parameter=region,
                )
            val = kwargs[region]
            if not isinstance(val, int) or val < 0 or val > 6:
                raise ClinicalRangeError(region, float(val), 0, 6)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the Injury Severity Score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with ISS (0--75) and per-region scores.
        """
        self.validate_inputs(**kwargs)

        scores: List[int] = [kwargs[region] for region in _ISS_BODY_REGIONS]
        components: Dict[str, float] = {
            region: float(kwargs[region]) for region in _ISS_BODY_REGIONS
        }

        if 6 in scores:
            total = 75
        else:
            top3 = sorted(scores, reverse=True)[:3]
            total = sum(s * s for s in top3)

        interpretation = self.interpret(total)
        risk = self._categorize(total)
        mortality = self._estimate_mortality(total)

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
            mortality_estimate=mortality,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum ISS values.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 75.0)``.
        """
        return (0.0, 75.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the Injury Severity Score.

        Parameters
        ----------
        score : int or float
            ISS value (0--75).

        Returns
        -------
        str
            Severity interpretation.
        """
        s = int(score)
        if s == 75:
            return "ISS 75: Unsurvivable injury (AIS 6 present)."
        if s >= 25:
            return f"ISS {s}: Severe injury. Major trauma center care indicated."
        if s >= 16:
            return f"ISS {s}: Moderate-to-severe injury."
        if s >= 9:
            return f"ISS {s}: Moderate injury."
        return f"ISS {s}: Minor injury."

    @staticmethod
    def _categorize(score: int) -> str:
        """Map ISS to a severity category.

        Parameters
        ----------
        score : int
            ISS value.

        Returns
        -------
        str
            Severity category.
        """
        if score >= 25:
            return "severe"
        if score >= 16:
            return "moderate_to_severe"
        if score >= 9:
            return "moderate"
        return "minor"

    @staticmethod
    def _estimate_mortality(score: int) -> float:
        """Provide a rough mortality estimate based on ISS.

        Parameters
        ----------
        score : int
            ISS value.

        Returns
        -------
        float
            Approximate mortality proportion.
        """
        if score == 75:
            return 1.0
        if score >= 25:
            return 0.50
        if score >= 16:
            return 0.10
        if score >= 9:
            return 0.04
        return 0.01


# ======================================================================
# RTS Score
# ======================================================================


class RTSScore(BaseScorer):
    """Revised Trauma Score (RTS).

    The RTS is a weighted sum of coded values for GCS, systolic blood
    pressure, and respiratory rate:

        RTS = 0.9368 * GCS_coded + 0.7326 * SBP_coded + 0.2908 * RR_coded

    Coded values range from 0--4 based on physiological thresholds.

    Parameters (passed to ``calculate``)
    -------------------------------------
    gcs : int
        Glasgow Coma Scale total (3--15).
    systolic_bp : float
        Systolic blood pressure in mmHg.
    respiratory_rate : float
        Respiratory rate in breaths per minute.

    References
    ----------
    Champion HR, et al. "A revision of the Trauma Score."
    J Trauma. 1989;29(5):623-629.
    """

    GCS_WEIGHT: float = 0.9368
    SBP_WEIGHT: float = 0.7326
    RR_WEIGHT: float = 0.2908

    def __init__(self) -> None:
        super().__init__()
        self._name = "Revised Trauma Score"
        self._description = (
            "Weighted sum of coded GCS, systolic BP, and respiratory "
            "rate values for trauma triage (0-7.8408)."
        )

    @staticmethod
    def _code_gcs(gcs: int) -> int:
        """Convert a raw GCS total into a coded value (0--4).

        Parameters
        ----------
        gcs : int
            GCS score (3--15).

        Returns
        -------
        int
            Coded GCS value.
        """
        if gcs >= 13:
            return 4
        if gcs >= 9:
            return 3
        if gcs >= 6:
            return 2
        if gcs >= 4:
            return 1
        return 0

    @staticmethod
    def _code_sbp(sbp: float) -> int:
        """Convert systolic BP into a coded value (0--4).

        Parameters
        ----------
        sbp : float
            Systolic blood pressure in mmHg.

        Returns
        -------
        int
            Coded SBP value.
        """
        if sbp > 89:
            return 4
        if sbp >= 76:
            return 3
        if sbp >= 50:
            return 2
        if sbp >= 1:
            return 1
        return 0

    @staticmethod
    def _code_rr(rr: float) -> int:
        """Convert respiratory rate into a coded value (0--4).

        Parameters
        ----------
        rr : float
            Respiratory rate in breaths per minute.

        Returns
        -------
        int
            Coded RR value.
        """
        if 10 <= rr <= 29:
            return 4
        if rr > 29:
            return 3
        if 6 <= rr <= 9:
            return 2
        if 1 <= rr <= 5:
            return 1
        return 0

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for RTS calculation.

        Parameters
        ----------
        **kwargs
            Must include ``gcs``, ``systolic_bp``, and
            ``respiratory_rate``.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a value is out of range.
        """
        required = ["gcs", "systolic_bp", "respiratory_rate"]
        for param in required:
            if param not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{param}' for RTS.",
                    parameter=param,
                )

        gcs = kwargs["gcs"]
        if not isinstance(gcs, int) or gcs < 3 or gcs > 15:
            raise ClinicalRangeError("gcs", float(gcs), 3, 15)

        sbp = kwargs["systolic_bp"]
        if not isinstance(sbp, (int, float)) or sbp < 0 or sbp > 400:
            raise ClinicalRangeError("systolic_bp", float(sbp), 0, 400)

        rr = kwargs["respiratory_rate"]
        if not isinstance(rr, (int, float)) or rr < 0 or rr > 80:
            raise ClinicalRangeError("respiratory_rate", float(rr), 0, 80)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the Revised Trauma Score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with RTS value and coded component breakdown.
        """
        self.validate_inputs(**kwargs)

        gcs_c = self._code_gcs(kwargs["gcs"])
        sbp_c = self._code_sbp(kwargs["systolic_bp"])
        rr_c = self._code_rr(kwargs["respiratory_rate"])

        rts = (
            self.GCS_WEIGHT * gcs_c
            + self.SBP_WEIGHT * sbp_c
            + self.RR_WEIGHT * rr_c
        )
        rts = round(rts, 4)

        components: Dict[str, float] = {
            "gcs_coded": float(gcs_c),
            "sbp_coded": float(sbp_c),
            "rr_coded": float(rr_c),
        }

        interpretation = self.interpret(rts)
        risk = self._categorize(rts)

        return ScoringResult(
            total_score=rts,
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum RTS values.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 7.8408)``.
        """
        max_rts = self.GCS_WEIGHT * 4 + self.SBP_WEIGHT * 4 + self.RR_WEIGHT * 4
        return (0.0, round(max_rts, 4))

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the Revised Trauma Score.

        Parameters
        ----------
        score : int or float
            RTS value.

        Returns
        -------
        str
            Clinical interpretation.
        """
        s = float(score)
        if s >= 7.0:
            return f"RTS {s:.2f}: Good prognosis. Survival probability >90%."
        if s >= 4.0:
            return f"RTS {s:.2f}: Moderate injury. Transport to trauma center advised."
        return f"RTS {s:.2f}: Severe injury. High mortality risk."

    @staticmethod
    def _categorize(rts: float) -> str:
        """Map RTS to a severity category.

        Parameters
        ----------
        rts : float
            RTS value.

        Returns
        -------
        str
            Severity category.
        """
        if rts >= 7.0:
            return "minor"
        if rts >= 4.0:
            return "moderate"
        return "severe"


# ======================================================================
# TRISS Score
# ======================================================================


class TRISSScore(BaseScorer):
    """Trauma and Injury Severity Score (TRISS).

    Combines RTS, ISS, and patient age to estimate the probability
    of survival (Ps) using logistic regression coefficients:

        b = b0 + b1*RTS + b2*ISS + b3*age_index
        Ps = 1 / (1 + e^(-b))

    ``age_index`` is 0 if age < 55, else 1. Separate coefficient sets
    are used for blunt and penetrating trauma.

    Parameters (passed to ``calculate``)
    -------------------------------------
    rts : float
        Revised Trauma Score.
    iss : int
        Injury Severity Score.
    age : int
        Patient age in years.
    mechanism : str
        ``"blunt"`` or ``"penetrating"``.

    References
    ----------
    Boyd CR, Tolson MA, Copes WS. "Evaluating trauma care: the TRISS
    method." J Trauma. 1987;27(4):370-378.
    """

    _BLUNT_COEFFICIENTS = (-0.4499, 0.8085, -0.0835, -1.7430)
    _PENETRATING_COEFFICIENTS = (-2.5355, 0.9934, -0.0651, -1.1360)

    def __init__(self) -> None:
        super().__init__()
        self._name = "TRISS Score"
        self._description = (
            "Trauma and Injury Severity Score combining RTS, ISS, and "
            "age to estimate survival probability (0-1)."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for TRISS calculation.

        Parameters
        ----------
        **kwargs
            Must include ``rts``, ``iss``, ``age``, and ``mechanism``.

        Raises
        ------
        ValidationError
            If a required parameter is missing or invalid.
        ClinicalRangeError
            If numeric values are out of range.
        """
        required = ["rts", "iss", "age", "mechanism"]
        for param in required:
            if param not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{param}' for TRISS.",
                    parameter=param,
                )

        rts = kwargs["rts"]
        if not isinstance(rts, (int, float)) or rts < 0 or rts > 8:
            raise ClinicalRangeError("rts", float(rts), 0, 8)

        iss = kwargs["iss"]
        if not isinstance(iss, int) or iss < 0 or iss > 75:
            raise ClinicalRangeError("iss", float(iss), 0, 75)

        age = kwargs["age"]
        if not isinstance(age, int) or age < 0 or age > 150:
            raise ClinicalRangeError("age", float(age), 0, 150)

        mechanism = kwargs["mechanism"]
        if mechanism not in ("blunt", "penetrating"):
            raise ValidationError(
                message=(
                    f"Parameter 'mechanism' must be 'blunt' or "
                    f"'penetrating', got '{mechanism}'."
                ),
                parameter="mechanism",
            )

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the TRISS survival probability.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result with survival probability (0--1) as total_score.
        """
        self.validate_inputs(**kwargs)

        rts = float(kwargs["rts"])
        iss = int(kwargs["iss"])
        age = int(kwargs["age"])
        mechanism: str = kwargs["mechanism"]

        age_index = 0 if age < 55 else 1

        if mechanism == "blunt":
            b0, b1, b2, b3 = self._BLUNT_COEFFICIENTS
        else:
            b0, b1, b2, b3 = self._PENETRATING_COEFFICIENTS

        b = b0 + b1 * rts + b2 * iss + b3 * age_index
        ps = 1.0 / (1.0 + math.exp(-b))

        components: Dict[str, float] = {
            "rts": rts,
            "iss": float(iss),
            "age_index": float(age_index),
            "b_coefficient": round(b, 4),
        }

        interpretation = self.interpret(ps)
        mortality = round(1.0 - ps, 4)
        risk = self._categorize(ps)

        return ScoringResult(
            total_score=round(ps, 4),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
            mortality_estimate=mortality,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the range of TRISS survival probability.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 1.0)`` representing probability bounds.
        """
        return (0.0, 1.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the TRISS survival probability.

        Parameters
        ----------
        score : int or float
            Probability of survival (0--1).

        Returns
        -------
        str
            Interpretation text.
        """
        ps = float(score)
        pct = round(ps * 100, 1)
        if ps >= 0.9:
            return f"TRISS Ps={pct}%: Good survival prognosis."
        if ps >= 0.5:
            return f"TRISS Ps={pct}%: Guarded prognosis."
        if ps >= 0.25:
            return f"TRISS Ps={pct}%: Poor prognosis. Aggressive care warranted."
        return f"TRISS Ps={pct}%: Very poor prognosis."

    @staticmethod
    def _categorize(ps: float) -> str:
        """Map survival probability to a risk category.

        Parameters
        ----------
        ps : float
            Probability of survival.

        Returns
        -------
        str
            Risk category (from the patient's perspective).
        """
        if ps >= 0.9:
            return "low_risk"
        if ps >= 0.5:
            return "moderate_risk"
        if ps >= 0.25:
            return "high_risk"
        return "very_high_risk"
