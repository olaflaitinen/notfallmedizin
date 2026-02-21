# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Sepsis-related clinical scoring systems.

This module implements the Sequential Organ Failure Assessment (SOFA),
Quick SOFA (qSOFA), and Systemic Inflammatory Response Syndrome (SIRS)
criteria as concrete subclasses of :class:`BaseScorer`.

Classes
-------
SOFAScore
    Sequential Organ Failure Assessment, scoring 6 organ systems (0--24).
qSOFAScore
    Quick SOFA bedside screen with 3 binary criteria (0--3).
SIRSCriteria
    Systemic Inflammatory Response Syndrome with 4 criteria (0--4).
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
# SOFA Score
# ======================================================================


class SOFAScore(BaseScorer):
    """Sequential Organ Failure Assessment (SOFA) score.

    Evaluates dysfunction across six organ systems, each scored 0--4,
    yielding a total range of 0--24.

    Parameters (passed to ``calculate``)
    -------------------------------------
    pao2_fio2_ratio : float
        PaO2/FiO2 ratio in mmHg.
    mechanical_ventilation : bool
        Whether the patient is on mechanical ventilation.
    platelets : float
        Platelet count in 10^3/uL.
    bilirubin : float
        Serum bilirubin in mg/dL.
    map_value : float
        Mean arterial pressure in mmHg.
    dopamine_dose : float
        Dopamine infusion rate in mcg/kg/min (0 if not administered).
    dobutamine_dose : float
        Dobutamine infusion rate in mcg/kg/min (0 if not administered).
    epinephrine_dose : float
        Epinephrine infusion rate in mcg/kg/min (0 if not administered).
    norepinephrine_dose : float
        Norepinephrine infusion rate in mcg/kg/min (0 if not administered).
    gcs : int
        Glasgow Coma Scale score (3--15).
    creatinine : float
        Serum creatinine in mg/dL.
    urine_output : float or None
        Urine output in mL/day. If ``None``, only creatinine is used.

    References
    ----------
    Vincent JL, et al. "The SOFA (Sepsis-related Organ Failure Assessment)
    score to describe organ dysfunction/failure." Intensive Care Med.
    1996;22(7):707-710.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "SOFA Score"
        self._description = (
            "Sequential Organ Failure Assessment scoring six organ "
            "systems (respiration, coagulation, liver, cardiovascular, "
            "CNS, renal) each 0-4 for a total of 0-24."
        )

    # ------------------------------------------------------------------
    # Component scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_respiration(
        pao2_fio2_ratio: float,
        mechanical_ventilation: bool,
    ) -> int:
        """Score the respiratory component (0--4).

        Parameters
        ----------
        pao2_fio2_ratio : float
            PaO2/FiO2 ratio in mmHg.
        mechanical_ventilation : bool
            Whether the patient is mechanically ventilated.

        Returns
        -------
        int
            Respiratory sub-score.
        """
        if pao2_fio2_ratio < 100 and mechanical_ventilation:
            return 4
        if pao2_fio2_ratio < 200 and mechanical_ventilation:
            return 3
        if pao2_fio2_ratio < 300:
            return 2
        if pao2_fio2_ratio < 400:
            return 1
        return 0

    @staticmethod
    def _score_coagulation(platelets: float) -> int:
        """Score the coagulation component (0--4).

        Parameters
        ----------
        platelets : float
            Platelet count in 10^3/uL.

        Returns
        -------
        int
            Coagulation sub-score.
        """
        if platelets < 20:
            return 4
        if platelets < 50:
            return 3
        if platelets < 100:
            return 2
        if platelets < 150:
            return 1
        return 0

    @staticmethod
    def _score_liver(bilirubin: float) -> int:
        """Score the hepatic component (0--4).

        Parameters
        ----------
        bilirubin : float
            Serum bilirubin in mg/dL.

        Returns
        -------
        int
            Liver sub-score.
        """
        if bilirubin >= 12.0:
            return 4
        if bilirubin >= 6.0:
            return 3
        if bilirubin >= 2.0:
            return 2
        if bilirubin >= 1.2:
            return 1
        return 0

    @staticmethod
    def _score_cardiovascular(
        map_value: float,
        dopamine_dose: float,
        dobutamine_dose: float,
        epinephrine_dose: float,
        norepinephrine_dose: float,
    ) -> int:
        """Score the cardiovascular component (0--4).

        Parameters
        ----------
        map_value : float
            Mean arterial pressure in mmHg.
        dopamine_dose : float
            Dopamine in mcg/kg/min.
        dobutamine_dose : float
            Dobutamine in mcg/kg/min.
        epinephrine_dose : float
            Epinephrine in mcg/kg/min.
        norepinephrine_dose : float
            Norepinephrine in mcg/kg/min.

        Returns
        -------
        int
            Cardiovascular sub-score.
        """
        if dopamine_dose > 15 or epinephrine_dose > 0.1 or norepinephrine_dose > 0.1:
            return 4
        if dopamine_dose > 5 or epinephrine_dose <= 0.1 and epinephrine_dose > 0:
            return 3
        if norepinephrine_dose <= 0.1 and norepinephrine_dose > 0:
            return 3
        if dopamine_dose > 0 or dobutamine_dose > 0:
            return 2
        if map_value < 70:
            return 1
        return 0

    @staticmethod
    def _score_cns(gcs: int) -> int:
        """Score the central nervous system component (0--4).

        Parameters
        ----------
        gcs : int
            Glasgow Coma Scale total (3--15).

        Returns
        -------
        int
            CNS sub-score.
        """
        if gcs < 6:
            return 4
        if gcs < 10:
            return 3
        if gcs < 13:
            return 2
        if gcs < 15:
            return 1
        return 0

    @staticmethod
    def _score_renal(
        creatinine: float,
        urine_output: Optional[float],
    ) -> int:
        """Score the renal component (0--4).

        Parameters
        ----------
        creatinine : float
            Serum creatinine in mg/dL.
        urine_output : float or None
            Urine output in mL/day, or ``None`` to ignore.

        Returns
        -------
        int
            Renal sub-score.
        """
        score_cr = 0
        if creatinine >= 5.0:
            score_cr = 4
        elif creatinine >= 3.5:
            score_cr = 3
        elif creatinine >= 2.0:
            score_cr = 2
        elif creatinine >= 1.2:
            score_cr = 1

        score_uo = 0
        if urine_output is not None:
            if urine_output < 200:
                score_uo = 4
            elif urine_output < 500:
                score_uo = 3

        return max(score_cr, score_uo)

    # ------------------------------------------------------------------
    # BaseScorer interface
    # ------------------------------------------------------------------

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate clinical parameters for SOFA calculation.

        Parameters
        ----------
        **kwargs
            Must include ``pao2_fio2_ratio``, ``platelets``,
            ``bilirubin``, ``map_value``, ``gcs``, ``creatinine``,
            and ``mechanical_ventilation``.

        Raises
        ------
        ValidationError
            If a required parameter is missing or has an invalid type.
        ClinicalRangeError
            If a value is outside the physiologically plausible range.
        """
        required = [
            "pao2_fio2_ratio",
            "mechanical_ventilation",
            "platelets",
            "bilirubin",
            "map_value",
            "gcs",
            "creatinine",
        ]
        for param in required:
            if param not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{param}' for SOFA score.",
                    parameter=param,
                )

        pf = kwargs["pao2_fio2_ratio"]
        if not isinstance(pf, (int, float)) or pf < 0:
            raise ClinicalRangeError("pao2_fio2_ratio", float(pf), 0, 700)

        platelets = kwargs["platelets"]
        if not isinstance(platelets, (int, float)) or platelets < 0:
            raise ClinicalRangeError("platelets", float(platelets), 0, 2000)

        bilirubin = kwargs["bilirubin"]
        if not isinstance(bilirubin, (int, float)) or bilirubin < 0:
            raise ClinicalRangeError("bilirubin", float(bilirubin), 0, 100)

        map_value = kwargs["map_value"]
        if not isinstance(map_value, (int, float)) or map_value < 0:
            raise ClinicalRangeError("map_value", float(map_value), 0, 300)

        gcs = kwargs["gcs"]
        if not isinstance(gcs, int) or gcs < 3 or gcs > 15:
            raise ClinicalRangeError("gcs", float(gcs), 3, 15)

        creatinine = kwargs["creatinine"]
        if not isinstance(creatinine, (int, float)) or creatinine < 0:
            raise ClinicalRangeError("creatinine", float(creatinine), 0, 50)

        urine_output = kwargs.get("urine_output")
        if urine_output is not None:
            if not isinstance(urine_output, (int, float)) or urine_output < 0:
                raise ClinicalRangeError("urine_output", float(urine_output), 0, 10000)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the SOFA score from clinical parameters.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result containing total score (0--24), component scores,
            interpretation, risk category, and mortality estimate.
        """
        self.validate_inputs(**kwargs)

        resp = self._score_respiration(
            kwargs["pao2_fio2_ratio"],
            kwargs["mechanical_ventilation"],
        )
        coag = self._score_coagulation(kwargs["platelets"])
        liver = self._score_liver(kwargs["bilirubin"])
        cardio = self._score_cardiovascular(
            kwargs["map_value"],
            kwargs.get("dopamine_dose", 0.0),
            kwargs.get("dobutamine_dose", 0.0),
            kwargs.get("epinephrine_dose", 0.0),
            kwargs.get("norepinephrine_dose", 0.0),
        )
        cns = self._score_cns(kwargs["gcs"])
        renal = self._score_renal(
            kwargs["creatinine"],
            kwargs.get("urine_output"),
        )

        total = resp + coag + liver + cardio + cns + renal

        components: Dict[str, float] = {
            "respiration": float(resp),
            "coagulation": float(coag),
            "liver": float(liver),
            "cardiovascular": float(cardio),
            "cns": float(cns),
            "renal": float(renal),
        }

        interpretation = self.interpret(total)
        risk, mortality = self._risk_and_mortality(total)

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
            mortality_estimate=mortality,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum possible SOFA scores.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 24.0)``.
        """
        return (0.0, 24.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Return a clinical interpretation of the SOFA score.

        Parameters
        ----------
        score : int or float
            SOFA score (0--24).

        Returns
        -------
        str
            Textual interpretation with mortality range.
        """
        s = int(score)
        if s <= 6:
            return f"SOFA {s}: Mortality <10%."
        if s <= 9:
            return f"SOFA {s}: Mortality 15-20%."
        if s <= 12:
            return f"SOFA {s}: Mortality 40-50%."
        return f"SOFA {s}: Mortality >80%."

    @staticmethod
    def _risk_and_mortality(score: int) -> Tuple[str, float]:
        """Map a SOFA total to risk category and mortality estimate.

        Parameters
        ----------
        score : int
            SOFA total score.

        Returns
        -------
        tuple of (str, float)
            Risk category label and midpoint mortality estimate.
        """
        if score <= 6:
            return ("low", 0.05)
        if score <= 9:
            return ("moderate", 0.175)
        if score <= 12:
            return ("high", 0.45)
        return ("very high", 0.85)


# ======================================================================
# qSOFA Score
# ======================================================================


class qSOFAScore(BaseScorer):
    """Quick SOFA (qSOFA) bedside screening tool.

    Three binary criteria, each worth one point (total 0--3).
    A score of 2 or more suggests increased risk of poor outcomes
    in patients with suspected infection.

    Parameters (passed to ``calculate``)
    -------------------------------------
    systolic_bp : float
        Systolic blood pressure in mmHg.
    respiratory_rate : float
        Respiratory rate in breaths per minute.
    altered_mentation : bool
        Whether the patient has altered mentation (GCS < 15).

    References
    ----------
    Seymour CW, et al. "Assessment of Clinical Criteria for Sepsis."
    JAMA. 2016;315(8):762-774.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "qSOFA Score"
        self._description = (
            "Quick SOFA bedside screen with 3 binary criteria: "
            "systolic BP <= 100 mmHg, respiratory rate >= 22, "
            "and altered mentation."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for qSOFA calculation.

        Parameters
        ----------
        **kwargs
            Must include ``systolic_bp``, ``respiratory_rate``, and
            ``altered_mentation``.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a numeric value is out of range.
        """
        required = ["systolic_bp", "respiratory_rate", "altered_mentation"]
        for param in required:
            if param not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{param}' for qSOFA.",
                    parameter=param,
                )

        sbp = kwargs["systolic_bp"]
        if not isinstance(sbp, (int, float)) or sbp < 0 or sbp > 400:
            raise ClinicalRangeError("systolic_bp", float(sbp), 0, 400)

        rr = kwargs["respiratory_rate"]
        if not isinstance(rr, (int, float)) or rr < 0 or rr > 80:
            raise ClinicalRangeError("respiratory_rate", float(rr), 0, 80)

        if not isinstance(kwargs["altered_mentation"], bool):
            raise ValidationError(
                message="Parameter 'altered_mentation' must be a boolean.",
                parameter="altered_mentation",
            )

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Compute the qSOFA score.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result containing total score (0--3) and component breakdown.
        """
        self.validate_inputs(**kwargs)

        sbp_point = 1 if kwargs["systolic_bp"] <= 100 else 0
        rr_point = 1 if kwargs["respiratory_rate"] >= 22 else 0
        mentation_point = 1 if kwargs["altered_mentation"] else 0

        total = sbp_point + rr_point + mentation_point

        components: Dict[str, float] = {
            "systolic_bp_le_100": float(sbp_point),
            "respiratory_rate_ge_22": float(rr_point),
            "altered_mentation": float(mentation_point),
        }

        interpretation = self.interpret(total)
        risk = "positive" if total >= 2 else "negative"

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum possible qSOFA scores.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 3.0)``.
        """
        return (0.0, 3.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the qSOFA score.

        Parameters
        ----------
        score : int or float
            qSOFA score (0--3).

        Returns
        -------
        str
            Clinical interpretation.
        """
        s = int(score)
        if s >= 2:
            return (
                f"qSOFA {s}: Positive. High risk of poor outcome in "
                "suspected infection. Further assessment recommended."
            )
        return f"qSOFA {s}: Negative. Low risk by bedside screening."


# ======================================================================
# SIRS Criteria
# ======================================================================


class SIRSCriteria(BaseScorer):
    """Systemic Inflammatory Response Syndrome (SIRS) criteria.

    Four binary criteria, each worth one point (total 0--4).
    SIRS is present when 2 or more criteria are met.

    Parameters (passed to ``calculate``)
    -------------------------------------
    temperature : float
        Body temperature in degrees Celsius.
    heart_rate : float
        Heart rate in beats per minute.
    respiratory_rate : float
        Respiratory rate in breaths per minute.
    paco2 : float or None
        Arterial CO2 partial pressure in mmHg. If provided and < 32,
        the respiratory criterion is satisfied regardless of rate.
    wbc : float
        White blood cell count in cells/uL (e.g. 12000).
    band_percentage : float
        Percentage of immature band forms (0--100).

    References
    ----------
    Bone RC, et al. "Definitions for sepsis and organ failure and
    guidelines for the use of innovative therapies in sepsis."
    Chest. 1992;101(6):1644-1655.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "SIRS Criteria"
        self._description = (
            "Systemic Inflammatory Response Syndrome evaluation with "
            "4 criteria: temperature, heart rate, respiratory "
            "rate/PaCO2, and white blood cell count."
        )

    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate inputs for SIRS evaluation.

        Parameters
        ----------
        **kwargs
            Must include ``temperature``, ``heart_rate``,
            ``respiratory_rate``, and ``wbc``.

        Raises
        ------
        ValidationError
            If a required parameter is missing.
        ClinicalRangeError
            If a numeric value is out of range.
        """
        required = ["temperature", "heart_rate", "respiratory_rate", "wbc"]
        for param in required:
            if param not in kwargs:
                raise ValidationError(
                    message=f"Missing required parameter '{param}' for SIRS.",
                    parameter=param,
                )

        temp = kwargs["temperature"]
        if not isinstance(temp, (int, float)) or temp < 20 or temp > 45:
            raise ClinicalRangeError("temperature", float(temp), 20, 45)

        hr = kwargs["heart_rate"]
        if not isinstance(hr, (int, float)) or hr < 0 or hr > 300:
            raise ClinicalRangeError("heart_rate", float(hr), 0, 300)

        rr = kwargs["respiratory_rate"]
        if not isinstance(rr, (int, float)) or rr < 0 or rr > 80:
            raise ClinicalRangeError("respiratory_rate", float(rr), 0, 80)

        wbc = kwargs["wbc"]
        if not isinstance(wbc, (int, float)) or wbc < 0:
            raise ClinicalRangeError("wbc", float(wbc), 0, 200000)

        paco2 = kwargs.get("paco2")
        if paco2 is not None:
            if not isinstance(paco2, (int, float)) or paco2 < 0 or paco2 > 150:
                raise ClinicalRangeError("paco2", float(paco2), 0, 150)

        band_pct = kwargs.get("band_percentage", 0.0)
        if not isinstance(band_pct, (int, float)) or band_pct < 0 or band_pct > 100:
            raise ClinicalRangeError("band_percentage", float(band_pct), 0, 100)

    def calculate(self, **kwargs: Any) -> ScoringResult:
        """Evaluate SIRS criteria.

        Parameters
        ----------
        **kwargs
            See class-level docstring for accepted parameters.

        Returns
        -------
        ScoringResult
            Result containing number of criteria met (0--4).
        """
        self.validate_inputs(**kwargs)

        temp = kwargs["temperature"]
        hr = kwargs["heart_rate"]
        rr = kwargs["respiratory_rate"]
        paco2 = kwargs.get("paco2")
        wbc = kwargs["wbc"]
        band_pct = kwargs.get("band_percentage", 0.0)

        temp_met = 1 if (temp > 38.0 or temp < 36.0) else 0
        hr_met = 1 if hr > 90 else 0

        resp_met = 0
        if rr > 20:
            resp_met = 1
        if paco2 is not None and paco2 < 32:
            resp_met = 1

        wbc_met = 0
        if wbc > 12000 or wbc < 4000 or band_pct > 10:
            wbc_met = 1

        total = temp_met + hr_met + resp_met + wbc_met

        components: Dict[str, float] = {
            "temperature": float(temp_met),
            "heart_rate": float(hr_met),
            "respiratory": float(resp_met),
            "wbc": float(wbc_met),
        }

        interpretation = self.interpret(total)
        risk = "sirs_positive" if total >= 2 else "sirs_negative"

        return ScoringResult(
            total_score=float(total),
            component_scores=components,
            interpretation=interpretation,
            risk_category=risk,
        )

    def get_score_range(self) -> Tuple[float, float]:
        """Return the minimum and maximum number of SIRS criteria.

        Returns
        -------
        tuple of (float, float)
            ``(0.0, 4.0)``.
        """
        return (0.0, 4.0)

    def interpret(self, score: Union[int, float]) -> str:
        """Interpret the SIRS criteria count.

        Parameters
        ----------
        score : int or float
            Number of SIRS criteria met (0--4).

        Returns
        -------
        str
            Clinical interpretation.
        """
        s = int(score)
        if s >= 2:
            return (
                f"SIRS criteria met: {s}/4. Systemic inflammatory "
                "response is present."
            )
        return f"SIRS criteria met: {s}/4. SIRS not present."
