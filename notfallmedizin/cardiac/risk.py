# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Cardiac risk calculators.

Implements the Framingham 10-year cardiovascular risk score, Wells
criteria for pulmonary embolism, and Wells criteria for deep vein
thrombosis.

References
----------
D'Agostino, R. B., et al. (2008). General cardiovascular risk profile for
    use in primary care. Circulation, 117(6), 743-753.
Wells, P. S., et al. (2000). Derivation of a simple clinical model to
    categorize patients' probability of pulmonary embolism. Thrombosis
    and Haemostasis, 83(3), 416-420.
Wells, P. S., et al. (2003). Evaluation of D-dimer in the diagnosis of
    suspected deep-vein thrombosis. NEJM, 349(13), 1227-1235.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class FraminghamResult:
    """Framingham 10-year risk result.

    Attributes
    ----------
    risk_percent : float
    risk_category : str
    points : int
    recommended_interventions : list of str
    """

    risk_percent: float
    risk_category: str
    points: int
    recommended_interventions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class WellsResult:
    """Wells score result.

    Attributes
    ----------
    total_score : float
    probability_category : str
    recommended_next_step : str
    components : dict
    """

    total_score: float
    probability_category: str
    recommended_next_step: str
    components: Dict[str, float] = field(default_factory=dict)


_MALE_AGE_POINTS = {
    (20, 34): -9, (35, 39): -4, (40, 44): 0, (45, 49): 3,
    (50, 54): 6, (55, 59): 8, (60, 64): 10, (65, 69): 11,
    (70, 74): 12, (75, 79): 13,
}

_FEMALE_AGE_POINTS = {
    (20, 34): -7, (35, 39): -3, (40, 44): 0, (45, 49): 3,
    (50, 54): 6, (55, 59): 8, (60, 64): 10, (65, 69): 12,
    (70, 74): 14, (75, 79): 16,
}

_MALE_CHOL_POINTS = {
    (20, 39): {160: 0, 200: 4, 240: 7, 280: 9, 999: 11},
    (40, 49): {160: 0, 200: 3, 240: 5, 280: 6, 999: 8},
    (50, 59): {160: 0, 200: 2, 240: 3, 280: 4, 999: 5},
    (60, 69): {160: 0, 200: 1, 240: 1, 280: 2, 999: 3},
    (70, 79): {160: 0, 200: 0, 240: 0, 280: 1, 999: 1},
}

_MALE_RISK_TABLE = {
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 4,
    9: 5, 10: 6, 11: 8, 12: 10, 13: 12, 14: 16, 15: 20, 16: 25,
}

_FEMALE_RISK_TABLE = {
    9: 1, 10: 1, 11: 1, 12: 1, 13: 2, 14: 2, 15: 3, 16: 4,
    17: 5, 18: 6, 19: 8, 20: 11, 21: 14, 22: 17, 23: 22, 24: 27,
}


class FraminghamRiskCalculator:
    """Framingham 10-year cardiovascular risk score.

    Uses the point-based system from D'Agostino et al. (2008).
    """

    def calculate(
        self,
        age: int,
        sex: str,
        total_cholesterol: float,
        hdl: float,
        systolic_bp: float,
        bp_treated: bool,
        smoker: bool,
        diabetic: bool,
    ) -> FraminghamResult:
        """Compute 10-year cardiovascular risk.

        Parameters
        ----------
        age : int
        sex : str
            'male' or 'female'.
        total_cholesterol : float
            mg/dL.
        hdl : float
            mg/dL.
        systolic_bp : float
            mmHg.
        bp_treated : bool
        smoker : bool
        diabetic : bool

        Returns
        -------
        FraminghamResult
        """
        if sex not in ("male", "female"):
            raise ValidationError("sex must be 'male' or 'female'.")
        if not 20 <= age <= 79:
            raise ValidationError("Age must be between 20 and 79.")

        points = 0
        age_table = _MALE_AGE_POINTS if sex == "male" else _FEMALE_AGE_POINTS
        for (lo, hi), pts in age_table.items():
            if lo <= age <= hi:
                points += pts
                break

        chol_table = _MALE_CHOL_POINTS
        for (alo, ahi), brackets in chol_table.items():
            if alo <= age <= ahi:
                for upper, pts in sorted(brackets.items()):
                    if total_cholesterol < upper:
                        points += pts
                        break
                break

        if sex == "male":
            if hdl >= 60:
                points -= 1
            elif 40 <= hdl < 50:
                points += 1
            elif hdl < 40:
                points += 2
        else:
            if hdl >= 60:
                points -= 1
            elif 40 <= hdl < 50:
                points += 1
            elif hdl < 40:
                points += 2

        if bp_treated:
            if systolic_bp >= 160:
                points += 3 if sex == "male" else 4
            elif systolic_bp >= 140:
                points += 2 if sex == "male" else 3
            elif systolic_bp >= 130:
                points += 1 if sex == "male" else 2
            elif systolic_bp >= 120:
                points += 1
        else:
            if systolic_bp >= 160:
                points += 2 if sex == "male" else 3
            elif systolic_bp >= 140:
                points += 1 if sex == "male" else 2
            elif systolic_bp >= 130:
                points += 1

        if smoker:
            points += 2

        if diabetic:
            points += 2 if sex == "male" else 4

        risk_table = _MALE_RISK_TABLE if sex == "male" else _FEMALE_RISK_TABLE
        risk_keys = sorted(risk_table.keys())
        risk_pct = risk_table.get(points, 0)
        if points > risk_keys[-1]:
            risk_pct = 30
        elif points < risk_keys[0]:
            risk_pct = risk_table[risk_keys[0]]

        if risk_pct < 10:
            category = "low"
            interventions = ["Lifestyle modifications", "Periodic reassessment"]
        elif risk_pct < 20:
            category = "moderate"
            interventions = [
                "Lifestyle modifications",
                "Consider statin therapy",
                "Blood pressure optimization",
            ]
        else:
            category = "high"
            interventions = [
                "Statin therapy recommended",
                "Aggressive BP control",
                "Aspirin therapy per guidelines",
                "Diabetes management if applicable",
            ]

        return FraminghamResult(
            risk_percent=float(risk_pct),
            risk_category=category,
            points=points,
            recommended_interventions=interventions,
        )


class WellsPEScore:
    """Wells criteria for pulmonary embolism.

    Point System (Wells et al., 2000)
    ----------------------------------
    Clinical signs/symptoms of DVT: 3.0
    PE is most likely diagnosis:    3.0
    Heart rate > 100:               1.5
    Immobilization/surgery:         1.5
    Previous DVT/PE:                1.5
    Hemoptysis:                     1.0
    Active malignancy:              1.0
    """

    def calculate(
        self,
        clinical_signs_dvt: bool = False,
        pe_most_likely: bool = False,
        heart_rate_gt_100: bool = False,
        immobilization_surgery: bool = False,
        previous_dvt_pe: bool = False,
        hemoptysis: bool = False,
        malignancy: bool = False,
    ) -> WellsResult:
        """Calculate the Wells PE score.

        Returns
        -------
        WellsResult
        """
        components = {
            "clinical_signs_dvt": 3.0 if clinical_signs_dvt else 0.0,
            "pe_most_likely": 3.0 if pe_most_likely else 0.0,
            "heart_rate_gt_100": 1.5 if heart_rate_gt_100 else 0.0,
            "immobilization_surgery": 1.5 if immobilization_surgery else 0.0,
            "previous_dvt_pe": 1.5 if previous_dvt_pe else 0.0,
            "hemoptysis": 1.0 if hemoptysis else 0.0,
            "malignancy": 1.0 if malignancy else 0.0,
        }
        total = sum(components.values())

        if total <= 4.0:
            category = "PE unlikely"
            step = "D-dimer testing; if negative, PE excluded"
        else:
            category = "PE likely"
            step = "CTPA (CT pulmonary angiography) recommended"

        return WellsResult(
            total_score=total,
            probability_category=category,
            recommended_next_step=step,
            components=components,
        )


class WellsDVTScore:
    """Wells criteria for deep vein thrombosis.

    Point System (Wells et al., 2003)
    ----------------------------------
    Active cancer:                1
    Paralysis/paresis/cast:       1
    Bedridden > 3 days or surgery: 1
    Localized tenderness:         1
    Entire leg swelling:          1
    Calf swelling > 3 cm:        1
    Pitting edema:                1
    Collateral superficial veins: 1
    Alternative diagnosis likely: -2
    Previously documented DVT:    1
    """

    def calculate(
        self,
        active_cancer: bool = False,
        paralysis_paresis_cast: bool = False,
        bedridden_or_surgery: bool = False,
        localized_tenderness: bool = False,
        entire_leg_swelling: bool = False,
        calf_swelling_gt_3cm: bool = False,
        pitting_edema: bool = False,
        collateral_veins: bool = False,
        alternative_diagnosis_likely: bool = False,
        previous_dvt: bool = False,
    ) -> WellsResult:
        """Calculate the Wells DVT score.

        Returns
        -------
        WellsResult
        """
        components = {
            "active_cancer": 1.0 if active_cancer else 0.0,
            "paralysis_paresis_cast": 1.0 if paralysis_paresis_cast else 0.0,
            "bedridden_or_surgery": 1.0 if bedridden_or_surgery else 0.0,
            "localized_tenderness": 1.0 if localized_tenderness else 0.0,
            "entire_leg_swelling": 1.0 if entire_leg_swelling else 0.0,
            "calf_swelling_gt_3cm": 1.0 if calf_swelling_gt_3cm else 0.0,
            "pitting_edema": 1.0 if pitting_edema else 0.0,
            "collateral_veins": 1.0 if collateral_veins else 0.0,
            "alternative_diagnosis_likely": -2.0 if alternative_diagnosis_likely else 0.0,
            "previous_dvt": 1.0 if previous_dvt else 0.0,
        }
        total = sum(components.values())

        if total <= 0:
            category = "DVT unlikely"
            step = "D-dimer testing; if negative, DVT excluded"
        elif total <= 2:
            category = "DVT moderately likely"
            step = "D-dimer and/or ultrasound"
        else:
            category = "DVT likely"
            step = "Compression ultrasound recommended"

        return WellsResult(
            total_score=total,
            probability_category=category,
            recommended_next_step=step,
            components=components,
        )
