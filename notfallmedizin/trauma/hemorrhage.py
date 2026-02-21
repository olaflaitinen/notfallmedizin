# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Hemorrhage classification and massive transfusion protocol.

References
----------
American College of Surgeons. (2018). ATLS Student Course Manual (10th ed.).
Nunez, T. C., et al. (2009). Early prediction of massive transfusion in
    trauma: simple as ABC. The Journal of Trauma, 66(2), 346-352.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class HemorrhageClass:
    """ATLS hemorrhage classification result.

    Attributes
    ----------
    class_level : int
        1 through 4.
    estimated_blood_loss_ml : float
    estimated_blood_loss_percent : float
    blood_volume_ml : float
    transfusion_needed : bool
    fluid_recommendation : str
    """

    class_level: int
    estimated_blood_loss_ml: float
    estimated_blood_loss_percent: float
    blood_volume_ml: float
    transfusion_needed: bool
    fluid_recommendation: str


@dataclass(frozen=True)
class MTPResult:
    """Massive transfusion protocol assessment result.

    Attributes
    ----------
    activate_mtp : bool
    abc_score : int
    component_scores : dict
    recommended_ratio : str
    """

    activate_mtp: bool
    abc_score: int
    component_scores: Dict[str, int] = field(default_factory=dict)
    recommended_ratio: str = ""


@dataclass(frozen=True)
class ShockIndexResult:
    """Shock index calculation result.

    Attributes
    ----------
    shock_index : float
    classification : str
    heart_rate : float
    systolic_bp : float
    """

    shock_index: float
    classification: str
    heart_rate: float
    systolic_bp: float


class HemorrhageClassifier:
    """ATLS hemorrhage classification (Classes I through IV).

    Estimated blood volume is calculated as 70 mL/kg for adults. Classes
    are determined by percentage of blood volume lost.
    """

    @staticmethod
    def estimate_blood_volume(
        weight_kg: float,
        age_years: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> float:
        """Estimate total blood volume.

        Parameters
        ----------
        weight_kg : float
        age_years : int, optional
        sex : str, optional
            'male' or 'female'.

        Returns
        -------
        float
            Estimated blood volume in mL.
        """
        if weight_kg <= 0:
            raise ValidationError("Weight must be positive.")
        ml_per_kg = 70.0
        if sex == "female":
            ml_per_kg = 65.0
        if age_years is not None and age_years < 1:
            ml_per_kg = 80.0
        return weight_kg * ml_per_kg

    def classify(
        self,
        estimated_blood_loss_ml: float,
        weight_kg: float,
        heart_rate: Optional[float] = None,
        systolic_bp: Optional[float] = None,
        sex: Optional[str] = None,
    ) -> HemorrhageClass:
        """Classify hemorrhage per ATLS guidelines.

        Parameters
        ----------
        estimated_blood_loss_ml : float
        weight_kg : float
        heart_rate : float, optional
        systolic_bp : float, optional
        sex : str, optional

        Returns
        -------
        HemorrhageClass
        """
        bv = self.estimate_blood_volume(weight_kg, sex=sex)
        pct = (estimated_blood_loss_ml / bv) * 100.0

        if pct < 15:
            level, transfuse, fluid = 1, False, "Crystalloid (optional)"
        elif pct < 30:
            level, transfuse, fluid = 2, False, "Crystalloid resuscitation"
        elif pct < 40:
            level, transfuse, fluid = 3, True, "Crystalloid + blood products"
        else:
            level, transfuse, fluid = 4, True, "Massive transfusion protocol"

        return HemorrhageClass(
            class_level=level,
            estimated_blood_loss_ml=round(estimated_blood_loss_ml, 0),
            estimated_blood_loss_percent=round(pct, 1),
            blood_volume_ml=round(bv, 0),
            transfusion_needed=transfuse,
            fluid_recommendation=fluid,
        )


class MassiveTransfusionProtocol:
    """Assessment for massive transfusion protocol activation.

    Uses the ABC (Assessment of Blood Consumption) score.

    ABC Score Components (Nunez et al., 2009)
    ------------------------------------------
    Penetrating mechanism:  1 point
    SBP <= 90 mmHg:         1 point
    HR >= 120 bpm:           1 point
    Positive FAST:          1 point

    Score >= 2 triggers MTP activation.
    """

    def assess_need(
        self,
        penetrating_mechanism: bool,
        systolic_bp_le_90: bool,
        heart_rate_ge_120: bool,
        positive_fast: bool,
    ) -> MTPResult:
        """Assess the need for massive transfusion protocol.

        Parameters
        ----------
        penetrating_mechanism : bool
        systolic_bp_le_90 : bool
        heart_rate_ge_120 : bool
        positive_fast : bool

        Returns
        -------
        MTPResult
        """
        components = {
            "penetrating_mechanism": int(penetrating_mechanism),
            "systolic_bp_le_90": int(systolic_bp_le_90),
            "heart_rate_ge_120": int(heart_rate_ge_120),
            "positive_fast": int(positive_fast),
        }
        total = sum(components.values())
        activate = total >= 2

        return MTPResult(
            activate_mtp=activate,
            abc_score=total,
            component_scores=components,
            recommended_ratio="1:1:1 (pRBC : FFP : Platelets)" if activate else "",
        )


class ShockIndexCalculator:
    """Shock Index (SI) = Heart Rate / Systolic Blood Pressure.

    Normal SI is approximately 0.5 to 0.7 in healthy adults.
    """

    @staticmethod
    def calculate(heart_rate: float, systolic_bp: float) -> ShockIndexResult:
        """Calculate the shock index.

        Parameters
        ----------
        heart_rate : float
        systolic_bp : float

        Returns
        -------
        ShockIndexResult
        """
        if systolic_bp <= 0:
            raise ValidationError("Systolic BP must be positive.")

        si = heart_rate / systolic_bp

        if si < 0.7:
            classification = "normal"
        elif si < 1.0:
            classification = "elevated"
        elif si < 1.5:
            classification = "concerning"
        else:
            classification = "critical"

        return ShockIndexResult(
            shock_index=round(si, 3),
            classification=classification,
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
        )
