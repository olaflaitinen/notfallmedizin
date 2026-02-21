# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Burn assessment, TBSA calculation, and fluid resuscitation.

References
----------
Baxter, C. R., & Shires, T. (1968). Physiological response to
    crystalloid resuscitation of severe burns. Annals of the New York
    Academy of Sciences, 150(3), 874-894.
Lund, C. C., & Browder, N. C. (1944). The estimation of areas of burns.
    Surgery, Gynecology & Obstetrics, 79, 352-358.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from notfallmedizin.core.exceptions import ValidationError


class BurnDepth(Enum):
    """Classification of burn depth."""

    SUPERFICIAL = "superficial"
    PARTIAL_SUPERFICIAL = "superficial_partial_thickness"
    PARTIAL_DEEP = "deep_partial_thickness"
    FULL_THICKNESS = "full_thickness"
    FOURTH_DEGREE = "fourth_degree"


@dataclass(frozen=True)
class TBSAResult:
    """Total body surface area burn result.

    Attributes
    ----------
    total_tbsa_percent : float
    regions : dict
    classification : str
        minor / moderate / major
    """

    total_tbsa_percent: float
    regions: Dict[str, float] = field(default_factory=dict)
    classification: str = ""


@dataclass(frozen=True)
class ResuscitationResult:
    """Parkland formula fluid resuscitation result.

    Attributes
    ----------
    total_volume_ml : float
        Total crystalloid volume for 24 hours.
    first_8hr_rate_ml_hr : float
    next_16hr_rate_ml_hr : float
    urine_output_target_ml_hr : float
    """

    total_volume_ml: float
    first_8hr_rate_ml_hr: float
    next_16hr_rate_ml_hr: float
    urine_output_target_ml_hr: float


_RULE_OF_NINES_ADULT: Dict[str, float] = {
    "head": 9.0,
    "neck": 1.0,
    "anterior_trunk": 18.0,
    "posterior_trunk": 18.0,
    "right_arm": 9.0,
    "left_arm": 9.0,
    "right_leg": 18.0,
    "left_leg": 18.0,
    "perineum": 1.0,
}

_LUND_BROWDER_BASE: Dict[str, float] = {
    "head": 7.0,
    "neck": 2.0,
    "anterior_trunk": 13.0,
    "posterior_trunk": 13.0,
    "right_upper_arm": 4.0,
    "left_upper_arm": 4.0,
    "right_lower_arm": 3.0,
    "left_lower_arm": 3.0,
    "right_hand": 2.5,
    "left_hand": 2.5,
    "buttocks": 5.0,
    "perineum": 1.0,
    "right_thigh": 9.5,
    "left_thigh": 9.5,
    "right_lower_leg": 7.0,
    "left_lower_leg": 7.0,
    "right_foot": 3.5,
    "left_foot": 3.5,
}

_LB_HEAD_ADJ = {0: 12.0, 1: 10.5, 5: 8.5, 10: 6.5, 15: 5.5}
_LB_THIGH_ADJ = {0: 5.5, 1: 6.5, 5: 8.0, 10: 8.5, 15: 9.0}
_LB_LOWER_LEG_ADJ = {0: 5.0, 1: 5.0, 5: 5.5, 10: 6.0, 15: 6.5}


class BurnAssessment:
    """Burn total body surface area (TBSA) estimation.

    Supports the Rule of Nines for adults and the Lund-Browder chart
    for pediatric patients.
    """

    def calculate_tbsa(
        self,
        burn_regions: Dict[str, float],
        age_years: Optional[int] = None,
        use_lund_browder: bool = False,
    ) -> TBSAResult:
        """Calculate TBSA percentage.

        Parameters
        ----------
        burn_regions : dict
            Mapping of region name to fraction of that region burned
            (0.0 to 1.0).
        age_years : int, optional
            Patient age. Required when ``use_lund_browder`` is True.
        use_lund_browder : bool
            If True, use the Lund-Browder chart (pediatric).

        Returns
        -------
        TBSAResult
        """
        if use_lund_browder:
            if age_years is None:
                raise ValidationError(
                    "age_years is required for Lund-Browder calculation."
                )
            chart = self._get_lund_browder_chart(age_years)
        else:
            chart = _RULE_OF_NINES_ADULT

        regions: Dict[str, float] = {}
        total = 0.0
        for region, fraction in burn_regions.items():
            if not 0.0 <= fraction <= 1.0:
                raise ValidationError(
                    f"Fraction for {region} must be in [0, 1], got {fraction}."
                )
            region_pct = chart.get(region, 0.0) * fraction
            if region_pct > 0:
                regions[region] = round(region_pct, 1)
                total += region_pct

        total = round(total, 1)

        if total < 10:
            classification = "minor"
        elif total < 20:
            classification = "moderate"
        else:
            classification = "major"

        return TBSAResult(
            total_tbsa_percent=total,
            regions=regions,
            classification=classification,
        )

    @staticmethod
    def classify_burn_depth(
        blanching: bool = True,
        blisters: bool = False,
        sensation: str = "intact",
        color: str = "red",
    ) -> BurnDepth:
        """Classify burn depth from clinical characteristics.

        Parameters
        ----------
        blanching : bool
            Whether the burn blanches with pressure.
        blisters : bool
            Presence of blisters.
        sensation : str
            One of 'intact', 'decreased', 'absent'.
        color : str
            Observed color: 'red', 'pink', 'white', 'brown', 'black'.

        Returns
        -------
        BurnDepth
        """
        if color in ("black", "brown") and sensation == "absent":
            return BurnDepth.FOURTH_DEGREE
        if sensation == "absent" and color == "white":
            return BurnDepth.FULL_THICKNESS
        if sensation == "decreased" and blisters:
            return BurnDepth.PARTIAL_DEEP
        if blisters and blanching:
            return BurnDepth.PARTIAL_SUPERFICIAL
        return BurnDepth.SUPERFICIAL

    @staticmethod
    def _get_lund_browder_chart(age_years: int) -> Dict[str, float]:
        chart = dict(_LUND_BROWDER_BASE)
        bracket = 15
        for threshold in sorted(_LB_HEAD_ADJ.keys()):
            if age_years >= threshold:
                bracket = threshold
        chart["head"] = _LB_HEAD_ADJ[bracket]
        chart["right_thigh"] = _LB_THIGH_ADJ[bracket]
        chart["left_thigh"] = _LB_THIGH_ADJ[bracket]
        chart["right_lower_leg"] = _LB_LOWER_LEG_ADJ[bracket]
        chart["left_lower_leg"] = _LB_LOWER_LEG_ADJ[bracket]
        return chart


class ParklandFormula:
    """Parkland (Baxter) formula for burn fluid resuscitation.

    Formula
    -------
    Total volume (mL) = 4 * weight_kg * TBSA%
    First 8 hours: 50% of total.
    Next 16 hours: remaining 50%.
    """

    def calculate_resuscitation(
        self,
        weight_kg: float,
        tbsa_percent: float,
        is_pediatric: bool = False,
    ) -> ResuscitationResult:
        """Calculate fluid resuscitation requirements.

        Parameters
        ----------
        weight_kg : float
        tbsa_percent : float
            Percentage of TBSA burned (0-100).
        is_pediatric : bool
            If True, adds maintenance fluids for children.

        Returns
        -------
        ResuscitationResult
        """
        if weight_kg <= 0:
            raise ValidationError("Weight must be positive.")
        if not 0 < tbsa_percent <= 100:
            raise ValidationError("TBSA percent must be in (0, 100].")

        total = 4.0 * weight_kg * tbsa_percent

        if is_pediatric:
            maintenance = self._pediatric_maintenance(weight_kg) * 24
            total += maintenance

        first_8hr = total / 2.0
        next_16hr = total / 2.0

        uo_target = 0.5 * weight_kg if not is_pediatric else 1.0 * weight_kg

        return ResuscitationResult(
            total_volume_ml=round(total, 0),
            first_8hr_rate_ml_hr=round(first_8hr / 8.0, 1),
            next_16hr_rate_ml_hr=round(next_16hr / 16.0, 1),
            urine_output_target_ml_hr=round(uo_target, 1),
        )

    @staticmethod
    def _pediatric_maintenance(weight_kg: float) -> float:
        """Holliday-Segar formula for maintenance fluid rate (mL/hr)."""
        if weight_kg <= 10:
            return 4.0 * weight_kg
        if weight_kg <= 20:
            return 40.0 + 2.0 * (weight_kg - 10)
        return 60.0 + 1.0 * (weight_kg - 20)
