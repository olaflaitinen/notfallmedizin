# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""STEMI detection and management protocol.

References
----------
O'Gara, P. T., et al. (2013). 2013 ACCF/AHA guideline for the
    management of ST-elevation myocardial infarction. Circulation,
    127(4), e362-e425.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class STEMIResult:
    """STEMI detection result.

    Attributes
    ----------
    is_stemi : bool
    affected_territory : str
    culprit_vessel : str
    st_elevations : dict
    reciprocal_depressions : dict
    confidence : float
    """

    is_stemi: bool
    affected_territory: str = ""
    culprit_vessel: str = ""
    st_elevations: Dict[str, float] = field(default_factory=dict)
    reciprocal_depressions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(frozen=True)
class FibrinolysisResult:
    """Fibrinolysis eligibility assessment.

    Attributes
    ----------
    eligible : bool
    contraindications_present : list of str
    recommended_agent : str
    time_window_remaining_hours : float
    """

    eligible: bool
    contraindications_present: List[str] = field(default_factory=list)
    recommended_agent: str = ""
    time_window_remaining_hours: float = 0.0


_TERRITORY_MAP = {
    "anterior": {
        "elevation_leads": ["V1", "V2", "V3", "V4"],
        "reciprocal_leads": ["II", "III", "aVF"],
        "vessel": "LAD (left anterior descending)",
    },
    "lateral": {
        "elevation_leads": ["I", "aVL", "V5", "V6"],
        "reciprocal_leads": ["III", "aVF"],
        "vessel": "LCx (left circumflex)",
    },
    "inferior": {
        "elevation_leads": ["II", "III", "aVF"],
        "reciprocal_leads": ["I", "aVL"],
        "vessel": "RCA (right coronary artery)",
    },
    "posterior": {
        "elevation_leads": ["V7", "V8", "V9"],
        "reciprocal_leads": ["V1", "V2", "V3"],
        "vessel": "RCA or LCx (posterior descending)",
    },
    "right_ventricular": {
        "elevation_leads": ["V3R", "V4R"],
        "reciprocal_leads": [],
        "vessel": "Proximal RCA",
    },
}

_ABSOLUTE_CONTRAINDICATIONS = [
    "active_internal_bleeding",
    "suspected_aortic_dissection",
    "prior_hemorrhagic_stroke",
    "ischemic_stroke_within_3_months",
    "intracranial_neoplasm",
    "significant_head_trauma_within_3_months",
    "structural_cerebrovascular_lesion",
]

_RELATIVE_CONTRAINDICATIONS = [
    "uncontrolled_hypertension_sbp_over_180",
    "current_anticoagulant_use",
    "recent_major_surgery_within_3_weeks",
    "internal_bleeding_within_4_weeks",
    "non_compressible_vascular_puncture",
    "pregnancy",
    "active_peptic_ulcer",
    "prolonged_cpr_over_10_minutes",
]


class STEMIDetector:
    """STEMI detection from 12-lead ECG ST-segment measurements.

    ST elevation criteria (O'Gara et al., 2013):
    - Limb leads: >= 1.0 mm (0.1 mV)
    - V2-V3 males >= 40 yr: >= 2.0 mm
    - V2-V3 males < 40 yr: >= 2.5 mm
    - V2-V3 females: >= 1.5 mm
    - Other precordial leads: >= 1.0 mm

    Requires ST elevation in >= 2 contiguous leads.
    """

    def detect(
        self,
        ecg_leads: Dict[str, float],
        age: int = 50,
        sex: str = "male",
    ) -> STEMIResult:
        """Evaluate ECG leads for STEMI criteria.

        Parameters
        ----------
        ecg_leads : dict
            Mapping of lead name to ST deviation in mm. Positive values
            indicate elevation; negative values indicate depression.
        age : int
        sex : str
            'male' or 'female'.

        Returns
        -------
        STEMIResult
        """
        elevations: Dict[str, float] = {}
        depressions: Dict[str, float] = {}

        for lead, value in ecg_leads.items():
            threshold = self._get_threshold(lead, age, sex)
            if value >= threshold:
                elevations[lead] = value
            elif value <= -1.0:
                depressions[lead] = value

        best_territory = ""
        best_vessel = ""
        best_count = 0

        for territory, info in _TERRITORY_MAP.items():
            matched = [l for l in info["elevation_leads"] if l in elevations]
            if len(matched) >= 2 and len(matched) > best_count:
                best_count = len(matched)
                best_territory = territory
                best_vessel = info["vessel"]

        is_stemi = best_count >= 2

        reciprocal: Dict[str, float] = {}
        if is_stemi and best_territory:
            recip_leads = _TERRITORY_MAP[best_territory]["reciprocal_leads"]
            reciprocal = {l: depressions[l] for l in recip_leads if l in depressions}

        confidence = 0.0
        if is_stemi:
            confidence = min(0.5 + best_count * 0.1 + len(reciprocal) * 0.1, 1.0)

        return STEMIResult(
            is_stemi=is_stemi,
            affected_territory=best_territory,
            culprit_vessel=best_vessel,
            st_elevations=elevations,
            reciprocal_depressions=reciprocal,
            confidence=round(confidence, 2),
        )

    @staticmethod
    def _get_threshold(lead: str, age: int, sex: str) -> float:
        if lead in ("V2", "V3"):
            if sex == "male":
                return 2.5 if age < 40 else 2.0
            return 1.5
        return 1.0


class STEMIProtocol:
    """STEMI management protocol utilities."""

    @staticmethod
    def calculate_door_to_balloon_target() -> int:
        """Return the target door-to-balloon time in minutes.

        Per ACC/AHA guidelines, the target is 90 minutes from first
        medical contact to primary PCI balloon inflation.

        Returns
        -------
        int
        """
        return 90

    @staticmethod
    def check_fibrinolysis_eligibility(
        onset_hours: float,
        contraindications: Optional[List[str]] = None,
    ) -> FibrinolysisResult:
        """Assess eligibility for fibrinolytic therapy.

        Parameters
        ----------
        onset_hours : float
            Hours since symptom onset.
        contraindications : list of str, optional
            List of contraindication codes.

        Returns
        -------
        FibrinolysisResult
        """
        contraindications = contraindications or []
        absolute_present = [
            c for c in contraindications if c in _ABSOLUTE_CONTRAINDICATIONS
        ]
        relative_present = [
            c for c in contraindications if c in _RELATIVE_CONTRAINDICATIONS
        ]

        window_remaining = max(12.0 - onset_hours, 0.0)
        eligible = (
            len(absolute_present) == 0
            and onset_hours <= 12.0
        )

        agent = ""
        if eligible:
            if onset_hours <= 6.0:
                agent = "tenecteplase (preferred) or alteplase"
            else:
                agent = "alteplase (consider PCI transfer if possible)"

        return FibrinolysisResult(
            eligible=eligible,
            contraindications_present=absolute_present + relative_present,
            recommended_agent=agent,
            time_window_remaining_hours=round(window_remaining, 1),
        )
