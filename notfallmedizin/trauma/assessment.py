# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Primary and secondary trauma survey implementations.

References
----------
American College of Surgeons. (2018). Advanced Trauma Life Support (ATLS)
    Student Course Manual (10th ed.).
Sasser, S. M., et al. (2012). Guidelines for Field Triage of Injured
    Patients. MMWR Recommendations and Reports, 61(RR-1), 1-20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from notfallmedizin.core.exceptions import ValidationError


class MechanismType(Enum):
    """Types of injury mechanism."""

    MVC = "motor_vehicle_collision"
    FALL = "fall"
    PENETRATING = "penetrating"
    BLAST = "blast"
    CRUSH = "crush"
    BURN = "burn"
    DROWNING = "drowning"
    ASSAULT = "assault"


@dataclass(frozen=True)
class PrimarySurveyResult:
    """Result of a primary trauma survey (ABCDE).

    Attributes
    ----------
    components : dict
        Status per component (normal / compromised / critical).
    overall_status : str
    immediate_interventions : list of str
    priority_level : int
        1 = immediate, 2 = urgent, 3 = delayed.
    """

    components: Dict[str, str] = field(default_factory=dict)
    overall_status: str = "stable"
    immediate_interventions: List[str] = field(default_factory=list)
    priority_level: int = 3


@dataclass(frozen=True)
class SecondarySurveyResult:
    """Result of a secondary trauma survey (head-to-toe)."""

    findings_by_region: Dict[str, List[str]] = field(default_factory=dict)
    total_findings: int = 0
    recommended_imaging: List[str] = field(default_factory=list)
    recommended_labs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MOIResult:
    """Result of mechanism-of-injury classification."""

    mechanism: MechanismType = MechanismType.FALL
    energy_classification: str = "low"
    trauma_activation_criteria_met: bool = False
    recommended_level: str = "standard"
    details: str = ""


_VALID_STATUS = {"normal", "compromised", "critical"}

_IMAGING_MAP: Dict[str, List[str]] = {
    "head": ["CT head without contrast"],
    "neck": ["CT cervical spine", "CT angiography neck"],
    "chest": ["Chest X-ray AP", "CT chest with contrast"],
    "abdomen": ["FAST ultrasound", "CT abdomen/pelvis with contrast"],
    "pelvis": ["Pelvic X-ray AP", "CT pelvis"],
    "extremities": ["X-ray of affected extremity"],
    "back": ["CT thoracolumbar spine"],
}


class PrimaryTraumaSurvey:
    """ABCDE primary trauma survey assessment.

    Each component (Airway, Breathing, Circulation, Disability, Exposure)
    is classified as *normal*, *compromised*, or *critical*.
    """

    def assess(
        self,
        airway: str = "normal",
        breathing: str = "normal",
        circulation: str = "normal",
        disability: str = "normal",
        exposure: str = "normal",
    ) -> PrimarySurveyResult:
        """Run the primary survey.

        Parameters
        ----------
        airway, breathing, circulation, disability, exposure : str
            Each must be one of ``'normal'``, ``'compromised'``, ``'critical'``.

        Returns
        -------
        PrimarySurveyResult
        """
        for name, value in [
            ("airway", airway),
            ("breathing", breathing),
            ("circulation", circulation),
            ("disability", disability),
            ("exposure", exposure),
        ]:
            if value not in _VALID_STATUS:
                raise ValidationError(
                    f"{name} must be one of {_VALID_STATUS}, got '{value}'."
                )

        components = {
            "airway": airway,
            "breathing": breathing,
            "circulation": circulation,
            "disability": disability,
            "exposure": exposure,
        }

        interventions: List[str] = []
        if airway == "critical":
            interventions.append("Definitive airway management (intubation)")
        elif airway == "compromised":
            interventions.append("Jaw thrust / suction / oral airway")

        if breathing == "critical":
            interventions.append("Needle decompression / chest tube")
        elif breathing == "compromised":
            interventions.append("Supplemental oxygen / BVM ventilation")

        if circulation == "critical":
            interventions.append("Massive transfusion protocol activation")
            interventions.append("Two large-bore IV access")
        elif circulation == "compromised":
            interventions.append("IV fluid resuscitation with crystalloid")

        if disability == "critical":
            interventions.append("Neuroprotective measures; consider mannitol")

        critical_count = sum(
            1 for v in components.values() if v == "critical"
        )
        compromised_count = sum(
            1 for v in components.values() if v == "compromised"
        )

        if critical_count > 0:
            overall = "critical"
            priority = 1
        elif compromised_count >= 2:
            overall = "unstable"
            priority = 1
        elif compromised_count == 1:
            overall = "potentially_unstable"
            priority = 2
        else:
            overall = "stable"
            priority = 3

        return PrimarySurveyResult(
            components=components,
            overall_status=overall,
            immediate_interventions=interventions,
            priority_level=priority,
        )


class SecondaryTraumaSurvey:
    """Head-to-toe secondary trauma survey."""

    REGIONS = (
        "head", "neck", "chest", "abdomen", "pelvis",
        "extremities", "back", "neuro",
    )

    def assess(self, **findings: List[str]) -> SecondarySurveyResult:
        """Run the secondary survey.

        Parameters
        ----------
        **findings
            Keyword arguments where each key is a body region and each
            value is a list of clinical findings (strings).

        Returns
        -------
        SecondarySurveyResult
        """
        by_region: Dict[str, List[str]] = {}
        for region in self.REGIONS:
            region_findings = findings.get(region, [])
            if region_findings:
                by_region[region] = list(region_findings)

        total = sum(len(v) for v in by_region.values())

        imaging: List[str] = []
        for region in by_region:
            imaging.extend(_IMAGING_MAP.get(region, []))

        labs: List[str] = ["CBC", "BMP", "Type and Screen", "Coagulation panel"]
        if "abdomen" in by_region or "pelvis" in by_region:
            labs.append("Lipase")
            labs.append("Urinalysis")
        if "chest" in by_region:
            labs.append("Troponin")

        return SecondarySurveyResult(
            findings_by_region=by_region,
            total_findings=total,
            recommended_imaging=list(dict.fromkeys(imaging)),
            recommended_labs=list(dict.fromkeys(labs)),
        )


class MechanismOfInjury:
    """Mechanism-of-injury classification per CDC field triage guidelines.

    Reference: Sasser et al. (2012). MMWR 61(RR-1).
    """

    _HIGH_ENERGY: Dict[MechanismType, List[str]] = {
        MechanismType.MVC: [
            "speed > 40 mph",
            "intrusion > 12 inches",
            "ejection from vehicle",
            "death in same passenger compartment",
            "vehicle telemetry consistent with high risk",
        ],
        MechanismType.FALL: [
            "adult fall > 20 feet (6 m)",
            "child fall > 10 feet (3 m) or 2-3 times the child height",
        ],
        MechanismType.PENETRATING: [
            "gunshot wound to head, neck, torso, or proximal extremities",
        ],
        MechanismType.BLAST: [
            "close proximity to detonation",
        ],
    }

    def classify(
        self,
        mechanism_type: MechanismType,
        details: Optional[Dict[str, object]] = None,
    ) -> MOIResult:
        """Classify mechanism of injury and recommend trauma activation level.

        Parameters
        ----------
        mechanism_type : MechanismType
        details : dict, optional
            Additional details such as ``speed_mph``, ``fall_height_feet``,
            ``penetrating_location``.

        Returns
        -------
        MOIResult
        """
        details = details or {}
        high_energy = False
        detail_text = ""

        if mechanism_type == MechanismType.MVC:
            speed = details.get("speed_mph", 0)
            if isinstance(speed, (int, float)) and speed > 40:
                high_energy = True
                detail_text = f"MVC at {speed} mph"
            ejected = details.get("ejected", False)
            if ejected:
                high_energy = True
                detail_text = "Ejection from vehicle"

        elif mechanism_type == MechanismType.FALL:
            height = details.get("fall_height_feet", 0)
            if isinstance(height, (int, float)) and height > 20:
                high_energy = True
                detail_text = f"Fall from {height} feet"

        elif mechanism_type == MechanismType.PENETRATING:
            high_energy = True
            location = details.get("location", "unspecified")
            detail_text = f"Penetrating injury to {location}"

        elif mechanism_type == MechanismType.BLAST:
            high_energy = True
            detail_text = "Blast injury"

        elif mechanism_type == MechanismType.CRUSH:
            high_energy = True
            detail_text = "Crush injury"

        energy = "high" if high_energy else "low"
        activation = high_energy
        level = "trauma_activation" if high_energy else "standard"

        return MOIResult(
            mechanism=mechanism_type,
            energy_classification=energy,
            trauma_activation_criteria_met=activation,
            recommended_level=level,
            details=detail_text,
        )
