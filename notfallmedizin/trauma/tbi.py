# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Traumatic brain injury assessment.

References
----------
Marshall, L. F., et al. (1991). A new classification of head injury based
    on computerized tomography. Journal of Neurosurgery, 75(Supplement),
    S14-S20.
Echemendia, R. J., et al. (2017). The Sport Concussion Assessment Tool
    5th Edition (SCAT5). British Journal of Sports Medicine, 51(11), 848-850.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from notfallmedizin.core.exceptions import ValidationError


@dataclass(frozen=True)
class TBIResult:
    """Traumatic brain injury classification result.

    Attributes
    ----------
    severity : str
        mild / moderate / severe
    gcs : int
    marshall_class : int
        Marshall CT classification I through VI.
    marshall_description : str
    recommended_management : list of str
    icp_monitoring_indicated : bool
    """

    severity: str
    gcs: int
    marshall_class: int = 0
    marshall_description: str = ""
    recommended_management: List[str] = field(default_factory=list)
    icp_monitoring_indicated: bool = False


@dataclass(frozen=True)
class ConcussionResult:
    """Concussion assessment result.

    Attributes
    ----------
    total_symptom_score : int
    symptom_count : int
    severity_rating : str
    return_to_play_eligible : bool
    symptoms : dict
    """

    total_symptom_score: int
    symptom_count: int
    severity_rating: str
    return_to_play_eligible: bool
    symptoms: Dict[str, int] = field(default_factory=dict)


_MARSHALL_DESCRIPTIONS = {
    1: "Diffuse injury I: no visible intracranial pathology on CT",
    2: "Diffuse injury II: cisterns present, midline shift 0-5mm, no lesion >25mL",
    3: "Diffuse injury III (swelling): cisterns compressed/absent, midline shift 0-5mm",
    4: "Diffuse injury IV (shift): midline shift >5mm, no lesion >25mL",
    5: "Evacuated mass lesion: any surgically evacuated lesion",
    6: "Non-evacuated mass lesion: high-density lesion >25mL, not surgically evacuated",
}

_SCAT5_SYMPTOMS = [
    "headache",
    "pressure_in_head",
    "neck_pain",
    "nausea_or_vomiting",
    "dizziness",
    "blurred_vision",
    "balance_problems",
    "sensitivity_to_light",
    "sensitivity_to_noise",
    "feeling_slowed_down",
    "feeling_in_a_fog",
    "dont_feel_right",
    "difficulty_concentrating",
    "difficulty_remembering",
    "fatigue_or_low_energy",
    "confusion",
    "drowsiness",
    "more_emotional",
    "irritability",
    "sadness",
    "nervous_or_anxious",
    "trouble_falling_asleep",
]


class TBIClassifier:
    """Traumatic brain injury severity classification.

    Integrates GCS, pupil reactivity, and CT findings into a composite
    assessment per the Marshall classification system.
    """

    def classify(
        self,
        gcs: int,
        pupil_reactivity: str = "both_reactive",
        ct_findings: Optional[Dict[str, object]] = None,
    ) -> TBIResult:
        """Classify TBI severity.

        Parameters
        ----------
        gcs : int
            Glasgow Coma Scale total (3-15).
        pupil_reactivity : str
            One of 'both_reactive', 'one_unreactive', 'both_unreactive'.
        ct_findings : dict, optional
            Keys may include: 'midline_shift_mm', 'lesion_volume_ml',
            'cisterns' ('normal'/'compressed'/'absent'),
            'evacuated_mass' (bool).

        Returns
        -------
        TBIResult
        """
        if not 3 <= gcs <= 15:
            raise ValidationError(f"GCS must be in [3, 15], got {gcs}.")
        valid_pupils = {"both_reactive", "one_unreactive", "both_unreactive"}
        if pupil_reactivity not in valid_pupils:
            raise ValidationError(
                f"pupil_reactivity must be one of {valid_pupils}."
            )

        if gcs >= 13:
            severity = "mild"
        elif gcs >= 9:
            severity = "moderate"
        else:
            severity = "severe"

        ct = ct_findings or {}
        marshall = self._marshall_classify(ct)

        management: List[str] = []
        icp_indicated = False

        if severity == "mild":
            management.append("Serial neurological examinations")
            management.append("CT head if risk factors present")
        elif severity == "moderate":
            management.append("ICU admission for neurological monitoring")
            management.append("Repeat CT in 6-8 hours")
        else:
            management.append("Intubation for airway protection (GCS <= 8)")
            management.append("ICP monitoring")
            management.append("Neurosurgical consultation")
            icp_indicated = True

        if pupil_reactivity == "both_unreactive":
            management.append("Emergent neurosurgical intervention")
            icp_indicated = True
        elif pupil_reactivity == "one_unreactive":
            management.append("Urgent CT; consider herniation")
            icp_indicated = True

        if marshall >= 3:
            icp_indicated = True

        return TBIResult(
            severity=severity,
            gcs=gcs,
            marshall_class=marshall,
            marshall_description=_MARSHALL_DESCRIPTIONS.get(marshall, ""),
            recommended_management=management,
            icp_monitoring_indicated=icp_indicated,
        )

    @staticmethod
    def _marshall_classify(ct: Dict[str, object]) -> int:
        evacuated = ct.get("evacuated_mass", False)
        if evacuated:
            return 5

        lesion_vol = ct.get("lesion_volume_ml", 0)
        if isinstance(lesion_vol, (int, float)) and lesion_vol > 25:
            return 6

        midline = ct.get("midline_shift_mm", 0)
        cisterns = ct.get("cisterns", "normal")

        if isinstance(midline, (int, float)) and midline > 5:
            return 4
        if cisterns in ("compressed", "absent"):
            return 3
        if any(
            ct.get(k) for k in ("contusion", "petechial_hemorrhage", "sah")
        ):
            return 2
        return 1


class PupilReactivityScore:
    """Pupil reactivity score for GCS-Pupils composite.

    Score: 0 = both reactive, 1 = one unreactive, 2 = both unreactive.
    GCS-P = GCS - PupilScore.
    """

    @staticmethod
    def calculate(
        left_reactive: bool,
        right_reactive: bool,
    ) -> int:
        """Return the pupil reactivity score (0, 1, or 2).

        Parameters
        ----------
        left_reactive : bool
        right_reactive : bool

        Returns
        -------
        int
        """
        unreactive_count = int(not left_reactive) + int(not right_reactive)
        return unreactive_count

    @staticmethod
    def gcs_pupils(gcs: int, left_reactive: bool, right_reactive: bool) -> int:
        """Compute the GCS-Pupils score.

        Parameters
        ----------
        gcs : int
        left_reactive : bool
        right_reactive : bool

        Returns
        -------
        int
            GCS-P score (1 to 15).
        """
        pr = int(not left_reactive) + int(not right_reactive)
        return max(gcs - pr, 1)


class ConcussionAssessment:
    """SCAT5-inspired concussion symptom assessment.

    Evaluates 22 symptoms each rated on a 0-6 Likert scale.
    """

    SYMPTOM_LIST = _SCAT5_SYMPTOMS

    def assess(self, symptoms: Dict[str, int]) -> ConcussionResult:
        """Perform concussion symptom evaluation.

        Parameters
        ----------
        symptoms : dict
            Mapping of symptom name to severity (0-6).

        Returns
        -------
        ConcussionResult
        """
        validated: Dict[str, int] = {}
        for symptom in self.SYMPTOM_LIST:
            score = symptoms.get(symptom, 0)
            if not 0 <= score <= 6:
                raise ValidationError(
                    f"Symptom '{symptom}' score must be in [0, 6], got {score}."
                )
            validated[symptom] = score

        total_score = sum(validated.values())
        count = sum(1 for v in validated.values() if v > 0)

        if total_score == 0:
            rating = "asymptomatic"
            eligible = True
        elif total_score <= 10:
            rating = "mild"
            eligible = False
        elif total_score <= 30:
            rating = "moderate"
            eligible = False
        else:
            rating = "severe"
            eligible = False

        return ConcussionResult(
            total_symptom_score=total_score,
            symptom_count=count,
            severity_rating=rating,
            return_to_play_eligible=eligible,
            symptoms=validated,
        )
