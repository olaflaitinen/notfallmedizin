# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Trauma assessment module.

Provides tools for primary and secondary trauma surveys, burn assessment
with the Parkland formula, hemorrhage classification per ATLS guidelines,
and traumatic brain injury evaluation.

References:
    American College of Surgeons. ATLS Student Course Manual. 10th ed. 2018.
    Baxter & Shires. Parkland formula. Ann N Y Acad Sci 1968.
"""

from notfallmedizin.trauma.assessment import (
    MechanismOfInjury,
    MechanismType,
    MOIResult,
    PrimarySurveyResult,
    PrimaryTraumaSurvey,
    SecondarySurveyResult,
    SecondaryTraumaSurvey,
)
from notfallmedizin.trauma.burns import (
    BurnAssessment,
    BurnDepth,
    ParklandFormula,
    ResuscitationResult,
    TBSAResult,
)
from notfallmedizin.trauma.hemorrhage import (
    HemorrhageClass,
    HemorrhageClassifier,
    MassiveTransfusionProtocol,
    MTPResult,
    ShockIndexCalculator,
    ShockIndexResult,
)
from notfallmedizin.trauma.tbi import (
    ConcussionAssessment,
    ConcussionResult,
    PupilReactivityScore,
    TBIClassifier,
    TBIResult,
)

__all__ = [
    "BurnAssessment",
    "BurnDepth",
    "ConcussionAssessment",
    "ConcussionResult",
    "HemorrhageClass",
    "HemorrhageClassifier",
    "MassiveTransfusionProtocol",
    "MechanismOfInjury",
    "MechanismType",
    "MOIResult",
    "MTPResult",
    "ParklandFormula",
    "PrimarySurveyResult",
    "PrimaryTraumaSurvey",
    "PupilReactivityScore",
    "ResuscitationResult",
    "SecondarySurveyResult",
    "SecondaryTraumaSurvey",
    "ShockIndexCalculator",
    "ShockIndexResult",
    "TBIClassifier",
    "TBIResult",
    "TBSAResult",
]
