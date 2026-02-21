# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Triage module for the notfallmedizin library.

Implements internationally recognised ED triage systems (ESI, MTS, CTAS),
a machine-learning triage classifier, and pediatric triage with PAT and PEWS.

Submodules:
    esi: Emergency Severity Index (ESI) v4.
    mts: Manchester Triage System (MTS).
    ctas: Canadian Triage and Acuity Scale (CTAS).
    ml_triage: ML classifier and feature extraction for triage.
    pediatric: Pediatric triage, PAT, PEWS.

References:
    Gilboy N et al. Emergency Severity Index (ESI) v4. AHRQ 2012.
    Mackway-Jones K et al. Manchester Triage Group. BMJ 1997.
"""

from notfallmedizin.triage.ctas import (
    CTASLevel,
    CTASResult,
    CTASTriageCalculator,
)
from notfallmedizin.triage.esi import (
    ESIMentalStatus,
    ESIResult,
    ESITriageCalculator,
)
from notfallmedizin.triage.ml_triage import (
    MLTriageClassifier,
    TriageFeatureExtractor,
)
from notfallmedizin.triage.mts import (
    MTSCategory,
    MTSDiscriminatorType,
    MTSResult,
    MTSTriageCalculator,
)
from notfallmedizin.triage.pediatric import (
    PediatricAgeGroup,
    PediatricTriageCalculator,
    PediatricTriageResult,
    calculate_pews,
    classify_age_group,
    evaluate_pat,
)

__all__ = [
    # ESI
    "ESITriageCalculator",
    "ESIResult",
    "ESIMentalStatus",
    # MTS
    "MTSTriageCalculator",
    "MTSResult",
    "MTSCategory",
    "MTSDiscriminatorType",
    # CTAS
    "CTASTriageCalculator",
    "CTASResult",
    "CTASLevel",
    # ML triage
    "MLTriageClassifier",
    "TriageFeatureExtractor",
    # Pediatric
    "PediatricTriageCalculator",
    "PediatricTriageResult",
    "PediatricAgeGroup",
    "classify_age_group",
    "evaluate_pat",
    "calculate_pews",
]
