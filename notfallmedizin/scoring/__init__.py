# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Scoring module for the notfallmedizin library.

Validated clinical scoring systems for risk stratification and
severity assessment in emergency medicine.

Submodules:
    sepsis: SOFA, qSOFA, SIRS.
    cardiac: HEART, TIMI, CHA2DS2-VASc.
    trauma: ISS, RTS, TRISS.
    neurological: GCS, NIHSS.
    pediatric: PEWS, APGAR.
    respiratory: CURB-65, ROX index.

References:
    Vincent et al. SOFA. Intensive Care Med 1996.
    Six et al. HEART score. Neth Heart J 2008.
"""

from notfallmedizin.scoring.cardiac import (
    CHA2DS2VAScScore,
    HEARTScore,
    TIMIScore,
)
from notfallmedizin.scoring.neurological import (
    GCSCalculator,
    NIHSSCalculator,
)
from notfallmedizin.scoring.pediatric import (
    APGARScore,
    PEWSScore,
)
from notfallmedizin.scoring.respiratory import (
    CURB65Score,
    ROXIndex,
)
from notfallmedizin.scoring.sepsis import (
    SIRSCriteria,
    SOFAScore,
    qSOFAScore,
)
from notfallmedizin.scoring.trauma import (
    ISSScore,
    RTSScore,
    TRISSScore,
)

__all__ = [
    # Sepsis
    "SOFAScore",
    "qSOFAScore",
    "SIRSCriteria",
    # Cardiac
    "HEARTScore",
    "TIMIScore",
    "CHA2DS2VAScScore",
    # Trauma
    "ISSScore",
    "RTSScore",
    "TRISSScore",
    # Neurological
    "GCSCalculator",
    "NIHSSCalculator",
    # Pediatric
    "PEWSScore",
    "APGARScore",
    # Respiratory
    "CURB65Score",
    "ROXIndex",
]
