# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Prediction module for emergency department outcome forecasting.

Provides machine-learning and statistical models for predicting patient
outcomes including mortality, readmission, clinical deterioration, length of
stay, and disposition decisions.

References:
    Knaus et al. APACHE II. Crit Care Med 1985.
    van Walraven et al. LACE index. CMAJ 2010.
"""

from notfallmedizin.prediction.mortality import (
    APACHE2Mortality,
    EDMortalityPredictor,
)
from notfallmedizin.prediction.readmission import (
    LACEIndex,
    LACEResult,
    ReadmissionPredictor,
)
from notfallmedizin.prediction.deterioration import (
    DeteriorationPredictor,
    EarlyWarningScorePredictor,
    MEWSResult,
)
from notfallmedizin.prediction.los import (
    EDThroughputAnalyzer,
    LOSPredictor,
    ThroughputMetrics,
)
from notfallmedizin.prediction.disposition import (
    BedAvailabilityEstimator,
    BedEstimate,
    DispositionCategory,
    DispositionPredictor,
)

__all__ = [
    "APACHE2Mortality",
    "BedAvailabilityEstimator",
    "BedEstimate",
    "DeteriorationPredictor",
    "DispositionCategory",
    "DispositionPredictor",
    "EarlyWarningScorePredictor",
    "EDMortalityPredictor",
    "EDThroughputAnalyzer",
    "LACEIndex",
    "LACEResult",
    "LOSPredictor",
    "MEWSResult",
    "ReadmissionPredictor",
    "ThroughputMetrics",
]
