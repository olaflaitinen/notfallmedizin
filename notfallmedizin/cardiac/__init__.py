# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Cardiac emergency analysis module.

Provides ECG signal processing (Pan-Tompkins R-peak detection, HRV
analysis), arrhythmia classification, STEMI detection with territory
mapping, and cardiac risk calculators (Framingham, Wells PE/DVT).

References:
    Pan & Tompkins. A real-time QRS detection algorithm. IEEE TBME 1985.
    Task Force ESC/NASPE. Heart rate variability. Circulation 1996.
"""

from notfallmedizin.cardiac.ecg import ECGProcessor, HRVMetrics
from notfallmedizin.cardiac.arrhythmia import (
    ArrhythmiaDetector,
    ArrhythmiaType,
    RhythmAnalysis,
    RhythmAnalyzer,
)
from notfallmedizin.cardiac.stemi import (
    FibrinolysisResult,
    STEMIDetector,
    STEMIProtocol,
    STEMIResult,
)
from notfallmedizin.cardiac.risk import (
    FraminghamResult,
    FraminghamRiskCalculator,
    WellsDVTScore,
    WellsPEScore,
    WellsResult,
)

__all__ = [
    "ArrhythmiaDetector",
    "ArrhythmiaType",
    "ECGProcessor",
    "FibrinolysisResult",
    "FraminghamResult",
    "FraminghamRiskCalculator",
    "HRVMetrics",
    "RhythmAnalysis",
    "RhythmAnalyzer",
    "STEMIDetector",
    "STEMIProtocol",
    "STEMIResult",
    "WellsDVTScore",
    "WellsPEScore",
    "WellsResult",
]
