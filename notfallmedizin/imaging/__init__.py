# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Imaging module for the notfallmedizin library.

This package provides medical image preprocessing utilities and
deep-learning-based analysis frameworks for emergency radiology:

* **Preprocessing** -- intensity normalization, CT windowing, spatial
  resizing, denoising, and augmentation.
* **Chest X-ray** -- multi-label finding classification with Grad-CAM
  visual explanations.
* **Head CT** -- intracranial hemorrhage detection, stroke-type
  classification, ABC/2 hemorrhage volume estimation, and midline
  shift measurement.
* **Point-of-care ultrasound** -- FAST / eFAST exam analysis with
  free-fluid and pneumothorax detection, and ejection fraction
  calculators (Simpson, Teichholz).

Submodules
----------
preprocessing
    Image normalization, resizing, CT windowing, denoising, and
    augmentation utilities.
xray
    Chest X-ray multi-label classifier and Grad-CAM heatmaps.
ct
    Head CT hemorrhage detection, stroke classification, volume
    estimation, and midline shift calculation.
ultrasound
    FAST/eFAST exam analyser and ejection fraction calculators.

References:
    Kothari et al. ABC/2 for intracerebral hemorrhage volume. Stroke 1996.
    FAST: Scalea et al. J Trauma 1999.
"""

from notfallmedizin.imaging.preprocessing import (
    CT_WINDOW_PRESETS,
    ImagePreprocessor,
)
from notfallmedizin.imaging.xray import (
    SUPPORTED_FINDINGS,
    ChestXrayClassifier,
    XrayFindingResult,
)
from notfallmedizin.imaging.ct import (
    CTAnalyzer,
    CTFinding,
    HemorrhageType,
    MiddlelineShiftCalculator,
    StrokeType,
    calculate_hemorrhage_volume,
)
from notfallmedizin.imaging.ultrasound import (
    EFAST_REGIONS,
    EFFunction,
    FAST_REGIONS,
    FASTExamAnalyzer,
    FASTResult,
)

__all__ = [
    # preprocessing
    "ImagePreprocessor",
    "CT_WINDOW_PRESETS",
    # xray
    "ChestXrayClassifier",
    "XrayFindingResult",
    "SUPPORTED_FINDINGS",
    # ct
    "CTAnalyzer",
    "CTFinding",
    "HemorrhageType",
    "StrokeType",
    "MiddlelineShiftCalculator",
    "calculate_hemorrhage_volume",
    # ultrasound
    "FASTExamAnalyzer",
    "FASTResult",
    "FAST_REGIONS",
    "EFAST_REGIONS",
    "EFFunction",
]
