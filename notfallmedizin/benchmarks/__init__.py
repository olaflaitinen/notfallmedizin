# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarking framework for evaluating emergency medicine AI models.

Provides synthetic dataset generation, comprehensive evaluation metrics,
model comparison utilities, and standardised report generation for
reproducible research.

References:
    DeLong et al. Comparing areas under ROC curves. Biometrics 1988.
"""

from notfallmedizin.benchmarks.datasets import (
    SyntheticEDDatasetGenerator,
    DatasetSplit,
)
from notfallmedizin.benchmarks.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    ClinicalMetrics,
)
from notfallmedizin.benchmarks.comparison import (
    ModelComparison,
    ComparisonResult,
)
from notfallmedizin.benchmarks.reporting import (
    BenchmarkReport,
    ReportSection,
)

__all__ = [
    "BenchmarkReport",
    "ClassificationMetrics",
    "ClinicalMetrics",
    "ComparisonResult",
    "DatasetSplit",
    "ModelComparison",
    "RegressionMetrics",
    "ReportSection",
    "SyntheticEDDatasetGenerator",
]
