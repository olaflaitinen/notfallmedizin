# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""notfallmedizin: Emergency Medicine meets Artificial Intelligence.

A comprehensive Python library integrating emergency medicine with
artificial intelligence for clinical decision support, predictive
analytics, and real-time patient monitoring.

Attributes:
    __version__: Package version string (e.g. "0.1.0").
    __version_tuple__: Version as (major, minor, patch) for comparison.

Modules:
    core: Base classes, configuration, validation, and exceptions.
    triage: ESI, MTS, CTAS scoring and ML-based triage prediction.
    vitals: Vital signs monitoring, anomaly detection, and trend analysis.
    imaging: Medical image preprocessing, chest X-ray, CT, and POCUS analysis.
    nlp: Clinical NER, text classification, summarization, and ICD coding.
    pharmacology: Drug dosing, interactions, pharmacokinetics, and safety alerts.
    scoring: Sepsis, cardiac, trauma, neurological, pediatric, respiratory scores.
    timeseries: Forecasting, decomposition, feature extraction, and real-time streaming.
    prediction: Mortality, readmission, deterioration, LOS, and disposition models.
    trauma: Primary/secondary surveys, burns, hemorrhage, and TBI assessment.
    cardiac: ECG processing, arrhythmia detection, STEMI analysis, and risk scores.
    statistics: Survival analysis, Bayesian inference, power analysis, and meta-analysis.
    benchmarks: Synthetic datasets, evaluation metrics, model comparison, and reporting.

References:
    Project repository: https://github.com/olaflaitinen/notfallmedizin
    Apache License 2.0: http://www.apache.org/licenses/LICENSE-2.0
"""

from notfallmedizin._version import __version__, __version_tuple__

__all__ = [
    "__version__",
    "__version_tuple__",
    "benchmarks",
    "cardiac",
    "core",
    "imaging",
    "nlp",
    "pharmacology",
    "prediction",
    "scoring",
    "statistics",
    "timeseries",
    "trauma",
    "triage",
    "vitals",
]
