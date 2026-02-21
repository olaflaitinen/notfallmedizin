# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Statistical methods module for emergency medicine research.

Provides survival analysis (Kaplan-Meier, Cox PH, log-rank), Bayesian
inference (A/B testing, diagnostic updating), power analysis, meta-analysis
(fixed/random effects), and diagnostic test evaluation.

References:
    Kaplan & Meier. Nonparametric estimation. JASA 1958.
    Cox. Regression models and life-tables. JRSS-B 1972.
    DeLong et al. Comparing AUCs. Biometrics 1988.
"""

from notfallmedizin.statistics.survival import (
    CoxPHModel,
    KaplanMeierEstimator,
    LogRankResult,
    LogRankTest,
)
from notfallmedizin.statistics.bayesian import (
    BayesianABTest,
    BayesianDiagnosticTest,
    BayesianResult,
)
from notfallmedizin.statistics.power import (
    MultiplicityCorrectionMethods,
    PowerAnalyzer,
)
from notfallmedizin.statistics.meta_analysis import (
    FixedEffectsMetaAnalysis,
    FunnelPlotData,
    MetaAnalysisResult,
    RandomEffectsMetaAnalysis,
)
from notfallmedizin.statistics.diagnostic import (
    DiagnosticMetrics,
    DiagnosticTestEvaluator,
    DeLongResult,
    ROCAnalyzer,
)

__all__ = [
    "BayesianABTest",
    "BayesianDiagnosticTest",
    "BayesianResult",
    "CoxPHModel",
    "DeLongResult",
    "DiagnosticMetrics",
    "DiagnosticTestEvaluator",
    "FixedEffectsMetaAnalysis",
    "FunnelPlotData",
    "KaplanMeierEstimator",
    "LogRankResult",
    "LogRankTest",
    "MetaAnalysisResult",
    "MultiplicityCorrectionMethods",
    "PowerAnalyzer",
    "RandomEffectsMetaAnalysis",
    "ROCAnalyzer",
]
