# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Time series analysis module for the notfallmedizin library.

This package provides forecasting, decomposition, feature extraction,
and real-time streaming analysis tools tailored to clinical time series
data such as vital signs and laboratory measurements.

Submodules
----------
forecasting
    Exponential smoothing, ARIMA, and multi-variate vital signs
    forecasting with clinical constraints.
decomposition
    Classical seasonal decomposition (additive and multiplicative) and
    Haar wavelet decomposition with denoising.
features
    Comprehensive feature extraction covering statistical, temporal,
    complexity, frequency-domain, and clinical threshold features.
realtime
    Sliding window streaming processor and Bayesian online changepoint
    detection.

References:
    Adams & MacKay. Bayesian online changepoint detection. arXiv 2007.
"""

from notfallmedizin.timeseries.decomposition import (
    DecompositionResult,
    SeasonalDecomposer,
    WaveletDecomposer,
)
from notfallmedizin.timeseries.features import (
    ClinicalTimeSeriesFeatureExtractor,
)
from notfallmedizin.timeseries.forecasting import (
    ExponentialSmoothingForecaster,
    VitalSignsForecaster,
)
from notfallmedizin.timeseries.realtime import (
    OnlineChangePointDetector,
    StreamingProcessor,
)

__all__ = [
    # forecasting
    "ExponentialSmoothingForecaster",
    "VitalSignsForecaster",
    # decomposition
    "SeasonalDecomposer",
    "DecompositionResult",
    "WaveletDecomposer",
    # features
    "ClinicalTimeSeriesFeatureExtractor",
    # realtime
    "StreamingProcessor",
    "OnlineChangePointDetector",
]

# ARIMAForecaster is excluded from top-level imports because it
# requires the optional dependency ``statsmodels``. Import it
# directly when needed:
#
#     from notfallmedizin.timeseries.forecasting import ARIMAForecaster
