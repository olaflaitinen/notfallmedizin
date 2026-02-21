# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Vital signs monitoring, anomaly detection, trend analysis, and alerting.

Toolkit for real-time vital sign processing: monitoring, anomaly
detection, trend analysis (Mann-Kendall, CUSUM), and NEWS2 alerting.

Submodules:
    monitor: Vital signs tracking and hemodynamic indices.
    anomaly: Isolation Forest and statistical anomaly detection.
    trends: Trend analysis and changepoint detection.
    alerts: Rule-based alerts and NEWS2.

References:
    Royal College of Physicians. National Early Warning Score (NEWS) 2. 2017.
"""

from notfallmedizin.vitals.alerts import (
    AlertSeverity,
    ClinicalAlert,
    ClinicalAlertEngine,
)
from notfallmedizin.vitals.anomaly import (
    StatisticalAnomalyDetector,
    VitalSignsAnomalyDetector,
)
from notfallmedizin.vitals.monitor import (
    VitalSignsMonitor,
    VitalSignsState,
)
from notfallmedizin.vitals.trends import (
    TrendDirection,
    TrendResult,
    VitalSignsTrendAnalyzer,
)

__all__ = [
    # monitor
    "VitalSignsMonitor",
    "VitalSignsState",
    # anomaly
    "VitalSignsAnomalyDetector",
    "StatisticalAnomalyDetector",
    # trends
    "VitalSignsTrendAnalyzer",
    "TrendResult",
    "TrendDirection",
    # alerts
    "ClinicalAlertEngine",
    "ClinicalAlert",
    "AlertSeverity",
]
