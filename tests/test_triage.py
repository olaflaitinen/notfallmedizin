# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Tests for notfallmedizin.triage module."""

import pytest

from notfallmedizin.triage.esi import ESITriageCalculator, ESIResult
from notfallmedizin.triage.ctas import CTASTriageCalculator
from notfallmedizin.triage.ml_triage import MLTriageClassifier, TriageFeatureExtractor
import numpy as np


def test_esi_calculate():
    esi = ESITriageCalculator()
    result = esi.calculate(
        chief_complaint="chest pain",
        vital_signs={"heart_rate": 110, "systolic_bp": 85, "spo2": 91},
        resource_estimate=3,
        mental_status="alert",
        severe_pain_distress=False,
        requires_immediate_intervention=False,
    )
    assert isinstance(result, ESIResult)
    assert 1 <= result.level <= 5
    assert isinstance(result.reasoning, (list, tuple))


def test_ctas_calculate():
    ctas = CTASTriageCalculator()
    result = ctas.calculate(
        complaint_group="cardiac",
        first_order_modifiers={},
        second_order_modifiers={},
    )
    assert 1 <= result.level <= 5
    assert result.target_time_to_physician_minutes >= 0


def test_ml_triage_fit_predict():
    clf = MLTriageClassifier()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    y = rng.integers(1, 6, size=50)
    clf.fit(X, y)
    pred = clf.predict(X[:5])
    assert pred.shape == (5,)
    assert np.all((pred >= 1) & (pred <= 5))


def test_triage_feature_extractor():
    ext = TriageFeatureExtractor()
    import pandas as pd
    df = pd.DataFrame({
        "heart_rate": [80, 90],
        "systolic_bp": [120, 110],
        "diastolic_bp": [80, 70],
        "respiratory_rate": [16, 18],
        "spo2": [98, 97],
        "temperature": [36.5, 37.0],
        "age": [45, 60],
        "chief_complaint": ["chest pain", "shortness of breath"],
    })
    X = ext.fit_transform(df)
    assert X.shape[0] == 2
    assert X.shape[1] >= 1
