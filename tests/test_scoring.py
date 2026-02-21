# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Tests for notfallmedizin.scoring module."""

import pytest

from notfallmedizin.scoring.sepsis import SOFAScore, qSOFAScore, SIRSCriteria
from notfallmedizin.scoring.cardiac import HEARTScore, TIMIScore, CHA2DS2VAScScore
from notfallmedizin.scoring.neurological import GCSCalculator, NIHSSCalculator
from notfallmedizin.scoring.respiratory import CURB65Score, ROXIndex


def test_qsofa_calculate():
    q = qSOFAScore()
    r = q.calculate(systolic_bp=95, respiratory_rate=24, altered_mentation=True)
    assert r.total_score == 3.0
    assert r.risk_category == "positive"


def test_sofa_calculate():
    sofa = SOFAScore()
    r = sofa.calculate(
        pao2_fio2_ratio=200,
        platelets=80,
        bilirubin=3.5,
        map_value=65,
        gcs=13,
        creatinine=2.1,
        mechanical_ventilation=True,
    )
    assert 0 <= r.total_score <= 24
    assert "component_scores" in dir(r) or hasattr(r, "component_scores")


def test_heart_score():
    heart = HEARTScore()
    r = heart.calculate(
        history=1,
        ecg=1,
        age=0,
        risk_factors=1,
        troponin=0,
    )
    assert 0 <= r.total_score <= 10
    assert r.risk_category in ("low", "moderate", "high")


def test_gcs_calculator():
    gcs = GCSCalculator()
    r = gcs.calculate(eye=4, verbal=5, motor=6)
    assert r.total_score == 15
    assert "mild" in r.interpretation.lower() or "15" in r.interpretation


def test_curb65():
    curb = CURB65Score()
    r = curb.calculate(
        confusion=False,
        bun=15,
        respiratory_rate=18,
        systolic_bp=120,
        diastolic_bp=80,
        age=40,
    )
    assert 0 <= r.total_score <= 5


def test_rox_index():
    rox = ROXIndex()
    r = rox.calculate(spo2=96, fio2=0.21, respiratory_rate=20)
    assert r.total_score > 0
