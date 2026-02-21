# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Tests for notfallmedizin.core module."""

import pytest

from notfallmedizin.core.exceptions import (
    ClinicalRangeError,
    NotfallmedizinError,
    ValidationError,
)
from notfallmedizin.core.validators import (
    validate_age,
    validate_gcs,
    validate_vital_signs,
)
from notfallmedizin.core.config import get_config, config_context


def test_validate_vital_signs_accepts_valid():
    """Valid vital signs should not raise."""
    validate_vital_signs(
        heart_rate=80,
        systolic_bp=120,
        diastolic_bp=80,
        respiratory_rate=16,
        spo2=98,
        temperature=37.0,
    )


def test_validate_vital_signs_rejects_out_of_range():
    """Out-of-range vital signs should raise ClinicalRangeError."""
    with pytest.raises(ClinicalRangeError):
        validate_vital_signs(
            heart_rate=80,
            systolic_bp=120,
            diastolic_bp=80,
            respiratory_rate=16,
            spo2=101,
            temperature=37.0,
        )


def test_validate_age_accepts_valid():
    """Valid age should not raise."""
    validate_age(30, unit="years")
    validate_age(12, unit="months")


def test_validate_gcs_accepts_valid():
    """Valid GCS components should not raise."""
    validate_gcs(eye=4, verbal=5, motor=6)


def test_validate_gcs_rejects_invalid():
    """Invalid GCS should raise."""
    with pytest.raises((ValidationError, ClinicalRangeError)):
        validate_gcs(eye=0, verbal=5, motor=6)


def test_config_get_set():
    """get_config and set_config should work."""
    cfg = get_config()
    assert hasattr(cfg, "random_state")
    assert hasattr(cfg, "n_jobs")


def test_config_context():
    """config_context should restore previous state."""
    cfg_before = get_config()
    with config_context(verbose=True):
        assert get_config().verbose is True
    assert get_config().verbose == cfg_before.verbose


def test_notfallmedizin_error():
    """NotfallmedizinError should be catchable."""
    with pytest.raises(NotfallmedizinError):
        raise ValidationError("test")
