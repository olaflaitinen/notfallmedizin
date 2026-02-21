# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Input validation utilities for clinical data.

Provides validation functions that enforce physiologically plausible
ranges on clinical parameters. Used throughout the library to catch
erroneous or implausible inputs early. Range limits are based on
broadly accepted clinical reference ranges.

References:
    Normal vital signs by age: PALS, APLS, and adult emergency guidelines.
    Glasgow Coma Scale: Teasdale & Jennett, Lancet 1974;2(7872):81-84.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from notfallmedizin.core.exceptions import (
    ClinicalRangeError,
    DataFormatError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Vital-sign reference ranges (parameter -> (lower, upper, unit))
# ---------------------------------------------------------------------------
_VITAL_SIGN_RANGES: Dict[str, Tuple[float, float, str]] = {
    "heart_rate": (0.0, 300.0, "bpm"),
    "systolic_bp": (0.0, 350.0, "mmHg"),
    "diastolic_bp": (0.0, 250.0, "mmHg"),
    "respiratory_rate": (0.0, 80.0, "breaths/min"),
    "spo2": (0.0, 100.0, "%"),
    "temperature": (20.0, 45.0, "deg C"),
}

# ---------------------------------------------------------------------------
# Laboratory reference ranges (parameter -> (lower, upper, unit))
# ---------------------------------------------------------------------------
_LAB_RANGES: Dict[str, Tuple[float, float, str]] = {
    "lactate": (0.0, 30.0, "mmol/L"),
    "ph": (6.5, 8.0, ""),
    "creatinine": (0.0, 30.0, "mg/dL"),
    "potassium": (1.0, 10.0, "mmol/L"),
    "sodium": (100.0, 180.0, "mmol/L"),
    "glucose": (0.0, 1500.0, "mg/dL"),
    "hemoglobin": (0.0, 25.0, "g/dL"),
    "platelets": (0.0, 2000.0, "x10^3/uL"),
    "wbc": (0.0, 500.0, "x10^3/uL"),
    "bilirubin": (0.0, 50.0, "mg/dL"),
    "troponin": (0.0, 100.0, "ng/mL"),
    "bnp": (0.0, 50000.0, "pg/mL"),
    "pco2": (5.0, 150.0, "mmHg"),
    "po2": (0.0, 700.0, "mmHg"),
    "bicarbonate": (1.0, 60.0, "mmol/L"),
    "inr": (0.5, 20.0, ""),
    "fibrinogen": (0.0, 1500.0, "mg/dL"),
    "alt": (0.0, 10000.0, "U/L"),
    "ast": (0.0, 10000.0, "U/L"),
    "albumin": (0.0, 7.0, "g/dL"),
    "crp": (0.0, 500.0, "mg/L"),
    "procalcitonin": (0.0, 200.0, "ng/mL"),
    "d_dimer": (0.0, 100000.0, "ng/mL"),
}


def _check_numeric(value: Any, name: str) -> float:
    """Coerce *value* to float and verify it is finite.

    Parameters
    ----------
    value : Any
        The value to check.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    float
        The validated numeric value.

    Raises
    ------
    ValidationError
        If *value* cannot be converted to a finite float.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            message=f"Parameter '{name}' must be numeric, got {type(value).__name__}.",
            parameter=name,
        ) from exc
    if not np.isfinite(result):
        raise ValidationError(
            message=f"Parameter '{name}' must be finite, got {result}.",
            parameter=name,
        )
    return result


def _check_range(value: float, name: str, lower: float, upper: float) -> float:
    """Verify that *value* falls within [*lower*, *upper*].

    Parameters
    ----------
    value : float
        The numeric value to check.
    name : str
        Parameter name used in error messages.
    lower : float
        Inclusive lower bound.
    upper : float
        Inclusive upper bound.

    Returns
    -------
    float
        The validated value (unchanged).

    Raises
    ------
    ClinicalRangeError
        If *value* is outside [*lower*, *upper*].
    """
    if value < lower or value > upper:
        raise ClinicalRangeError(
            parameter=name,
            value=value,
            lower=lower,
            upper=upper,
        )
    return value


# ======================================================================
# Public validation functions
# ======================================================================


def validate_vital_signs(
    heart_rate: float,
    systolic_bp: float,
    diastolic_bp: float,
    respiratory_rate: float,
    spo2: float,
    temperature: float,
) -> Dict[str, float]:
    """Validate a set of vital-sign measurements.

    Each parameter is checked for numeric type and physiologically
    plausible range. An additional consistency check ensures that
    diastolic blood pressure does not exceed systolic blood pressure.

    Parameters
    ----------
    heart_rate : float
        Heart rate in beats per minute.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    diastolic_bp : float
        Diastolic blood pressure in mmHg.
    respiratory_rate : float
        Respiratory rate in breaths per minute.
    spo2 : float
        Peripheral oxygen saturation as a percentage (0--100).
    temperature : float
        Body temperature in degrees Celsius.

    Returns
    -------
    dict of str to float
        A dictionary of validated vital signs keyed by parameter name.

    Raises
    ------
    ValidationError
        If any value is non-numeric or non-finite.
    ClinicalRangeError
        If any value falls outside its acceptable range.
    """
    params = {
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "spo2": spo2,
        "temperature": temperature,
    }

    validated: Dict[str, float] = {}
    for name, value in params.items():
        v = _check_numeric(value, name)
        lo, hi, _ = _VITAL_SIGN_RANGES[name]
        validated[name] = _check_range(v, name, lo, hi)

    if validated["diastolic_bp"] > validated["systolic_bp"]:
        raise ValidationError(
            message=(
                f"Diastolic BP ({validated['diastolic_bp']} mmHg) cannot "
                f"exceed systolic BP ({validated['systolic_bp']} mmHg)."
            ),
            parameter="diastolic_bp",
        )

    return validated


def validate_age(
    age: float,
    unit: str = "years",
) -> float:
    """Validate a patient age value.

    Parameters
    ----------
    age : float
        The age value to validate.
    unit : {"years", "months", "days"}, optional
        Unit of the age value. Default is ``"years"``.

    Returns
    -------
    float
        The validated age.

    Raises
    ------
    ValidationError
        If *unit* is not one of the recognized options or *age* is
        non-numeric.
    ClinicalRangeError
        If *age* is outside the physiologically plausible range for
        the given unit.
    """
    allowed_units = ("years", "months", "days")
    if unit not in allowed_units:
        raise ValidationError(
            message=f"Age unit must be one of {allowed_units}, got '{unit}'.",
            parameter="unit",
        )

    age_val = _check_numeric(age, "age")

    upper_limits = {
        "years": 130.0,
        "months": 130.0 * 12,
        "days": 130.0 * 365.25,
    }
    return _check_range(age_val, "age", 0.0, upper_limits[unit])


def validate_gcs(
    eye: int,
    verbal: int,
    motor: int,
) -> Dict[str, int]:
    """Validate Glasgow Coma Scale component scores.

    Parameters
    ----------
    eye : int
        Eye-opening response (1--4).
    verbal : int
        Verbal response (1--5).
    motor : int
        Motor response (1--6).

    Returns
    -------
    dict of str to int
        A dictionary with keys ``"eye"``, ``"verbal"``, ``"motor"``,
        and ``"total"``.

    Raises
    ------
    ValidationError
        If any component is not an integer.
    ClinicalRangeError
        If any component is outside its valid range.
    """
    components: Dict[str, Tuple[int, int, int]] = {
        "eye": (eye, 1, 4),
        "verbal": (verbal, 1, 5),
        "motor": (motor, 1, 6),
    }

    validated: Dict[str, int] = {}
    for name, (value, lo, hi) in components.items():
        if not isinstance(value, int):
            raise ValidationError(
                message=f"GCS component '{name}' must be an integer, got {type(value).__name__}.",
                parameter=name,
            )
        _check_range(float(value), name, float(lo), float(hi))
        validated[name] = value

    validated["total"] = validated["eye"] + validated["verbal"] + validated["motor"]
    return validated


def validate_lab_values(**kwargs: float) -> Dict[str, float]:
    """Validate common laboratory values.

    Each keyword argument should be a laboratory parameter name (e.g.
    ``lactate``, ``ph``, ``creatinine``) mapped to its numeric value.
    Recognized parameter names and their acceptable ranges are defined
    in the internal ``_LAB_RANGES`` dictionary.

    Parameters
    ----------
    **kwargs : float
        Laboratory parameter names and their values.

    Returns
    -------
    dict of str to float
        A dictionary of validated lab values.

    Raises
    ------
    ValidationError
        If a parameter name is unrecognized or a value is non-numeric.
    ClinicalRangeError
        If a value falls outside its acceptable range.

    Examples
    --------
    >>> validate_lab_values(lactate=2.1, ph=7.35, creatinine=1.2)
    {'lactate': 2.1, 'ph': 7.35, 'creatinine': 1.2}
    """
    validated: Dict[str, float] = {}
    for name, value in kwargs.items():
        name_lower = name.lower()
        if name_lower not in _LAB_RANGES:
            raise ValidationError(
                message=(
                    f"Unrecognized lab parameter '{name}'. "
                    f"Recognized parameters: {sorted(_LAB_RANGES.keys())}."
                ),
                parameter=name,
            )
        v = _check_numeric(value, name)
        lo, hi, _ = _LAB_RANGES[name_lower]
        validated[name_lower] = _check_range(v, name, lo, hi)
    return validated


def validate_probability(
    value: float,
    name: str = "probability",
) -> float:
    """Validate that a value represents a valid probability in [0, 1].

    Parameters
    ----------
    value : float
        The value to validate.
    name : str, optional
        Descriptive name for the probability (used in error messages).
        Default is ``"probability"``.

    Returns
    -------
    float
        The validated probability.

    Raises
    ------
    ValidationError
        If *value* is non-numeric or falls outside [0, 1].
    """
    v = _check_numeric(value, name)
    if v < 0.0 or v > 1.0:
        raise ValidationError(
            message=f"'{name}' must be in [0, 1], got {v}.",
            parameter=name,
        )
    return v


def validate_dataframe(
    df: Any,
    required_columns: Sequence[str],
    numeric_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Validate the structure and content of a pandas DataFrame.

    Parameters
    ----------
    df : Any
        The object to validate. Must be a :class:`pandas.DataFrame`.
    required_columns : sequence of str
        Column names that must be present in the DataFrame.
    numeric_columns : sequence of str or None, optional
        Subset of columns that must contain numeric (non-NaN) data.
        If ``None``, no numeric check is performed.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame (unchanged).

    Raises
    ------
    ValidationError
        If *df* is not a DataFrame.
    DataFormatError
        If required columns are missing or numeric columns contain
        non-numeric data.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            message=f"Expected a pandas DataFrame, got {type(df).__name__}.",
            parameter="df",
        )

    if df.empty:
        raise DataFormatError("The provided DataFrame is empty.")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise DataFormatError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}."
        )

    if numeric_columns is not None:
        for col in numeric_columns:
            if col not in df.columns:
                raise DataFormatError(
                    f"Numeric column '{col}' is not present in the DataFrame."
                )
            if not np.issubdtype(df[col].dtype, np.number):
                raise DataFormatError(
                    f"Column '{col}' must be numeric, "
                    f"got dtype '{df[col].dtype}'."
                )
            if df[col].isna().any():
                n_missing = int(df[col].isna().sum())
                raise DataFormatError(
                    f"Column '{col}' contains {n_missing} missing (NaN) "
                    f"value(s). Impute or drop these before proceeding."
                )

    return df
