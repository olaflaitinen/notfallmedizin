# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Custom exception hierarchy for the notfallmedizin library.

Defines a structured exception hierarchy used throughout the library
to signal specific error conditions related to clinical data validation,
model fitting, configuration, and computation. All exceptions inherit
from NotfallmedizinError, which inherits from built-in Exception.

References:
    PEP 3151: Reworking the OS and IO exception hierarchy.
    scikit-learn: sklearn.exceptions.NotFittedError.
"""


class NotfallmedizinError(Exception):
    """Base exception for all notfallmedizin-specific errors.

    Args:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(self.message)


class ValidationError(NotfallmedizinError):
    """Raised when input validation fails.

    Used for general-purpose validation failures such as incorrect
    types, missing required fields, or structurally invalid inputs.

    Args:
        message: Description of the validation failure.
        parameter: Name of the parameter that failed validation.
    """

    def __init__(self, message: str = "", parameter: str = "") -> None:
        self.parameter = parameter
        if parameter and not message:
            message = f"Validation failed for parameter '{parameter}'."
        super().__init__(message)


class ConfigurationError(NotfallmedizinError):
    """Raised when library configuration is invalid or inconsistent.

    Args:
        message: Description of the configuration error.
    """


class ModelNotFittedError(NotfallmedizinError):
    """Raised when a method is called on an unfitted model.

    Analogous to scikit-learn NotFittedError. Raised when predict,
    transform, or score is invoked before fit has been called.

    Args:
        message: Description of the error.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = (
                "This estimator has not been fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        super().__init__(message)


class DataFormatError(NotfallmedizinError):
    """Raised when data does not conform to the expected format.

    Covers unexpected column names, incompatible dtypes, or
    structurally malformed datasets.

    Args:
        message: Description of the format violation.
    """


class ClinicalRangeError(ValidationError):
    """Raised when a clinical value is outside physiologically plausible range.

    Args:
        parameter: Name of the clinical parameter (e.g. heart_rate).
        value: The value that was provided.
        lower: Lower bound of the acceptable range (inclusive).
        upper: Upper bound of the acceptable range (inclusive).
    """

    def __init__(
        self,
        parameter: str,
        value: float,
        lower: float,
        upper: float,
    ) -> None:
        self.value = value
        self.lower = lower
        self.upper = upper
        message = (
            f"Clinical parameter '{parameter}' has value {value}, "
            f"which is outside the acceptable range [{lower}, {upper}]."
        )
        super().__init__(message=message, parameter=parameter)


class InsufficientDataError(NotfallmedizinError):
    """Raised when the dataset contains too few observations.

    Args:
        message: Description of the data insufficiency.
        n_samples: Number of samples that were provided.
        n_required: Minimum number of samples required.
    """

    def __init__(
        self,
        message: str = "",
        n_samples: int = 0,
        n_required: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.n_required = n_required
        if not message and n_samples and n_required:
            message = (
                f"Insufficient data: received {n_samples} samples, "
                f"but at least {n_required} are required."
            )
        super().__init__(message)


class ComputationError(NotfallmedizinError):
    """Raised when a numerical or algorithmic computation fails.

    Covers convergence failures, singular matrices, or numerical
    overflow/underflow during model fitting or evaluation.

    Args:
        message: Description of the computational failure.
    """
