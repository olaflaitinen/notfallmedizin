# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Core module for the notfallmedizin library.

Provides foundational building blocks used across all domain-specific
subpackages: custom exceptions, input validators, global configuration
management, and abstract base classes for estimators, scorers, and
transformers.

Submodules:
    exceptions: Structured exception hierarchy for domain-specific error handling.
    validators: Clinical data validation utilities (vitals, age, GCS, lab values).
    config: Global configuration management (get/set/context manager).
    base: Abstract base classes and mixin classes for estimators and scorers.

References:
    scikit-learn BaseEstimator interface: https://scikit-learn.org/stable/developers/develop.html
"""

from notfallmedizin.core.base import (
    BaseEstimator,
    BaseScorer,
    BaseTransformer,
    ClassifierMixin,
    ClinicalModel,
    ClusterMixin,
    RegressorMixin,
)
from notfallmedizin.core.config import (
    NotfallmedizinConfig,
    config_context,
    get_config,
    reset_config,
    set_config,
)
from notfallmedizin.core.exceptions import (
    ClinicalRangeError,
    ComputationError,
    ConfigurationError,
    DataFormatError,
    InsufficientDataError,
    ModelNotFittedError,
    NotfallmedizinError,
    ValidationError,
)
from notfallmedizin.core.validators import (
    validate_age,
    validate_dataframe,
    validate_gcs,
    validate_lab_values,
    validate_probability,
    validate_vital_signs,
)

__all__ = [
    "NotfallmedizinError",
    "ValidationError",
    "ConfigurationError",
    "ModelNotFittedError",
    "DataFormatError",
    "ClinicalRangeError",
    "InsufficientDataError",
    "ComputationError",
    "validate_vital_signs",
    "validate_age",
    "validate_gcs",
    "validate_lab_values",
    "validate_probability",
    "validate_dataframe",
    "NotfallmedizinConfig",
    "get_config",
    "set_config",
    "reset_config",
    "config_context",
    "BaseEstimator",
    "BaseScorer",
    "BaseTransformer",
    "ClinicalModel",
    "ClassifierMixin",
    "RegressorMixin",
    "ClusterMixin",
]
