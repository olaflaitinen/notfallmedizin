# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for estimators, scorers, and transformers.

Foundational abstract interfaces for all domain-specific models and
scoring systems. Design follows scikit-learn conventions (parameter
management, fit/predict/transform) with clinical extensions such as
evidence levels, ICD codes, and score interpretation.

References:
    scikit-learn developer API: https://scikit-learn.org/stable/developers/develop.html
    Buitinck et al. (2013). API design for machine learning software. ECML PKDD.
"""

from __future__ import annotations

import copy
import inspect
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    NotfallmedizinError,
    ValidationError,
)


# ======================================================================
# Base Estimator
# ======================================================================


class BaseEstimator(ABC):
    """Abstract base class for all notfallmedizin estimators.

    Provides a scikit-learn-compatible interface for parameter
    introspection (``get_params`` / ``set_params``), a ``fit`` /
    ``predict`` / ``score`` protocol, and automatic ``__repr__``
    generation.

    Subclasses must implement :meth:`fit` and :meth:`predict`.

    Attributes
    ----------
    is_fitted_ : bool
        ``True`` after :meth:`fit` has been called successfully.
    fit_time_ : float or None
        Wall-clock time in seconds consumed by the last call to
        :meth:`fit`, or ``None`` if :meth:`fit` has not been called.
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self.fit_time_: Optional[float] = None

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the parameters of this estimator.

        Parameters
        ----------
        deep : bool, optional
            If ``True``, return parameters of nested sub-estimators as
            well. Default is ``True``.

        Returns
        -------
        dict of str to Any
            Parameter names mapped to their current values.
        """
        init_sig = inspect.signature(self.__init__)  # type: ignore[misc]
        params: Dict[str, Any] = {}
        for name, param in init_sig.parameters.items():
            if name == "self":
                continue
            value = getattr(self, name, param.default)
            params[name] = value
            if deep and hasattr(value, "get_params"):
                nested = value.get_params(deep=True)
                for nk, nv in nested.items():
                    params[f"{name}__{nk}"] = nv
        return params

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Keyword arguments corresponding to estimator parameters.

        Returns
        -------
        self
            The estimator instance.

        Raises
        ------
        ValidationError
            If a parameter name is not recognized.
        """
        valid_params = self.get_params(deep=True)
        nested_params: Dict[str, Dict[str, Any]] = {}

        for key, value in params.items():
            if "__" in key:
                prefix, sub_key = key.split("__", 1)
                nested_params.setdefault(prefix, {})[sub_key] = value
            elif key not in valid_params:
                raise ValidationError(
                    message=(
                        f"Invalid parameter '{key}' for {type(self).__name__}. "
                        f"Valid parameters: {sorted(valid_params.keys())}."
                    ),
                    parameter=key,
                )
            else:
                setattr(self, key, value)

        for prefix, sub_params in nested_params.items():
            sub_estimator = getattr(self, prefix, None)
            if sub_estimator is None or not hasattr(sub_estimator, "set_params"):
                raise ValidationError(
                    message=(
                        f"Parameter prefix '{prefix}' does not correspond to "
                        f"a nested estimator on {type(self).__name__}."
                    ),
                    parameter=prefix,
                )
            sub_estimator.set_params(**sub_params)

        return self

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any) -> "BaseEstimator":
        """Fit the estimator to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,) or None, optional
            Target values. May be ``None`` for unsupervised methods.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        self
            The fitted estimator.
        """

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Generate predictions for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """

    def score(self, X: ArrayLike, y: ArrayLike, **kwargs: Any) -> float:
        """Evaluate the estimator on the given test data and labels.

        The default implementation returns the negative mean squared error.
        Subclasses should override this with a domain-appropriate metric.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test input samples.
        y : array-like of shape (n_samples,)
            True target values.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        float
            Score value.
        """
        self._check_is_fitted()
        predictions = self.predict(X)
        y_arr = np.asarray(y)
        return -float(np.mean((predictions - y_arr) ** 2))

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise if the estimator has not been fitted.

        Raises
        ------
        ModelNotFittedError
            If ``is_fitted_`` is ``False``.
        """
        if not self.is_fitted_:
            raise ModelNotFittedError(
                f"{type(self).__name__} has not been fitted. "
                f"Call 'fit' before using this method."
            )

    def _set_fitted(self) -> None:
        """Mark the estimator as fitted and record fit timing."""
        self.is_fitted_ = True

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"{type(self).__name__}({param_str})"

    def clone(self) -> "BaseEstimator":
        """Create a deep copy of this estimator with the same parameters.

        Returns
        -------
        BaseEstimator
            A new, unfitted estimator with identical parameters.
        """
        return copy.deepcopy(self)


# ======================================================================
# Base Scorer
# ======================================================================


class BaseScorer(ABC):
    """Abstract base class for clinical scoring systems.

    Clinical scoring systems (e.g. SOFA, qSOFA, NEWS2, APACHE II)
    translate a set of clinical observations into a numeric score that
    stratifies patient acuity or risk.

    Subclasses must implement :meth:`calculate`, :meth:`validate_inputs`,
    :meth:`get_score_range`, and :meth:`interpret`.

    Attributes
    ----------
    name : str
        Human-readable name of the scoring system.
    version : str
        Version or revision of the scoring algorithm.
    references : list of str
        Bibliographic references for the scoring system.
    """

    def __init__(
        self,
        name: str = "",
        version: str = "1.0",
        references: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.references = references if references is not None else []

    @abstractmethod
    def calculate(self, **kwargs: Any) -> Union[int, float]:
        """Compute the score from the provided clinical parameters.

        Parameters
        ----------
        **kwargs
            Clinical parameter values required by the scoring system.

        Returns
        -------
        int or float
            The computed score.
        """

    @abstractmethod
    def validate_inputs(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate and normalize the inputs required for score calculation.

        Parameters
        ----------
        **kwargs
            Raw clinical parameter values.

        Returns
        -------
        dict of str to Any
            Validated (and possibly coerced) parameter values.

        Raises
        ------
        ValidationError
            If any input is missing or invalid.
        ClinicalRangeError
            If any value is out of range.
        """

    @abstractmethod
    def get_score_range(self) -> Tuple[float, float]:
        """Return the theoretical minimum and maximum score.

        Returns
        -------
        tuple of (float, float)
            ``(minimum_score, maximum_score)``.
        """

    @abstractmethod
    def interpret(self, score: Union[int, float]) -> str:
        """Return a clinical interpretation of the computed score.

        Parameters
        ----------
        score : int or float
            A score previously returned by :meth:`calculate`.

        Returns
        -------
        str
            A textual interpretation or risk category.
        """

    def __call__(self, **kwargs: Any) -> Union[int, float]:
        """Shorthand: validate inputs, then calculate and return the score.

        Parameters
        ----------
        **kwargs
            Clinical parameter values.

        Returns
        -------
        int or float
            The computed score.
        """
        validated = self.validate_inputs(**kwargs)
        return self.calculate(**validated)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"


# ======================================================================
# Base Transformer
# ======================================================================


class BaseTransformer(ABC):
    """Abstract base class for data transformers.

    Transformers convert input data into a transformed representation
    (e.g. scaling, encoding, feature engineering). They follow the
    scikit-learn ``fit`` / ``transform`` / ``fit_transform`` protocol.

    Attributes
    ----------
    is_fitted_ : bool
        ``True`` after :meth:`fit` has been called successfully.
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "BaseTransformer":
        """Learn transformation parameters from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,) or None, optional
            Target values (ignored in most transformers).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        self
            The fitted transformer.
        """

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply the learned transformation to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        numpy.ndarray
            Transformed data.
        """

    def fit_transform(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Fit to the data and then transform it.

        Equivalent to calling ``fit(X, y, **kwargs).transform(X)`` but
        may be optimized in subclasses.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,) or None, optional
            Target values.
        **kwargs
            Additional keyword arguments passed to :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            Transformed data.
        """
        return self.fit(X, y, **kwargs).transform(X)

    def _check_is_fitted(self) -> None:
        """Raise if the transformer has not been fitted.

        Raises
        ------
        ModelNotFittedError
            If ``is_fitted_`` is ``False``.
        """
        if not self.is_fitted_:
            raise ModelNotFittedError(
                f"{type(self).__name__} has not been fitted. "
                f"Call 'fit' before using 'transform'."
            )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the parameters of this transformer.

        Parameters
        ----------
        deep : bool, optional
            If ``True``, include parameters of nested objects.
            Default is ``True``.

        Returns
        -------
        dict of str to Any
            Parameter names mapped to their current values.
        """
        init_sig = inspect.signature(self.__init__)  # type: ignore[misc]
        params: Dict[str, Any] = {}
        for name, param in init_sig.parameters.items():
            if name == "self":
                continue
            params[name] = getattr(self, name, param.default)
        return params

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"{type(self).__name__}({param_str})"


# ======================================================================
# Clinical Model
# ======================================================================


class ClinicalModel(BaseEstimator):
    """Estimator augmented with clinical metadata.

    Extends :class:`BaseEstimator` with fields that document the
    clinical context of a model: applicable ICD-10 codes, the level
    of evidence supporting the model, and bibliographic references.

    Parameters
    ----------
    icd_codes : list of str, optional
        Applicable ICD-10 codes. Default is an empty list.
    evidence_level : str, optional
        Level of evidence (e.g. ``"I"``, ``"II-a"``, ``"III"``).
        Default is an empty string.
    references : list of str, optional
        Bibliographic references (formatted citations or DOIs).
        Default is an empty list.
    description : str, optional
        Free-text description of the model's clinical purpose.
        Default is an empty string.

    Attributes
    ----------
    icd_codes : list of str
    evidence_level : str
    references : list of str
    description : str
    """

    def __init__(
        self,
        icd_codes: Optional[List[str]] = None,
        evidence_level: str = "",
        references: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        super().__init__()
        self.icd_codes: List[str] = icd_codes if icd_codes is not None else []
        self.evidence_level: str = evidence_level
        self.references: List[str] = references if references is not None else []
        self.description: str = description

    def get_clinical_metadata(self) -> Dict[str, Any]:
        """Return clinical metadata as a dictionary.

        Returns
        -------
        dict of str to Any
            Keys: ``"icd_codes"``, ``"evidence_level"``, ``"references"``,
            ``"description"``.
        """
        return {
            "icd_codes": list(self.icd_codes),
            "evidence_level": self.evidence_level,
            "references": list(self.references),
            "description": self.description,
        }

    def add_reference(self, reference: str) -> None:
        """Append a bibliographic reference.

        Parameters
        ----------
        reference : str
            Formatted citation string or DOI.
        """
        if not reference:
            raise ValidationError("Reference string must not be empty.")
        self.references.append(reference)

    def add_icd_code(self, code: str) -> None:
        """Append an ICD-10 code to the model metadata.

        Parameters
        ----------
        code : str
            A valid ICD-10 code (e.g. ``"R65.20"``).
        """
        if not code:
            raise ValidationError("ICD code must not be empty.")
        self.icd_codes.append(code)


# ======================================================================
# Mixin Classes
# ======================================================================


class ClassifierMixin:
    """Mixin class for classification estimators.

    Provides a default :meth:`score` implementation using accuracy and
    a :meth:`predict_proba` stub.

    The ``_estimator_type`` attribute is set to ``"classifier"`` so that
    utilities can introspect the estimator's purpose.
    """

    _estimator_type: str = "classifier"

    def score(self, X: ArrayLike, y: ArrayLike, **kwargs: Any) -> float:
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        predictions = self.predict(X)  # type: ignore[attr-defined]
        y_arr = np.asarray(y)
        return float(np.mean(predictions == y_arr))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Class probability estimates.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'predict_proba'."
        )


class RegressorMixin:
    """Mixin class for regression estimators.

    Provides a default :meth:`score` implementation using the
    coefficient of determination (R-squared).

    The ``_estimator_type`` attribute is set to ``"regressor"``.
    """

    _estimator_type: str = "regressor"

    def score(self, X: ArrayLike, y: ArrayLike, **kwargs: Any) -> float:
        """Return the R-squared score on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            R-squared (coefficient of determination). Best possible
            score is 1.0; it can be negative if the model is
            arbitrarily bad.
        """
        predictions = self.predict(X)  # type: ignore[attr-defined]
        y_arr = np.asarray(y, dtype=np.float64)
        ss_res = float(np.sum((y_arr - predictions) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot


class ClusterMixin:
    """Mixin class for clustering estimators.

    Provides :meth:`fit_predict` as a convenience method.

    The ``_estimator_type`` attribute is set to ``"clusterer"``.
    """

    _estimator_type: str = "clusterer"

    def fit_predict(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Fit the model and predict cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like or None, optional
            Ignored. Present for API consistency.
        **kwargs
            Additional keyword arguments passed to :meth:`fit`.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        self.fit(X, y, **kwargs)  # type: ignore[attr-defined]
        return self.predict(X)  # type: ignore[attr-defined]
