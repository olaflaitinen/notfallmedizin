# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Machine-learning-based triage level prediction.

This module provides a supervised classification model for predicting
emergency department triage levels from structured patient data, as
well as a feature extraction transformer for converting raw clinical
records into numerical feature vectors.

The :class:`MLTriageClassifier` wraps a scikit-learn gradient-boosted
tree or random-forest classifier and exposes a clinical-model interface
(fit/predict/predict_proba) with additional methods for feature
importance analysis. The :class:`TriageFeatureExtractor` handles
encoding of vital signs, chief-complaint categories, demographics,
and arrival-mode information.

References
----------
.. [1] Levin S, Toerper M, Hamrock E, et al. Machine-learning-based
       electronic triage more accurately differentiates patients with
       respect to clinical outcomes compared with the Emergency
       Severity Index. Ann Emerg Med. 2018;71(5):565-574.e2.
       doi:10.1016/j.annemergmed.2017.08.005
.. [2] Raita Y, Goto T, Faridi MK, et al. Emergency department triage
       prediction of clinical outcomes using machine learning models.
       Crit Care. 2019;23(1):64. doi:10.1186/s13054-019-2351-7
.. [3] Klug M, Barash Y, Bechler S, et al. A gradient boosting machine
       learning model for predicting early mortality in the emergency
       department triage: devising a nine-point triage score. J Gen
       Intern Med. 2020;35(1):220-227.
       doi:10.1007/s11606-019-05512-7
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from notfallmedizin.core.base import (
    BaseTransformer,
    ClassifierMixin,
    ClinicalModel,
)
from notfallmedizin.core.config import get_config
from notfallmedizin.core.exceptions import (
    DataFormatError,
    ModelNotFittedError,
    ValidationError,
)


# ======================================================================
# Feature extractor
# ======================================================================

_DEFAULT_VITAL_SIGN_COLUMNS: Tuple[str, ...] = (
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature",
)

_DEFAULT_DEMOGRAPHIC_COLUMNS: Tuple[str, ...] = (
    "age",
    "sex",
)

_DEFAULT_CATEGORICAL_COLUMNS: Tuple[str, ...] = (
    "chief_complaint",
    "arrival_mode",
)

_ARRIVAL_MODE_VALUES: Tuple[str, ...] = (
    "ambulance",
    "walk_in",
    "helicopter",
    "transfer",
    "police",
    "other",
)


class TriageFeatureExtractor(BaseTransformer):
    """Extract numerical features from raw patient data for triage ML.

    Transforms a :class:`pandas.DataFrame` of raw patient data into
    a numerical feature matrix suitable for machine-learning models.
    The transformer handles:

    - **Vital signs**: Passed through as continuous features with
      optional standardisation.
    - **Chief complaint**: One-hot encoded from a learned vocabulary.
    - **Demographics**: Age passed through as continuous; sex one-hot
      encoded.
    - **Arrival mode**: One-hot encoded.
    - **Derived features**: Shock index (HR / SBP), mean arterial
      pressure (MAP), pulse pressure.

    Parameters
    ----------
    vital_sign_columns : tuple of str, optional
        Names of vital-sign columns to extract. Defaults to the
        standard six vital-sign parameters.
    demographic_columns : tuple of str, optional
        Names of demographic columns. Defaults to ``("age", "sex")``.
    categorical_columns : tuple of str, optional
        Names of categorical columns to one-hot encode. Defaults to
        ``("chief_complaint", "arrival_mode")``.
    add_derived_features : bool, optional
        If ``True`` (default), compute derived vital-sign features
        (shock index, MAP, pulse pressure).
    standardize : bool, optional
        If ``True`` (default ``False``), standardize continuous features
        to zero mean and unit variance based on training-set statistics.

    Attributes
    ----------
    categories_ : dict of str to list of str
        Learned categorical vocabularies (populated after ``fit``).
    continuous_mean_ : dict of str to float
        Training-set mean of each continuous feature (if
        ``standardize=True``).
    continuous_std_ : dict of str to float
        Training-set standard deviation of each continuous feature.
    feature_names_ : list of str
        Ordered list of output feature names (populated after ``fit``).

    Examples
    --------
    >>> import pandas as pd
    >>> extractor = TriageFeatureExtractor(standardize=True)
    >>> df = pd.DataFrame({
    ...     "heart_rate": [80, 120],
    ...     "systolic_bp": [120, 90],
    ...     "diastolic_bp": [80, 60],
    ...     "respiratory_rate": [16, 24],
    ...     "spo2": [98, 91],
    ...     "temperature": [37.0, 38.5],
    ...     "age": [45, 70],
    ...     "sex": ["male", "female"],
    ...     "chief_complaint": ["chest_pain", "dyspnea"],
    ...     "arrival_mode": ["ambulance", "walk_in"],
    ... })
    >>> X = extractor.fit_transform(df)
    >>> X.shape[1] == len(extractor.feature_names_)
    True
    """

    def __init__(
        self,
        vital_sign_columns: Tuple[str, ...] = _DEFAULT_VITAL_SIGN_COLUMNS,
        demographic_columns: Tuple[str, ...] = _DEFAULT_DEMOGRAPHIC_COLUMNS,
        categorical_columns: Tuple[str, ...] = _DEFAULT_CATEGORICAL_COLUMNS,
        add_derived_features: bool = True,
        standardize: bool = False,
    ) -> None:
        super().__init__()
        self.vital_sign_columns = vital_sign_columns
        self.demographic_columns = demographic_columns
        self.categorical_columns = categorical_columns
        self.add_derived_features = add_derived_features
        self.standardize = standardize

        self.categories_: Dict[str, List[str]] = {}
        self.continuous_mean_: Dict[str, float] = {}
        self.continuous_std_: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "TriageFeatureExtractor":
        """Learn vocabularies and (optionally) standardisation parameters.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw patient data with columns matching the configured
            vital-sign, demographic, and categorical column names.
        y : array-like or None, optional
            Ignored. Present for API consistency.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        self
            The fitted transformer.

        Raises
        ------
        DataFormatError
            If ``X`` is not a DataFrame or required columns are missing.
        """
        df = self._validate_dataframe(X)

        for col in self.categorical_columns:
            if col in df.columns:
                self.categories_[col] = sorted(df[col].dropna().unique().tolist())
            else:
                self.categories_[col] = []

        if "sex" in self.demographic_columns and "sex" in df.columns:
            self.categories_["sex"] = sorted(df["sex"].dropna().unique().tolist())

        continuous_cols = self._get_continuous_columns(df)
        if self.standardize and continuous_cols:
            for col in continuous_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                self.continuous_mean_[col] = float(series.mean())
                std_val = float(series.std())
                self.continuous_std_[col] = std_val if std_val > 0 else 1.0

        self.feature_names_ = self._build_feature_names()
        self.is_fitted_ = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform raw patient data into a numerical feature matrix.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw patient data.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            Numerical feature matrix.

        Raises
        ------
        ModelNotFittedError
            If the transformer has not been fitted.
        DataFormatError
            If ``X`` is not a DataFrame or required columns are missing.
        """
        self._check_is_fitted()
        df = self._validate_dataframe(X)
        n_samples = len(df)
        features: List[np.ndarray] = []

        for col in self.vital_sign_columns:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan,
                )
            else:
                vals = np.full(n_samples, np.nan, dtype=np.float64)
            features.append(vals.reshape(-1, 1))

        if self.add_derived_features:
            features.extend(self._compute_derived(df, n_samples))

        for col in self.demographic_columns:
            if col == "sex":
                features.extend(self._one_hot_encode(df, "sex", n_samples))
            elif col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan,
                )
                features.append(vals.reshape(-1, 1))
            else:
                features.append(np.full((n_samples, 1), np.nan, dtype=np.float64))

        for col in self.categorical_columns:
            features.extend(self._one_hot_encode(df, col, n_samples))

        result = np.hstack(features) if features else np.empty((n_samples, 0))

        if self.standardize and self.continuous_mean_:
            continuous_cols = self._get_continuous_columns(df)
            col_idx = 0
            for name in self.feature_names_:
                if name in self.continuous_mean_:
                    mean = self.continuous_mean_[name]
                    std = self.continuous_std_[name]
                    idx = self.feature_names_.index(name)
                    result[:, idx] = (result[:, idx] - mean) / std

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dataframe(self, X: Any) -> pd.DataFrame:
        """Ensure X is a non-empty DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise DataFormatError(
                f"Expected a pandas DataFrame, got {type(X).__name__}."
            )
        if X.empty:
            raise DataFormatError("The provided DataFrame is empty.")
        return X

    def _get_continuous_columns(self, df: pd.DataFrame) -> List[str]:
        """Return the list of continuous column names present in df."""
        continuous = []
        for col in self.vital_sign_columns:
            if col in df.columns:
                continuous.append(col)
        for col in self.demographic_columns:
            if col != "sex" and col in df.columns:
                continuous.append(col)
        return continuous

    def _one_hot_encode(
        self,
        df: pd.DataFrame,
        column: str,
        n_samples: int,
    ) -> List[np.ndarray]:
        """One-hot encode a categorical column using the learned vocabulary."""
        categories = self.categories_.get(column, [])
        if not categories:
            return []

        encoded: List[np.ndarray] = []
        if column in df.columns:
            values = df[column].astype(str).str.strip().str.lower()
            for cat in categories:
                encoded.append(
                    (values == cat.lower()).to_numpy(dtype=np.float64).reshape(-1, 1)
                )
        else:
            for _ in categories:
                encoded.append(np.zeros((n_samples, 1), dtype=np.float64))
        return encoded

    def _compute_derived(
        self,
        df: pd.DataFrame,
        n_samples: int,
    ) -> List[np.ndarray]:
        """Compute derived vital-sign features."""
        derived: List[np.ndarray] = []

        hr = pd.to_numeric(df.get("heart_rate"), errors="coerce") if "heart_rate" in df.columns else None
        sbp = pd.to_numeric(df.get("systolic_bp"), errors="coerce") if "systolic_bp" in df.columns else None
        dbp = pd.to_numeric(df.get("diastolic_bp"), errors="coerce") if "diastolic_bp" in df.columns else None

        # Shock index = HR / SBP
        if hr is not None and sbp is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                shock_idx = np.where(
                    sbp.to_numpy(dtype=np.float64) > 0,
                    hr.to_numpy(dtype=np.float64) / sbp.to_numpy(dtype=np.float64),
                    np.nan,
                )
            derived.append(shock_idx.reshape(-1, 1))
        else:
            derived.append(np.full((n_samples, 1), np.nan, dtype=np.float64))

        # Mean arterial pressure = DBP + (SBP - DBP) / 3
        if sbp is not None and dbp is not None:
            sbp_arr = sbp.to_numpy(dtype=np.float64)
            dbp_arr = dbp.to_numpy(dtype=np.float64)
            map_arr = dbp_arr + (sbp_arr - dbp_arr) / 3.0
            derived.append(map_arr.reshape(-1, 1))
        else:
            derived.append(np.full((n_samples, 1), np.nan, dtype=np.float64))

        # Pulse pressure = SBP - DBP
        if sbp is not None and dbp is not None:
            pp = sbp.to_numpy(dtype=np.float64) - dbp.to_numpy(dtype=np.float64)
            derived.append(pp.reshape(-1, 1))
        else:
            derived.append(np.full((n_samples, 1), np.nan, dtype=np.float64))

        return derived

    def _build_feature_names(self) -> List[str]:
        """Build the ordered list of output feature names."""
        names: List[str] = []
        names.extend(self.vital_sign_columns)

        if self.add_derived_features:
            names.extend(["shock_index", "mean_arterial_pressure", "pulse_pressure"])

        for col in self.demographic_columns:
            if col == "sex":
                for cat in self.categories_.get("sex", []):
                    names.append(f"sex_{cat}")
            else:
                names.append(col)

        for col in self.categorical_columns:
            for cat in self.categories_.get(col, []):
                names.append(f"{col}_{cat}")

        return names


# ======================================================================
# ML triage classifier
# ======================================================================


class MLTriageClassifier(ClinicalModel, ClassifierMixin):
    """Machine-learning classifier for triage level prediction.

    Wraps a scikit-learn ensemble classifier (gradient-boosted trees or
    random forest) and exposes the :class:`ClinicalModel` interface
    with additional methods for feature-importance analysis.

    Parameters
    ----------
    base_estimator : object or None, optional
        A scikit-learn-compatible classifier instance. If ``None``
        (default), a :class:`~sklearn.ensemble.GradientBoostingClassifier`
        with sensible defaults is used.
    n_estimators : int, optional
        Number of boosting stages (trees) when using the default
        estimator. Ignored if ``base_estimator`` is provided.
        Default is ``200``.
    max_depth : int, optional
        Maximum tree depth for the default estimator. Default is ``5``.
    learning_rate : float, optional
        Learning rate for the default estimator. Default is ``0.1``.
    random_state : int or None, optional
        Random seed. If ``None``, the library-wide configuration value
        is used. Default is ``None``.
    icd_codes : list of str, optional
        Applicable ICD-10 codes. Default is an empty list.
    evidence_level : str, optional
        Level of evidence. Default is ``""``.
    references : list of str, optional
        Bibliographic references. Default is ``None``.
    description : str, optional
        Free-text description. Default is ``""``.

    Attributes
    ----------
    classes_ : numpy.ndarray or None
        Unique class labels learned during ``fit``.
    model_ : object or None
        The fitted scikit-learn estimator.

    Examples
    --------
    >>> import numpy as np
    >>> clf = MLTriageClassifier(n_estimators=50, max_depth=3)
    >>> X_train = np.random.randn(200, 10)
    >>> y_train = np.random.randint(1, 6, size=200)
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    MLTriageClassifier(...)
    >>> predictions = clf.predict(X_train)  # doctest: +SKIP

    References
    ----------
    .. [1] Levin S, et al. ML-based electronic triage. Ann Emerg Med.
           2018;71(5):565-574.e2.
    .. [2] Raita Y, et al. ED triage prediction using ML models. Crit
           Care. 2019;23(1):64.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None,
        icd_codes: Optional[List[str]] = None,
        evidence_level: str = "",
        references: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        super().__init__(
            icd_codes=icd_codes,
            evidence_level=evidence_level,
            references=references or [
                "Levin S, et al. ML-based electronic triage. Ann Emerg "
                "Med. 2018;71(5):565-574.e2.",
                "Raita Y, et al. ED triage prediction using ML models. "
                "Crit Care. 2019;23(1):64.",
            ],
            description=description or (
                "Gradient-boosted tree classifier for emergency department "
                "triage level prediction."
            ),
        )
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.model_: Optional[Any] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> "MLTriageClassifier":
        """Fit the triage classifier on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Triage level labels. Expected values are integers in
            {1, 2, 3, 4, 5}.
        **kwargs
            Additional keyword arguments passed to the underlying
            scikit-learn ``fit`` method.

        Returns
        -------
        self
            The fitted classifier.

        Raises
        ------
        ValidationError
            If ``y`` is ``None`` or contains values outside {1..5}.
        """
        if y is None:
            raise ValidationError(
                message="Target labels 'y' are required for supervised training.",
                parameter="y",
            )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        unique_labels = set(np.unique(y_arr).tolist())
        valid_labels = {1, 2, 3, 4, 5}
        if not unique_labels.issubset(valid_labels):
            invalid = unique_labels - valid_labels
            raise ValidationError(
                message=(
                    f"Target labels must be in {{1,2,3,4,5}}, "
                    f"found invalid labels: {sorted(invalid)}."
                ),
                parameter="y",
            )

        rs = self.random_state
        if rs is None:
            cfg = get_config()
            rs = cfg.random_state

        start_time = time.perf_counter()

        if self.base_estimator is not None:
            import copy
            self.model_ = copy.deepcopy(self.base_estimator)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=rs,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
            )

        self.model_.fit(X_arr, y_arr, **kwargs)
        self.classes_ = np.array(sorted(unique_labels))
        self.fit_time_ = time.perf_counter() - start_time
        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict triage levels for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted triage levels (integers in {1, 2, 3, 4, 5}).

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X_arr)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Probability distribution over triage levels for each
            sample. Columns correspond to ``self.classes_``.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model_.predict_proba(X_arr)

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importances from the fitted model.

        For tree-based models this returns the Gini importance (mean
        decrease in impurity). The keys are ``"feature_0"``,
        ``"feature_1"``, etc., unless a ``feature_names_`` attribute
        has been set.

        Returns
        -------
        dict of str to float
            Feature names mapped to their importance values, sorted
            in descending order of importance.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted.
        ValidationError
            If the underlying estimator does not expose
            ``feature_importances_``.
        """
        self._check_is_fitted()
        if not hasattr(self.model_, "feature_importances_"):
            raise ValidationError(
                message=(
                    "The underlying estimator does not provide "
                    "'feature_importances_'. Use a tree-based model."
                ),
                parameter="base_estimator",
            )

        importances = self.model_.feature_importances_
        n_features = len(importances)

        if hasattr(self, "_feature_names") and self._feature_names is not None:
            names = list(self._feature_names)
        else:
            names = [f"feature_{i}" for i in range(n_features)]

        pairs = sorted(zip(names, importances), key=lambda p: p[1], reverse=True)
        return dict(pairs)

    def set_feature_names(self, names: Sequence[str]) -> "MLTriageClassifier":
        """Assign feature names for interpretability.

        Parameters
        ----------
        names : sequence of str
            Ordered feature names corresponding to the columns of the
            training matrix.

        Returns
        -------
        self
            The classifier instance.
        """
        self._feature_names: List[str] = list(names)
        return self

    # ------------------------------------------------------------------
    # Overridden score method (from ClassifierMixin)
    # ------------------------------------------------------------------

    def score(self, X: ArrayLike, y: ArrayLike, **kwargs: Any) -> float:
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True triage levels.
        **kwargs
            Unused.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        self._check_is_fitted()
        predictions = self.predict(X)
        y_arr = np.asarray(y)
        return float(np.mean(predictions == y_arr))
