# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Mortality prediction models for the emergency department.

Implements ensemble machine-learning classifiers and the APACHE II predicted
mortality equation for estimating in-hospital and ED mortality risk.

References
----------
Knaus, W. A., Draper, E. A., Wagner, D. P., & Zimmerman, J. E. (1985).
    APACHE II: a severity of disease classification system. Critical Care
    Medicine, 13(10), 818-829.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from notfallmedizin.core.base import ClinicalModel, ClassifierMixin
from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    ValidationError,
)
from notfallmedizin.core.config import get_config


class EDMortalityPredictor(ClinicalModel, ClassifierMixin):
    """Ensemble classifier for emergency department mortality prediction.

    Combines gradient boosting, logistic regression, and random forest
    into a soft-voting ensemble.  Supports post-hoc probability
    calibration via Platt scaling.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of boosting / forest trees.
    calibrated : bool, default=False
        Whether to apply Platt scaling after initial fit.

    Attributes
    ----------
    is_fitted_ : bool
        True after ``fit`` has been called.
    feature_names_ : list of str or None
        Column names seen during ``fit`` when a DataFrame is passed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        calibrated: bool = False,
    ) -> None:
        super().__init__()
        self._name = "EDMortalityPredictor"
        self._description = (
            "Ensemble model for ED in-hospital mortality prediction"
        )
        self.n_estimators = n_estimators
        self.calibrated = calibrated
        self.is_fitted_: bool = False
        self.feature_names_: Optional[List[str]] = None
        self._scaler = StandardScaler()
        self._ensemble: Optional[VotingClassifier] = None
        self._calibrated_model: Optional[CalibratedClassifierCV] = None

    def fit(
        self,
        X: Any,
        y: Any,
    ) -> "EDMortalityPredictor":
        """Fit the ensemble on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary labels (0 = survived, 1 = deceased).

        Returns
        -------
        self
        """
        import pandas as pd

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        if X_arr.ndim != 2:
            raise ValidationError("X must be two-dimensional.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValidationError("X and y must have the same number of rows.")

        cfg = get_config()
        rs = cfg.random_state

        X_scaled = self._scaler.fit_transform(X_arr)

        gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=rs,
        )
        lr = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=rs,
        )
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            random_state=rs,
            n_jobs=cfg.n_jobs,
        )

        self._ensemble = VotingClassifier(
            estimators=[("gb", gb), ("lr", lr), ("rf", rf)],
            voting="soft",
        )
        self._ensemble.fit(X_scaled, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return binary mortality predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        model = self._calibrated_model or self._ensemble
        return model.predict(X_scaled)  # type: ignore[union-attr]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return probability estimates for mortality.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0 = P(survived), column 1 = P(deceased).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        model = self._calibrated_model or self._ensemble
        return model.predict_proba(X_scaled)  # type: ignore[union-attr]

    def calibrate(self, X_cal: Any, y_cal: Any) -> "EDMortalityPredictor":
        """Apply Platt scaling on a held-out calibration set.

        Parameters
        ----------
        X_cal : array-like of shape (n_samples, n_features)
        y_cal : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X_cal, dtype=np.float64))
        y_arr = np.asarray(y_cal, dtype=np.int64)
        self._calibrated_model = CalibratedClassifierCV(
            self._ensemble, method="sigmoid", cv="prefit"
        )
        self._calibrated_model.fit(X_scaled, y_arr)
        return self

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance from the gradient boosting sub-model.

        Returns
        -------
        dict
            Mapping of feature name (or index) to importance value.
        """
        self._check_fitted()
        gb = self._ensemble.named_estimators_["gb"]  # type: ignore[union-attr]
        importances = gb.feature_importances_
        names = self.feature_names_ or [
            f"feature_{i}" for i in range(len(importances))
        ]
        return dict(zip(names, importances.tolist()))

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ModelNotFittedError(
                "EDMortalityPredictor has not been fitted yet."
            )


class APACHE2Mortality:
    """APACHE II predicted mortality calculator.

    Uses the logistic regression equation published by Knaus et al. (1985):
        ln(R / (1 - R)) = -3.517 + 0.146 * APACHE_II + diagnostic_weight

    Parameters
    ----------
    None

    Notes
    -----
    Diagnostic category weights are derived from Knaus et al. (1985),
    Table 7.  A subset of common ED diagnostic categories is included.
    """

    _DIAGNOSTIC_WEIGHTS: Dict[str, float] = {
        "nonoperative_trauma": -1.228,
        "postoperative_cardiovascular": -1.376,
        "postoperative_respiratory": -0.802,
        "nonoperative_cardiovascular": -0.191,
        "nonoperative_respiratory": 0.0,
        "nonoperative_gastrointestinal": 0.501,
        "nonoperative_neurological": -0.759,
        "nonoperative_metabolic_renal": 0.196,
        "postoperative_gastrointestinal": -0.613,
        "postoperative_neurological": -1.150,
        "nonoperative_hematological": 0.891,
        "default": 0.0,
    }

    def calculate(
        self,
        apache_score: int,
        diagnostic_category: str = "default",
    ) -> float:
        """Compute predicted mortality probability from APACHE II score.

        Parameters
        ----------
        apache_score : int
            APACHE II score in the range [0, 71].
        diagnostic_category : str
            Key from the diagnostic weight table.

        Returns
        -------
        float
            Predicted mortality probability in [0, 1].

        Raises
        ------
        ValidationError
            If the score is outside the valid range.
        """
        if not 0 <= apache_score <= 71:
            raise ValidationError(
                f"APACHE II score must be in [0, 71], got {apache_score}."
            )
        weight = self._DIAGNOSTIC_WEIGHTS.get(
            diagnostic_category,
            self._DIAGNOSTIC_WEIGHTS["default"],
        )
        logit = -3.517 + 0.146 * apache_score + weight
        probability = 1.0 / (1.0 + math.exp(-logit))
        return round(probability, 4)
