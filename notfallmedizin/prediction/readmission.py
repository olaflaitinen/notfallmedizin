# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Thirty-day readmission prediction models.

Implements the LACE index and a gradient-boosting classifier for
predicting unplanned hospital readmissions within 30 days of discharge.

References
----------
van Walraven, C., et al. (2010). Derivation and validation of an index to
    predict early death or unplanned readmission after discharge from
    hospital to the community. CMAJ, 182(6), 551-557.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from notfallmedizin.core.base import ClinicalModel, ClassifierMixin
from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    ValidationError,
)
from notfallmedizin.core.config import get_config


@dataclass(frozen=True)
class LACEResult:
    """Result of a LACE index calculation.

    Attributes
    ----------
    total_score : int
    components : dict
    risk_category : str
    expected_readmission_rate : float
    """

    total_score: int
    components: Dict[str, int] = field(default_factory=dict)
    risk_category: str = ""
    expected_readmission_rate: float = 0.0


class LACEIndex:
    """LACE index for 30-day readmission risk stratification.

    Components
    ----------
    L : Length of stay (days)
    A : Acuity of admission
    C : Charlson comorbidity index
    E : Emergency department visits in prior 6 months

    Scoring Tables (van Walraven et al., 2010)
    -------------------------------------------
    L:  1d=1, 2d=2, 3d=3, 4-6d=4, 7-13d=5, >=14d=7
    A:  acute/emergent=3, elective=0
    C:  0=0, 1=1, 2=2, 3=3, >=4=5
    E:  0=0, 1=1, 2=2, 3=3, >=4=4
    """

    def calculate(
        self,
        length_of_stay_days: int,
        is_acute_admission: bool,
        charlson_index: int,
        ed_visits_6months: int,
    ) -> LACEResult:
        """Compute the LACE index.

        Parameters
        ----------
        length_of_stay_days : int
            Hospital length of stay in days (>=1).
        is_acute_admission : bool
            True if the admission was acute or emergent.
        charlson_index : int
            Charlson comorbidity index (>=0).
        ed_visits_6months : int
            Number of ED visits in the prior 6 months (>=0).

        Returns
        -------
        LACEResult
        """
        if length_of_stay_days < 1:
            raise ValidationError("Length of stay must be at least 1 day.")
        if charlson_index < 0:
            raise ValidationError("Charlson index must be non-negative.")
        if ed_visits_6months < 0:
            raise ValidationError("ED visits count must be non-negative.")

        l_score = self._score_los(length_of_stay_days)
        a_score = 3 if is_acute_admission else 0
        c_score = self._score_charlson(charlson_index)
        e_score = min(ed_visits_6months, 4)

        total = l_score + a_score + c_score + e_score

        if total <= 4:
            category = "low"
            rate = 0.02
        elif total <= 9:
            category = "moderate"
            rate = 0.08
        else:
            category = "high"
            rate = 0.21

        return LACEResult(
            total_score=total,
            components={
                "L_length_of_stay": l_score,
                "A_acuity": a_score,
                "C_comorbidity": c_score,
                "E_ed_visits": e_score,
            },
            risk_category=category,
            expected_readmission_rate=rate,
        )

    @staticmethod
    def _score_los(days: int) -> int:
        if days <= 3:
            return days
        if days <= 6:
            return 4
        if days <= 13:
            return 5
        return 7

    @staticmethod
    def _score_charlson(index: int) -> int:
        if index <= 3:
            return index
        return 5


class ReadmissionPredictor(ClinicalModel, ClassifierMixin):
    """Gradient-boosting classifier for 30-day unplanned readmission.

    Handles class imbalance through ``class_weight='balanced'`` support
    in the underlying model via sample weighting.

    Parameters
    ----------
    n_estimators : int, default=300
    max_depth : int, default=5
    learning_rate : float, default=0.05
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self._name = "ReadmissionPredictor"
        self._description = "30-day unplanned readmission predictor"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.is_fitted_: bool = False
        self.feature_names_: Optional[List[str]] = None
        self._scaler = StandardScaler()
        self._model: Optional[GradientBoostingClassifier] = None

    def fit(self, X: Any, y: Any) -> "ReadmissionPredictor":
        """Fit the readmission model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary labels (0 = not readmitted, 1 = readmitted).

        Returns
        -------
        self
        """
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        X_scaled = self._scaler.fit_transform(X_arr)

        n_pos = int(y_arr.sum())
        n_neg = len(y_arr) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValidationError("y must contain both positive and negative samples.")

        sample_weight = np.where(
            y_arr == 1,
            len(y_arr) / (2.0 * n_pos),
            len(y_arr) / (2.0 * n_neg),
        )

        cfg = get_config()
        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            random_state=cfg.random_state,
        )
        self._model.fit(X_scaled, y_arr, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict readmission labels."""
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict(X_scaled)  # type: ignore[union-attr]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict readmission probabilities."""
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict_proba(X_scaled)  # type: ignore[union-attr]

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance mapping."""
        self._check_fitted()
        importances = self._model.feature_importances_  # type: ignore[union-attr]
        names = self.feature_names_ or [
            f"feature_{i}" for i in range(len(importances))
        ]
        return dict(zip(names, importances.tolist()))

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ModelNotFittedError("ReadmissionPredictor has not been fitted.")
