# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Clinical deterioration prediction.

Provides the Modified Early Warning Score (MEWS) and a machine-learning
model for predicting ICU transfer or rapid-response-team activation
within 6 to 24 hours.

References
----------
Subbe, C. P., Kruger, M., Rutherford, P., & Gemmel, L. (2001). Validation
    of a modified Early Warning Score in medical admissions. QJM, 94(10),
    521-526.
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
class MEWSResult:
    """Result of a Modified Early Warning Score calculation.

    Attributes
    ----------
    total_score : int
    component_scores : dict
    risk_level : str
    escalation_recommendation : str
    """

    total_score: int
    component_scores: Dict[str, int] = field(default_factory=dict)
    risk_level: str = ""
    escalation_recommendation: str = ""


class EarlyWarningScorePredictor:
    """Modified Early Warning Score (MEWS) calculator.

    Scoring (Subbe et al., 2001)
    ----------------------------
    SBP: <=70->3, 71-80->2, 81-100->1, 101-199->0, >=200->2
    HR:  <40->2, 40-50->1, 51-100->0, 101-110->1, 111-129->2, >=130->3
    RR:  <9->2, 9-14->0, 15-20->1, 21-29->2, >=30->3
    Temp: <35->2, 35-38.4->0, >=38.5->2
    AVPU: A->0, V->1, P->2, U->3
    """

    def calculate(
        self,
        heart_rate: float,
        systolic_bp: float,
        respiratory_rate: float,
        temperature: float,
        avpu: str = "A",
    ) -> MEWSResult:
        """Calculate the Modified Early Warning Score.

        Parameters
        ----------
        heart_rate : float
        systolic_bp : float
        respiratory_rate : float
        temperature : float
            Degrees Celsius.
        avpu : str
            One of 'A' (alert), 'V' (voice), 'P' (pain), 'U' (unresponsive).

        Returns
        -------
        MEWSResult
        """
        avpu = avpu.upper()
        if avpu not in ("A", "V", "P", "U"):
            raise ValidationError(f"AVPU must be one of A, V, P, U; got '{avpu}'.")

        sbp_score = self._score_sbp(systolic_bp)
        hr_score = self._score_hr(heart_rate)
        rr_score = self._score_rr(respiratory_rate)
        temp_score = self._score_temp(temperature)
        avpu_score = {"A": 0, "V": 1, "P": 2, "U": 3}[avpu]

        total = sbp_score + hr_score + rr_score + temp_score + avpu_score

        if total <= 2:
            risk = "low"
            rec = "Continue routine monitoring."
        elif total <= 4:
            risk = "medium"
            rec = "Increase monitoring frequency to every 1-2 hours."
        elif total <= 6:
            risk = "high"
            rec = "Urgent review by physician; consider ICU consultation."
        else:
            risk = "critical"
            rec = "Immediate physician review; activate rapid response team."

        return MEWSResult(
            total_score=total,
            component_scores={
                "systolic_bp": sbp_score,
                "heart_rate": hr_score,
                "respiratory_rate": rr_score,
                "temperature": temp_score,
                "avpu": avpu_score,
            },
            risk_level=risk,
            escalation_recommendation=rec,
        )

    @staticmethod
    def _score_sbp(sbp: float) -> int:
        if sbp <= 70:
            return 3
        if sbp <= 80:
            return 2
        if sbp <= 100:
            return 1
        if sbp <= 199:
            return 0
        return 2

    @staticmethod
    def _score_hr(hr: float) -> int:
        if hr < 40:
            return 2
        if hr <= 50:
            return 1
        if hr <= 100:
            return 0
        if hr <= 110:
            return 1
        if hr <= 129:
            return 2
        return 3

    @staticmethod
    def _score_rr(rr: float) -> int:
        if rr < 9:
            return 2
        if rr <= 14:
            return 0
        if rr <= 20:
            return 1
        if rr <= 29:
            return 2
        return 3

    @staticmethod
    def _score_temp(temp: float) -> int:
        if temp < 35.0:
            return 2
        if temp <= 38.4:
            return 0
        return 2


class DeteriorationPredictor(ClinicalModel, ClassifierMixin):
    """Gradient-boosting model for clinical deterioration prediction.

    Predicts ICU transfer or rapid-response-team activation within a
    configurable time horizon (default 12 hours).

    Parameters
    ----------
    n_estimators : int, default=300
    max_depth : int, default=5
    horizon_hours : int, default=12
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        horizon_hours: int = 12,
    ) -> None:
        super().__init__()
        self._name = "DeteriorationPredictor"
        self._description = (
            "Predicts clinical deterioration requiring ICU transfer"
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.horizon_hours = horizon_hours
        self.is_fitted_: bool = False
        self.feature_names_: Optional[List[str]] = None
        self._scaler = StandardScaler()
        self._model: Optional[GradientBoostingClassifier] = None

    def fit(self, X: Any, y: Any) -> "DeteriorationPredictor":
        """Fit the deterioration model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Clinical features including vitals and trends.
        y : array-like of shape (n_samples,)
            Binary labels (0 = stable, 1 = deteriorated).

        Returns
        -------
        self
        """
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        X_scaled = self._scaler.fit_transform(X_arr)

        cfg = get_config()
        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=cfg.random_state,
        )
        self._model.fit(X_scaled, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict deterioration labels."""
        self._check_fitted()
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict(X_scaled)  # type: ignore[union-attr]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict deterioration probabilities."""
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
            raise ModelNotFittedError(
                "DeteriorationPredictor has not been fitted."
            )
