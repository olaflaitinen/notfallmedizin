# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Model comparison utilities for benchmarking.

Provides paired statistical comparison of models with cross-validation
and significance testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from sklearn.model_selection import StratifiedKFold, KFold

from notfallmedizin.core.exceptions import ValidationError
from notfallmedizin.core.config import get_config


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a model comparison.

    Attributes
    ----------
    model_names : list of str
    mean_scores : dict
    std_scores : dict
    best_model : str
    pairwise_p_values : dict
    training_times_seconds : dict
    inference_times_seconds : dict
    """

    model_names: List[str] = field(default_factory=list)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    best_model: str = ""
    pairwise_p_values: Dict[str, float] = field(default_factory=dict)
    training_times_seconds: Dict[str, float] = field(default_factory=dict)
    inference_times_seconds: Dict[str, float] = field(default_factory=dict)


class ModelComparison:
    """Compare multiple models using cross-validated evaluation.

    Parameters
    ----------
    n_splits : int, default=5
    metric : str, default='accuracy'
    stratified : bool, default=True
    """

    def __init__(
        self,
        n_splits: int = 5,
        metric: str = "accuracy",
        stratified: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.metric = metric
        self.stratified = stratified

    def compare(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring_fn: Optional[Callable] = None,
    ) -> ComparisonResult:
        """Run cross-validated comparison.

        Parameters
        ----------
        models : dict
            Mapping of model name to estimator object (must have
            ``fit`` and ``predict`` methods).
        X : np.ndarray
        y : np.ndarray
        scoring_fn : callable, optional
            Function(y_true, y_pred) -> float.  Defaults to accuracy.

        Returns
        -------
        ComparisonResult
        """
        if len(models) < 2:
            raise ValidationError("At least 2 models required for comparison.")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        cfg = get_config()
        if self.stratified:
            kf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True,
                random_state=cfg.random_state,
            )
        else:
            kf = KFold(
                n_splits=self.n_splits, shuffle=True,
                random_state=cfg.random_state,
            )

        if scoring_fn is None:
            from sklearn.metrics import accuracy_score
            scoring_fn = accuracy_score

        all_scores: Dict[str, List[float]] = {name: [] for name in models}
        train_times: Dict[str, List[float]] = {name: [] for name in models}
        infer_times: Dict[str, List[float]] = {name: [] for name in models}

        for train_idx, test_idx in kf.split(X_arr, y_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            for name, model in models.items():
                from sklearn.base import clone
                m = clone(model)

                t0 = time.perf_counter()
                m.fit(X_train, y_train)
                train_times[name].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                preds = m.predict(X_test)
                infer_times[name].append(time.perf_counter() - t0)

                score = scoring_fn(y_test, preds)
                all_scores[name].append(score)

        mean_scores = {k: round(float(np.mean(v)), 6) for k, v in all_scores.items()}
        std_scores = {k: round(float(np.std(v, ddof=1)), 6) for k, v in all_scores.items()}

        best = max(mean_scores, key=mean_scores.get)  # type: ignore[arg-type]

        names = list(models.keys())
        pairwise: Dict[str, float] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = np.array(all_scores[names[i]])
                b = np.array(all_scores[names[j]])
                stat, p = sp_stats.wilcoxon(a, b) if self.n_splits >= 5 else sp_stats.ttest_rel(a, b)
                key = f"{names[i]}_vs_{names[j]}"
                pairwise[key] = round(float(p), 6)

        avg_train = {k: round(float(np.mean(v)), 6) for k, v in train_times.items()}
        avg_infer = {k: round(float(np.mean(v)), 6) for k, v in infer_times.items()}

        return ComparisonResult(
            model_names=names,
            mean_scores=mean_scores,
            std_scores=std_scores,
            best_model=best,
            pairwise_p_values=pairwise,
            training_times_seconds=avg_train,
            inference_times_seconds=avg_infer,
        )
