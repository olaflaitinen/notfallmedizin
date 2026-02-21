# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic dataset generation for benchmarking EM-AI models.

Generates realistic emergency department patient data with configurable
class distributions, feature correlations, and missing-data patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from notfallmedizin.core.config import get_config
from notfallmedizin.core.exceptions import ValidationError


@dataclass
class DatasetSplit:
    """Train / validation / test split.

    Attributes
    ----------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : np.ndarray
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class SyntheticEDDatasetGenerator:
    """Generator for synthetic emergency department datasets.

    Creates patient records with realistic vital signs, demographics,
    lab values, and outcome labels suitable for training and evaluating
    triage, mortality, LOS, and disposition models.

    Parameters
    ----------
    n_samples : int, default=1000
    missing_rate : float, default=0.05
        Fraction of values to mask as NaN.
    """

    VITAL_RANGES: Dict[str, Tuple[float, float, float]] = {
        "heart_rate": (60.0, 90.0, 15.0),
        "systolic_bp": (110.0, 130.0, 20.0),
        "diastolic_bp": (65.0, 80.0, 12.0),
        "respiratory_rate": (14.0, 18.0, 4.0),
        "spo2": (97.0, 99.0, 2.0),
        "temperature": (36.6, 37.2, 0.5),
    }

    LAB_RANGES: Dict[str, Tuple[float, float]] = {
        "wbc": (6.0, 3.0),
        "hemoglobin": (14.0, 2.0),
        "platelets": (250.0, 60.0),
        "creatinine": (1.0, 0.4),
        "lactate": (1.2, 0.8),
        "troponin": (0.01, 0.02),
    }

    def __init__(
        self,
        n_samples: int = 1000,
        missing_rate: float = 0.05,
    ) -> None:
        if n_samples < 10:
            raise ValidationError("n_samples must be at least 10.")
        self.n_samples = n_samples
        self.missing_rate = missing_rate

    def generate_triage_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate a dataset for triage-level prediction (5 classes).

        Returns
        -------
        tuple of (X, y)
        """
        cfg = get_config()
        rng = np.random.default_rng(cfg.random_state)

        df = self._generate_base(rng)
        labels = rng.choice([1, 2, 3, 4, 5], size=self.n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.15])
        return df, labels

    def generate_mortality_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate a dataset for binary mortality prediction.

        Returns
        -------
        tuple of (X, y)
        """
        cfg = get_config()
        rng = np.random.default_rng(cfg.random_state)

        df = self._generate_base(rng)
        prob = 0.02 + 0.03 * (df["heart_rate"].values > 100).astype(float)
        prob += 0.05 * (df["systolic_bp"].values < 90).astype(float)
        prob += 0.01 * (df["age"].values / 100.0)
        prob = np.clip(prob, 0, 1)
        labels = rng.binomial(1, prob)
        return df, labels

    def generate_los_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate a dataset for length-of-stay regression.

        Returns
        -------
        tuple of (X, y)
            y is LOS in hours.
        """
        cfg = get_config()
        rng = np.random.default_rng(cfg.random_state)

        df = self._generate_base(rng)
        base_los = 4.0 + rng.exponential(3.0, size=self.n_samples)
        base_los += 0.05 * df["age"].values
        base_los += 2.0 * (df["lactate"].values > 2.0).astype(float)
        return df, np.maximum(base_los, 0.5)

    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> DatasetSplit:
        """Split data into train / validation / test sets.

        Parameters
        ----------
        X : pd.DataFrame
        y : np.ndarray
        train_frac : float
        val_frac : float

        Returns
        -------
        DatasetSplit
        """
        n = len(X)
        cfg = get_config()
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.permutation(n)

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        return DatasetSplit(
            X_train=X.iloc[train_idx].reset_index(drop=True),
            X_val=X.iloc[val_idx].reset_index(drop=True),
            X_test=X.iloc[test_idx].reset_index(drop=True),
            y_train=y[train_idx],
            y_val=y[val_idx],
            y_test=y[test_idx],
        )

    def _generate_base(self, rng: np.random.Generator) -> pd.DataFrame:
        n = self.n_samples
        data: Dict[str, np.ndarray] = {}

        data["age"] = rng.normal(55, 20, n).clip(1, 100).astype(int)
        data["sex"] = rng.choice([0, 1], n)

        for name, (mean, _, std) in self.VITAL_RANGES.items():
            data[name] = rng.normal(mean, std, n)

        data["spo2"] = np.clip(data["spo2"], 70, 100)
        data["temperature"] = np.clip(data["temperature"], 33, 42)

        for name, (mean, std) in self.LAB_RANGES.items():
            data[name] = np.abs(rng.normal(mean, std, n))

        data["gcs"] = rng.choice(range(3, 16), n, p=self._gcs_probs())

        df = pd.DataFrame(data)

        if self.missing_rate > 0:
            mask = rng.random(df.shape) < self.missing_rate
            mask[:, :2] = False
            df = df.mask(mask)

        return df

    @staticmethod
    def _gcs_probs() -> List[float]:
        probs = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.15, 0.55]
        total = sum(probs)
        return [p / total for p in probs]
