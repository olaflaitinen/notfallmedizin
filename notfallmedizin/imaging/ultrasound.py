# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Point-of-care ultrasound (POCUS) analysis.

This module provides analysis tools for Focused Assessment with
Sonography for Trauma (FAST / eFAST) exams and echocardiographic
ejection fraction estimation.

Classes
-------
FASTExamAnalyzer
    FAST and eFAST exam analysis with free-fluid detection.
EFFunction
    Ejection fraction calculations (Simpson, Teichholz) and
    classification.

Data classes
------------
FASTResult
    Result for a single FAST exam region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from notfallmedizin.core.base import ClinicalModel
from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    ValidationError,
)

# Lazy torch import
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import torchvision.models as tv_models

    _TORCHVISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    tv_models = None  # type: ignore[assignment]
    _TORCHVISION_AVAILABLE = False


def _require_torch(operation: str = "This operation") -> None:
    """Raise ``ImportError`` when PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{operation} requires PyTorch. "
            f"Install it with: pip install torch torchvision"
        )


# ======================================================================
# Constants
# ======================================================================

FAST_REGIONS: Tuple[str, ...] = (
    "right_upper_quadrant",
    "left_upper_quadrant",
    "suprapubic",
    "subxiphoid",
)

EFAST_REGIONS: Tuple[str, ...] = FAST_REGIONS + (
    "left_lung",
    "right_lung",
)


# ======================================================================
# Data class
# ======================================================================


@dataclass
class FASTResult:
    """Result for a single FAST exam region.

    Attributes
    ----------
    region : str
        Anatomical region examined.
    free_fluid_detected : bool
        Whether free fluid was detected.
    confidence : float
        Model confidence in [0, 1].
    volume_estimate : float
        Rough estimate of free-fluid volume in millilitres.
        ``0.0`` if no fluid detected or estimation is not applicable.
    pneumothorax_detected : bool or None
        Whether pneumothorax was detected (eFAST lung regions only).
        ``None`` for standard FAST regions.
    """

    region: str
    free_fluid_detected: bool
    confidence: float
    volume_estimate: float = 0.0
    pneumothorax_detected: Optional[bool] = None

    def __post_init__(self) -> None:
        all_regions = set(EFAST_REGIONS)
        if self.region not in all_regions:
            raise ValidationError(
                f"Unknown FAST/eFAST region '{self.region}'. "
                f"Supported: {sorted(all_regions)}.",
                parameter="region",
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(
                f"Confidence must be in [0, 1], got {self.confidence}.",
                parameter="confidence",
            )
        if self.volume_estimate < 0.0:
            raise ValidationError(
                f"Volume estimate must be non-negative, "
                f"got {self.volume_estimate}.",
                parameter="volume_estimate",
            )


# ======================================================================
# FAST Exam Analyzer
# ======================================================================


class FASTExamAnalyzer(ClinicalModel):
    """FAST and eFAST exam analyser.

    Classifies point-of-care ultrasound images by FAST region and
    detects the presence of free intraperitoneal/pericardial fluid.
    The extended FAST (eFAST) mode additionally assesses bilateral
    lung fields for pneumothorax (absent lung sliding).

    When PyTorch is available, a backbone CNN is used for
    classification. Without PyTorch the class is importable but
    inference raises ``ImportError``.

    Parameters
    ----------
    backbone_name : str, optional
        Torchvision model name. Default is ``"resnet18"``.
    pretrained : bool, optional
        Use ImageNet-pretrained weights. Default is ``True``.
    threshold : float, optional
        Decision threshold. Default is ``0.5``.
    device : str, optional
        PyTorch device string. Default is ``"cpu"``.
    efast : bool, optional
        If ``True``, enable eFAST mode with pneumothorax assessment
        on lung regions. Default is ``False``.

    Attributes
    ----------
    model_ : torch.nn.Module or None
        The classification backbone, set after fitting or weight
        loading.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        threshold: float = 0.5,
        device: str = "cpu",
        efast: bool = False,
    ) -> None:
        super().__init__(
            description="FAST/eFAST exam free-fluid and pneumothorax detector.",
        )
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.threshold = threshold
        self.device = device
        self.efast = efast
        self.model_: Optional[Any] = None

    @property
    def regions(self) -> Tuple[str, ...]:
        """Active regions depending on eFAST mode."""
        return EFAST_REGIONS if self.efast else FAST_REGIONS

    @property
    def _num_outputs(self) -> int:
        """Number of model output units.

        For each region: 1 unit for free-fluid probability.
        For eFAST lung regions: an additional unit per lung for
        pneumothorax probability.
        """
        n = len(FAST_REGIONS)
        if self.efast:
            n += 2  # left_lung fluid, right_lung fluid
            n += 2  # left_lung pneumothorax, right_lung pneumothorax
        return n

    def _build_model(self) -> Any:
        _require_torch("FASTExamAnalyzer")
        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required. "
                "Install with: pip install torchvision"
            )
        weights = "IMAGENET1K_V1" if self.pretrained else None
        constructor = getattr(tv_models, self.backbone_name, None)
        if constructor is None:
            raise ValidationError(
                f"Unknown backbone '{self.backbone_name}'.",
                parameter="backbone_name",
            )
        model = constructor(weights=weights)

        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, self._num_outputs)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                last = model.classifier[-1]
                if isinstance(last, nn.Linear):
                    model.classifier[-1] = nn.Linear(
                        last.in_features, self._num_outputs
                    )
            elif isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(
                    model.classifier.in_features, self._num_outputs
                )
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(
                model.head.in_features, self._num_outputs
            )

        return model.to(self.device)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> FASTExamAnalyzer:
        """Train the FAST exam classifier.

        Parameters
        ----------
        X : array-like
            Ultrasound images of shape ``(N, C, H, W)`` or
            ``(N, H, W)``.
        y : array-like of shape (N, n_outputs)
            Label matrix with columns corresponding to region-level
            binary indicators (fluid / pneumothorax).
        **kwargs
            ``batch_size`` and ``epochs`` overrides.

        Returns
        -------
        self
        """
        _require_torch("FASTExamAnalyzer.fit")
        if y is None:
            raise ValidationError(
                "Labels (y) required for training.",
                parameter="y",
            )

        images = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.float32)

        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if labels.shape[1] != self._num_outputs:
            raise ValidationError(
                f"Expected {self._num_outputs} label columns, "
                f"got {labels.shape[1]}.",
                parameter="y",
            )

        self.model_ = self._build_model()
        self.model_.train()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(images).to(self.device),
            torch.from_numpy(labels).to(self.device),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs.get("batch_size", 16),
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=kwargs.get("learning_rate", 1e-4),
        )
        criterion = nn.BCEWithLogitsLoss()
        n_epochs = kwargs.get("epochs", 10)

        for _ in range(n_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(bx), by)
                loss.backward()
                optimizer.step()

        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary indicators for each output unit.

        Parameters
        ----------
        X : array-like
            Ultrasound images.

        Returns
        -------
        numpy.ndarray of shape (N, n_outputs)
        """
        proba = self._predict_proba_array(X)
        return (proba >= self.threshold).astype(np.int32)

    def _predict_proba_array(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        _require_torch("FASTExamAnalyzer inference")

        images = np.asarray(X, dtype=np.float32)
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]

        self.model_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(images).to(self.device)
            logits = self.model_(tensor)
            proba = torch.sigmoid(logits).cpu().numpy()

        return proba.astype(np.float64)

    # ------------------------------------------------------------------
    # Region-level interface
    # ------------------------------------------------------------------

    def classify_region(
        self,
        image: np.ndarray,
        region: str,
    ) -> FASTResult:
        """Classify a single ultrasound image for a specific FAST region.

        Parameters
        ----------
        image : numpy.ndarray
            Single ultrasound image of shape ``(C, H, W)`` or
            ``(H, W)``.
        region : str
            Target region name from :data:`FAST_REGIONS` or
            :data:`EFAST_REGIONS`.

        Returns
        -------
        FASTResult
            Classification result for the specified region.

        Raises
        ------
        ValidationError
            If *region* is not recognised.
        ModelNotFittedError
            If the model has not been fitted.
        ImportError
            If PyTorch is not installed.
        """
        allowed = set(self.regions)
        if region not in allowed:
            raise ValidationError(
                f"Region '{region}' not recognised. "
                f"Available: {sorted(allowed)}.",
                parameter="region",
            )

        img = image.astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        proba_row = self._predict_proba_array(img[np.newaxis])[0]

        region_idx = self._region_fluid_index(region)
        fluid_prob = float(proba_row[region_idx])
        fluid_detected = fluid_prob >= self.threshold

        volume_est = 0.0
        if fluid_detected:
            volume_est = self._rough_volume_estimate(fluid_prob)

        ptx_detected: Optional[bool] = None
        if self.efast and region in ("left_lung", "right_lung"):
            ptx_idx = self._region_ptx_index(region)
            ptx_prob = float(proba_row[ptx_idx])
            ptx_detected = ptx_prob >= self.threshold

        return FASTResult(
            region=region,
            free_fluid_detected=fluid_detected,
            confidence=fluid_prob,
            volume_estimate=volume_est,
            pneumothorax_detected=ptx_detected,
        )

    def bilateral_pneumothorax(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> Dict[str, FASTResult]:
        """Assess bilateral pneumothorax from eFAST lung images.

        Parameters
        ----------
        left_image : numpy.ndarray
            Left lung ultrasound image.
        right_image : numpy.ndarray
            Right lung ultrasound image.

        Returns
        -------
        dict
            Keys ``"left_lung"`` and ``"right_lung"`` mapped to
            :class:`FASTResult` instances.

        Raises
        ------
        ValidationError
            If eFAST mode is not enabled.
        """
        if not self.efast:
            raise ValidationError(
                "Bilateral pneumothorax assessment requires eFAST mode. "
                "Set efast=True.",
                parameter="efast",
            )

        return {
            "left_lung": self.classify_region(left_image, "left_lung"),
            "right_lung": self.classify_region(right_image, "right_lung"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _region_fluid_index(self, region: str) -> int:
        """Return the output index for a region's fluid probability."""
        fast_order = list(FAST_REGIONS)
        if region in fast_order:
            return fast_order.index(region)
        if region == "left_lung":
            return len(FAST_REGIONS)
        if region == "right_lung":
            return len(FAST_REGIONS) + 1
        raise ValidationError(
            f"Cannot map region '{region}' to output index.",
            parameter="region",
        )

    def _region_ptx_index(self, region: str) -> int:
        """Return the output index for a region's pneumothorax probability."""
        base = len(FAST_REGIONS) + 2  # after all fluid outputs
        if region == "left_lung":
            return base
        if region == "right_lung":
            return base + 1
        raise ValidationError(
            f"Pneumothorax index not applicable for region '{region}'.",
            parameter="region",
        )

    @staticmethod
    def _rough_volume_estimate(confidence: float) -> float:
        """Heuristic volume estimate (ml) proportional to confidence.

        This is a rough linear mapping for demonstration purposes.
        Clinical volume estimation from ultrasound requires dedicated
        measurement algorithms.
        """
        return float(np.clip(confidence * 500.0, 0.0, 500.0))

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""
        self._check_is_fitted()
        _require_torch("save_weights")
        torch.save(self.model_.state_dict(), path)

    def load_weights(self, path: str) -> FASTExamAnalyzer:
        """Load model weights from disk.

        Returns
        -------
        self
        """
        _require_torch("load_weights")
        self.model_ = self._build_model()
        state = torch.load(path, map_location=self.device)
        self.model_.load_state_dict(state)
        self._set_fitted()
        return self


# ======================================================================
# Ejection Fraction Calculator
# ======================================================================


class EFFunction:
    """Left ventricular ejection fraction (EF) calculator.

    Provides two estimation methods:

    * **Simpson's biplane method**: ``EF = (EDV - ESV) / EDV * 100``
    * **Teichholz formula**: volume derived from M-mode LV internal
      dimensions.

    Also provides an EF classification utility based on standard
    echocardiographic guidelines.

    Examples
    --------
    >>> ef_calc = EFFunction()
    >>> ef_calc.calculate_ef_simpson(120.0, 50.0)
    58.333...
    >>> ef_calc.classify_ef(58.0)
    'normal'
    """

    # ------------------------------------------------------------------
    # Simpson's method
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ef_simpson(
        volumes_edv: float,
        volumes_esv: float,
    ) -> float:
        """Calculate ejection fraction using Simpson's method.

        Parameters
        ----------
        volumes_edv : float
            End-diastolic volume in millilitres.
        volumes_esv : float
            End-systolic volume in millilitres.

        Returns
        -------
        float
            Ejection fraction as a percentage (0--100).

        Raises
        ------
        ValidationError
            If volumes are negative or ESV exceeds EDV.
        """
        if volumes_edv < 0 or volumes_esv < 0:
            raise ValidationError(
                "Volumes must be non-negative. "
                f"Got EDV={volumes_edv}, ESV={volumes_esv}.",
                parameter="volumes_edv",
            )
        if volumes_esv > volumes_edv:
            raise ValidationError(
                f"ESV ({volumes_esv}) cannot exceed EDV ({volumes_edv}).",
                parameter="volumes_esv",
            )
        if volumes_edv < 1e-12:
            raise ValidationError(
                "EDV must be greater than zero for EF calculation.",
                parameter="volumes_edv",
            )

        ef = (volumes_edv - volumes_esv) / volumes_edv * 100.0
        return float(ef)

    # ------------------------------------------------------------------
    # Teichholz formula
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ef_teichholz(
        lvedd: float,
        lvesd: float,
    ) -> float:
        """Calculate ejection fraction using the Teichholz formula.

        The Teichholz formula converts M-mode LV internal diameters to
        volumes and then computes EF::

            V = 7 * D^3 / (2.4 + D)
            EF = (V_ed - V_es) / V_ed * 100

        Parameters
        ----------
        lvedd : float
            Left ventricular end-diastolic diameter in centimetres.
        lvesd : float
            Left ventricular end-systolic diameter in centimetres.

        Returns
        -------
        float
            Ejection fraction as a percentage (0--100).

        Raises
        ------
        ValidationError
            If diameters are non-positive or LVESD exceeds LVEDD.
        """
        if lvedd <= 0 or lvesd <= 0:
            raise ValidationError(
                f"Diameters must be positive. "
                f"Got LVEDD={lvedd}, LVESD={lvesd}.",
                parameter="lvedd",
            )
        if lvesd > lvedd:
            raise ValidationError(
                f"LVESD ({lvesd}) cannot exceed LVEDD ({lvedd}).",
                parameter="lvesd",
            )

        def _teichholz_volume(d: float) -> float:
            return 7.0 * d ** 3 / (2.4 + d)

        edv = _teichholz_volume(lvedd)
        esv = _teichholz_volume(lvesd)

        if edv < 1e-12:
            raise ValidationError(
                "Computed EDV is effectively zero.",
                parameter="lvedd",
            )

        ef = (edv - esv) / edv * 100.0
        return float(ef)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_ef(ef_value: float) -> str:
        """Classify ejection fraction into clinical categories.

        Classification follows standard echocardiographic guidelines:

        * ``"normal"``: EF > 55 %
        * ``"mild_dysfunction"``: 45 % <= EF <= 55 %
        * ``"moderate_dysfunction"``: 30 % <= EF < 45 %
        * ``"severe_dysfunction"``: EF < 30 %

        Parameters
        ----------
        ef_value : float
            Ejection fraction percentage.

        Returns
        -------
        str
            Classification label.

        Raises
        ------
        ValidationError
            If *ef_value* is outside [0, 100].
        """
        if ef_value < 0.0 or ef_value > 100.0:
            raise ValidationError(
                f"EF must be in [0, 100], got {ef_value}.",
                parameter="ef_value",
            )

        if ef_value > 55.0:
            return "normal"
        elif ef_value >= 45.0:
            return "mild_dysfunction"
        elif ef_value >= 30.0:
            return "moderate_dysfunction"
        else:
            return "severe_dysfunction"

    def __repr__(self) -> str:
        return "EFFunction()"
