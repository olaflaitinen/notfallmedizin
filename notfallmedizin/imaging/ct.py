# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""CT scan analysis framework.

This module provides tools for analysing non-contrast and contrast-
enhanced head CT scans in the emergency setting, focusing on
intracranial hemorrhage detection, stroke-type classification,
hemorrhage volume estimation, and midline shift measurement.

Classes
-------
CTAnalyzer
    Hemorrhage detection and stroke classification model.
MiddlelineShiftCalculator
    Midline shift estimation from axial CT slices.

Enumerations
------------
HemorrhageType
    Types of intracranial hemorrhage.

Data classes
------------
CTFinding
    Structured CT analysis result.
"""

from __future__ import annotations

import enum
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
# Enumerations & data classes
# ======================================================================


class HemorrhageType(enum.Enum):
    """Types of intracranial hemorrhage."""

    SUBDURAL = "subdural"
    EPIDURAL = "epidural"
    SUBARACHNOID = "subarachnoid"
    INTRAPARENCHYMAL = "intraparenchymal"
    INTRAVENTRICULAR = "intraventricular"


class StrokeType(enum.Enum):
    """Stroke classification categories."""

    ISCHEMIC = "ischemic"
    HEMORRHAGIC = "hemorrhagic"
    UNDETERMINED = "undetermined"


@dataclass
class CTFinding:
    """Structured result from CT analysis.

    Attributes
    ----------
    finding_type : str
        Type of finding (e.g. ``"subdural_hemorrhage"``).
    location : str
        Anatomical location of the finding (e.g. ``"right_frontal"``).
    probability : float
        Confidence probability in [0, 1].
    volume_ml : float
        Estimated volume in millilitres. ``0.0`` if not applicable.
    metadata : dict
        Additional finding-specific key-value metadata.
    """

    finding_type: str
    location: str
    probability: float
    volume_ml: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValidationError(
                f"Probability must be in [0, 1], got {self.probability}.",
                parameter="probability",
            )
        if self.volume_ml < 0.0:
            raise ValidationError(
                f"Volume must be non-negative, got {self.volume_ml}.",
                parameter="volume_ml",
            )


# ======================================================================
# Volume estimation
# ======================================================================


def calculate_hemorrhage_volume(
    segmentation_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
) -> float:
    """Estimate hemorrhage volume using the ABC/2 method.

    The ABC/2 formula approximates the volume of an ellipsoidal lesion
    as ``V = A * B * C / 2`` where A, B, and C are the maximal
    extents of the hemorrhage along the three principal axes measured in
    centimetres.

    Parameters
    ----------
    segmentation_mask : numpy.ndarray
        Binary 3-D mask (depth, height, width) where non-zero voxels
        represent hemorrhage.
    voxel_spacing : tuple of (float, float, float)
        Physical spacing ``(dz, dy, dx)`` in millimetres per voxel
        along each axis.

    Returns
    -------
    float
        Estimated volume in millilitres (equivalent to cm^3).

    Raises
    ------
    ValidationError
        If the mask is not 3-D or the spacing tuple has the wrong
        length.
    """
    if segmentation_mask.ndim != 3:
        raise ValidationError(
            f"Segmentation mask must be 3-D, got {segmentation_mask.ndim}-D.",
            parameter="segmentation_mask",
        )
    if len(voxel_spacing) != 3:
        raise ValidationError(
            f"voxel_spacing must have 3 elements, got {len(voxel_spacing)}.",
            parameter="voxel_spacing",
        )

    binary = (segmentation_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0

    nonzero = np.argwhere(binary)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)

    dz, dy, dx = voxel_spacing
    a_cm = (maxs[2] - mins[2] + 1) * dx / 10.0
    b_cm = (maxs[1] - mins[1] + 1) * dy / 10.0
    c_cm = (maxs[0] - mins[0] + 1) * dz / 10.0

    volume_ml = a_cm * b_cm * c_cm / 2.0
    return float(volume_ml)


# ======================================================================
# CT Analyzer
# ======================================================================


class CTAnalyzer(ClinicalModel):
    """Intracranial hemorrhage detection and stroke classification.

    Provides a framework for detecting five hemorrhage subtypes and
    classifying stroke as ischemic or hemorrhagic. When PyTorch is
    available, a configurable backbone CNN is used. Without PyTorch,
    the class remains importable but inference methods raise
    ``ImportError``.

    Parameters
    ----------
    backbone_name : str, optional
        Torchvision model name used as the feature extractor. Default
        is ``"resnet18"``.
    pretrained : bool, optional
        Load ImageNet-pretrained weights. Default is ``True``.
    hemorrhage_types : tuple of HemorrhageType, optional
        Hemorrhage subtypes the model is trained to detect. Default
        is all five types.
    threshold : float, optional
        Decision threshold for binary classification. Default is
        ``0.5``.
    device : str, optional
        PyTorch device string. Default is ``"cpu"``.

    Attributes
    ----------
    model_ : torch.nn.Module or None
        The backbone after fitting or weight loading.
    """

    _ALL_HEMORRHAGE_TYPES: Tuple[HemorrhageType, ...] = tuple(HemorrhageType)

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        hemorrhage_types: Optional[Tuple[HemorrhageType, ...]] = None,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            description=(
                "Intracranial hemorrhage detection and stroke "
                "classification from head CT."
            ),
        )
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.hemorrhage_types = (
            hemorrhage_types
            if hemorrhage_types is not None
            else self._ALL_HEMORRHAGE_TYPES
        )
        self.threshold = threshold
        self.device = device
        # n_classes: one per hemorrhage subtype + stroke_ischemic + stroke_hemorrhagic
        self._num_classes = len(self.hemorrhage_types) + 2
        self.model_: Optional[Any] = None

    @property
    def class_names(self) -> List[str]:
        """Ordered list of output class names."""
        names = [ht.value for ht in self.hemorrhage_types]
        names.extend(["stroke_ischemic", "stroke_hemorrhagic"])
        return names

    def _build_model(self) -> Any:
        _require_torch("CTAnalyzer")
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
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, self._num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                last = model.classifier[-1]
                if isinstance(last, nn.Linear):
                    model.classifier[-1] = nn.Linear(
                        last.in_features, self._num_classes
                    )
            elif isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(
                    model.classifier.in_features, self._num_classes
                )
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(
                model.head.in_features, self._num_classes
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
    ) -> CTAnalyzer:
        """Train the hemorrhage/stroke classifier.

        Parameters
        ----------
        X : array-like
            Training CT images of shape ``(N, C, H, W)`` or
            ``(N, H, W)``.
        y : array-like of shape (N, n_classes)
            Multi-hot label matrix. Columns correspond to
            :attr:`class_names`.
        **kwargs
            ``batch_size`` (int) and ``epochs`` (int) may be overridden
            here.

        Returns
        -------
        self
            The fitted analyzer.
        """
        _require_torch("CTAnalyzer.fit")

        if y is None:
            raise ValidationError(
                "Labels (y) are required for supervised training.",
                parameter="y",
            )

        images = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.float32)

        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]

        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if labels.shape[1] != self._num_classes:
            raise ValidationError(
                f"Expected {self._num_classes} label columns, "
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

        for _epoch in range(n_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model_(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary labels for each hemorrhage type and stroke.

        Parameters
        ----------
        X : array-like
            CT images in the same format accepted by :meth:`fit`.

        Returns
        -------
        numpy.ndarray of shape (N, n_classes)
            Binary predictions.
        """
        proba = self._predict_proba_array(X)
        return (proba >= self.threshold).astype(np.int32)

    def predict_proba(self, X: ArrayLike) -> List[Dict[str, float]]:
        """Predict class probabilities for each image.

        Parameters
        ----------
        X : array-like
            CT images.

        Returns
        -------
        list of dict
            One dictionary per image mapping class names to
            probabilities.
        """
        proba = self._predict_proba_array(X)
        results: List[Dict[str, float]] = []
        for row in proba:
            results.append(
                {name: float(p) for name, p in zip(self.class_names, row)}
            )
        return results

    def _predict_proba_array(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        _require_torch("CTAnalyzer.predict_proba")

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
    # Hemorrhage detection helpers
    # ------------------------------------------------------------------

    def detect_hemorrhages(
        self,
        X: ArrayLike,
    ) -> List[List[CTFinding]]:
        """Detect hemorrhage subtypes and return structured findings.

        Parameters
        ----------
        X : array-like
            CT images.

        Returns
        -------
        list of list of CTFinding
            For each image, a list of findings that exceed the
            decision threshold.
        """
        proba_dicts = self.predict_proba(X)
        all_findings: List[List[CTFinding]] = []

        for pdict in proba_dicts:
            findings: List[CTFinding] = []
            for ht in self.hemorrhage_types:
                p = pdict.get(ht.value, 0.0)
                if p >= self.threshold:
                    findings.append(
                        CTFinding(
                            finding_type=ht.value,
                            location="intracranial",
                            probability=p,
                        )
                    )
            all_findings.append(findings)

        return all_findings

    def classify_stroke(
        self,
        X: ArrayLike,
    ) -> List[StrokeType]:
        """Classify stroke type for each image.

        Parameters
        ----------
        X : array-like
            CT images.

        Returns
        -------
        list of StrokeType
            Predicted stroke type per image.
        """
        proba_dicts = self.predict_proba(X)
        results: List[StrokeType] = []

        for pdict in proba_dicts:
            p_isch = pdict.get("stroke_ischemic", 0.0)
            p_hemo = pdict.get("stroke_hemorrhagic", 0.0)

            if p_isch < self.threshold and p_hemo < self.threshold:
                results.append(StrokeType.UNDETERMINED)
            elif p_isch >= p_hemo:
                results.append(StrokeType.ISCHEMIC)
            else:
                results.append(StrokeType.HEMORRHAGIC)

        return results

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""
        self._check_is_fitted()
        _require_torch("save_weights")
        torch.save(self.model_.state_dict(), path)

    def load_weights(self, path: str) -> CTAnalyzer:
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
# Midline Shift Calculator
# ======================================================================


class MiddlelineShiftCalculator:
    """Estimate midline shift from axial head CT images.

    Midline shift is the displacement (in millimetres) of the septum
    pellucidum or falx cerebri from the ideal midline. This
    implementation uses a symmetry-based heuristic on axial slices:

    1. The ideal midline is defined as the vertical centre of the
       image.
    2. A weighted centre-of-mass is computed for the high-intensity
       region on each side.
    3. The shift is the horizontal displacement of the overall
       centre-of-mass from the ideal midline, converted to physical
       units using the provided pixel spacing.

    For production use, replace this heuristic with a trained
    segmentation model.

    Parameters
    ----------
    intensity_threshold : float, optional
        Hounsfield-unit threshold above which tissue is considered
        high-density (potential mass effect). Default is ``40.0``.
    """

    def __init__(self, intensity_threshold: float = 40.0) -> None:
        self.intensity_threshold = intensity_threshold

    def calculate(
        self,
        ct_image: np.ndarray,
        pixel_spacing: float = 0.5,
    ) -> float:
        """Estimate midline shift in millimetres.

        Parameters
        ----------
        ct_image : numpy.ndarray
            2-D axial CT slice in Hounsfield units, shape ``(H, W)``.
        pixel_spacing : float, optional
            Physical size of one pixel in millimetres. Default is
            ``0.5`` mm.

        Returns
        -------
        float
            Estimated midline shift in millimetres. Positive values
            indicate rightward displacement; negative values indicate
            leftward displacement.

        Raises
        ------
        ValidationError
            If the input is not a 2-D array.
        """
        if ct_image.ndim != 2:
            raise ValidationError(
                f"ct_image must be a 2-D array, got {ct_image.ndim}-D.",
                parameter="ct_image",
            )

        h, w = ct_image.shape
        ideal_midline = w / 2.0

        mask = ct_image > self.intensity_threshold
        if not mask.any():
            return 0.0

        cols = np.arange(w)
        col_weights = mask.sum(axis=0).astype(np.float64)
        total = col_weights.sum()

        if total < 1e-12:
            return 0.0

        com_x = float(np.dot(col_weights, cols) / total)
        shift_pixels = com_x - ideal_midline
        shift_mm = shift_pixels * pixel_spacing

        return float(shift_mm)

    def calculate_3d(
        self,
        ct_volume: np.ndarray,
        pixel_spacing: float = 0.5,
    ) -> float:
        """Estimate midline shift from a 3-D CT volume.

        Computes the shift on every axial slice and returns the maximum
        absolute shift value, preserving sign.

        Parameters
        ----------
        ct_volume : numpy.ndarray
            3-D CT volume of shape ``(D, H, W)``.
        pixel_spacing : float, optional
            Physical pixel size in millimetres. Default is ``0.5``.

        Returns
        -------
        float
            Maximum absolute midline shift in millimetres.

        Raises
        ------
        ValidationError
            If the volume is not 3-D.
        """
        if ct_volume.ndim != 3:
            raise ValidationError(
                f"ct_volume must be 3-D, got {ct_volume.ndim}-D.",
                parameter="ct_volume",
            )

        shifts = [
            self.calculate(ct_volume[i], pixel_spacing)
            for i in range(ct_volume.shape[0])
        ]

        if not shifts:
            return 0.0

        abs_shifts = [abs(s) for s in shifts]
        max_idx = int(np.argmax(abs_shifts))
        return shifts[max_idx]

    def __repr__(self) -> str:
        return (
            f"MiddlelineShiftCalculator("
            f"intensity_threshold={self.intensity_threshold!r})"
        )
