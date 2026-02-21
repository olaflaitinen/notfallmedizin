# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Chest X-ray analysis framework.

This module provides a configurable chest X-ray classification model
with Grad-CAM visualization support. The underlying deep learning
backbone (PyTorch / torchvision) is an optional dependency imported
lazily at runtime.

Classes
-------
ChestXrayClassifier
    Multi-label chest X-ray finding classifier with Grad-CAM support.
XrayFindingResult
    Structured result for a single radiographic finding.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from notfallmedizin.core.base import ClinicalModel
from notfallmedizin.core.exceptions import (
    ModelNotFittedError,
    ValidationError,
)

# Lazy torch imports -- guarded at module level so the rest of the
# module stays importable without PyTorch installed.
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


SUPPORTED_FINDINGS: Tuple[str, ...] = (
    "pneumonia",
    "pneumothorax",
    "pleural_effusion",
    "cardiomegaly",
    "fracture",
    "consolidation",
    "atelectasis",
    "edema",
)


# ======================================================================
# Data class
# ======================================================================


@dataclass
class XrayFindingResult:
    """Structured result for a single radiographic finding.

    Attributes
    ----------
    finding : str
        Name of the radiographic finding (e.g. ``"pneumonia"``).
    probability : float
        Estimated probability in [0, 1].
    confidence_interval : tuple of (float, float)
        Lower and upper bounds of a 95 % confidence interval.
    laterality : str
        Affected side: ``"left"``, ``"right"``, ``"bilateral"``, or
        ``"not_applicable"``.
    """

    finding: str
    probability: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    laterality: str = "not_applicable"

    def __post_init__(self) -> None:
        if self.finding not in SUPPORTED_FINDINGS:
            raise ValidationError(
                f"Unknown finding '{self.finding}'. "
                f"Supported: {SUPPORTED_FINDINGS}.",
                parameter="finding",
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValidationError(
                f"Probability must be in [0, 1], got {self.probability}.",
                parameter="probability",
            )
        valid_lat = ("left", "right", "bilateral", "not_applicable")
        if self.laterality not in valid_lat:
            raise ValidationError(
                f"Laterality must be one of {valid_lat}, "
                f"got '{self.laterality}'.",
                parameter="laterality",
            )


def _require_torch(operation: str = "This operation") -> None:
    """Raise ``ImportError`` when PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{operation} requires PyTorch. "
            f"Install it with: pip install torch torchvision"
        )


# ======================================================================
# Backbone helpers
# ======================================================================


def _build_backbone(
    backbone_name: str,
    num_classes: int,
    pretrained: bool,
) -> Any:
    """Construct a torchvision backbone and replace its classifier head.

    Parameters
    ----------
    backbone_name : str
        Name of the torchvision model (e.g. ``"resnet18"``).
    num_classes : int
        Number of output classes.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    torch.nn.Module
        The modified backbone.
    """
    _require_torch("Building a backbone model")

    if not _TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is required for backbone construction. "
            "Install with: pip install torchvision"
        )

    weights = "IMAGENET1K_V1" if pretrained else None

    constructor = getattr(tv_models, backbone_name, None)
    if constructor is None:
        raise ValidationError(
            f"Unknown backbone '{backbone_name}'. Check torchvision "
            f"documentation for available model names.",
            parameter="backbone_name",
        )

    model = constructor(weights=weights)

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head"):
        if isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)

    return model


# ======================================================================
# Grad-CAM
# ======================================================================


class _GradCAM:
    """Minimal Grad-CAM implementation for a single target layer.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model.
    target_layer : torch.nn.Module
        Convolutional layer to extract gradients from.
    """

    def __init__(self, model: Any, target_layer: Any) -> None:
        _require_torch("GradCAM")
        self._model = model
        self._target_layer = target_layer
        self._gradients: Optional[Any] = None
        self._activations: Optional[Any] = None
        self._hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: Any, _input: Any, output: Any) -> None:
            self._activations = output.detach()

        def backward_hook(_module: Any, _grad_in: Any, grad_out: Any) -> None:
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            self._target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self._target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(
        self,
        input_tensor: Any,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed input tensor of shape ``(1, C, H, W)``.
        target_class : int or None, optional
            Class index to visualize. If ``None``, the predicted class
            is used.

        Returns
        -------
        numpy.ndarray
            2-D heatmap of shape ``(H, W)`` with values in [0, 1].
        """
        self._model.eval()
        output = self._model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self._model.zero_grad()
        score = output[0, target_class]
        score.backward()

        gradients = self._gradients
        activations = self._activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float64)

    def remove_hooks(self) -> None:
        """Remove all registered forward/backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ======================================================================
# Chest X-ray Classifier
# ======================================================================


class ChestXrayClassifier(ClinicalModel):
    """Multi-label chest X-ray finding classifier.

    Wraps a configurable torchvision backbone CNN and exposes a
    scikit-learn-style ``fit`` / ``predict`` interface together with
    Grad-CAM-based visual explanations.

    Parameters
    ----------
    backbone_name : str, optional
        Torchvision model name. Default is ``"resnet18"``.
    pretrained : bool, optional
        Load ImageNet-pretrained weights. Default is ``True``.
    findings : tuple of str, optional
        Target findings to classify. Default is
        :data:`SUPPORTED_FINDINGS`.
    threshold : float, optional
        Decision threshold for converting probabilities to binary
        predictions. Default is ``0.5``.
    device : str, optional
        PyTorch device string. Default is ``"cpu"``.
    learning_rate : float, optional
        Learning rate for the Adam optimizer during ``fit``. Default
        is ``1e-4``.
    epochs : int, optional
        Number of training epochs in ``fit``. Default is ``10``.

    Attributes
    ----------
    model_ : torch.nn.Module or None
        The backbone model. Set after :meth:`fit` or
        :meth:`load_weights`.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        findings: Tuple[str, ...] = SUPPORTED_FINDINGS,
        threshold: float = 0.5,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        epochs: int = 10,
    ) -> None:
        super().__init__(
            description="Multi-label chest X-ray classifier.",
        )
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.findings = findings
        self.threshold = threshold
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_: Optional[Any] = None
        self._gradcam: Optional[_GradCAM] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Instantiate the backbone and move it to the target device."""
        _require_torch("ChestXrayClassifier")
        model = _build_backbone(
            self.backbone_name,
            num_classes=len(self.findings),
            pretrained=self.pretrained,
        )
        model = model.to(self.device)
        return model

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> ChestXrayClassifier:
        """Train the classifier on labelled chest X-ray images.

        Parameters
        ----------
        X : array-like
            Training images. Accepted formats:

            * numpy array of shape ``(N, C, H, W)`` or ``(N, H, W)``
            * list of numpy arrays (each ``(C, H, W)`` or ``(H, W)``)

            If PyTorch is available, images are converted to tensors
            internally.
        y : array-like of shape (N, n_findings), optional
            Binary label matrix. Each row is a multi-hot vector over
            :attr:`findings`.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        self
            The fitted classifier.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValidationError
            If *y* is ``None`` or has the wrong shape.
        """
        _require_torch("ChestXrayClassifier.fit")

        if y is None:
            raise ValidationError(
                "Labels (y) are required for supervised training.",
                parameter="y",
            )

        images = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.float32)

        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        if labels.shape[1] != len(self.findings):
            raise ValidationError(
                f"Label matrix must have {len(self.findings)} columns "
                f"(one per finding), got {labels.shape[1]}.",
                parameter="y",
            )

        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]

        self.model_ = self._build_model()
        self.model_.train()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(images).to(self.device),
            torch.from_numpy(labels).to(self.device),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=kwargs.get("batch_size", 16), shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
        )
        criterion = nn.BCEWithLogitsLoss()

        for _epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self._set_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary findings for the input images.

        Parameters
        ----------
        X : array-like
            Images with the same format accepted by :meth:`fit`.

        Returns
        -------
        numpy.ndarray of shape (N, n_findings)
            Binary prediction matrix.
        """
        proba = self._predict_proba_array(X)
        return (proba >= self.threshold).astype(np.int32)

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> List[Dict[str, float]]:
        """Predict finding probabilities for each input image.

        Parameters
        ----------
        X : array-like
            Images with the same format accepted by :meth:`fit`.

        Returns
        -------
        list of dict
            One dictionary per image mapping finding names to
            estimated probabilities.
        """
        proba = self._predict_proba_array(X)
        results: List[Dict[str, float]] = []
        for row in proba:
            results.append(
                {name: float(p) for name, p in zip(self.findings, row)}
            )
        return results

    def _predict_proba_array(self, X: ArrayLike) -> np.ndarray:
        """Return probability matrix ``(N, n_findings)``."""
        self._check_is_fitted()
        _require_torch("ChestXrayClassifier.predict_proba")

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
    # Grad-CAM
    # ------------------------------------------------------------------

    def _get_target_layer(self) -> Any:
        """Heuristically select the last convolutional layer."""
        _require_torch("GradCAM target layer selection")
        model = self.model_

        if hasattr(model, "layer4"):
            return model.layer4[-1]
        if hasattr(model, "features"):
            for layer in reversed(list(model.features.children())):
                if isinstance(layer, nn.Conv2d):
                    return layer
                if hasattr(layer, "children"):
                    for sub in reversed(list(layer.children())):
                        if isinstance(sub, nn.Conv2d):
                            return sub

        raise ValidationError(
            f"Could not automatically locate a target convolutional "
            f"layer in backbone '{self.backbone_name}'. Provide the "
            f"layer manually.",
            parameter="backbone_name",
        )

    def generate_heatmap(
        self,
        image: np.ndarray,
        target_finding: Optional[str] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        image : numpy.ndarray
            Single image of shape ``(C, H, W)`` or ``(H, W)``.
        target_finding : str or None, optional
            Finding to visualize. If ``None``, the finding with the
            highest predicted probability is used.

        Returns
        -------
        numpy.ndarray
            2-D heatmap of shape ``(H, W)`` normalized to [0, 1].

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.
        ImportError
            If PyTorch is not installed.
        """
        self._check_is_fitted()
        _require_torch("generate_heatmap")

        img = image.astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        input_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        target_class: Optional[int] = None
        if target_finding is not None:
            if target_finding not in self.findings:
                raise ValidationError(
                    f"Unknown finding '{target_finding}'. "
                    f"Available: {self.findings}.",
                    parameter="target_finding",
                )
            target_class = self.findings.index(target_finding)

        target_layer = self._get_target_layer()
        gradcam = _GradCAM(self.model_, target_layer)
        try:
            heatmap = gradcam.generate(input_tensor, target_class)
        finally:
            gradcam.remove_hooks()

        from scipy.ndimage import zoom as ndi_zoom

        spatial_h, spatial_w = image.shape[-2], image.shape[-1]
        if heatmap.shape != (spatial_h, spatial_w):
            zh = spatial_h / heatmap.shape[0]
            zw = spatial_w / heatmap.shape[1]
            heatmap = ndi_zoom(heatmap, (zh, zw), order=1)

        return heatmap

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """Save model weights to disk.

        Parameters
        ----------
        path : str
            File path for the saved state dict.
        """
        self._check_is_fitted()
        _require_torch("save_weights")
        torch.save(self.model_.state_dict(), path)

    def load_weights(self, path: str) -> ChestXrayClassifier:
        """Load model weights from disk.

        Parameters
        ----------
        path : str
            Path to a state dict file previously saved with
            :meth:`save_weights`.

        Returns
        -------
        self
            The classifier with loaded weights.
        """
        _require_torch("load_weights")
        self.model_ = self._build_model()
        state = torch.load(path, map_location=self.device)
        self.model_.load_state_dict(state)
        self._set_fitted()
        return self
