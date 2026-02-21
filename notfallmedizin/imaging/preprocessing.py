# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Medical image preprocessing utilities.

This module provides numpy-based preprocessing operations commonly
required in emergency medicine imaging pipelines: intensity
normalization, spatial resizing, CT windowing with clinical presets,
denoising filters, and data augmentation.

Classes
-------
ImagePreprocessor
    Stateless collection of image preprocessing methods.

Constants
---------
CT_WINDOW_PRESETS
    Dictionary of standard CT window presets mapping preset names to
    ``(center, width)`` tuples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

from notfallmedizin.core.exceptions import ValidationError


CT_WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "lung": (-600.0, 1500.0),
    "bone": (300.0, 1500.0),
    "brain": (40.0, 80.0),
    "soft_tissue": (50.0, 400.0),
    "liver": (60.0, 160.0),
    "mediastinum": (50.0, 350.0),
}


class ImagePreprocessor:
    """Stateless medical image preprocessing toolkit.

    All methods are class-level utilities that operate on numpy arrays
    and return new arrays without mutating the input.

    Examples
    --------
    >>> import numpy as np
    >>> proc = ImagePreprocessor()
    >>> img = np.random.rand(512, 512).astype(np.float32)
    >>> normed = proc.normalize(img, method="zscore")
    >>> windowed = proc.windowing(img * 2000 - 1000, preset="lung")
    """

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(
        image: np.ndarray,
        method: str = "minmax",
        *,
        clip_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Normalize pixel intensities.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of arbitrary dimensionality.
        method : str, optional
            Normalization strategy. One of ``"minmax"``, ``"zscore"``,
            or ``"clahe"``. Default is ``"minmax"``.
        clip_range : tuple of (float, float) or None, optional
            If provided, clip the image to this ``(low, high)`` range
            before normalization.

        Returns
        -------
        numpy.ndarray
            Normalized image as float64.

        Raises
        ------
        ValidationError
            If *method* is not one of the supported values.
        """
        supported = ("minmax", "zscore", "clahe")
        if method not in supported:
            raise ValidationError(
                f"Unsupported normalization method '{method}'. "
                f"Choose from {supported}.",
                parameter="method",
            )

        img = image.astype(np.float64, copy=True)

        if clip_range is not None:
            img = np.clip(img, clip_range[0], clip_range[1])

        if method == "minmax":
            return ImagePreprocessor._normalize_minmax(img)
        elif method == "zscore":
            return ImagePreprocessor._normalize_zscore(img)
        else:
            return ImagePreprocessor._normalize_clahe(img)

    @staticmethod
    def _normalize_minmax(img: np.ndarray) -> np.ndarray:
        """Scale values to the [0, 1] range."""
        vmin = img.min()
        vmax = img.max()
        if vmax - vmin < 1e-12:
            return np.zeros_like(img)
        return (img - vmin) / (vmax - vmin)

    @staticmethod
    def _normalize_zscore(img: np.ndarray) -> np.ndarray:
        """Subtract the mean and divide by standard deviation."""
        mean = img.mean()
        std = img.std()
        if std < 1e-12:
            return img - mean
        return (img - mean) / std

    @staticmethod
    def _normalize_clahe(img: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization (numpy).

        Implements a simplified, tile-free CLAHE approximation using
        rank-based equalization with a clip limit applied to the global
        histogram.  For full tile-based CLAHE, consider using an
        external library such as scikit-image.
        """
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min < 1e-12:
            return np.zeros_like(img)

        scaled = (img - img_min) / (img_max - img_min) * 255.0
        scaled = scaled.astype(np.uint8) if scaled.max() <= 255 else scaled

        flat = scaled.flatten().astype(np.float64)
        hist, bin_edges = np.histogram(flat, bins=256, range=(0, 256))

        clip_limit = int(np.ceil(flat.size / 256 * 2.0))
        excess = 0
        for i in range(len(hist)):
            if hist[i] > clip_limit:
                excess += hist[i] - clip_limit
                hist[i] = clip_limit

        redistrib = excess // 256
        hist = hist + redistrib

        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        denom = float(flat.size) - float(cdf_min)
        if denom < 1e-12:
            return ImagePreprocessor._normalize_minmax(img)

        cdf_normalized = (cdf - cdf_min) / denom
        cdf_normalized = np.clip(cdf_normalized, 0.0, 1.0)

        indices = np.clip(scaled.astype(np.intp).flatten(), 0, 255)
        equalized = cdf_normalized[indices].reshape(img.shape)
        return equalized

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    @staticmethod
    def resize(
        image: np.ndarray,
        target_size: Tuple[int, ...],
        order: int = 1,
    ) -> np.ndarray:
        """Resize an image to the given spatial dimensions.

        Uses ``scipy.ndimage.zoom`` with the specified interpolation
        order.

        Parameters
        ----------
        image : numpy.ndarray
            Input image. Can be 2-D (H, W) or 3-D (H, W, C).
        target_size : tuple of int
            Desired spatial dimensions ``(height, width)`` or
            ``(depth, height, width)`` for volumetric data.
        order : int, optional
            Spline interpolation order passed to
            ``scipy.ndimage.zoom``. Default is ``1`` (bilinear).

        Returns
        -------
        numpy.ndarray
            Resized image.

        Raises
        ------
        ValidationError
            If *target_size* length does not match the image
            dimensionality (excluding a trailing channel axis).
        """
        spatial_ndim = image.ndim
        has_channels = False

        if image.ndim == 3 and len(target_size) == 2:
            spatial_ndim = 2
            has_channels = True
        elif image.ndim == 4 and len(target_size) == 3:
            spatial_ndim = 3
            has_channels = True

        if len(target_size) != spatial_ndim:
            raise ValidationError(
                f"target_size has {len(target_size)} elements but image "
                f"has {spatial_ndim} spatial dimensions.",
                parameter="target_size",
            )

        zoom_factors: List[float] = []
        for i in range(spatial_ndim):
            zoom_factors.append(target_size[i] / image.shape[i])

        if has_channels:
            zoom_factors.append(1.0)

        return ndi.zoom(image, zoom_factors, order=order).astype(image.dtype)

    # ------------------------------------------------------------------
    # CT Windowing
    # ------------------------------------------------------------------

    @staticmethod
    def windowing(
        image: np.ndarray,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        *,
        preset: Optional[str] = None,
    ) -> np.ndarray:
        """Apply CT windowing (level/width) to a Hounsfield-unit image.

        You may specify *window_center* and *window_width* directly, or
        pass one of the standard *preset* names defined in
        :data:`CT_WINDOW_PRESETS`.

        Parameters
        ----------
        image : numpy.ndarray
            Image in Hounsfield units.
        window_center : float or None, optional
            Window center (level). Ignored when *preset* is given.
        window_width : float or None, optional
            Window width. Ignored when *preset* is given.
        preset : str or None, optional
            Name of a standard preset (``"lung"``, ``"bone"``,
            ``"brain"``, ``"soft_tissue"``, ``"liver"``,
            ``"mediastinum"``).

        Returns
        -------
        numpy.ndarray
            Windowed image scaled to [0, 1] as float64.

        Raises
        ------
        ValidationError
            If neither a preset nor both center/width are supplied, or
            if the preset name is unknown.
        """
        if preset is not None:
            if preset not in CT_WINDOW_PRESETS:
                raise ValidationError(
                    f"Unknown CT window preset '{preset}'. "
                    f"Available presets: {sorted(CT_WINDOW_PRESETS.keys())}.",
                    parameter="preset",
                )
            window_center, window_width = CT_WINDOW_PRESETS[preset]

        if window_center is None or window_width is None:
            raise ValidationError(
                "Provide both 'window_center' and 'window_width', or a "
                "valid 'preset' name.",
                parameter="window_center",
            )

        lower = window_center - window_width / 2.0
        upper = window_center + window_width / 2.0

        img = image.astype(np.float64, copy=True)
        img = np.clip(img, lower, upper)
        if upper - lower < 1e-12:
            return np.zeros_like(img)
        return (img - lower) / (upper - lower)

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    @staticmethod
    def denoise(
        image: np.ndarray,
        method: str = "gaussian",
        *,
        sigma: float = 1.0,
        size: int = 3,
    ) -> np.ndarray:
        """Apply a denoising filter to the image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        method : str, optional
            Filter type: ``"gaussian"`` or ``"median"``. Default is
            ``"gaussian"``.
        sigma : float, optional
            Standard deviation for the Gaussian filter. Only used when
            ``method="gaussian"``. Default is ``1.0``.
        size : int, optional
            Kernel size for the median filter. Only used when
            ``method="median"``. Default is ``3``.

        Returns
        -------
        numpy.ndarray
            Denoised image.

        Raises
        ------
        ValidationError
            If *method* is not ``"gaussian"`` or ``"median"``.
        """
        if method == "gaussian":
            return ndi.gaussian_filter(
                image.astype(np.float64), sigma=sigma
            )
        elif method == "median":
            return ndi.median_filter(image, size=size).astype(np.float64)
        else:
            raise ValidationError(
                f"Unsupported denoise method '{method}'. "
                f"Choose 'gaussian' or 'median'.",
                parameter="method",
            )

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def augment(
        image: np.ndarray,
        operations: List[str],
        *,
        rotate_angle: float = 15.0,
        brightness_factor: float = 0.1,
        noise_std: float = 0.01,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Apply a chain of augmentation operations to an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image (2-D or 3-D with channel last).
        operations : list of str
            Ordered list of operations to apply. Supported values:
            ``"flip_horizontal"``, ``"flip_vertical"``, ``"rotate"``,
            ``"brightness"``, ``"noise"``.
        rotate_angle : float, optional
            Maximum rotation angle in degrees (uniformly sampled from
            ``[-rotate_angle, rotate_angle]``). Default is ``15.0``.
        brightness_factor : float, optional
            Maximum additive brightness shift (uniformly sampled from
            ``[-brightness_factor, brightness_factor]``). Default is
            ``0.1``.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise. Default is
            ``0.01``.
        random_state : int or None, optional
            Seed for the random number generator. Default is ``None``.

        Returns
        -------
        numpy.ndarray
            Augmented image as float64.

        Raises
        ------
        ValidationError
            If an unrecognised operation name is encountered.
        """
        valid_ops = {
            "flip_horizontal",
            "flip_vertical",
            "rotate",
            "brightness",
            "noise",
        }
        for op in operations:
            if op not in valid_ops:
                raise ValidationError(
                    f"Unknown augmentation operation '{op}'. "
                    f"Supported: {sorted(valid_ops)}.",
                    parameter="operations",
                )

        rng = np.random.default_rng(random_state)
        img = image.astype(np.float64, copy=True)

        for op in operations:
            if op == "flip_horizontal":
                img = np.flip(img, axis=1)
            elif op == "flip_vertical":
                img = np.flip(img, axis=0)
            elif op == "rotate":
                angle = rng.uniform(-rotate_angle, rotate_angle)
                img = ndi.rotate(img, angle, reshape=False, order=1)
            elif op == "brightness":
                shift = rng.uniform(-brightness_factor, brightness_factor)
                img = img + shift
            elif op == "noise":
                img = img + rng.normal(0, noise_std, size=img.shape)

        return np.ascontiguousarray(img)
