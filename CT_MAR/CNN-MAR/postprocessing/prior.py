"""Construct the tissue prior used by the original CNN-MAR post-processing.

This is a NumPy/SciPy port of ``cnnPrior.m`` from Zhang and Yu's release.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class PriorInfo:
    """Segmentation values useful for logging and stage-wise evaluation."""

    cluster_centers_hu: tuple[float, float, float]
    air_threshold_hu: float
    bone_threshold_hu: float
    water_mean_hu: float


def _kmeans_1d(
    values: np.ndarray,
    initial_centers: tuple[float, float, float] = (-900.0, 0.0, 1000.0),
    *,
    max_iter: int = 100,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic three-class k-means for scalar CT values."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.asarray(initial_centers, dtype=np.float64)

    labels = np.zeros(x.size, dtype=np.int8)
    for _ in range(max_iter):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in range(3):
            members = x[labels == index]
            if members.size:
                updated[index] = members.mean()
        if np.max(np.abs(updated - centers)) <= tolerance:
            centers = updated
            break
        centers = updated

    order = np.argsort(centers)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return inverse[labels], centers[order]


def _nearest_ring_fill(image: np.ndarray, metal: np.ndarray) -> np.ndarray:
    """Fill a one-pixel-dilated metal region from its nearest outer ring."""
    structure = np.ones((3, 3), dtype=bool)
    fill_region = ndi.binary_dilation(metal, structure=structure)
    outer_ring = ndi.binary_dilation(fill_region, structure=structure) & ~fill_region
    if not np.any(fill_region) or not np.any(outer_ring):
        return image

    # distance_transform_edt returns the indices of the nearest zero.  Making
    # the outer ring zero.
    _, nearest = ndi.distance_transform_edt(
        ~outer_ring, return_distances=True, return_indices=True
    )
    result = image.copy()
    nearest_values = image[tuple(nearest)]
    result[fill_region] = nearest_values[fill_region]
    return result


def build_prior_image(
    cnn_hu: np.ndarray,
    metal_mask: np.ndarray,
    *,
    transition_pixels: int = 5,
    minimum_bone_hu: float = 350.0,
) -> tuple[np.ndarray, PriorInfo]:
    """Build the air/water/bone prior from a direct CNN prediction.

    The water-equivalent region is smoothly homogenized over five pixels,
    bone and air are retained, and the metal region is filled from its nearest
    surrounding non-metal pixels.
    """
    image = np.asarray(cnn_hu, dtype=np.float32)
    metal = np.asarray(metal_mask) > 0
    if image.ndim != 2 or metal.shape != image.shape:
        raise ValueError(
            f"cnn_hu and metal_mask must be equal-size 2-D arrays; "
            f"got {image.shape} and {metal.shape}"
        )
    if transition_pixels <= 0:
        raise ValueError("transition_pixels must be positive")
    if not np.all(np.isfinite(image)):
        raise ValueError("cnn_hu contains NaN or infinite values")

    # Water attenuation corresponds to 0 HU.
    working = image.copy()
    working[metal] = 0.0
    labels, centers = _kmeans_1d(working)

    water_members = working.reshape(-1)[labels == 1]
    bone_members = working.reshape(-1)[labels == 2]
    if water_members.size == 0 or bone_members.size == 0:
        raise ValueError("three-class k-means produced an empty tissue class")

    # These definitions intentionally follow cnnPrior.m.
    air_threshold = float(water_members.min())
    bone_threshold = max(float(bone_members.min()), float(minimum_bone_hu))

    bone = working > bone_threshold
    water = (working > air_threshold) & ~bone
    water_weight = ndi.distance_transform_edt(water)
    water_weight = np.minimum(water_weight, float(transition_pixels))
    water_weight /= float(transition_pixels)

    weight_sum = float(water_weight.sum())
    if weight_sum == 0.0:
        raise ValueError("no water-equivalent pixels were available for the prior")
    water_mean = float(np.sum(water_weight * working) / weight_sum)
    prior = water_mean * water_weight + working * (1.0 - water_weight)
    prior = _nearest_ring_fill(prior, metal).astype(np.float32, copy=False)

    info = PriorInfo(
        cluster_centers_hu=tuple(float(value) for value in centers),
        air_threshold_hu=air_threshold,
        bone_threshold_hu=bone_threshold,
        water_mean_hu=water_mean,
    )
    return prior, info
