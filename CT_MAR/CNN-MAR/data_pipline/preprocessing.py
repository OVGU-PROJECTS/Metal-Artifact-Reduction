"""Input preparation matching the original Zhang/Yu CNN-MAR MATLAB code."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from mar_common.units import hu2mu, mu2hu


MU_WATER = 0.02  # 1/mm, equivalent to the original code's ~0.19 1/cm scale.
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _as_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def _to_mu(img_hu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    return hu2mu(np.asarray(img_hu, dtype=np.float32), mu_water=mu_water).astype(np.float32, copy=False)


def prepare_training_images(
    raw_hu: np.ndarray,
    bhc_hu: np.ndarray,
    li_hu: np.ndarray,
    ref_hu: np.ndarray,
    metal_mask: np.ndarray,
    *,
    mu_water: float = MU_WATER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert training images to mu and set metal pixels to water.

    This mirrors ``dataGen.m`` in the original MATLAB code.
    """
    metal = _as_mask(metal_mask)
    raw = _to_mu(raw_hu, mu_water)
    bhc = _to_mu(bhc_hu, mu_water)
    li = _to_mu(li_hu, mu_water)
    ref = _to_mu(ref_hu, mu_water)

    for img in (raw, bhc, li, ref):
        img[metal] = mu_water

    return raw, bhc, li, ref


def prepare_inference_input(
    raw_hu: np.ndarray,
    bhc_hu: np.ndarray,
    li_hu: np.ndarray,
    metal_mask: np.ndarray,
    *,
    mu_water: float = MU_WATER,
) -> np.ndarray:
    """Prepare a full-image CNN input like original ``dataInput.m``."""
    metal = ndi.binary_dilation(_as_mask(metal_mask), structure=_CROSS)
    raw = _to_mu(raw_hu, mu_water)
    bhc = _to_mu(bhc_hu, mu_water)
    li = _to_mu(li_hu, mu_water)

    for img in (raw, bhc, li):
        img[metal] = mu_water

    x = np.stack([raw, bhc, li], axis=-1).astype(np.float32, copy=False)
    x[x < 0] = 0
    return x


def prediction_mu_to_hu(pred_mu: np.ndarray, *, mu_water: float = MU_WATER) -> np.ndarray:
    return mu2hu(np.asarray(pred_mu, dtype=np.float32), mu_water=mu_water).astype(np.float32, copy=False)
