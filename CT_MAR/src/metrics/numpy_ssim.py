"""Thesis SSIM metrics on HU images, using numpy and skimage.

    ssim_aapm         AAPM masked SSIM, skimage, data_range=5000
    ssim_aapm_masked  same, plus an optional external patient mask
    ssim_trace        SSIM inside the dilated metal region
"""
from pathlib import Path
from typing import Optional
import numpy as np
from skimage.metrics import structural_similarity
from mar_common.io import read_mha


# =============================================================================
# AAPM masked SSIM, HU domain
# =============================================================================
def ssim_aapm_masked(
    reference_hu: np.ndarray,
    prediction_hu: np.ndarray,
    metal_exclusion_mask: np.ndarray,
    *,
    external_mask: Optional[np.ndarray] = None,
) -> float:
    """AAPM CT-MAR SSIM style """
    reference_hu = np.asarray(reference_hu, dtype=np.float32)
    prediction_hu = np.asarray(prediction_hu, dtype=np.float32)
    metal_exclusion_mask = np.asarray(metal_exclusion_mask, dtype=bool)

    if external_mask is None:
        external_mask = reference_hu > -950.0
    evaluation_mask = np.asarray(external_mask, dtype=bool) & ~metal_exclusion_mask
    if not evaluation_mask.any():
        raise ValueError("AAPM SSIM evaluation mask is empty")

    reference_masked = reference_hu.copy()
    prediction_masked = prediction_hu.copy()
    reference_masked[metal_exclusion_mask] = 100.0
    prediction_masked[metal_exclusion_mask] = 100.0

    _, ssim_map = structural_similarity(
        reference_masked,
        prediction_masked,
        sigma=1.5,
        gaussian_weights=True,
        use_sample_covariance=False,
        data_range=5000,
        full=True,
        gradient=False,
    )
    return float(np.mean(ssim_map[evaluation_mask]))


def ssim_aapm(reference_hu, prediction_hu, dilated_mask):
    """metal is filled with 100 HU, then the score is taken outside it"""
    return ssim_aapm_masked(reference_hu, prediction_hu, dilated_mask)


def ssim_trace(reference_hu, prediction_hu, dilated_mask):
    """AAPM-style SSIM inside the dilated mask, the region ssim_aapm drops.
    no constant fill here, real values on both sides."""
    dilated_mask = np.asarray(dilated_mask, dtype=bool)
    if not dilated_mask.any():
        raise ValueError("trace SSIM mask is empty")
    _, ssim_map = structural_similarity(
        np.asarray(reference_hu, dtype=np.float32),
        np.asarray(prediction_hu, dtype=np.float32),
        sigma=1.5,
        gaussian_weights=True,
        use_sample_covariance=False,
        data_range=5000,
        full=True,
        gradient=False,
    )
    return float(ssim_map[dilated_mask].mean())


# =============================================================================
# mask loaders (DuDoNet)
# =============================================================================
def _load_mask(case_dir: Path, name: str) -> np.ndarray:
    """read one .mha mask as bool, and say which file is missing if it is"""
    path = Path(case_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing metal mask: {path}")
    return read_mha(str(path)) > 0


def load_metal_exclusion_mask(case_dir: Path) -> np.ndarray:
    """the pre-dilated image mask the model's metal trace was built from."""
    return _load_mask(case_dir, "input_mask_dilated.mha")

