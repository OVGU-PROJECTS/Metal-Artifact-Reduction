"""Thesis SSIM metrics on attenuation inputs in mm^-1, using PyTorch.

    ssim_aapm   AAPM SSIM outside dilated metal, within the patient region
    ssim_trace  SSIM inside dilated metal, used in the PConv development study
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from mar_common.io import read_mha
from mar_common.units import mu2hu
from metrics.torch_hu_metrics import masked_mean


# =============================================================================
# internal helpers
# =============================================================================
def _aapm_ssim_map(reference_hu, prediction_hu):
    """full-size AAPM SSIM map. 11x11 gaussian, sigma 1.5, reflect pad, range 5000"""
    coords = torch.arange(11, device=reference_hu.device, dtype=reference_hu.dtype) - 5
    gaussian = torch.exp(-(coords.square()) / (2.0 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian.outer(gaussian).view(1, 1, 11, 11)

    def smooth(image):
        return F.conv2d(F.pad(image, (5, 5, 5, 5), mode="reflect"), kernel)

    mu1 = smooth(reference_hu)
    mu2 = smooth(prediction_hu)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1_sq = smooth(reference_hu.square()) - mu1_sq
    sigma2_sq = smooth(prediction_hu.square()) - mu2_sq
    sigma12 = smooth(reference_hu * prediction_hu) - mu1_mu2
    c1, c2 = (0.01 * 5000.0) ** 2, (0.03 * 5000.0) ** 2
    return ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

# =============================================================================
# AAPM masked SSIM
# =============================================================================
def ssim_aapm(img_true_mu, img_pred_mu, dilated_mask):
    """mu goes to HU, metal is filled with 100 HU, then scored outside it."""
    reference = mu2hu(img_true_mu)
    prediction = mu2hu(img_pred_mu)
    metal = dilated_mask > 0.5
    reference_masked = torch.where(metal, reference.new_tensor(100.0), reference)
    prediction_masked = torch.where(
        metal, prediction.new_tensor(100.0), prediction
    )
    ssim_map = _aapm_ssim_map(reference_masked, prediction_masked)
    evaluation_mask = (reference > -950.0) & ~metal
    return masked_mean(ssim_map, evaluation_mask, "AAPM SSIM evaluation")


def ssim_trace(img_true_mu, img_pred_mu, dilated_mask):
    """AAPM-style SSIM inside the dilated mask, the exact region ssim_aapm
    drops. no constant fill here, real values on both sides"""
    reference = mu2hu(img_true_mu)
    prediction = mu2hu(img_pred_mu)
    metal = dilated_mask > 0.5
    ssim_map = _aapm_ssim_map(reference, prediction)
    return masked_mean(ssim_map, metal, "trace SSIM")


# =============================================================================
# mask loaders
# =============================================================================
def load_image_masks(case_paths, size, device):
    """original and dilated metal masks for a batch, resized to the image grid"""
    original = np.stack([
        read_mha(os.path.join(path, "input_mask.mha")) > 0 for path in case_paths
    ])
    dilated = np.stack([
        read_mha(os.path.join(path, "input_mask_dilated.mha")) > 0
        for path in case_paths
    ])
    original = torch.from_numpy(original[:, None].astype(np.float32)).to(device)
    dilated = torch.from_numpy(dilated[:, None].astype(np.float32)).to(device)
    original = F.interpolate(original, size=size, mode="nearest")
    dilated = F.interpolate(dilated, size=size, mode="nearest")
    return original, dilated
