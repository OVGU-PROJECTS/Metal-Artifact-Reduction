"""Image-domain metrics and regions for MAR scoring.
Shared by PConv-MAR and DuDoNet. Every metric takes (ref_hu, pred_hu, boolean mask)
and returns a float, score_case is the composition the two models are compared on
"""
import numpy as np
from scipy import ndimage as ndi
from metrics.numpy_ssim import ssim_aapm_masked

FOV_MM = {"head": 220.16, "body": 400.0}


# ----------- regions ----------------------------------------------------------------

def fov_circle(fov_mm, diameter_mm=470.0, size=512):
    """circular field of view, so the scan background never enters a metric"""
    radius_px = diameter_mm / (fov_mm / size) / 2.0
    rows, cols = np.indices((size, size))
    centre = (size - 1) / 2.0
    return (rows - centre) ** 2 + (cols - centre) ** 2 < radius_px ** 2

# --- pointwise ---
def rmse(ref, pred, mask):
    return float(np.sqrt(np.mean((pred[mask] - ref[mask]) ** 2)))

def mae(ref, pred, mask):
    return float(np.mean(np.abs(pred[mask] - ref[mask])))


def psnr(ref, pred, mask, data_range=5000.0):
    return float(20.0 * np.log10(data_range / rmse(ref, pred, mask)))


# ---------------------------------- structure -------------------------------------------

def sobel_mag(img):
    return np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1))


def edge_ratio(ref, pred, mask):
    """<1 means edges were smoothed away, >1 means they were sharpened """
    return float(sobel_mag(pred)[mask].mean() / sobel_mag(ref)[mask].mean())

def streak_roi(ref, mask, quiet_percent=25.0):
    """The flat part of the mask: where streaks show up and anatomical edges do not

    Streaks cross homogeneous tissue, so the quietest quarter of the reference is
    where they are visible on their own.
    """
    edges = sobel_mag(ref)
    limit = np.percentile(edges[mask], quiet_percent)
    return mask & (edges <= limit)


def streak_amplitude(ref, pred, mask, quiet_percent=25.0, tail_percent=5.0):
    """streak amplitude in HU. Lower is better, 0 means no streaks left
    """
    values = np.sort((pred - ref)[streak_roi(ref, mask, quiet_percent)])
    cutoff = max(1, int(len(values) * tail_percent / 100.0))
    return float(values[-cutoff:].mean() - values[:cutoff].mean())


def noise_mad(img, mask):
    """noise sigma from the finest-scale detail (Donoho MAD), the median outvotes edges."""
    detail = np.diff(np.diff(img, axis=0), axis=1)[mask[1:, 1:]]
    return float(np.median(np.abs(detail - np.median(detail))) / 0.6745 / 2.0)

def noise_ratio(ref, pred, mask):
    """<1 means the prediction is smoother than the reference, >1 noisier."""
    return noise_mad(pred, mask) / noise_mad(ref, mask)


# -------------------------------------------- scoring --------------------------------------
def score_region(ref, pred, region, metal):
    """The seven image metrics reported in the thesis"""
    return {
        "rmse": rmse(ref, pred, region),
        "mae": mae(ref, pred, region),
        "psnr": psnr(ref, pred, region),
        "ssim": ssim_aapm_masked(ref, pred, metal, external_mask=region),
        "edge_ratio": edge_ratio(ref, pred, region),
        "streak_amp": streak_amplitude(ref, pred, region),
        "noise_ratio": noise_ratio(ref, pred, region),
    }


def score_case(ref, pred, metal, fov_mm):
    """Thesis image metrics inside the 470 mm FOV, excluding dilated metal
    "metal" is the dilated mask and is excluded: partial volume puts metal-level
    attenuation within 1 px, so the reference value there is not recoverable
    """
    circle = fov_circle(fov_mm)
    r_global = circle & ~metal
    metrics = score_region(ref, pred, r_global, metal)
    row = {}
    for name, value in metrics.items():
        row[f"global_{name}"] = value
    return row




