"""Python implemention of the NMAR and LI MATLAB code of Zhang and Yu (IEEE TMI 2018),
https://github.com/yanbozhang007/CNN-MAR, which follows Meyer et al. (Med Phys 2010).
LI ("mar_li") fills the metal trace by linear interpolation in each view. NMAR ("nmar")
builds a prior image from the LI result with 3-means thresholds, forward projects it,
and interpolates the trace in the sinogram normalized by that prior. Units are mu per mm.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

from ct_ops.fbp import reconstruct_fdk
from ct_ops.forward_proj import compute_sinogram
from mar_common.units import hu2mu

# nmar.m: smFilter = fspecial('gaussian',[5 5],1)
GAUSS_SIZE, GAUSS_SIGMA = 5, 1.0
# nmar.m: eps = 10^(-6), added to the prior sinogram before dividing
PRIOR_EPS = 1e-6
# nmar.m: threshBone = max([threshBone, 1.2*miuWater]) -- "not smaller than 200 HU"
BONE_FLOOR_FACTOR = 1.2


def fspecial_gaussian(size=GAUSS_SIZE, sigma=GAUSS_SIGMA):
    """MATLAB fspecial('gaussian', [size size], sigma), same normalization."""
    half = (size - 1) / 2.0
    axis = np.arange(size, dtype=np.float64) - half
    y, x = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel[kernel < np.finfo(np.float64).eps * kernel.max()] = 0.0
    total = kernel.sum()
    return kernel / total if total != 0 else kernel


def proj_interp(proj, metal_trace):
    """cnnmar/projInterp.m: linear interpolation across the metal trace, view by view.
    - on a detector edge np.interp keeps the end value, MATLAB interp1 gives NaN
    """
    proj = np.asarray(proj, dtype=np.float64)
    trace = np.asarray(metal_trace) > 0
    out = proj.copy()
    for view in range(proj.shape[0]):
        row = trace[view]
        if not row.any():
            continue
        metal_pos = np.flatnonzero(row)
        valid_pos = np.flatnonzero(~row)
        if valid_pos.size < 2:
            continue  # nothing to interpolate from; leave the view untouched
        out[view, metal_pos] = np.interp(metal_pos, valid_pos, proj[view, valid_pos])
    return out


def edge_views(metal_trace):
    """Number of views where the metal trace touches a detector edge."""
    trace = np.asarray(metal_trace) > 0
    return int(np.count_nonzero(trace[:, 0] | trace[:, -1]))


def nmar_thresholds(img_mu, mu_water):
    """nmar.m: water and bone thresholds from 3-means (seeds: air, water, 2x water).
    - metal in "img_mu" must already be set to water
    - returns (thresh_water, thresh_bone) in mu per mm
    """
    flat = img_mu.reshape(-1, 1).astype(np.float64)
    start = np.array([[0.0], [mu_water], [2.0 * mu_water]])
    # one run from this fixed start, like the MATLAB call
    labels = KMeans(n_clusters=3, init=start, n_init=1, random_state=0).fit_predict(flat)
    water = flat[labels == 1]
    bone = flat[labels == 2]
    thresh_water = float(water.min()) if water.size else 0.0
    floor = BONE_FLOOR_FACTOR * mu_water
    thresh_bone = max(float(bone.min()), floor) if bone.size else floor
    return thresh_water, thresh_bone


def nmar_prior(img_mu, thresh_water, thresh_bone, mu_water, mu_air=0.0):
    """nmar.m::nmarPrior: smooth, then set air and soft tissue flat.
    - bone keeps the smoothed value
    """
    smoothed = ndimage.correlate(np.asarray(img_mu, dtype=np.float64),
                                 fspecial_gaussian(), mode="nearest")
    prior = smoothed.copy()
    prior[smoothed <= thresh_water] = mu_air
    prior[(smoothed > thresh_water) & (smoothed < thresh_bone)] = mu_water
    return prior


def nmar_project(proj, proj_prior, metal_trace):
    """nmar.m::nmar_proj: normalize, interpolate, denormalize, keep the measured data."""
    prior = np.asarray(proj_prior, dtype=np.float64).copy()
    prior[prior < 0] = 0.0
    prior = prior + PRIOR_EPS
    normalized = np.asarray(proj, dtype=np.float64) / prior
    out = proj_interp(normalized, metal_trace) * prior
    # keep the measured data outside the trace
    outside = np.asarray(metal_trace) <= 0
    out[outside] = np.asarray(proj, dtype=np.float64)[outside]
    return out


def mar_li(proj, metal_trace, config):
    """cnnmar/marLI.m: interpolate the trace, then reconstruct.
    - returns (corrected sinogram, image in HU)
    """
    proj_li = proj_interp(proj, metal_trace)
    return proj_li, reconstruct_fdk(config, proj_li)


def nmar(proj, metal_trace, metal_bw, config, prior_source="li"):
    """The full NMAR pipeline.
    - prior_source "li": prior from the LI image (Meyer, imNMAR2), "raw": from the raw image (imNMAR1)
    - returns a dict with all the intermediate images
    """
    if prior_source not in ("li", "raw"):
        raise ValueError("prior_source must be 'li' or 'raw'")

    mu_water = config.recon.mu
    trace = np.asarray(metal_trace) > 0
    metal = np.asarray(metal_bw) > 0

    img_raw_hu = reconstruct_fdk(config, proj)
    proj_li, img_li_hu = mar_li(proj, trace, config)

    base_hu = img_li_hu if prior_source == "li" else img_raw_hu
    base_mu = hu2mu(np.asarray(base_hu, dtype=np.float64), mu_water)
    # metal set to water before clustering, the same image is used for the prior
    base_mu[metal] = mu_water

    thresh_water, thresh_bone = nmar_thresholds(base_mu, mu_water)
    prior_mu = nmar_prior(base_mu, thresh_water, thresh_bone, mu_water)
    proj_prior = compute_sinogram(prior_mu, config)
    proj_nmar = nmar_project(proj, proj_prior, trace)

    return {
        "img_raw_hu": img_raw_hu,
        "img_li_hu": img_li_hu,
        "img_nmar_hu": reconstruct_fdk(config, proj_nmar),
        "prior_mu": prior_mu,
        "proj_li": proj_li,
        "proj_nmar": proj_nmar,
        "proj_prior": proj_prior,
        "thresh_water_hu": (thresh_water / mu_water) * 1000.0 - 1000.0,
        "thresh_bone_hu": (thresh_bone / mu_water) * 1000.0 - 1000.0,
        "prior_source": prior_source,
        "edge_views": edge_views(trace),
    }
