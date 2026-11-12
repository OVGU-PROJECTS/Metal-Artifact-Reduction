"""Sinogram-domain metrics, scored inside the metal trace where the network predicts

Shared by PConv-MAR (predicted projection) and DuDoNet (SE-Net Y_out). Values are
line integrals. "hole" is a boolean array, True inside the trace.
"""
import numpy as np
from scipy import ndimage as ndi

STRUCT = ndi.generate_binary_structure(2, 5)


def fill_mae(ref, pred, hole):
    return float(np.mean(np.abs(pred[hole] - ref[hole])))

def seam_gradient(ref, pred, hole, radius=1):
    """gradient error across the trace boundary, where a step becomes a streak"""
    band = ndi.binary_dilation(hole, STRUCT, radius) & ~ndi.binary_erosion(hole, STRUCT, radius)
    grad = lambda a: np.hypot(ndi.sobel(a, 0), ndi.sobel(a, 1))
    return float(np.abs(grad(pred)[band] - grad(ref)[band]).mean())


def trace_gradient_err(ref, pred, hole):
    """Detector-axis gradient error inside the trace, catches an over-smoothed fill"""
    inner = hole[:, 1:] & hole[:, :-1]
    return float(np.abs(np.diff(pred, axis=1)[inner] - np.diff(ref, axis=1)[inner]).mean())


def trace_width(hole):
    """Mean trace width in detector bins over the views that contain it"""
    per_view = hole.sum(axis=1)
    return float(per_view[per_view > 0].mean())


def score_sinogram(ref, pred, hole):
    """Three thesis quality metrics and the two reported trace descriptors"""
    return {
        "fill_mae": fill_mae(ref, pred, hole),
        "seam_gradient": seam_gradient(ref, pred, hole),
        "trace_gradient_err": trace_gradient_err(ref, pred, hole),
        "trace_width": trace_width(hole),
        "hole_frac": float(hole.mean()),
    }





