""" score the leftover streaks:
    streak_edge_err  mean |Sobel gradient| error in HU/px
    The torch twin of this metric lives in metrics.torch_hu_metrics.
"""
import numpy as np
from scipy.signal import convolve2d


def streak_edge_err(reference_hu, prediction_hu, dilated_mask):
    """Mean |Sobel gradient| error in HU/px over the patient region
    - dilated metal excluded"""
    reference_hu = np.asarray(reference_hu, dtype=np.float32)
    prediction_hu = np.asarray(prediction_hu, dtype=np.float32)
    kx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32)

    def sobel_mag(image):
        gx = convolve2d(image, kx, mode="same", boundary="symm")
        gy = convolve2d(image, kx.T, mode="same", boundary="symm")
        return np.sqrt(gx * gx + gy * gy)

    region = (reference_hu > -950.0) & ~np.asarray(dilated_mask, dtype=bool)
    if not region.any():
        raise ValueError("streak metric region is empty")
    return float(np.abs(sobel_mag(reference_hu) - sobel_mag(prediction_hu))[region].mean())
