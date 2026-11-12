"""Builds the fan -> parallel rebinning table which the DC loss need
"""
import numpy as np
import torch

from geometry.fanbeam import (
    compute_bin_pos_parallel,
    compute_fan_angles,
    compute_rot_angles,
    wrap_angle_0_2pi,
)

def build_dc_pack_from_cfg(cfg, eps=1e-7):
    eps = np.float32(eps)
    view_count = int(cfg.protocol.viewCount)        # sinogram rows
    det_cols = int(cfg.scanner.detectorColCount)    # sinogram columns

    rot = compute_rot_angles(cfg).astype(np.float32)   # angle of each view, source to detector
    fan = compute_fan_angles(cfg).astype(np.float32)   # angle of each detector column

    # searchsorted needs sorted angles, so sort and keep the original positions
    x_idx = np.argsort(fan)
    y_idx = np.argsort(rot)
    x_sorted = fan[x_idx]
    y_sorted = rot[y_idx]
    two_pi = np.float32(2.0 * np.pi)
    # the views listed twice, so an angle just past 2pi still finds a pair
    y_sorted_ext = np.concatenate([y_sorted, y_sorted + two_pi]).astype(np.float32)

    # the fan angle each parallel bin would have been measured at
    bin_pos = compute_bin_pos_parallel(cfg).astype(np.float32)
    sid = np.float32(cfg.scanner.sid)
    fan_tgt = np.arcsin(np.clip(bin_pos / sid, -0.99999, 0.99999)).astype(np.float32)
    x_new = np.clip(fan_tgt, np.min(x_sorted), np.max(x_sorted))

    # x direction: the two detector columns either side of that angle.
    # every view shares the same detector, so these stay 1-D
    pos = np.searchsorted(x_sorted, x_new, side="left")
    x2s = np.clip(pos, 0, det_cols - 1)
    x1s = np.clip(x2s - 1, 0, det_cols - 1)
    x1 = x_sorted[x1s]
    x2 = x_sorted[x2s]
    # at bin 0 the pair collapses and x2 - x1 is 0, eps keeps that 0/0 away
    wx1 = (x2 - x_new) / ((x2 - x1) + eps)
    wx2 = (x_new - x1) / ((x2 - x1) + eps)
    x1u = x_idx[x1s].astype(np.int32)
    x2u = x_idx[x2s].astype(np.int32)

    # y direction: which two views to read. the answer shifts with the fan
    # angle, so it depends on the view and the bin, which makes it 2-D
    pi_half = np.float32(np.pi / 2.0)
    rot_par = wrap_angle_0_2pi(rot + pi_half)
    y_new = wrap_angle_0_2pi(rot_par[:, None] - pi_half + fan_tgt[None, :])

    posy = np.searchsorted(y_sorted_ext, y_new, side="left")
    y2s = np.clip(posy, 1, 2 * view_count - 1)
    y1s = y2s - 1
    y1v = y_sorted_ext[y1s]
    y2v = y_sorted_ext[y2s]
    wy1 = (y2v - y_new) / ((y2v - y1v) + eps)
    wy2 = (y_new - y1v) / ((y2v - y1v) + eps)
    # back to real view numbers, undoing the doubled list
    y1u = y_idx[np.mod(y1s, view_count)].astype(np.int32)
    y2u = y_idx[np.mod(y2s, view_count)].astype(np.int32)

    # read the sinogram as one long row: element (view, column) sits at
    # view * det_cols + column. one flat index per bilinear corner
    i11 = y1u * det_cols + x1u[None, :]
    i21 = y1u * det_cols + x2u[None, :]
    i12 = y2u * det_cols + x1u[None, :]
    i22 = y2u * det_cols + x2u[None, :]

    # how much each corner contributes, the four summing to 1
    w11 = (wy1 * wx1[None, :])
    w21 = (wy1 * wx2[None, :])
    w12 = (wy2 * wx1[None, :])
    w22 = (wy2 * wx2[None, :])

    # long for indexing, float for the weighted sum in rebin_proj
    return {
        "V": view_count,
        "U": det_cols,
        "S": torch.from_numpy(bin_pos).float(),
        "I11": torch.from_numpy(i11).long(),
        "I21": torch.from_numpy(i21).long(),
        "I12": torch.from_numpy(i12).long(),
        "I22": torch.from_numpy(i22).long(),
        "W11": torch.from_numpy(w11).float(),
        "W21": torch.from_numpy(w21).float(),
        "W12": torch.from_numpy(w12).float(),
        "W22": torch.from_numpy(w22).float(),
    }
