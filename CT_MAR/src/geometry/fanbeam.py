# Source: https://github.com/xcist/main/tree/master/gecatsim/reconstruction
# https://github.com/xcist/main/blob/master/gecatsim/reconstruction/pyfiles/mapConfigVariablesToFDK.py#L10
# https://github.com/xcist/main/blob/master/gecatsim/reconstruction/pyfiles/fdk_equiAngle.py#L115
# https://github.com/xcist/main/blob/master/gecatsim/reconstruction/src/interface_fdk_angle.c#L73
# https://github.com/xcist/main/blob/master/gecatsim/reconstruction/pyfiles/helical_equiAngle.py#L144
# https://github.com/xcist/example/blob/main/AAPM_datachallenge/proj_and_recon/AAPMProj.py
"""Equi-angular fan-beam acquisition geometry (NumPy, float32).

"""
import numpy as np

# Reconstruction field-of-view diameter (mm) per scanner FOV mode.
RECO_FOV = {"head": 220.16, "body": 400.0}


def wrap_angle_0_2pi(angle_rad):
    """Wrap an angle (scalar or array) into ``[0, 2*pi)`` without ``%``/FloorMod."""
    two_pi = np.float32(2.0 * np.pi)
    return angle_rad - two_pi * np.floor(angle_rad / two_pi)


def compute_det_bin_width(cfg):
    """Detector column pitch covered by one module (mm)."""
    return np.float32(cfg.scanner.detectorColsPerMod * cfg.scanner.detectorColSize)


def compute_fan_ang_incr_rad(cfg):
    """Fan angular increment of an equi-angular fan-beam projection (rad)."""
    det_bin_width = compute_det_bin_width(cfg)
    return np.float32(2.0 * np.arctan(det_bin_width / (2.0 * np.float32(cfg.scanner.sdd))))


def compute_cent_det_pix(cfg):
    """Center detector pixel index, accounting for the column offset."""
    return np.float32(((cfg.scanner.detectorColCount - 1.0) / 2.0) + (-cfg.scanner.detectorColOffset))


def compute_rot_angles(cfg):
    """Per-view gantry rotation angles in DICOM LPS order (rad), wrapped to [0, 2pi).

    XCAT->DICOM LPS: source at 12 o'clock => +3*pi/2 offset; angle decreases during
    anti-clockwise rotation.
    """
    start = np.float32(np.deg2rad(float(cfg.recon.startAngle)) + 3.0 * np.pi / 2.0)
    view_count = int(cfg.protocol.viewCount)
    delta = np.float32(2.0 * np.pi / float(view_count))
    rot_dir = np.float32(getattr(cfg.protocol, "rotationDirection", 1.0))
    idx = np.arange(view_count, dtype=np.float32)
    return wrap_angle_0_2pi(start - rot_dir * idx * delta)


def compute_fan_angles(cfg):
    """Per-detector-column fan angles (rad), centered on the detector center."""
    inc = compute_fan_ang_incr_rad(cfg)
    cent = compute_cent_det_pix(cfg)
    det_cols = int(cfg.scanner.detectorColCount)
    cols = np.arange(det_cols, dtype=np.float32)
    return (cols - cent) * inc


def compute_bin_pos_parallel(cfg):
    """Parallel-beam detector bin positions (mm) after fan->parallel rebinning."""
    fan = compute_fan_angles(cfg)
    sid = np.float32(cfg.scanner.sid)
    return np.linspace(
        np.float32(sid * np.sin(np.min(fan))),
        np.float32(sid * np.sin(np.max(fan))),
        num=fan.shape[0],
        dtype=np.float32,
    )
