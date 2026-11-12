# Source: https://github.com/xcist/example/blob/main/AAPM_datachallenge/proj_and_recon/AAPMProj.py
"""CT forward projection: turn an image into a sinogram """
from ctypes import *
from gecatsim.pyfiles.CommonTools import *
from numpy.ctypeslib import ndpointer
import numpy as np

from geometry.fanbeam import compute_fan_ang_incr_rad


def dist_driven_fan_arc_proj(nrdet, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows,
                             originalImgPtr):
    clib = load_C_lib()

    func = clib.DD2FanProj
    func.argtypes = [c_int, c_float, c_float, ndpointer(c_float), ndpointer(c_float), c_float, c_float,
                     ndpointer(c_float), c_int, ndpointer(c_float), c_int, c_int, ndpointer(c_float)]
    func.restype = None

    func(nrdet, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows, originalImgPtr)

    return sinogram


def compute_sinogram(img, config):
    """Forward projection of image."""
    if len(img.shape) == 2:  # Forward projection code requires a 3rd (slice) axis.
        img = np.expand_dims(img, axis=2)

    sid_mm = config.scanner.sid  # source-to-isocenter distance
    sdd_mm = config.scanner.sdd  # source-to-detector distance

    num_det_cols = config.scanner.detectorColCount
    num_img_cols = config.recon.imageSize
    num_img_rows = config.recon.imageSize
    img_pix_size_mm = config.recon.fov / config.recon.imageSize
    num_views = config.protocol.viewCount

    x0 = 0.0 / img_pix_size_mm
    y0 = sid_mm / img_pix_size_mm
    xCor = 0.0 / img_pix_size_mm
    yCor = 0.0 / img_pix_size_mm

    det_pix_width_mm = 1.0
    # Fan angle of each detector column, centered on the detector and shifted by the offset.
    fan_ang_incr_rad = compute_fan_ang_incr_rad(config)
    fan_angles_rad = (np.arange(num_det_cols) - (
            num_det_cols - 1) / 2 + config.scanner.detectorColOffset) * fan_ang_incr_rad
    # (x,y) position of each detector cell on the arc, in pixel units.
    xds = np.single(sdd_mm * np.sin(fan_angles_rad) / img_pix_size_mm)
    yds = np.single((sid_mm - sdd_mm * np.cos(fan_angles_rad)) / img_pix_size_mm)

    # Gantry angle of each view, spread evenly over one full rotation (0..2*pi).
    view_angles = np.single(1 * (0 + np.arange(num_views) / num_views * 2 * np.pi))

    orig_img_ptr = np.single(img)
    sinogram = np.zeros([num_views, num_det_cols, 1], dtype=np.single)

    sinogram = dist_driven_fan_arc_proj(num_det_cols, x0, y0, xds, yds, xCor, yCor, view_angles, num_views,
                                        sinogram,
                                        num_img_cols,
                                        num_img_rows,
                                        orig_img_ptr)
    sinogram = sinogram * img_pix_size_mm

    sinogram = sinogram.squeeze()

    return sinogram




