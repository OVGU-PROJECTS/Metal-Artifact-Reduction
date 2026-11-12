"""Precompute the FDK reconstruction geometry once, reuse it every step.
- Recon is called for every training sample
- so all the fixed geometry such as source positions per view,
detector spacing, image-grid coordinates,cosine weights,
the 1000->500 view downsampling indices is computed
a single time into an "FDKCache" and cached on the recon config.
- "createFdkCache" is the builder.
"""
import math
from dataclasses import dataclass
import torch

from geometry.fanbeam import RECO_FOV

@dataclass
class FDKCache:
    """bundle of all geometry tensors the FDK kernels need (see createFdkCache)"""
    # views
    ds_idx: torch.Tensor            # which raw views to keep (every 2nd, for 1000->500)
    raw_views_full: int             # views the incoming sinogram must have (1000)
    num_views: int                  # protocol view count, divides the final sum

    # detector
    YL: int                         # detector columns (900)
    ZL_orig: int                    # detector rows (1)
    ScanR: torch.Tensor             # source to isocenter, mm (550)
    DistD: torch.Tensor             # source to detector, mm (950)
    DeltaY: torch.Tensor            # fan angle of one column, rad
    DeltaZ: torch.Tensor            # height of one row, mm
    YCtr: torch.Tensor              # center column index (450.75)
    ZCtr: torch.Tensor              # center row index (0.5)
    fan_ang_incr_rad: torch.Tensor  # same angle as DeltaY, used by the ramp filter
    cos_w2d: torch.Tensor           # FDK cosine weights, [columns, rows]

    # image volume
    RecSize: int                    # image is RecSize x RecSize (256)
    FOIHeight: int                  # number of slices (1)
    xGrid: torch.Tensor             # pixel x coordinates in mm
    yGrid: torch.Tensor             # pixel y coordinates in mm
    zGrid: torch.Tensor             # slice z coordinates in mm

    # x-ray source, one entry per kept view
    SrcX_all: torch.Tensor          # source x, mm
    SrcY_all: torch.Tensor          # source y, mm
    DirX_all: torch.Tensor          # cos of the view angle
    DirY_all: torch.Tensor          # sin of the view angle


def fanAngleIncrement(cfg):
    det_bin_width = cfg.scanner.detectorColsPerMod * cfg.scanner.detectorColSize
    return 2 * math.atan(det_bin_width / 2 / cfg.scanner.sdd)


def selectViews(raw_view_count, target_views, view_block):
    """keep "target_views" evenly spaced views out of the raw scan"""
    if target_views % view_block != 0:
        raise ValueError("view_block must divide target_views.")

    stride = max(raw_view_count // target_views, 1)
    return torch.arange(0, raw_view_count, step=stride, dtype=torch.long)[:target_views]


def cosineWeights(det_cols, det_rows, detector_height, fan_angle_increment, source_detector_distance):
    center_col = (det_cols - 1) * 0.5
    center_row = (det_rows - 1) * 0.5
    pixel_height = detector_height / float(det_rows)

    col_indices = torch.arange(det_cols, dtype=torch.float32)
    row_indices = torch.arange(det_rows, dtype=torch.float32)
    in_plane = torch.cos((col_indices - center_col) * fan_angle_increment)
    row_offsets = (row_indices - center_row) * pixel_height
    out_plane = source_detector_distance / torch.sqrt(
        source_detector_distance.square() + row_offsets.square()
    )
    return torch.einsum("i,j->ij", in_plane, out_plane)


def detectorGeometry(cfg):
    """detector dimensions, spacing, centers, distances, and the cosine weights.
    Keys are the FDKCache field names, so createFdkCache can pass them straight through.
    """
    det_cols = int(cfg.scanner.detectorColCount)
    det_rows = int(cfg.scanner.detectorRowCount)
    effective_rows = 2 if det_rows == 1 else det_rows

    scan_radius = torch.tensor(float(cfg.scanner.sid), dtype=torch.float32)
    source_detector_distance = torch.tensor(float(cfg.scanner.sdd), dtype=torch.float32)

    num_modules = math.ceil(cfg.scanner.detectorColCount / cfg.scanner.detectorColsPerMod)
    fan_angle_increment = float(fanAngleIncrement(cfg))
    detector_fan_angle = float(num_modules) * fan_angle_increment
    detector_height = float(cfg.scanner.detectorRowSize) * float(det_rows)

    return {
        "YL": det_cols,
        "ZL_orig": det_rows,
        "ScanR": scan_radius,
        "DistD": source_detector_distance,
        "DeltaY": torch.tensor(detector_fan_angle / float(det_cols), dtype=torch.float32),
        "DeltaZ": torch.tensor(detector_height / float(effective_rows), dtype=torch.float32),
        "YCtr": torch.tensor((det_cols - 1) * 0.5 - float(cfg.scanner.detectorColOffset), dtype=torch.float32),
        "ZCtr": torch.tensor((effective_rows - 1) * 0.5 + float(cfg.scanner.detectorRowOffset), dtype=torch.float32),
        "fan_ang_incr_rad": torch.tensor(fan_angle_increment, dtype=torch.float32),
        "cos_w2d": cosineWeights(det_cols, det_rows, detector_height,
                                 fan_angle_increment, source_detector_distance),
    }


def imageGrid(cfg):
    """create the x, y, and z coordinates of the reconstruction volume."""
    radius = torch.tensor(float(cfg.recon.fov) * 0.5, dtype=torch.float32)
    image_size = int(cfg.recon.imageSize)
    slice_count = int(cfg.recon.sliceCount)
    slice_thickness = torch.tensor(float(cfg.recon.sliceThickness), dtype=torch.float32)

    original_x, original_y, z_offset = cfg.recon.centerOffset
    x_offset = -float(original_y)
    y_offset = float(original_x)

    center_x = (image_size - 1) * 0.5
    center_y = (image_size - 1) * 0.5
    center_z = (slice_count - 1) * 0.5
    pixel_size = (2.0 * radius) / torch.tensor(float(image_size), dtype=torch.float32)

    x_coords = (torch.arange(image_size, dtype=torch.float32) - center_x) * pixel_size - x_offset
    y_coords = (torch.arange(image_size, dtype=torch.float32) - center_y) * pixel_size - y_offset
    z_coords = (torch.arange(slice_count, dtype=torch.float32) - center_z) * slice_thickness - float(z_offset)

    return {
        "RecSize": image_size,
        "FOIHeight": slice_count,
        "xGrid": x_coords.view(-1, 1, 1),
        "yGrid": y_coords.view(1, -1, 1),
        "zGrid": z_coords.view(1, 1, -1),
    }


def sourceGeometry(cfg, target_views, scan_radius):
    """Calculate source positions and directions for every selected view."""
    start_view_at_zero = cfg.protocol.viewCount / 2
    angle_offset = int(cfg.protocol.viewCount * cfg.recon.startAngle / 360)
    if cfg.recon.startAngle <= 180:
        start_view = start_view_at_zero + angle_offset
    else:
        start_view = start_view_at_zero - angle_offset

    angle_step = torch.tensor(2.0 * math.pi / float(target_views), dtype=torch.float32)
    rotation_direction = torch.tensor(float(cfg.protocol.rotationDirection), dtype=torch.float32)
    view_numbers = torch.arange(target_views, dtype=torch.float32)
    angles = (rotation_direction * view_numbers + float(start_view)) * angle_step

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    return {"SrcX_all": scan_radius * cos_angles, "SrcY_all": scan_radius * sin_angles,
            "DirX_all": cos_angles, "DirY_all": sin_angles}


def createFdkCache(cfg, target_views=500, view_block=50, raw_view_count=1000):
    """create all reusable geometry tensors required by FDK reconstruction."""
    target_views = int(target_views)
    raw_view_count = int(raw_view_count)
    detector = detectorGeometry(cfg)

    return FDKCache(
        ds_idx=selectViews(raw_view_count, target_views, int(view_block)),
        raw_views_full=raw_view_count,
        num_views=int(cfg.protocol.viewCount),
        **detector,                                              # the detector fields
        **imageGrid(cfg),                                        # the image-volume fields
        **sourceGeometry(cfg, target_views, detector["ScanR"]),  # the source fields
    )
