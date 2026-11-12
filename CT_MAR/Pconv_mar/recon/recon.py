"""FDK reconstruction 
- "reconstruct" is the public entry point.
- uses the compiled CUDA kernel.
"""
import math

import torch

try:
    from recon.cuda_backproj import backproject_cuda
    _CUDA_BACKPROJ_AVAILABLE = True
except Exception:
    backproject_cuda = None
    _CUDA_BACKPROJ_AVAILABLE = False

from recon.filtering import filter_proj


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device=device)
    return x


def reconstruct(config, proj, loaded_tensor):
    """[views, columns] sinogram in, reconstructed image out"""
    if not hasattr(config, "_fdk_cache"):
        raise AttributeError("config._fdk_cache is required.")
    cache = config._fdk_cache

    proj = torch.as_tensor(proj, dtype=torch.float32).transpose(0, 1).unsqueeze(1)
    if proj.shape[-1] != cache.raw_views_full:
        raise ValueError(f"Expected {cache.raw_views_full} views, got {proj.shape[-1]}")
    # keep only the views the cache selected
    proj = torch.index_select(proj, 2, cache.ds_idx.to(device=proj.device))
    return recon_fdk(config, proj, loaded_tensor).squeeze()


def recon_fdk(cfg, proj, loaded_tensor):
    """core FDK:
    1- cosine-weight the projections
    2- ramp-filter them,
    3- then backproject (CUDA kernel) into a single-slice image.
    4- Returns the (RecSize, RecSize, 1) volume
    """
    cache = cfg._fdk_cache
    device = proj.device

    if device.type != "cuda":
        raise RuntimeError("FDK reconstruction requires CUDA.")
    if not _CUDA_BACKPROJ_AVAILABLE:
        raise RuntimeError("CUDA backprojection dependencies are unavailable.")
    if cache.FOIHeight != 1:
        raise RuntimeError("CUDA backprojection supports one slice only.")

    # FDK stage 1: cosine weighting (precomputed per-pixel weights).
    proj_cos = proj * to_device(cache.cos_w2d, device)[:, :, None]

    # FDK stage 2: ramp filtering along the detector.
    proj_filtered = filter_proj(proj_cos, to_device(cache.fan_ang_incr_rad, device),
                                cache.YL, to_device(loaded_tensor, device))

    # a single detector row is duplicated so the kernel has two rows to interpolate between
    if cache.ZL_orig == 1:
        proj_filtered = torch.cat([proj_filtered, proj_filtered], dim=1)

    # the kernel wants the pixel coordinates on the GPU and the geometry as plain floats
    x_coords = to_device(cache.xGrid, device).view(-1)
    y_coords = to_device(cache.yGrid, device).view(-1)
    z0 = float(cache.zGrid.view(-1)[0])
    scan_r = to_device(cache.ScanR, device)
    scale = float(-scan_r * torch.tensor(math.pi, dtype=torch.float32, device=device) / cache.num_views)

    # FDK stage 3: backproject every filtered view into the image grid.
    rec2d = backproject_cuda(
        proj_filtered, x_coords, y_coords,
        to_device(cache.SrcX_all, device), to_device(cache.SrcY_all, device),
        to_device(cache.DirX_all, device), to_device(cache.DirY_all, device),
        float(cache.ScanR), float(cache.DistD), float(cache.DeltaY), float(cache.DeltaZ),
        float(cache.YCtr), float(cache.ZCtr), z0, scale,
    )

    rec = torch.zeros((cache.RecSize, cache.RecSize, cache.FOIHeight), dtype=torch.float32, device=device)
    rec[:, :, 0] = rec2d
    return rec
