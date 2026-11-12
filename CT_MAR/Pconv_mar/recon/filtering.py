"""Ramp (Ram-Lak) filtering of projections — the FDK filter step.

Filtering is done as an FFT multiply with a precomputed frequency-domain kernel
("load_createhsp" builds it once per FOV via gecatsim's "createHSP").
"""
import math
import torch
from gecatsim.reconstruction.pyfiles.createHSP import createHSP

from recon.cache import RECO_FOV
from mar_common.scanner_config import config_downsampled


def filter_proj(proj, fan_ang_incr_rad, num_det_cols, loaded_tensor):
    """Apply the ramp filter to each detector row:
    - zero-pad -> FFT -> multiply by the ramp kernel -> inverse FFT.
    Returns the filtered projections, sign/scale corrected
    """
    filt_fft = loaded_tensor if loaded_tensor.dtype == torch.complex64 else loaded_tensor.to(torch.complex64)
    fft_size = filt_fft.shape[-1]

    pad_cols = max(0, fft_size - num_det_cols)
    proj_padded = torch.nn.functional.pad(proj, (0, 0, 0, 0, 0, pad_cols))
    proj_padded = proj_padded.permute(2, 1, 0)
    proj_fft = torch.fft.fft(proj_padded.to(torch.complex64), dim=-1)

    filt_fft = filt_fft.view(1, 1, fft_size)
    filtered_fft = proj_fft * filt_fft
    filtered = torch.fft.ifft(filtered_fft, dim=-1)
    filtered_im = filtered.imag
    filtered_im = filtered_im.permute(2, 1, 0)
    filtered_im = filtered_im[:num_det_cols, :, :]
    return -filtered_im / fan_ang_incr_rad


def load_createhsp(recon_fov=str, kernel_type="standard"):
    """Precompute the FFT-domain ramp filter for a given FOV ("head"/"body") 
    - sized to the next power of two above the detector width
    Returned once and reused every recon."""
    fov_mm = RECO_FOV[recon_fov]
    cfg = config_downsampled(fov_mm)
    det_cols = int(cfg.scanner.detectorColCount)
    nn = int(math.pow(2, (math.ceil(math.log2(abs(det_cols))) + 1)))
    filt_fft = createHSP(nn, kernel_type)
    return torch.as_tensor(filt_fft, dtype=torch.complex64)
