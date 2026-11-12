"""RIL: a differentiable CT reconstruction layer (DuDoNet step 2 of 3).
- Reconstructs an image from a sinogram via FBP
- written so gradients flow back through it 
- the image-domain loss must reach the upstream sinogram network.

"RIL.forwar" pipeline:
  1. Fan -> parallel rebinning ("Fan2ParallelRebin"), the form FBP expects.
  2. Ramp (Ram-Lak) filtering along the detector axis.
  3. Backprojection: smear each filtered view back across the image and sum.

Runs entirely on GPU: Numba CUDA kernels bound to torch tensors zero-copy via
CuPy/DLPack.
"""
import math, os
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import cuda

from mar_common.scanner_config import recon_config
from geometry.fanbeam import (
    compute_rot_angles,
    compute_bin_pos_parallel,
    wrap_angle_0_2pi,
    compute_fan_ang_incr_rad,
    compute_cent_det_pix,
)
# =============================================================================
# DLPack bridge (GPU, no CPU)
# =============================================================================
try:
    import cupy as cp
except ImportError as e:
    raise ImportError(
        "This hybrid Torch+Numba version requires CuPy for zero-copy GPU interop via DLPack.\n"
        "Install a CUDA-matching CuPy build (e.g. cupy-cuda11x / cupy-cuda12x)."
    ) from e

# params indices 
P_TMIN   = 0
P_INV_DT = 1
P_DX     = 2
P_OFFX   = 3
P_OFFY   = 4
P_DTHETA = 5

# =============================================================================
# CUDA kernels (unchanged logic; only called on GPU tensors)
# =============================================================================
@cuda.jit
def bp_fwd_kernel(trig, proj, rec, params):
    """backprojection (forward pass). One GPU thread per output pixel (x,y,n).
    For each pixel, sum over all views: project the pixel onto the detector
    (t = x*cos - y*sin), linearly interpolate the filtered sinogram there, and
    accumulate. Pixel value = (sum over views) * dtheta.

    trig  : (V,2)  [cos(theta), sin(theta)] per view
    proj  : (N,V,B) filtered parallel sinogram
    rec   : (N,H,W) output image
    params: (6,) [t_min, inv_dt, dx, offx, offy, dtheta]
    """
    
    x, y, n = cuda.grid(3)

    N = proj.shape[0]
    V = proj.shape[1]
    B = proj.shape[2]
    H = rec.shape[1]
    W = rec.shape[2]
    if n >= N or x >= H or y >= W:
        return

    t_min  = params[P_TMIN]
    inv_dt = params[P_INV_DT]
    dx     = params[P_DX]
    offx   = params[P_OFFX]
    offy   = params[P_OFFY]
    dtheta = params[P_DTHETA]

    cx = (H - 1) * 0.5
    cy = (W - 1) * 0.5

    # pixel -> mm (with centerOffset)
    x_mm = (x - cx) * dx - offx
    y_mm = (y - cy) * dx - offy

    acc = 0.0
    for i in range(V):
        c = trig[i, 0]
        s = trig[i, 1]

        # Minus sign is the mirroring fix; should be kept consistent with the adjoint.
        t = x_mm * c - y_mm * s

        u = (t - t_min) * inv_dt
        j0 = int(math.floor(u))
        if j0 < 0 or j0 >= (B - 1):
            continue

        w = u - j0
        v0 = proj[n, i, j0]
        v1 = proj[n, i, j0 + 1]
        acc += (1.0 - w) * v0 + w * v1

    rec[n, x, y] = acc *  dtheta #math.pi / V


@cuda.jit
def bp_adj_kernel(trig, grad_img, grad_proj, params):
    """adjoint of backprojection — the gradient (backward pass)- to sinogram space
    trig     : (V,2)
    grad_img : (N,H,W)   incoming gradient w.r.t. the reconstructed image
    grad_proj: (N,V,B)   outgoing gradient w.r.t. the sinogram (atomic adds)
    params   : (6,)
    """
    x, y, n = cuda.grid(3)

    N = grad_proj.shape[0]
    V = grad_proj.shape[1]
    B = grad_proj.shape[2]
    H = grad_img.shape[1]
    W = grad_img.shape[2]
    if n >= N or x >= H or y >= W:
        return

    t_min  = params[P_TMIN]
    inv_dt = params[P_INV_DT]
    dx     = params[P_DX]
    offx   = params[P_OFFX]
    offy   = params[P_OFFY]
    dtheta = params[P_DTHETA]

    cx = (H - 1) * 0.5
    cy = (W - 1) * 0.5

    x_mm = (x - cx) * dx - offx
    y_mm = (y - cy) * dx - offy

    g = grad_img[n, x, y] * dtheta

    for i in range(V):
        c = trig[i, 0]
        s = trig[i, 1]

        t = x_mm * c - y_mm * s
        u = (t - t_min) * inv_dt
        j0 = int(math.floor(u))
        if j0 < 0 or j0 >= (B - 1):
            continue

        w = u - j0
        cuda.atomic.add(grad_proj, (n, i, j0),     (1.0 - w) * g)
        cuda.atomic.add(grad_proj, (n, i, j0 + 1), w * g)


# =============================================================================
# Torch <-> CuPy <-> Numba helpers (GPU zero-copy)
# =============================================================================
def torch_to_cupy(t: torch.Tensor) -> "cp.ndarray":
    """zero-copy view of a CUDA torch Tensor as a CuPy array via DLPack"""
    assert t.is_cuda, "Expected CUDA tensor"
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t))

def cupy_to_numba(a: "cp.ndarray"):
    """zero-copy view of a CuPy array as a Numba device array """
    return cuda.as_cuda_array(a)

# =============================================================================
# Custom autograd Function: backprojection forward + adjoint backward
# =============================================================================
class BackprojFn(torch.autograd.Function):
    """
    - forward = backproject
    - backward = adjoint.
    - Gives autograd a backward pass for the otherwise-opaque kernel.
    """
    @staticmethod
    def forward(ctx, proj_filt: torch.Tensor, trig_t: torch.Tensor, params_t: torch.Tensor, img_size: int):
        """
        proj_filt: (N,V,B) CUDA float32
        trig_t   : (V,2)   CUDA float32 (no grad)
        params_t : (6,)    CUDA float32 (no grad)
        returns  : (N,H,W) CUDA float32
        """
        assert proj_filt.is_cuda and trig_t.is_cuda and params_t.is_cuda
        assert proj_filt.ndim == 3 and trig_t.ndim == 2 and params_t.ndim == 1
        assert proj_filt.dtype == torch.float32, "Use float32 for kernels"
        assert trig_t.dtype == torch.float32 and params_t.dtype == torch.float32

        proj_filt = proj_filt.contiguous()
        trig_t    = trig_t.contiguous()
        params_t  = params_t.contiguous()

        N, V, B = proj_filt.shape
        H = W = int(img_size)

        rec = torch.zeros((N, H, W), device=proj_filt.device, dtype=torch.float32)

        # Zero-copy views into the same GPU memory
        proj_cp   = torch_to_cupy(proj_filt)
        trig_cp   = torch_to_cupy(trig_t)
        params_cp = torch_to_cupy(params_t)
        rec_cp    = torch_to_cupy(rec)

        proj_nb   = cupy_to_numba(proj_cp)
        trig_nb   = cupy_to_numba(trig_cp)
        params_nb = cupy_to_numba(params_cp)
        rec_nb    = cupy_to_numba(rec_cp)

        TPB = 16
        blocks  = (math.ceil(H / TPB), math.ceil(W / TPB), N)
        threads = (TPB, TPB, 1)

        # Sync around the kernel
        torch.cuda.synchronize(device=proj_filt.device)
        bp_fwd_kernel[blocks, threads](trig_nb, proj_nb, rec_nb, params_nb)
        cuda.synchronize()
        torch.cuda.synchronize(device=proj_filt.device)

        # Save what backward needs
        ctx.save_for_backward(trig_t, params_t)
        ctx.shape = (N, V, B)
        ctx.img_size = H
        return rec  # CUDA tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        grad_output: (N,H,W) CUDA
        returns grad_proj: (N,V,B) CUDA
        """
        trig_t, params_t = ctx.saved_tensors
        assert grad_output.is_cuda and grad_output.dtype == torch.float32

        grad_output = grad_output.contiguous()
        N, H, W = grad_output.shape
        N2, V, B = ctx.shape
        assert N == N2 and H == ctx.img_size and W == ctx.img_size

        grad_proj = torch.zeros((N, V, B), device=grad_output.device, dtype=torch.float32)

        grad_img_cp  = torch_to_cupy(grad_output)
        grad_proj_cp = torch_to_cupy(grad_proj)
        trig_cp      = torch_to_cupy(trig_t)
        params_cp    = torch_to_cupy(params_t)

        grad_img_nb  = cupy_to_numba(grad_img_cp)
        grad_proj_nb = cupy_to_numba(grad_proj_cp)
        trig_nb      = cupy_to_numba(trig_cp)
        params_nb    = cupy_to_numba(params_cp)

        TPB = 16
        blocks  = (math.ceil(H / TPB), math.ceil(W / TPB), N)
        threads = (TPB, TPB, 1)

        torch.cuda.synchronize(device=grad_output.device)
        bp_adj_kernel[blocks, threads](trig_nb, grad_img_nb, grad_proj_nb, params_nb)
        cuda.synchronize()
        torch.cuda.synchronize(device=grad_output.device)

        return grad_proj, None, None, None


def ramp_H_rfft(bins: int, dt: float, device, dtype):
    """Ram-Lak ramp filter |f| in the frequency domain (the FBP filtering step) """
    f = torch.fft.rfftfreq(bins, d=float(dt)).to(device=device, dtype=dtype)
    return torch.abs(f)  # (bins//2+1,)

# =============================================================================
# Fan -> Parallel rebin 
# =============================================================================
class Fan2ParallelRebin(nn.Module):
    """Resample a fan-beam sinogram onto parallel-beam geometry
    - Assumes:
        1- uniformly sampled views (Δβ = 2π/V).
        2- equi-angular fan bins.
        3- the acquisition convention beta_k = wrap(start - k*Δβ). 
    - output keeps the input's view/bin order.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        V = int(cfg.protocol.viewCount)
        B = int(cfg.scanner.detectorColCount)

        sid = float(cfg.scanner.sid)

        # angles and increments from scanner config helpers
        fan_inc = float(compute_fan_ang_incr_rad(cfg))     # rad per detector column
        cent    = float(compute_cent_det_pix(cfg))         # detector center (cols)

        start = float(cfg.recon.startAngle)
        if abs(start) > 2.0 * math.pi:  
            start = start * math.pi / 180.0
        start = start + 3.0 * math.pi / 2.0  

        d_beta = 2.0 * math.pi / float(V)

        # parallel detector coordinate t and corresponding gamma
        t_bins = np.asarray(compute_bin_pos_parallel(cfg), dtype=np.float32)  # (B,)
        gamma  = np.arcsin(np.clip(t_bins / sid, -1.0, 1.0)).astype(np.float32)    # (B,)

        # build 2D lookup (V,B)
        beta = np.asarray(compute_rot_angles(cfg), dtype=np.float32)    # (V,) acquisition order
        beta = wrap_angle_0_2pi(beta).astype(np.float32)

        # mapping: lookup angle = beta + gamma  (shape V,B)
        beta2d = beta[:, None]                      # (V,1)
        g2d    = gamma[None, :]                     # (1,B)
        beta_lookup = wrap_angle_0_2pi(beta2d + g2d).astype(np.float32)  # (V,B)

        # view index from beta_lookup using modular arithmetic:
        # beta_k = wrap(start - k*d_beta)  =>  k = ((start - beta_lookup) mod 2π)/d_beta
        two_pi = np.float32(2.0 * math.pi)
        delta = (np.float32(start) - beta_lookup) % two_pi
        v_idx = (delta / np.float32(d_beta)).astype(np.float32)  #(V,B) in [0,V)

        # detector bin index from gamma (equi-angular fan):
        # gamma = (col - cent)*fan_inc  => col = cent + gamma/fan_inc
        bin_idx = (np.float32(cent) + (g2d / np.float32(fan_inc))).astype(np.float32)  # (1,B)
        bin_idx = np.broadcast_to(bin_idx, v_idx.shape).astype(np.float32)             # (V,B)

        # pad views to allow fractional sampling near the wrap seam safely
        pad_v = int(np.ceil(np.max(np.abs(gamma)) / max(d_beta, 1e-6))) + 2
        Vext = V + 2 * pad_v
        v_idx_pad = v_idx + np.float32(pad_v)

        # normalized grid for align_corners=True
        y = (2.0 * v_idx_pad / np.float32(Vext - 1)) - 1.0     # (V,B)
        x = (2.0 * bin_idx    / np.float32(B    - 1)) - 1.0    # (V,B)
        grid = np.stack([x, y], axis=-1).astype(np.float32)    # (V,B,2)

        self.register_buffer("grid", torch.from_numpy(grid)[None, ...], persistent=False)  # (1,V,B,2)
        self.V = V
        self.B = B
        self.pad_v = pad_v

    def forward(self, fan_sino: torch.Tensor):
        """
        fan_sino: (N,1,V,B) in ORIGINAL acquisition/bin order
        returns : (N,1,V,B) parallel-beam sinogram, SAME order (no sorting)
        """
        assert fan_sino.ndim == 4 and fan_sino.shape[2] == self.V and fan_sino.shape[3] == self.B

        p = self.pad_v
        x_ext = torch.cat([fan_sino[:, :, -p:, :], fan_sino, fan_sino[:, :, :p, :]], dim=2)  # (N,1,Vext,B)

        grid = self.grid.to(device=fan_sino.device, dtype=fan_sino.dtype)
        grid = grid.expand(fan_sino.shape[0], -1, -1, -1)  # (N,V,B,2)

        para = F.grid_sample(
            x_ext, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        return para  # (N,1,V,B)

# =============================================================================
# RIL (Reconstruction Inversion Layer): rebin -> filter -> backproject
# =============================================================================
class RIL(nn.Module):
    """Reconstruction layer: rebin -> ramp filter -> backproject.

    Everything stays in acquisition view order (rebin output, theta, trig), and
    dtheta is positive since the kernel computes rec = acc * dtheta.
    """
    def __init__(self, fov_mm: float, img_size: int = 512):
        super().__init__()
        self.cfg = recon_config(float(fov_mm))
        self.img_size = int(img_size)

        V = int(self.cfg.protocol.viewCount)
        B = int(self.cfg.scanner.detectorColCount)

        self.rebin = Fan2ParallelRebin(self.cfg)

        # theta in ACQUISITION order
        beta = np.asarray(compute_rot_angles(self.cfg), dtype=np.float32)  # (V,) acquisition order
        theta = wrap_angle_0_2pi(beta).astype(np.float32) # (V,) parallel angles

        trig = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)  # (V,2)
        self.register_buffer("trig_t", torch.from_numpy(trig), persistent=False)

        # t-grid in mm
        t_bins = np.asarray(compute_bin_pos_parallel(self.cfg), dtype=np.float32)  # (B,)
        t_min = float(t_bins[0])
        dt = float((t_bins[-1] - t_bins[0]) / (B - 1))
        inv_dt = 1.0 / dt

        # pixel size and offsets
        dx = float(fov_mm) / float(self.img_size)
        offx = float(self.cfg.recon.centerOffset[0])
        offy = float(self.cfg.recon.centerOffset[1])

        # per-view angular weight: pi/V (not 2pi/V) since 360° covers each line twice.
        dtheta = float(math.pi / float(V)) if V > 1 else float(math.pi)

        params = np.array([t_min, inv_dt, dx, offx, offy, dtheta], dtype=np.float32)
        self.register_buffer("params_t", torch.from_numpy(params), persistent=False)

        self.dt = dt
        self.V = V
        self.B = B

    def forward(self, fan_sino: torch.Tensor):
        """
        fan_sino: (N,1,V,B) fan sinogram in ORIGINAL acquisition/bin order
        returns : (N,1,H,W) reconstructed image
        """
        assert fan_sino.ndim == 4 and fan_sino.shape[2] == self.V and fan_sino.shape[3] == self.B
        assert fan_sino.is_cuda, "This version is GPU-only end-to-end."

        # 1) fan -> parallel (same order as input)
        para = self.rebin(fan_sino).squeeze(1)  # (N,V,B)

        # 2) ramp filter along bins
        Hf = ramp_H_rfft(self.B, self.dt, para.device, para.dtype)  # (B//2+1,)
        S  = torch.fft.rfft(para, dim=-1)
        para_filt = torch.fft.irfft(S * Hf[None, None, :], n=self.B, dim=-1).contiguous()

        # 3) backprojection (custom autograd, CUDA kernels)
        trig_t   = self.trig_t.to(device=para.device, dtype=torch.float32)
        params_t = self.params_t.to(device=para.device, dtype=torch.float32)

        img = BackprojFn.apply(para_filt.float(), trig_t, params_t, self.img_size)  # (N,H,W)
        return img.unsqueeze(1)                                                     # (N,1,H,W)





