"""Differentiable fan-beam backprojection as custom CUDA kernels.

- This is the FDK backprojection step written so it has a gradient.
- The forward kernel "bp_fwd_kernel":
    reconstructs one image pixel per GPU thread by summing 
    bilinearly-interpolated, distance-weighted contributions from every projection view.
- The backward kernel "bp_adj_kernel" is its adjoint:
    it scatters image-space gradients back onto the detector bins (atomic adds).
- "BackprojFanbeamFn" wires both into PyTorch autograd;
- "backproject_cuda" is the public entry point.
Torch tensors are shared with the Numba kernels zero-copy via CuPy/DLPack.
"""
import math

import torch
from numba import cuda

try:
    import cupy as cp
except ImportError:
    cp = None


@cuda.jit
def _bp_fwd_kernel(proj_filtered, x_coords, y_coords, src_x, src_y, dir_x, dir_y, out,
                   scan_r, dist_d, delta_y, delta_z, y_ctr, z_ctr, z0, scale):
    """ reconstruct one image pixel per CUDA thread """
    x_idx, y_idx = cuda.grid(2)

    h = out.shape[0]
    w = out.shape[1]
    if x_idx >= h or y_idx >= w:
        return

    yl = proj_filtered.shape[0]
    zl_eff = proj_filtered.shape[1]
    view_count = proj_filtered.shape[2]

    x_mm = x_coords[x_idx]
    y_mm = y_coords[y_idx]
    acc = 0.0
    eps = 1e-30

    # the contribution of each projection view
    for i in range(view_count):
        sx = src_x[i]
        sy = src_y[i]
        dxv = dir_x[i]
        dyv = dir_y[i]

        dx = x_mm - sx
        dy = y_mm - sy
        dis = abs(x_mm * dxv + y_mm * dyv - scan_r)
        d2 = dx * dx + dy * dy + z0 * z0

        v_cor = (z0 * dist_d / dis) / delta_z + z_ctr
        u_cor = math.atan2(dx * dyv - dy * dxv, dis) / delta_y + y_ctr

        # find the four detector pixels around the projected image point
        ul = int(math.floor(u_cor))
        uu = ul + 1
        vl = int(math.floor(v_cor))
        vv = vl + 1
        if ul < 0 or uu >= yl or vl < 0 or vv >= zl_eff:
            continue

        alfa = uu - u_cor
        beta = vv - v_cor

        g_ul_vl = proj_filtered[ul, vl, i]
        g_ul_vv = proj_filtered[ul, vv, i]
        g_uu_vl = proj_filtered[uu, vl, i]
        g_uu_vv = proj_filtered[uu, vv, i]

        # bilinear interpolation for the detector vaules and apply FDK weights
        acc += (
            g_uu_vl * (1.0 - alfa) * beta
            + g_uu_vv * (1.0 - alfa) * (1.0 - beta)
            + g_ul_vl * alfa * beta
            + g_ul_vv * alfa * (1.0 - beta)
        ) / max(d2, eps)

    out[x_idx, y_idx] = acc * scale


@cuda.jit
def _bp_adj_kernel(grad_img, proj_grad, x_coords, y_coords, src_x, src_y, dir_x, dir_y,
                   scan_r, dist_d, delta_y, delta_z, y_ctr, z_ctr, z0, scale):
    """ sedn image gradients back to the filtered prjections"""
    x_idx, y_idx = cuda.grid(2)

    h = grad_img.shape[0]
    w = grad_img.shape[1]
    if x_idx >= h or y_idx >= w:
        return

    yl = proj_grad.shape[0]
    zl_eff = proj_grad.shape[1]
    view_count = proj_grad.shape[2]

    x_mm = x_coords[x_idx]
    y_mm = y_coords[y_idx]
    g = grad_img[x_idx, y_idx] * scale
    eps = 1e-30

    for i in range(view_count):
        sx = src_x[i]
        sy = src_y[i]
        dxv = dir_x[i]
        dyv = dir_y[i]

        dx = x_mm - sx
        dy = y_mm - sy
        dis = abs(x_mm * dxv + y_mm * dyv - scan_r)
        d2 = dx * dx + dy * dy + z0 * z0

        v_cor = (z0 * dist_d / dis) / delta_z + z_ctr
        u_cor = math.atan2(dx * dyv - dy * dxv, dis) / delta_y + y_ctr

        ul = int(math.floor(u_cor))
        uu = ul + 1
        vl = int(math.floor(v_cor))
        vv = vl + 1
        if ul < 0 or uu >= yl or vl < 0 or vv >= zl_eff:
            continue

        alfa = uu - u_cor
        beta = vv - v_cor
        coeff = g / max(d2, eps)
        # Several image pixels can update the same detecot pixel
        cuda.atomic.add(proj_grad, (uu, vl, i), (1.0 - alfa) * beta * coeff)
        cuda.atomic.add(proj_grad, (uu, vv, i), (1.0 - alfa) * (1.0 - beta) * coeff)
        cuda.atomic.add(proj_grad, (ul, vl, i), alfa * beta * coeff)
        cuda.atomic.add(proj_grad, (ul, vv, i), alfa * (1.0 - beta) * coeff)


def _to_numba(t):
    """ share a CUDA tensor with a Numba kernel witohut copying its data"""
    if cp is None:
        raise RuntimeError("CuPy is required for the CUDA backprojection path.")
    if not t.is_cuda:
        raise ValueError("Expected a CUDA tensor.")
    return cuda.as_cuda_array(cp.from_dlpack(t.detach()))


class BackprojFanbeamFn(torch.autograd.Function):
    """ connect the custom CUDA kerenels to pytorch autograd """
    @staticmethod
    def forward(ctx, proj_filtered, x_coords, y_coords, src_x, src_y, dir_x, dir_y,
                scan_r, dist_d, delta_y, delta_z, y_ctr, z_ctr, z0, scale):
        # run CUDA backprop and save vaules required for gradient
        if not proj_filtered.is_cuda:
            raise ValueError("CUDA backprojection requires CUDA tensors.")

        # CUDA kernels needs contiguous RAM for predictable indexing
        proj_filtered = proj_filtered.contiguous()
        x_coords, y_coords = x_coords.contiguous(), y_coords.contiguous()
        src_x, src_y = src_x.contiguous(), src_y.contiguous()
        dir_x, dir_y = dir_x.contiguous(), dir_y.contiguous()

        out = torch.zeros((x_coords.shape[0], y_coords.shape[0]), device=proj_filtered.device, dtype=torch.float32)
        # the same eight numbers go to both kernels, so build them once
        scalars = (float(scan_r), float(dist_d), float(delta_y), float(delta_z),
                   float(y_ctr), float(z_ctr), float(z0), float(scale))

        # here launch enough 16x16 thread blocks to cover the output image
        threads = (16, 16)
        blocks = (math.ceil(out.shape[0] / threads[0]), math.ceil(out.shape[1] / threads[1]))

        torch.cuda.synchronize(device=proj_filtered.device)
        _bp_fwd_kernel[blocks, threads](
            _to_numba(proj_filtered), _to_numba(x_coords), _to_numba(y_coords),
            _to_numba(src_x), _to_numba(src_y), _to_numba(dir_x), _to_numba(dir_y),
            _to_numba(out), *scalars,
        )
        cuda.synchronize()
        torch.cuda.synchronize(device=proj_filtered.device)

        #save tensors and scalar geometry for the backward CUDA kernel
        ctx.save_for_backward(proj_filtered, x_coords, y_coords, src_x, src_y, dir_x, dir_y)
        ctx.scalars = scalars
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """ calc gradients with respect to the input projection """
        proj_filtered, x_coords, y_coords, src_x, src_y, dir_x, dir_y = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_proj = torch.zeros_like(proj_filtered)

        threads = (16, 16)
        blocks = (math.ceil(grad_output.shape[0] / threads[0]),
                  math.ceil(grad_output.shape[1] / threads[1]))

        torch.cuda.synchronize(device=grad_output.device)
        _bp_adj_kernel[blocks, threads](
            _to_numba(grad_output), _to_numba(grad_proj),
            _to_numba(x_coords), _to_numba(y_coords),
            _to_numba(src_x), _to_numba(src_y), _to_numba(dir_x), _to_numba(dir_y),
            *ctx.scalars,
        )
        cuda.synchronize()
        torch.cuda.synchronize(device=grad_output.device)

        # only the projections get a gradient, the geometry arguments do not
        return (grad_proj,) + (None,) * 15


def backproject_cuda(proj_filtered, x_coords, y_coords, src_x, src_y, dir_x, dir_y,
                     scan_r, dist_d, delta_y, delta_z, y_ctr, z_ctr, z0, scale):
    """ Public entry point for the differentibale CUDA backprop """
    return BackprojFanbeamFn.apply(proj_filtered, x_coords, y_coords, src_x, src_y,
                                   dir_x, dir_y, scan_r, dist_d, delta_y, delta_z,
                                   y_ctr, z_ctr, z0, scale)
