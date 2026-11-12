"""direct order moment data-consistency losses motivated by  Helgason-Ludwig 
data consistency conditions in parallel beam.
"dc_loss" is the public training entrypoint used by "model.loss" and the
calibration tools
"""
import torch
from .rebin import rebin_proj, squeeze

DC_MODES = {
    "direct_m0": (1.0,),                          # M0 only
    "direct_m0_m1": (1.0, 0.25),                  # + light M1
    "direct_m0_m1_m2_light": (1.0, 0.25, 0.05),   # + very light M2
}


def norm_detector_coord(dc_pack, det_cols, device, dtype):
    """scale physical detector coordinate S to [-1, 1]"""
    s = dc_pack["S"].to(device=device, dtype=dtype)
    return (s / s.abs().amax()).view(1, 1, det_cols)


def direct_moment_loss(y_true, y_pred, dc_pack, order_betas):
    """return per-sample [B] low-order moment matching loss.
    "order_betas" is onee weight per moment order.
    """
    yt = squeeze(y_true)
    yp = squeeze(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must share [B,V,U] shape.")

    par_t = rebin_proj(yt, dc_pack)
    par_p = rebin_proj(yp, dc_pack)
    coord = norm_detector_coord(dc_pack, par_p.shape[-1], par_p.device, par_p.dtype)

    total = par_p.new_zeros((par_p.shape[0],))
    for order, beta in enumerate(order_betas):
        basis = coord.pow(order)
        pred_moment = (par_p * basis).sum(dim=-1)
        with torch.no_grad():
            true_moment = (par_t * basis).sum(dim=-1)
        total = total + beta * torch.mean(torch.abs(pred_moment - true_moment), dim=1)
    return total


def dc_loss(y_true, y_pred, dc_pack, mode="direct_m0_m1"):
    """public direct-moment DC loss returning per-sample [B]"""
    return direct_moment_loss(y_true, y_pred, dc_pack, order_betas=DC_MODES[mode])

