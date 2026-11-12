# DuDoNet of Lin et al. (CVPR 2019), rebuilt from the paper: https://doi.org/10.1109/CVPR.2019.01076
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import SinoNet, IENetUNet, RIL

class DuDoNet2RIL(nn.Module):
    """SE-Net + two RILs (head/body) + IE-Net.
    Inputs:
      y_li: (B,1,V,Bins)
      mt:   (B,1,V,Bins)
      fov:  either an int (0=head,1=body) or a tensor of shape (B,)

    Output dict keys:
      Y_out  sinogram-domain: SE-Net enhanced projection      (vs y_gt -> L_Gs)
      X_hat  image-domain: RIL reconstruction of Y_out         (vs x_gt -> L_RC)
      X_LI   image-domain: RIL reconstruction of the LI input  (IE-Net prior)
      X_out  image-domain: X_LI + IE-Net residual = final MAR  (vs x_gt -> L_Gi)
    """

    def __init__(self, fov_head_mm: float, fov_body_mm: float, *, img_size: int = 512):
        super().__init__()
        self.se_net = SinoNet()
        self.ril_head = RIL(fov_mm=float(fov_head_mm), img_size=int(img_size))
        self.ril_body = RIL(fov_mm=float(fov_body_mm), img_size=int(img_size))
        self.ie_net = IENetUNet(in_ch=2)

    def _ril_apply_mixed(self, fan_sino: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        """apply head/body RIL per sample and stitch back into a full tensor."""
        assert fan_sino.ndim == 4 and fov.ndim == 1 and fan_sino.shape[0] == fov.shape[0]
        B = fan_sino.shape[0]

        # Output buffer (same device/dtype as RIL output)
        out = torch.empty((B, 1, self.ril_head.img_size, self.ril_head.img_size),
                          device=fan_sino.device, dtype=torch.float32)

        idx_head = (fov == 0).nonzero(as_tuple=False).squeeze(1)
        idx_body = (fov == 1).nonzero(as_tuple=False).squeeze(1)

        if idx_head.numel() > 0:
            out[idx_head] = self.ril_head(fan_sino[idx_head])
        if idx_body.numel() > 0:
            out[idx_body] = self.ril_body(fan_sino[idx_body])

        return out

    def forward(
        self,
        y_li: torch.Tensor,
        mt: torch.Tensor,
        fov: Union[int, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # 1) sinogram enhancement
        y_out, _ = self.se_net(y_li, mt)

        # 2) geometry selection
        if isinstance(fov, int):
            ril = self.ril_head if fov == 0 else self.ril_body
            x_hat = ril(y_out)
            x_li = ril(y_li)
        else:
            fov_t = fov.to(device=y_li.device, dtype=torch.long).view(-1)
            x_hat = self._ril_apply_mixed(y_out, fov_t)
            x_li = self._ril_apply_mixed(y_li, fov_t)

        # 3) image enhancement (residual)
        ie_in = torch.cat([x_li, x_hat], dim=1)
        x_out = x_li + self.ie_net(ie_in)

        return {"Y_out": y_out, "X_hat": x_hat, "X_LI": x_li, "X_out": x_out}


def dudonet_loss(outputs: Dict[str, torch.Tensor], y_gt: torch.Tensor, x_gt: torch.Tensor):
    """paper-stylr loss: L = L_Gs + L_RC + L_Gi using L1."""
    L_Gs = F.l1_loss(outputs["Y_out"], y_gt)
    L_RC = F.l1_loss(outputs["X_hat"], x_gt)
    L_Gi = F.l1_loss(outputs["X_out"], x_gt)
    L = L_Gs + L_RC + L_Gi
    terms = {"L": float(L.detach()), "Gs": float(L_Gs.detach()), "RC": float(L_RC.detach()), "Gi": float(L_Gi.detach())}
    return L, terms


def dudonet_loss_mixed(
    model: nn.Module,
    y_li: torch.Tensor,
    mt: torch.Tensor,
    y_gt: torch.Tensor,
    x_gt: torch.Tensor,
    fov: torch.Tensor,
):
    """Convenience wrapper used by trainer.py.
    works with a plain model or a DDP-wrapped model
    """
    outputs = model(y_li, mt, fov)
    return dudonet_loss(outputs, y_gt, x_gt)
