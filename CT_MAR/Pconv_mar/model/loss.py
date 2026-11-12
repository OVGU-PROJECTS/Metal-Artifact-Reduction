"""six loss terms, calibrated and scheduled.
1- rmse (whole sinogram)
2- hole (inside the metal trace)
3- image (recon RMSE)
4- edge (recon Sobel)
5- perceptual (recon VGG)
6- dc (physics data-consistency)

Two ideas keep them balanced:
- Calibration: each term is divided by a precomputed scale (from tools/calibrate.py) 
    so all six produce start with simialr magnitude.
- Curriculum (LossWeightScheduler): start with rmse only, then phase and introduce the harder
    image/edge/perc/dc terms in over training, so the
    network learns a basic sinogram fill before being pushed on image quality.
"""
import math
from dataclasses import dataclass

import torch

from data_inconsistency import DC_MODES, dc_loss


@dataclass
class LossWeightScheduler:
    """ controls when each loss term is activated and how fast it reaches full weight """
    # when each loss term is activated.
    start_ep_image: int
    start_ep_edge: int
    start_ep_hole: int
    start_ep_dc: int
    start_ep_perc: int
    
    # number of epochs the loss term takes to reach full weight.
    ramp_duration: int
    ramp_duration_dc: int
    
    # final training weight for each loss term
    weight_rmse: float
    weight_image: float
    weight_edge: float
    weight_hole: float
    weight_dc: float
    weight_perc: float

    def __post_init__(self):
        self.loss_weights = {
            "rmse": self.weight_rmse,
            "hole": 0.0,
            "image": 0.0,
            "edge": 0.0,
            "dc": 0.0,
            "perc": 0.0,
        }

    def _ramp(self, epoch, start, final, duration=None):
        """smoothly raise a weight from 0 over "duration" epochs,
        following a cosine curve (gentle on/off, no jumps)
        """
        if epoch < start:
            return 0.0
        t = (epoch - start) / (duration if duration is not None else self.ramp_duration)
        if t >= 1.0:
            return final
        t_cos = (1.0 - math.cos(math.pi * t)) / 2.0
        return final * t_cos

    def step_epoch(self, epoch):
        """recomputes every term's current weight for this epoch (call once per epoch)"""
        epoch += 1
        self.loss_weights["rmse"] = self.weight_rmse
        self.loss_weights["image"] = self._ramp(epoch, self.start_ep_image, self.weight_image)
        self.loss_weights["hole"] = self._ramp(epoch, self.start_ep_hole, self.weight_hole)
        self.loss_weights["dc"] = self._ramp(epoch, self.start_ep_dc, self.weight_dc, duration=self.ramp_duration_dc)
        self.loss_weights["edge"] = self._ramp(epoch, self.start_ep_edge, self.weight_edge)
        self.loss_weights["perc"] = self._ramp(epoch, self.start_ep_perc, self.weight_perc)


class CombinedLoss:
    """Weighted sum of the six calibrated loss terms, using the scheduler's current weights """
    
    def __init__(self, loss_weight_scheduler, pconv_model, dc_mode):
        if dc_mode not in DC_MODES:
            raise ValueError(
                f"Unknown dc mode: {dc_mode}. Expected one of {DC_MODES}."
            )
        self.loss_weight_scheduler = loss_weight_scheduler
        self.pconv_model = pconv_model
        self.dc_mode = dc_mode


    def __call__(self, y_true, y_pred):
        """
        - y_pred pack: [predicted sinogram, mask, raw_pred]
        - y_true packs: [target sinogram, image-pack]. The image-pack carries the 256x256 GT image
        and stores the head/body FOV code (head:0, body:1) in its bottom-right corner pixel.
        """
        weights = self.loss_weight_scheduler.loss_weights
        w_rmse = y_true.new_tensor(weights["rmse"])
        w_hole = y_true.new_tensor(weights["hole"])
        w_dc = y_true.new_tensor(weights["dc"])
        w_img = y_true.new_tensor(weights["image"])
        w_edge = y_true.new_tensor(weights["edge"])
        w_perc = y_true.new_tensor(weights["perc"])

        y_pred_proj = y_pred[:, 0:1]
        mask = y_pred[:, 1:2]

        y_true_proj = y_true[:, 0:1]
        img_pack = y_true[:, 1:2]
        img_true256 = img_pack[:, :, :256, :256]

        def zeros_b():
            return torch.zeros((y_true_proj.shape[0],), device=y_true_proj.device, dtype=y_true_proj.dtype)

        def denorm_true_pred():
            """
            - The network sees z-scored sinograms,
            - but the image and physics (DC) losses need true attenuation values 
            — therefore undo the normalization here.
            """
            fov_codes = img_pack[:, 0, -1, -1]
            fov_i = torch.round(fov_codes).long()
            mu = self.pconv_model.fov_mean[fov_i].view(-1, 1, 1, 1)
            sg = self.pconv_model.fov_std[fov_i].view(-1, 1, 1, 1)
            y_true_den = y_true_proj * sg + mu
            y_pred_den = y_pred_proj * sg + mu
            return fov_codes, y_true_den, y_pred_den

        rmse_term = (
            self.pconv_model.RMSE(y_true_proj, y_pred_proj) / (self.pconv_model.rmse_scale + 1e-8)
            if weights["rmse"] > 1e-8
            else zeros_b()
        )
        hole_term = (
            self.pconv_model.loss_hole(mask, y_true_proj, y_pred_proj) / (self.pconv_model.hole_scale + 1e-8)
            if weights["hole"] > 1e-8
            else zeros_b()
        )

        if weights["dc"] > 1e-8:
            _, y_true_den, y_pred_den = denorm_true_pred()
            dc_term = dc_loss(y_true_den, y_pred_den, self.pconv_model.dc_pack,
                              mode=self.dc_mode) / (self.pconv_model.dc_scale + 1e-8)
        else:
            dc_term = zeros_b()

        do_img_edge_perc = (weights["image"] > 1e-8) or (weights["edge"] > 1e-8) or (weights["perc"] > 1e-8)
        if do_img_edge_perc:
            fov_codes, _, y_pred_den = denorm_true_pred()
            if weights["perc"] > 1e-8:
                img_rmse_val, edge_val, perc_val = self.pconv_model.img_loss(
                    img_true256, y_pred_den, fov_codes=fov_codes, proj_mask=mask, return_perc=True)
                perc_term = perc_val / (self.pconv_model.perc_scale + 1e-8)
            else:
                img_rmse_val, edge_val = self.pconv_model.img_loss(
                    img_true256, y_pred_den, fov_codes=fov_codes, proj_mask=mask, return_perc=False)
                perc_term = zeros_b()
            img_term = img_rmse_val / (self.pconv_model.img_scale + 1e-8)
            edge_term = edge_val / (self.pconv_model.edge_scale + 1e-8)
        else:
            img_term = zeros_b()
            edge_term = zeros_b()
            perc_term = zeros_b()

        # Normalize by the sum of active weights so the loss scale stays stable as
        # terms ramp in/out over the curriculum.
        w_sum = torch.maximum(y_true.new_tensor(1e-8), w_rmse + w_hole + w_img + w_edge + w_dc + w_perc)
        total_loss_val = (
            w_rmse * rmse_term
            + w_hole * hole_term
            + w_img * img_term
            + w_edge * edge_term
            + w_dc * dc_term
            + w_perc * perc_term
        ) / w_sum
        return total_loss_val

    def breakdown(self, y_true, y_pred):
        """Return calibrated, unweighted term means for diagnostic CSV logging.
        Unlike __call__, this does not apply curriculum weights or divide by their sum.
        """
        
        y_pred_proj = y_pred[:, 0:1]
        mask        = y_pred[:, 1:2]
        y_true_proj = y_true[:, 0:1]
        img_pack    = y_true[:, 1:2]
        img_true256 = img_pack[:, :, :256, :256]

        fov_codes = img_pack[:, 0, -1, -1]
        fov_i = torch.round(fov_codes).long()
        mu = self.pconv_model.fov_mean[fov_i].view(-1, 1, 1, 1)
        sg = self.pconv_model.fov_std[fov_i].view(-1, 1, 1, 1)
        y_true_den = y_true_proj * sg + mu
        y_pred_den = y_pred_proj * sg + mu

        rmse = self.pconv_model.RMSE(y_true_proj, y_pred_proj) / (self.pconv_model.rmse_scale + 1e-8)
        hole = self.pconv_model.loss_hole(mask, y_true_proj, y_pred_proj) / (self.pconv_model.hole_scale + 1e-8)
        dc   = dc_loss(y_true_den, y_pred_den, self.pconv_model.dc_pack,
                       mode=self.dc_mode) / (self.pconv_model.dc_scale + 1e-8)
        img_rmse, edge, perc = self.pconv_model.img_loss(
            img_true256, y_pred_den, fov_codes=fov_codes, proj_mask=mask, return_perc=True
        )
        img  = img_rmse / (self.pconv_model.img_scale  + 1e-8)
        edge = edge     / (self.pconv_model.edge_scale  + 1e-8)
        perc = perc     / (self.pconv_model.perc_scale  + 1e-8)

        result = {
            "rmse": rmse.mean().item(),
            "hole": hole.mean().item(),
            "dc":   dc.mean().item(),
            "img":  img.mean().item(),
            "edge": edge.mean().item(),
            "perc": perc.mean().item(),
        }

        return result, y_pred_den, fov_codes
