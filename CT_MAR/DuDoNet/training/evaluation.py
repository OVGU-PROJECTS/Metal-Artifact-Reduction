"""loss and SSIM evaluation for DuDoNet training"""
import math

import torch

from mar_common.units import mu2hu
from metrics.numpy_ssim import (
    ssim_aapm_masked,
    load_metal_exclusion_mask,
)

from models import dudonet_loss
from .data import load_batch


def evaluate_cases(model, case_order, batch_size, device, *, calculate_loss):
    """Evaluate a fixed case order; optionally skip loss."""
    run = {"L": 0.0, "Gs": 0.0, "RC": 0.0, "Gi": 0.0}
    ssim_sums = {"ssim_aapm": 0.0}
    count = 0
    steps = math.ceil(len(case_order) / batch_size)

    with torch.no_grad():
        for step in range(steps):
            case_batch = case_order[step * batch_size:(step + 1) * batch_size]
            if not case_batch:
                break

            y_li, mt, y_gt, x_gt, fov = load_batch(case_batch, device=device)
            outputs = model(y_li, mt, fov)
            if calculate_loss:
                _, terms = dudonet_loss(outputs, y_gt, x_gt)
                for key in run:
                    run[key] += terms[key]

            dilated_masks = [load_metal_exclusion_mask(case) for case in case_batch]

            reference_hu = mu2hu(x_gt[:, 0].detach().cpu().numpy())
            prediction_hu = mu2hu(outputs["X_out"][:, 0].detach().cpu().numpy())
            for index, dilated_mask in enumerate(dilated_masks):
                ssim_sums["ssim_aapm"] += ssim_aapm_masked(
                    reference_hu[index], prediction_hu[index], dilated_mask
                )
                count += 1

    if count == 0:
        raise ValueError("SSIM evaluation is empty")
    run = {key: value / max(steps, 1) for key, value in run.items()}
    scores = {key: value / count for key, value in ssim_sums.items()}
    return run, scores
