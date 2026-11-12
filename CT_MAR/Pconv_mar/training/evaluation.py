"""Validation routines for training/trainer.py.

- "evaluate" returns the weighted combined loss.
- "evaluate_breakdown" returns the calibrated, unweighted term averages
plus the image-domain metrics.
"""
import os

import numpy as np
import torch

from mar_common.io import read_mha
from mar_common.units import hu2mu
from recon import metric_fdk_setups, reconstruct

from metrics.torch_ssim import ssim_aapm, ssim_trace, load_image_masks
from metrics.torch_hu_metrics import mae_near_metal, streak_edge_err
from training.distributed import to_device, reduce_mean


def load_img_true512(case_paths, device):
    imgs = np.stack([hu2mu(read_mha(os.path.join(p, "output_img.mha"))) for p in case_paths])
    return torch.from_numpy(imgs[:, None].astype(np.float32)).to(device)


def reconstruct_metric_grid(y_pred_den, fov_codes):
    """Reconstruct a batch of denormalized sinograms for the metrics."""
    ctx = metric_fdk_setups()
    imgs = []
    for idx in range(y_pred_den.shape[0]):
        proj2d = y_pred_den[idx, 0]
        fov_name = "body" if fov_codes[idx].item() > 0.5 else "head"
        cfg, hsp = ctx[fov_name]
        imgs.append(reconstruct(cfg, proj2d, hsp.to(device=proj2d.device)))
    return torch.stack(imgs, dim=0).unsqueeze(1)


def evaluate(model, loss_fn, loader, device, max_steps):
    """combined validation loss over up to max_steps batches, reduced across GPUs."""
    model.eval()
    loss_sum = 0.0
    loss_count = 0.0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            (proj, mask), y_true = to_device(batch, device)
            y_pred = model(proj, mask)
            loss_sum += loss_fn(y_true, y_pred).mean().item()
            loss_count += 1.0
    return reduce_mean(loss_sum, loss_count, device)



def evaluate_breakdown(model, loss_fn, loader, device, max_steps):
    """Calibrated, unweighted loss terms averaged per batch; image metrics per sample."""
    model.eval()
    term_sums = {"rmse": 0.0, "hole": 0.0, "dc": 0.0, "img": 0.0, "edge": 0.0, "perc": 0.0}
    metric_sums = {key: 0.0 for key in ("ssim_aapm", "ssim_trace",
                                        "mae_near_metal", "streak_edge_err")}
    batch_count = 0
    sample_count = 0
    case_offset = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            (proj, mask), y_true = to_device(batch, device)
            y_pred = model(proj, mask)
            terms, y_pred_den, fov_codes = loss_fn.breakdown(y_true, y_pred)
            pred4d = reconstruct_metric_grid(y_pred_den, fov_codes)

            # the loader keeps the dataset's case order, so this slice is the batch
            batch_size = pred4d.shape[0]
            case_paths = loader.dataset.case_paths[case_offset:case_offset + batch_size]
            case_offset += batch_size

            img_true512 = load_img_true512(case_paths, device)
            metal, dilated = load_image_masks(case_paths, img_true512.shape[-2:], device)
            per_sample = {
                "ssim_aapm": ssim_aapm(img_true512, pred4d, dilated),
                "ssim_trace": ssim_trace(img_true512, pred4d, dilated),
                "mae_near_metal": mae_near_metal(img_true512, pred4d, metal, dilated),
                "streak_edge_err": streak_edge_err(img_true512, pred4d, dilated),
            }

            for key in term_sums:
                term_sums[key] += terms[key]
            for key, value in per_sample.items():
                metric_sums[key] += value.sum().item()
            batch_count += 1
            sample_count += batch_size

    result = {key: value / batch_count for key, value in term_sums.items()}
    result.update({key: value / sample_count for key, value in metric_sums.items()})
    return result

#########