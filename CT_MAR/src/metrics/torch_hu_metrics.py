"""Error metrics on mu inputs. torch only.
    mae_near_metal   same, over the dilated-minus-original annulus
    streak_edge_err  mean |Sobel gradient| error in HU/px

masked_mean is the shared masked reduction, metrics.torch_ssim imports it too.
"""
import torch
import torch.nn.functional as F

from mar_common.units import mu2hu


def masked_mean(values, mask, what):
    """per-image mean of values inside mask. raises instead of returning 0/0"""
    count = mask.sum(dim=(1, 2, 3))
    if not bool(count.all()):
        raise ValueError(f"{what} mask is empty")
    return (values * mask).sum(dim=(1, 2, 3)) / count


def streak_edge_err(img_true_mu, img_pred_mu, dilated_mask):
    """Mean |Sobel gradient| error in HU/px over the patient region,
    -Dilated metal excluded.
    """
    reference = mu2hu(img_true_mu)
    prediction = mu2hu(img_pred_mu)
    kx = reference.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    def sobel_mag(image):
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
        return torch.sqrt(F.conv2d(padded, kx).square() + F.conv2d(padded, ky).square() + 1e-8)

    error = (sobel_mag(reference) - sobel_mag(prediction)).abs()
    region = ((reference > -950.0) & ~(dilated_mask > 0.5)).to(error.dtype)
    return masked_mean(error, region, "streak metric")


def mae_near_metal(img_true_mu, img_pred_mu, original_mask, dilated_mask):
    """Mean absolute HU error in the dilated-minus-original annulus."""
    error_hu = (mu2hu(img_true_mu) - mu2hu(img_pred_mu)).abs()
    ring = ((dilated_mask > 0.5) & ~(original_mask > 0.5)).to(error_hu.dtype)
    return masked_mean(error_hu, ring, "near-metal ring")
