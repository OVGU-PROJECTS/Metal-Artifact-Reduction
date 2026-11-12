# The structure of the model is partially motivated by https://github.com/MathiasGruber/PConv-Keras
"""PConvUNet — the sinogram-inpainting network.

- A partial-convolution U-Net that takes the metal-corrupted sinogram + its hole
mask and predicts the missing projection values.
- What makes this model distinctive is its loss stack
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from data_inconsistency import build_dc_pack_from_cfg
from model.pconv_nvidia_layer import PConv2d
from recon import RECO_FOV, createFdkCache, load_createhsp, reconstruct
from mar_common.scanner_config import config_downsampled, recon_config


_VGG16_WEIGHTS = VGG16_Weights.IMAGENET1K_V1


def match_size(x, ref):
    """Pad or center-crop "x" to match "ref" in H,W. 
    decoder upsampling rarely lands on the exact encoder size
    for the 1000x900 sinogram """
    _, _, ref_h, ref_w = ref.shape
    _, _, h, w = x.shape
    dh = ref_h - h          # rows to add, or negative to remove
    dw = ref_w - w          # same for columns

    if dh >= 0:             # too short, pad top and bottom
        top = dh // 2
        x = F.pad(x, (0, 0, top, dh - top))
    else:                   # too tall, cut evenly off both ends
        top = (-dh) // 2
        x = x[:, :, top : top + ref_h, :]

    if dw >= 0:             # too narrow, pad left and right
        left = dw // 2
        x = F.pad(x, (left, dw - left, 0, 0))
    else:                   # too wide, cut evenly off both sides
        left = (-dw) // 2
        x = x[:, :, :, left : left + ref_w]
    return x


class EncoderBlock(nn.Module):
    """U-Net down step
    - stride-2 partial conv (halves resolution)
    - Batchnormalizion 
    - optional refine
    - propagating both the features and the validity mask
    """
    def __init__(self, in_channels, out_channels, bn_down=True, bn_ref=True, do_refine=True):
        super().__init__()
        self.down = PConv2d(in_channels, out_channels, 3, stride=2)
        self.down_norm = nn.GroupNorm(8, out_channels) if bn_down else None
        self.refine = PConv2d(out_channels, out_channels, 3, stride=1) if do_refine else None
        self.refine_norm = nn.GroupNorm(8, out_channels) if do_refine and bn_ref else None

    def forward(self, img, mask):
        conv, mask = self.down(img, mask)
        if self.down_norm is not None:
            conv = self.down_norm(conv)
        conv = F.relu(conv)

        if self.refine is not None:
            conv, mask = self.refine(conv, mask)
            if self.refine_norm is not None:
                conv = self.refine_norm(conv)
            conv = F.relu(conv)
        return conv, mask


class DecoderBlock(nn.Module):
    """U-Net up step:
    - upsample
    - concatenate the matching encoder skip connection
    - then partial-conv.
    - Carries the mask through alongside the features.
    """
    def __init__(self, in_channels, skip_channels, out_channels, *, bn=True, relu=True):
        super().__init__()
        self.relu = relu
        self.upsample_img = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample_mask = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = PConv2d(in_channels + skip_channels, out_channels, 3)
        self.norm1 = nn.GroupNorm(8, out_channels) if bn else None
        second = relu and out_channels != 1
        self.conv2 = PConv2d(out_channels, out_channels, 3) if second else None
        self.norm2 = nn.GroupNorm(8, out_channels) if second and bn else None

    def forward(self, img, mask, e_conv, e_mask):
        up_img = self.upsample_img(img)
        up_mask = self.upsample_mask(mask)
        up_img = match_size(up_img, e_conv)
        up_mask = match_size(up_mask, e_mask)

        concat_img = torch.cat([e_conv, up_img], dim=1)
        concat_mask = torch.cat([e_mask, up_mask], dim=1)

        conv, mask = self.conv1(concat_img, concat_mask)
        if self.norm1 is not None:
            conv = self.norm1(conv)
        if self.relu:
            conv = F.leaky_relu(conv, negative_slope=0.2)

        if self.conv2 is not None:
            conv, mask = self.conv2(conv, mask)
            if self.norm2 is not None:
                conv = self.norm2(conv)
            conv = F.leaky_relu(conv, negative_slope=0.2)
        return conv, mask


class VGGFeatureExtractor(nn.Module):
    """Frozen ImageNet-pretrained VGG16,
    sliced into 4 stages. Used for the perceptual loss:
    - comparing deep feature (not raw pixels)
    - makes the recon look perceptually right, not just numerically close.
    - Weights never train.
    """
    def __init__(self):
        super().__init__()
        base = vgg16(weights=_VGG16_WEIGHTS)
        features = base.features
        self.slices = nn.ModuleList(
            [
                nn.Sequential(*features[:4]),
                nn.Sequential(*features[4:9]),
                nn.Sequential(*features[9:16]),
                nn.Sequential(*features[16:23]),
            ]
        )
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        outs = []
        cur = x
        for block in self.slices:
            cur = block(cur)
            outs.append(cur)
        return outs


class PConvUNet(nn.Module):
    """Network that fills in the missing part of a metal-corrupted sinogram.
    - Takes the 1000x900 sinogram + a mask (1 = known, 0 = hole) and predicts
      the missing values with an 8-level partial-conv U-Net that carries the
      mask through every layer.
    - Output keeps the known values and inserts predictions only in the hole.
    - Trained on how the filled sinogram looks once reconstructed into a CT
      image: pixel error (RMSE), edge sharpness (Sobel), perceptual (VGG).
    - Holds two reconstruction setups (head / body scans) so each sample uses
      its matching geometry, plus the frozen VGG and per-loss scales.
    """
    def __init__(self, global_mean=None, global_std=None):
        super().__init__()

        self.edge_mask_alpha = 0.5

        cfg_dc = recon_config(RECO_FOV["head"])
        self.dc_pack = build_dc_pack_from_cfg(cfg_dc)

        self.recon_cfg_head = config_downsampled(RECO_FOV["head"])
        self.recon_cfg_body = config_downsampled(RECO_FOV["body"])
        for cfg in (self.recon_cfg_head, self.recon_cfg_body):
            cfg._fdk_cache = createFdkCache(cfg, target_views=500, view_block=50, raw_view_count=1000)
        # complex64 — not registered as buffer to avoid NCCL incompatibility in DDP;
        # img_loss calls .to(device=proj2d.device) explicitly so auto-move is not needed.
        self.loaded_tensor_head = load_createhsp("head")
        self.loaded_tensor_body = load_createhsp("body")

        self.register_buffer("fov_mean", torch.tensor(list(global_mean), dtype=torch.float32))
        self.register_buffer("fov_std", torch.tensor(list(global_std), dtype=torch.float32))

        self.perc_mu_min = 0.0
        self.perc_mu_max = 0.060
        vgg_preprocessing = _VGG16_WEIGHTS.transforms()
        self.perc_vgg_size = vgg_preprocessing.crop_size[0]
        self.vgg_feat = VGGFeatureExtractor()
        self.register_buffer(
            "vgg_mean_rgb",
            torch.tensor(vgg_preprocessing.mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vgg_std_rgb",
            torch.tensor(vgg_preprocessing.std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.enc1 = EncoderBlock(1, 32, bn_down=False, bn_ref=True)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        self.enc5 = EncoderBlock(256, 512)
        self.enc6 = EncoderBlock(512, 512)
        self.enc7 = EncoderBlock(512, 512)
        self.enc8 = EncoderBlock(512, 1024)

        self.dec9 = DecoderBlock(1024, 512, 512)
        self.dec10 = DecoderBlock(512, 512, 512)
        self.dec11 = DecoderBlock(512, 512, 512)
        self.dec12 = DecoderBlock(512, 256, 256)
        self.dec13 = DecoderBlock(256, 128, 128)
        self.dec14 = DecoderBlock(128, 64, 64)
        self.dec15 = DecoderBlock(64, 32, 32)
        self.dec16 = DecoderBlock(32, 1, 1, bn=False, relu=False)

    def set_calibration_scales(self, rmse_scale, hole_scale, dc_scale, img_scale, edge_scale, perc_scale):
        self.rmse_scale = rmse_scale
        self.hole_scale = hole_scale
        self.dc_scale = dc_scale
        self.img_scale = img_scale
        self.edge_scale = edge_scale
        self.perc_scale = perc_scale

    def forward(self, proj, mask):
        e_conv1, e_mask1 = self.enc1(proj, mask)
        e_conv2, e_mask2 = self.enc2(e_conv1, e_mask1)
        e_conv3, e_mask3 = self.enc3(e_conv2, e_mask2)
        e_conv4, e_mask4 = self.enc4(e_conv3, e_mask3)
        e_conv5, e_mask5 = self.enc5(e_conv4, e_mask4)
        e_conv6, e_mask6 = self.enc6(e_conv5, e_mask5)
        e_conv7, e_mask7 = self.enc7(e_conv6, e_mask6)
        e_conv8, e_mask8 = self.enc8(e_conv7, e_mask7)

        d_conv9, d_mask9 = self.dec9(e_conv8, e_mask8, e_conv7, e_mask7)
        d_conv10, d_mask10 = self.dec10(d_conv9, d_mask9, e_conv6, e_mask6)
        d_conv11, d_mask11 = self.dec11(d_conv10, d_mask10, e_conv5, e_mask5)
        d_conv12, d_mask12 = self.dec12(d_conv11, d_mask11, e_conv4, e_mask4)
        d_conv13, d_mask13 = self.dec13(d_conv12, d_mask12, e_conv3, e_mask3)
        d_conv14, d_mask14 = self.dec14(d_conv13, d_mask13, e_conv2, e_mask2)
        d_conv15, d_mask15 = self.dec15(d_conv14, d_mask14, e_conv1, e_mask1)
        raw_pred, _ = self.dec16(d_conv15, d_mask15, proj, mask)
        # Y-comp style output: preserve measured rays and learn only the metal trace.
        completed_pred = mask * proj + (1.0 - mask) * raw_pred
        mask_passthrough = mask.detach()
        return torch.cat([completed_pred, mask_passthrough, raw_pred], dim=1)

    @staticmethod
    def RMSE(y_true, y_pred):
        return torch.sqrt(torch.mean((y_pred - y_true).square(), dim=(1, 2, 3)) + 1e-8)

    # RMSE over only the masked ("hole") pixels 
    @staticmethod
    def loss_hole(mask, y_true, y_pred):
        hole = 1 - mask
        sum_dif_sq = ((hole * y_pred) - (hole * y_true)).square().sum(dim=(1, 2, 3))
        return torch.sqrt(sum_dif_sq / (hole.sum(dim=(1, 2, 3)) + 1e-8) + 1e-8)

    def sobel_mag(self, x):
        """Edge-strength map (Sobel gradient magnitude).
        The edge loss matches these between recon and
        target so sharp anatomical boundaries are preserved."""
        gx = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        gy = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(x, gx, padding=1)
        grad_y = F.conv2d(x, gy, padding=1)
        return torch.sqrt(torch.clamp(grad_x.square() + grad_y.square(), min=1e-12))
    
    def to_vgg(self, x_mu_4d):
        """Prepare a one-channel attenuation image for TorchVision VGG16."""
        x = x_mu_4d.clamp(self.perc_mu_min, self.perc_mu_max)
        x = (x - self.perc_mu_min) / (self.perc_mu_max - self.perc_mu_min)
        x = F.interpolate(
            x,
            size=(self.perc_vgg_size, self.perc_vgg_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)
        mean = self.vgg_mean_rgb.to(device=x.device, dtype=x.dtype)
        std = self.vgg_std_rgb.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def img_weight_from_proj_mask(self, proj_mask, fov_codes):
        """Backproject the metal-trace projection mask into image space as an
        artifact-attention weight for image RMSE.
        - "proj_mask" is 1 on valid rays and 0 in the hole,
        matching the model output channel 1
        """
        weights = []
        for idx in range(proj_mask.shape[0]):
            hole_proj = (1.0 - proj_mask[idx, 0]).detach()
            if fov_codes[idx].item() > 0.5:
                w = reconstruct(self.recon_cfg_body, hole_proj, self.loaded_tensor_body.to(device=hole_proj.device))
            else:
                w = reconstruct(self.recon_cfg_head, hole_proj, self.loaded_tensor_head.to(device=hole_proj.device))
            w = w.abs()
            w = w / (w.mean().clamp_min(1e-8))
            w = w.clamp(max=5.0)
            weights.append(w / w.mean().clamp_min(1e-8))
        return torch.stack(weights, dim=0).unsqueeze(1)

    def img_loss(self, img_true256, y_pred, fov_codes, proj_mask, return_perc=True):
        """Reconstruct the predicted sinogram and compare in image space.

        -this is the bridge that lets a sinogram-domain network be trained on image quality.
        - Each sample is FDK-reconstructed with its head/body FOV config,
        - then scored by image RMSE, edge (Sobel) loss, and VGG perceptual loss.
        Image RMSE and edge loss are focused by a backprojected metal-trace map.
        """
        # Reconstruct each predicted sinogram with the correct (head/body) geometry.
        pred_imgs = []
        for idx in range(y_pred.shape[0]):
            proj2d = y_pred[idx, 0]
            if fov_codes[idx].item() > 0.5:
                pred_img = reconstruct(self.recon_cfg_body, proj2d, self.loaded_tensor_body.to(device=proj2d.device))
            else:
                pred_img = reconstruct(self.recon_cfg_head, proj2d, self.loaded_tensor_head.to(device=proj2d.device))
            pred_imgs.append(pred_img)
        pred_img = torch.stack(pred_imgs, dim=0)

        pred4d = pred_img.unsqueeze(1)
        img_w = self.img_weight_from_proj_mask(proj_mask, fov_codes).to(device=pred4d.device, dtype=pred4d.dtype)
        img_rmse = torch.sqrt(((pred4d - img_true256).square() * img_w).mean(dim=(1, 2, 3)) + 1e-8)

        grad_true = self.sobel_mag(img_true256)
        grad_pred = self.sobel_mag(pred4d)
        edge_diff = torch.abs(grad_true - grad_pred)
        edge_global = torch.mean(edge_diff, dim=(1, 2, 3))
        edge_masked = torch.mean(edge_diff * img_w, dim=(1, 2, 3))
        edge_loss = (1.0 - self.edge_mask_alpha) * edge_global + self.edge_mask_alpha * edge_masked
        if not return_perc:
            return img_rmse, edge_loss

        vgg_t = self.to_vgg(img_true256)
        vgg_p = self.to_vgg(pred4d)

        ft_list = [t.detach() for t in self.vgg_feat(vgg_t)]
        fp_list = self.vgg_feat(vgg_p)
        perc_terms = [torch.mean(torch.abs(p - t), dim=(1, 2, 3)) for p, t in zip(fp_list, ft_list)]
        perc = torch.stack(perc_terms, dim=0).sum(dim=0)
        return img_rmse, edge_loss, perc
    
