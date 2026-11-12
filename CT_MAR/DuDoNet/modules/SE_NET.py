"""SE-Net: the sinogram-domain stage of DuDoNet (step 1 of 3).
- DuDoNet works in two domains: sinogram and image domains
- learns to repair residual metal-corrupted regions since it uses
the linearinterpolated (li) sinogram as input. 
- It is a U-Net with two DuDoNet-specific tricks:
  1. The metal-trace mask is re-injected at every encoder scale, so the network
    always knows which bins are corrupted.
  2. The output is fused with the input by the mask, so trusted (non-metal)
    readings are passed through unchanged and only the metal gap is replaced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinoNet(nn.Module):
    """sinogram-domain SE-Net
    Inputs:
      y_li: (B, 1, H, W)   LI sinogram
      mt : (B, 1, H, W)   metal trace mask
    Outputs:
      y_out: (B, 1, H, W)  fused output
      g_s  : (B, 1, H, W)  raw network output G_s
    """

    def __init__(self, upsample_mode: str = "bilinear"):
        super().__init__()
        self.upsample_mode = upsample_mode

        # Encoder (down) 
        self.down_conv0 = self._conv_block(2,   64, k=4, s=2, p=1, activation="relu")
        self.down_conv1 = self._conv_block(65,  128, k=4, s=2, p=1, activation="lrelu")
        self.down_conv2 = self._conv_block(129, 256, k=4, s=2, p=1, activation="lrelu")
        self.down_conv3 = self._conv_block(257, 512, k=4, s=2, p=1, activation="lrelu")
        self.down_conv4 = self._conv_block(513, 1024, k=4, s=2, p=1, activation="lrelu")

        # Decoder (up)
        self.up_conv5 = self._conv_block(1025,        512, k=3, s=1, p=1, activation="relu")
        self.up_conv6 = self._conv_block(512 + 513,   256, k=3, s=1, p=1, activation="relu")
        self.up_conv7 = self._conv_block(256 + 257,   128, k=3, s=1, p=1, activation="relu")
        self.up_conv8 = self._conv_block(128 + 129,   64,  k=3, s=1, p=1, activation="relu")
        self.up_conv9 = self._conv_block(64  + 65,    1,   k=3, s=1, p=1, activation=None)

    @staticmethod
    def _conv_block(in_ch, out_ch, k, s, p, activation="relu"):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)]
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unknown activation '{activation}'")
        return nn.Sequential(*layers)

    def _upsample_to(self, x, ref):
        # upsamples x so that spatial size == ref.shape[2:]
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ["bilinear", "bicubic"] else None
        )

    def forward(self, y_li, mt):
        """
        y_li: (B, 1, H, W)  e.g. H=1000, W=900
        mt : (B, 1, H, W)
        """
        assert y_li.shape == mt.shape, "y_li and mt must have same shape"
        B, C, H, W = y_li.shape

        # Encoder
        x0_in = torch.cat([y_li, mt], dim=1)        # (B,2,H,W)
        d0 = self.down_conv0(x0_in)                  # (B,64, H/2, W/2 approx)
        m2 = F.interpolate(mt, size=d0.shape[-2:], mode="nearest")
        concat0 = torch.cat([d0, m2], dim=1)         # (B,65, ...)

        d1 = self.down_conv1(concat0)
        m4 = F.interpolate(mt, size=d1.shape[-2:], mode="nearest")
        concat1 = torch.cat([d1, m4], dim=1)         # (B,129, ...)

        d2 = self.down_conv2(concat1)
        m8 = F.interpolate(mt, size=d2.shape[-2:], mode="nearest")
        concat2 = torch.cat([d2, m8], dim=1)         # (B,257, ...)

        d3 = self.down_conv3(concat2)
        m16 = F.interpolate(mt, size=d3.shape[-2:], mode="nearest")
        concat3 = torch.cat([d3, m16], dim=1)        # (B,513, ...)

        d4 = self.down_conv4(concat3)
        m32 = F.interpolate(mt, size=d4.shape[-2:], mode="nearest")
        concat4 = torch.cat([d4, m32], dim=1)        # (B,1025, ...)

        # Decoder
        # Stage 5
        u5 = self._upsample_to(concat4, concat3)
        u5 = self.up_conv5(u5)                       # (B,512, ...)
        u5_cat = torch.cat([u5, concat3], dim=1)     # (B,512+513=1025, ...)

        # Stage 6
        u6 = self._upsample_to(u5_cat, concat2)
        u6 = self.up_conv6(u6)                       # (B,256, ...)
        u6_cat = torch.cat([u6, concat2], dim=1)     # (B,256+257=513, ...)

        # Stage 7
        u7 = self._upsample_to(u6_cat, concat1)
        u7 = self.up_conv7(u7)                       # (B,128, ...)
        u7_cat = torch.cat([u7, concat1], dim=1)     # (B,128+129=257, ...)

        # Stage 8
        u8 = self._upsample_to(u7_cat, concat0)
        u8 = self.up_conv8(u8)                       # (B,64, ...)
        u8_cat = torch.cat([u8, concat0], dim=1)     # (B,64+65=129, ...)

        # Stage 9 – upsample back to exactly (H, W)
        u9 = self._upsample_to(u8_cat, y_li)         # (B,129,H,W)
        g_s = self.up_conv9(u9)                      # (B,1,H,W)

        # keep the original (trusted) LI sinogram. Y_out = mt*G_s + (1-mt)*Y_LI
        y_out = mt * g_s + (1.0 - mt) * y_li

        return y_out, g_s


