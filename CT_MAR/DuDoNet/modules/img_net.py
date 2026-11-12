"""IE-Net: the image-domain stage of DuDoNet (step 3 of 3).
- After the sinogram is repaired (SE-Net) and reconstructed into an image (RIL layer)
- this standard U-Net cleans up the residual streak artifacts that survive in image space.
- It predicts a residual error not the final image directly.
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """two 3x3 conv + ReLU layers """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class IENetUNet(nn.Module):
    """
    Image Enhancement Network G_i (U-Net).
    Input:  concat(X_LI, X_hat)  -> (B, 2, Hc, Wc)
    Output: residual image  (B, 1, Hc, Wc)
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 64):
        super().__init__()
        # Down
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, base_ch * 8),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 8, base_ch * 8),
        )

        # Up
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 16, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 8, base_ch * 2)

        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 4, base_ch)

        self.up4 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv4(u4)

        out = self.outc(u4)   # residual
        return out
