"""
Code in this file is partially or fully based on, and in some cases adapted from, the implementation provided in the following repository:https://github.com/eeulig/ldct-invariances
"""
"""
cINN model wrapper for the MAR invariance pipeline.

This module defines the conditional transformer that maps between the VAE
latent space and a standard normal latent distribution. The conditioning
inputs are the MAR prediction and metal mask, which are embedded and fed into
every flow block.
"""

import numpy as np
import torch
import torch.nn as nn

from MAR_Inv.cinn.blocks import ConditionalFlow, FeatureLayer
from MAR_Inv.vae.blocks import ActNorm, DenseEncoderLayer


class DenseEmbedder(nn.Module):
    """A tiny bottleneck MLP that turns per-pixel features into a dense vector.
    """

    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims) - 2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)

    # This embedder is often used for small vector-like conditions that have
    # already been reduced to 1x1 spatial size.


class Embedder(nn.Module):
    """Embeds a conditioning image into a fixed-size dense code.

    The cINN uses this embedder to turn [prediction, mask] condition maps into
    a vector that can modulate every flow block with the same size as the latent code.
    """

    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = "an"  # hard coded
        bottleneck_size = in_spatial_size // 2**n_down
        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))
        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, emb_dim)
        if n_down == 1:
            print(
                " Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                "increase its capacity."
            )

    def forward(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return h.squeeze(-1).squeeze(-1)

    # The result is a compact vector that represents the conditioning maps.


class ConditionalTransformer(nn.Module):
    def __init__(self, args):
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        super().__init__()
        conditioning_option = "none"
        flowactivation = "lrelu"
        embedding_channels = args.in_channels

        self.flow = ConditionalFlow(
            in_channels=args.in_channels,
            embedding_dim=embedding_channels,
            hidden_dim=args.mid_channels,
            hidden_depth=args.hidden_depth,
            n_flows=args.n_flows,
            conditioning_option=conditioning_option,
            activation=flowactivation,
        )
        self.embedder = Embedder(
            args.conditioning_spatial_size, args.conditioning_in_ch, args.in_channels, n_down=args.n_down
        )

    def embed(self, conditioning):
        # embed it via embedding layer
        embedding = self.embedder(conditioning)
        return embedding

    # Conditioning is always the raw [prediction, mask] maps in this project.

    def sample(self, shape, conditioning):
        # Flow convention:
        #   forward:  z (VAE latent) -> z_tilde (invariance variable v, simple prior) + log-det
        #   reverse:  z_tilde (v) -> z (map back to VAE latent) by applying inverse flow
        # Here we sample z_tilde ~ N(0, I) (the invariance variable `v`) and
        # map it back through the cINN reverse path to obtain a VAE latent `z`
        # conditioned on `predicted image and mask` (so it can be decoded by the VAE).
        z_tilde = torch.randn(shape, device=conditioning.device)
        return self.reverse(z_tilde, conditioning)

    # Note: sample() directly creates a latent sample in the VAE space from
    # a standard normal prior under the learned flow.


    def forward(self, input, conditioning, train=False):
        embedding = self.embed(conditioning)
        out, logdet = self.flow(input, embedding)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    # Forward returns the transformed latent and its log-det; reverse maps
    # the flow output back to the original latent space.

    def reverse(self, out, conditioning):
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight")
