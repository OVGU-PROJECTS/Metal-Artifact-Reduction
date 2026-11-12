"""
Code in this file is partially or fully based on, and in some cases adapted from, the implementation provided in the following repository:https://github.com/eeulig/ldct-invariances
"""
# These are the training helper functions used across the project.
# They handle data loader setup, model loading, optimizer setup, loss helpers,
# and perceptual loss utilities so the main training scripts stay clean.

import argparse
import importlib
import os
import random
import warnings
from argparse import Namespace
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from .auxiliaries import load_yaml


def _seed_worker(worker_id: int) -> None:
    """
    Ensure numpy/python RNG are different per worker.
    PyTorch sets torch.initial_seed() per worker; we derive others from it.

    Ref: https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_dataloader(args: Namespace, datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]:
    """Create the DataLoaders used for training and validation

    Wraps each dataset in a DataLoader and enables shuffle only during training.
    The goal is to keep data loading simple while still allowing randomized
    training order without skipping any patches within an epoch.
    """
    dataloader: Dict[str, DataLoader] = {}
    for phase, data in datasets.items():
        is_train = (phase == "train")
        drop_last = bool(getattr(args, "drop_last", True)) if is_train else False

        dataloader[phase] = DataLoader(
            dataset=data,
            batch_size=int(args.mbs),
            num_workers=int(args.num_workers),
            pin_memory=bool(getattr(args, "cuda", False)),
            drop_last=drop_last,
            shuffle=is_train,
            worker_init_fn=_seed_worker,
            persistent_workers=(int(args.num_workers) > 0),
        )
    return dataloader


def setup_trained_model(
    run_name,
    device=torch.device("cuda"),
    network_name="Model",
    state_dict=None,
    return_args=False,
    return_model=True,
    pretrained_run=None,
    **model_kwargs,
) -> Union[Tuple[nn.Module, argparse.Namespace], nn.Module, argparse.Namespace]:
    """used to resume training, run inference, or load a
    pretrained model with the exact arguments used in the original run.
    """
    savepath = os.path.join("wandb", "run-" + run_name, "files")
    if not os.path.exists(savepath):
        warnings.warn(f"Folder run-{run_name} not found in wandb directory. We'll try finding an offline-run!")
        savepath = os.path.join("wandb", "offline-run-" + run_name, "files")
        if not os.path.exists(savepath):
            raise FileNotFoundError(f"Neither found run-{run_name} nor offline-run-{run_name} in wandb directory!")

    args = argparse.Namespace(**load_yaml(os.path.join(savepath, "args.yaml")))
    if return_args and not return_model:
        return args

    model_class = getattr(importlib.import_module("ldctinv.{}.network".format(args.trainer)), network_name)
    model_args = args
    if pretrained_run:
        model_args = argparse.Namespace(**load_yaml(os.path.join("wandb", "run-" + pretrained_run, "files", "args.yaml")))

    model = model_class(model_args, **model_kwargs).to(device)
    if state_dict:
        state = torch.load(os.path.join(savepath, "{}.pt".format(state_dict)), weights_only=True)
        print(f"Restore state dict {state_dict} of {network_name} from epoch", state["epoch"])
        model.load_state_dict(state["model_state_dict"])

    if return_args:
        return model, args
    return model


def setup_optimizer(args, parameters):
    """Create the optimizer used by the project based on command line args."""
    # This function centralizes optimizer creation so all experiments use the
    # same naming and hyperparameter conventions.
    if args.optimizer == "sgd":
        return optim.SGD(parameters, lr=args.lr, momentum=args.sgd_momentum, dampening=args.sgd_dampening)
    elif args.optimizer == "adam":
        return optim.Adam(parameters, lr=args.lr, betas=(args.adam_b1, args.adam_b2))
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(parameters, lr=args.lr)
    else:
        raise ValueError("Optimizer unknown. Must be one of sgd | adam | rmsprop")


def l1(input, target):
    """Pixel-wise L1 loss helper."""
    # used by the VAE reconstruction loss when args.criterion == 'l1'
    return torch.abs(input - target)


def l2(input, target):
    """Pixel-wise L2 loss helper."""
    # used when args.criterion == 'l2'
    return torch.pow((input - target), 2)


def setup_loss(loss):
    """Pick the requested loss function by name."""
    try:
        return globals()[loss]
    except Exception:
        raise ValueError("Loss function unknown.")


def nll(sample):
    """Negative log likelihood for the normal distribution used in INN loss."""
    return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])


class AELoss(nn.Module):
    """Autoencoder loss wrapper combining reconstruction and KL loss."""

    # This is the main loss used by the VAE trainer in the project.
   
    def __init__(self, args):
        super().__init__()
        self.pixel_loss = setup_loss(args.criterion)

    def forward(self, reconstructions, targets, posteriors, is_kl=True):
        rec_loss = self.pixel_loss(reconstructions, targets)
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        if is_kl:
            kl_loss = posteriors
        else:
            kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return rec_loss, kl_loss


class UnsupervisedINNLoss(nn.Module):
    """Loss used for invertible neural networks without supervision."""

    def __init__(self):
        super().__init__()

    def forward(self, sample, logdet):
        """Compute the normalizing flow loss from sample and log-det term."""
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        return loss




class normalize(object):
    """Normalize image tensors using ImageNet mean/std for VGG preprocessing."""
    # helper class for preprocessing image batches before
    # they go into a pretrained VGG network for perceptual loss.

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        for ch in range(3):
            x[:, ch, :, :] = (x[:, ch, :, :] - self.mean[ch]) / self.std[ch]
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


class repeat_ch(object):
    """Repeat a single-channel input to 3 channels when needed for VGG."""
    # Keep CT patches compatible with VGG by converting 1-channel inputs to 3-channel.

    def __init__(self, in_ch):
        self.in_ch = in_ch

    def __call__(self, x):
        if self.in_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"

class PerceptualLoss(torch.nn.Module):
    """
    VGG perceptual loss with built-in preprocessing.

    Supports BOTH:
      - PerceptualLoss(args)                               # current usage
      - PerceptualLoss(network="vgg16", device=device, ...) # original style

    Preprocessing (default):
      - assumes inputs are z-scored CT patches (can be negative / unbounded)
      - clamps to [-z_clip, z_clip]
      - maps to [0,1]
      - repeats to 3 channels if needed
      - applies ImageNet normalization (mean/std) expected by torchvision weights :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        network="vgg16",
        device=None,
        in_ch=1,
        layers=[3, 8, 15, 22],
        norm="l1",
        return_features=False,
        input_mode="zscore",   # "zscore" (default) or "none"
        z_clip=6.0,
    ):
        super().__init__()

        # ---- allow PerceptualLoss(args) ----
        if not isinstance(network, str):
            args = network
            network = getattr(args, "perc_network", "vgg16")

            # pick device from args if not explicitly provided
            if device is None:
                use_cuda = bool(getattr(args, "cuda", True)) and torch.cuda.is_available()
                dev_id = getattr(args, "device", 0)
                device = torch.device(f"cuda:{dev_id}" if use_cuda else "cpu")

            # optional overrides from args (no need to add them)
            in_ch = int(getattr(args, "perc_in_ch", in_ch))
            layers = list(getattr(args, "perc_layers", layers))
            norm = str(getattr(args, "perc_norm", norm))
            return_features = bool(getattr(args, "perc_return_features", return_features))
            input_mode = str(getattr(args, "perc_input_mode", input_mode))
            z_clip = float(getattr(args, "perc_z_clip", z_clip))
        else:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.layers = [str(l) for l in layers]
        self.return_features = return_features
        self.input_mode = input_mode
        self.z_clip = float(z_clip)

        # ---- load VGG ----
        if network == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        elif network == "vgg19":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
        else:
            raise ValueError("network must be 'vgg16' or 'vgg19'")

        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.vgg_features = vgg.features

        if norm == "l1":
            self.norm = nn.L1Loss()
        elif norm == "mse":
            self.norm = nn.MSELoss()
        else:
            raise ValueError(f"Norm {norm} not known for PerceptualLoss")

        # ImageNet normalization buffers (torchvision pretrained models) :contentReference[oaicite:2]{index=2}
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("im_mean", mean)
        self.register_buffer("im_std", std)

        self.in_ch = int(in_ch)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # ensure float
        x = x.float()

        # 1ch -> 3ch (VGG expects 3ch)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Map CT-ish tensors to what torchvision weights expect:
        # rescale to [0,1] then normalize with ImageNet mean/std :contentReference[oaicite:3]{index=3}
        if self.input_mode == "zscore":
            zc = self.z_clip
            x = torch.clamp(x, -zc, zc)
            x = (x + zc) / (2.0 * zc)  # -> [0,1]
        elif self.input_mode == "none":
            # assume caller already prepared x as [0,1] float
            pass
        else:
            raise ValueError(f"Unknown input_mode='{self.input_mode}'. Use 'zscore' or 'none'.")

        x = (x - self.im_mean) / self.im_std
        return x

    def forward(self, input, target):
        input = self._prep(input)
        target = self._prep(target)

        loss = 0.0
        if self.return_features:
            features = {"input": [], "target": []}

        for i, m in self.vgg_features._modules.items():
            input = m(input)
            target = m(target)

            if i in self.layers:
                loss = loss + self.norm(input, target)
                if self.return_features:
                    features["input"].append(input.clone())
                    features["target"].append(target.clone())

                if i == self.layers[-1]:
                    break

        return (loss, features) if self.return_features else loss