"""Trainer for the cINN part of the MAR invariance.

This file trains a conditional invertible network that operates on the VAE
latent space. It keeps the VAE frozen, conditions on stored MAR prediction
and metal mask maps, and learns a flow-based latent mapping for invariance.
"""


# trainer_cinn.py
import os
import copy
import torch.nn as nn
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from argparse import Namespace
from MAR_Inv.vae.network import BigAE
from MAR_Inv import utils
from MAR_Inv.cinn.network import ConditionalTransformer
from data.patches_generator import PatchesGenerator


class Trainer(object):
    """
    Simple cINN trainer for MAR invariance pipeline.

    Assumptions:
      - The CVAE is always conditional.
      - Dataset returns:
            x : corrupted CT patch
            p : stored MAR prediction
            m : metal mask
      - CVAE condition is always [p, m].
      - cINN condition is always [p, m].
      - CVAE encoder input is [x, p, m].
      - CINN input is [z, p, m]

    """

    def __init__(self, args, device):
        self.args = args
        self.dev = device

        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------
        train_args = copy.deepcopy(args)
        train_args.datafolder = args.train_datafolder

        val_args = copy.deepcopy(args)
        val_args.datafolder = args.val_datafolder

        self.data = {
            "train": PatchesGenerator(
                train_args,
                fn_x=getattr(args, "fn_x", "input_img.mha"),
                fn_y=getattr(args, "fn_y", "output_img.mha"),
                fn_mask=getattr(args, "fn_mask", "input_mask.mha"),
                fn_pred=getattr(args, "fn_pred", None),
                patches_per_image=int(getattr(args, "patches_per_image", 1)),
                strong_ratio=float(getattr(args, "strong_ratio", 0.7)),
                topk=int(getattr(args, "topk", 5000)),
                use_strength=bool(getattr(args, "use_strength", True)),
                return_meta=False,
                stats_path=args.stats,
                conditional=True,
                require_cond_files=True,
            ),
            "val": PatchesGenerator(
                val_args,
                fn_x=getattr(args, "fn_x", "input_img.mha"),
                fn_y=getattr(args, "fn_y", "output_img.mha"),
                fn_mask=getattr(args, "fn_mask", "input_mask.mha"),
                fn_pred=getattr(args, "fn_pred", None),
                patches_per_image=int(getattr(args, "patches_per_image", 1)),
                strong_ratio=float(getattr(args, "strong_ratio", 0.7)),
                topk=int(getattr(args, "topk", 5000)),
                use_strength=bool(getattr(args, "use_strength", True)),
                return_meta=False,
                stats_path=args.stats,
                conditional=True,
                require_cond_files=True,
            ),
        }

        
        # The cINN dataset provides the same condition maps as the VAE:
        # stored MAR prediction and the metal mask.
        self.dataloader = utils.setup_dataloader(args, self.data)

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        self.criterion = utils.UnsupervisedINNLoss()

        ckpt = torch.load(args.vae_checkpoint, map_location=self.dev)

        cfg = ckpt["config"]
        ae_args = Namespace(**cfg)

        self.ae = BigAE(
            ae_args,
            in_ch=3,      # x + p + m
            out_ch=1,
            cond_ch=2,    # p + m
        ).to(self.dev)

        self.ae.load_state_dict(ckpt["generator_state_dict"])
    

        self.ae.to(self.dev)
        self.ae.eval()

        for p in self.ae.parameters():
            p.requires_grad_(False)

        if getattr(ae_args, "domain", "x") != "x":
            raise ValueError("For this MAR invariance setup, the VAE must be trained with domain='x'.")

        # --------------------------------------------------
        # cINN model
        # --------------------------------------------------
         # cINN is conditional on [prediction, mask] and only changes the VAE
        # latent z distribution, not the decoder or image domain.
        if int(args.conditioning_in_ch) != 2:
            raise ValueError("Set conditioning_in_ch: 2 because cINN condition is [prediction, mask].")

        self.model = ConditionalTransformer(args).to(self.dev)

        if self.dev.type == "cuda" and len(getattr(args, "devices", [])) > 1:
            print(f"Using DataParallel on devices: {args.devices}")
            self.model = nn.DataParallel(self.model, device_ids=args.devices)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.adam_b1, args.adam_b2),
        )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        self.losses = utils.metrics.Losses(
            self.dataloader,
            [
                "cinn loss",
                "v mean",
                "v std",
                "v abs mean",
                "logdet mean",
            ],
)
        self.savedir = wandb.run.dir if wandb.run is not None else args.out_dir
        os.makedirs(self.savedir, exist_ok=True)
        self.epoch = 0
        self.images = {}

        self.best_val_loss = float("inf")
        self.current_val_loss = None

    def move_batch_to_device(self, batch):
        return {
            k: v.to(self.dev, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def cinn_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def train_step(self, batch):
        x = batch["x"]
        pred = batch["p"]
        mask = batch["m"]

        # condition = [MAR prediction, metal mask]
        cond = torch.cat([pred, mask], dim=1)

        # VAE encoder input = [x, pred, mask]
        ae_inp = torch.cat([x, cond], dim=1)

        with torch.no_grad():
            z = self.ae.encode(ae_inp).sample()

        # cINN forward: z -> zz
        zz, logdet = self.model(z, cond)

        loss = self.criterion(zz, logdet)

        if not torch.isfinite(loss):
            print("Non-finite loss detected.")
            print("loss:", loss)
            print("v mean:", zz.detach().mean())
            print("v std:", zz.detach().std())
            print("logdet mean:", logdet.detach().mean())
            raise FloatingPointError("cINN loss became NaN or Inf.")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        metrics = {
            "cinn loss": loss.detach(),
            "v mean": zz.detach().mean(),
            "v std": zz.detach().std(),
            "v abs mean": zz.detach().abs().mean(),
            "logdet mean": logdet.detach().mean(),
        }

        self.losses.push(loss=metrics, phase="train")


    def train(self):
        self.model.train()

        for bi, batch in enumerate(tqdm(self.dataloader["train"])):
            batch = self.move_batch_to_device(batch)
            self.train_step(batch)

            if bool(getattr(self.args, "dryrun", False)) and bi >= 1:
                break

        self.losses.summarize("train")

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        x = batch["x"]
        pred = batch["p"]
        mask = batch["m"]

        cond = torch.cat([pred, mask], dim=1)
        ae_inp = torch.cat([x, cond], dim=1)

        z = self.ae.encode(ae_inp).sample()
        zz, logdet = self.model(z, cond)

        loss = self.criterion(zz, logdet)
        metrics = {
            "cinn loss": loss.detach(),
            "v mean": zz.detach().mean(),
            "v std": zz.detach().std(),
            "v abs mean": zz.detach().abs().mean(),
            "logdet mean": logdet.detach().mean(),
       }

        self.losses.push(loss=metrics, phase="val")

                
        if wandb.run is not None and batch_idx < int(getattr(self.args, "loginterv", 4)):
            n_vis = int(getattr(self.args, "n_vis", 4))

            zz_small = zz[:n_vis]
            cond_small = cond[:n_vis]
            x_small = x[:n_vis]
            pred_small = pred[:n_vis]
            mask_small = mask[:n_vis]

            zz_sample = torch.randn_like(zz_small)

            # cINN reverse: random v -> sampled VAE latent z
            z_sample = self.cinn_model().reverse(zz_sample, cond_small)

            # VAE decoder: sampled z + condition -> image sample
            x_sample = self.ae.decode(z_sample, im_cond=cond_small)

            self.log_wandb_images(
                {
                    "input_x": x_small,
                    "prediction": pred_small,
                    "mask": mask_small,
                    "sample": x_sample,
                }
            )
        return loss.detach()

    def validate(self):
        self.images = {}
        self.model.eval()

        val_losses = []

        for bi, batch in enumerate(tqdm(self.dataloader["val"])):
            batch = self.move_batch_to_device(batch)
            loss = self.val_step(bi, batch)
            val_losses.append(loss.item())

            if bool(getattr(self.args, "dryrun", False)) and bi >= 1:
                break

        self.current_val_loss = sum(val_losses) / max(len(val_losses), 1)

        print(f"Validation cINN loss: {self.current_val_loss:.6f}")

        self.losses.summarize("val")

    def log_wandb_images(self, images):
        if wandb.run is None:
            return

        for tag, img in images.items():
            wimg = wandb.Image(img.detach().cpu()[0], caption=tag)
            self.images.setdefault(tag, []).append(wimg)

    def save_checkpoint(self):
        os.makedirs(self.savedir, exist_ok=True)

        state = {
            "epoch": self.epoch,
            "model_state_dict": self.cinn_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": vars(self.args),
            "current_val_loss": self.current_val_loss,
            "best_val_loss": self.best_val_loss,
        }

        # Always save latest checkpoint
        latest_path = os.path.join(self.savedir, "cinn_latest.pt")
        torch.save(state, latest_path)

        # Save best checkpoint based on validation loss
        if self.current_val_loss is not None and self.current_val_loss < self.best_val_loss:
            self.best_val_loss = self.current_val_loss
            state["best_val_loss"] = self.best_val_loss

            best_path = os.path.join(self.savedir, "cinn_best.pt")
            torch.save(state, best_path)

            print(f"New best checkpoint saved: val loss = {self.best_val_loss:.6f}")

        # Save periodic checkpoints
        save_every = int(getattr(self.args, "save_every", 50))
        if (self.epoch + 1) % save_every == 0:
            epoch_path = os.path.join(
                self.savedir,
                f"cinn_epoch_{self.epoch + 1:04d}.pt"
            )
            torch.save(state, epoch_path)
    
    def log(self):
        self.save_checkpoint()

        # Save local loss plots/logs only if this function does not require wandb
        try:
            self.losses.plot(self.savedir)
        except Exception as e:
            print(f"Could not plot losses: {e}")

        # Always save the local CSV. Losses.log handles wandb conditionally.
        self.losses.log(self.savedir, self.epoch)

        # Only log images if wandb is initialized.
        if wandb.run is not None:
            wandb.log(self.images, step=self.epoch + 1)

    def fit(self):
        for epoch in range(int(self.args.epochs)):
            torch.manual_seed(int(self.args.seed) + epoch)

            if hasattr(self.data["train"], "set_epoch"):
                self.data["train"].set_epoch(epoch)
            if hasattr(self.data["val"], "set_epoch"):
                self.data["val"].set_epoch(epoch)

            self.train()
            self.validate()

            self.log()

            self.losses.reset()
            self.epoch += 1

if __name__ == "__main__":
    import argparse
    import yaml
    from argparse import Namespace

    
    # loads the config, builds the Trainer, fetches
    # runs it through the encoder + cINN forward/backward pass
    # 
    # Example shapes and decode flow (for reference):
    #  - batch['x']  -> (B, 1, 128, 128)         # corrupted CT patch
    #  - batch['p']  -> (B, 1, 128, 128)         # stored MAR prediction map
    #  - batch['m']  -> (B, 1, 128, 128)         # metal mask
    #  - cond = torch.cat([p, m], dim=1)     -> (B, 2,128, 128)
    #  - ae_inp = torch.cat([x, cond], dim=1) -> (B, 3, 128, 128)
    #  - z = ae.encode(ae_inp).sample()      -> (B, C, 1, 1)  (VAE latent)
    #  - zz, logdet = model(z, cond)         -> zz same shape as z (flow output)
    #  - v = zz (also called z_tilde or `v`) is the invariance variable sampled from N(0,I)
    #  - z_sample = model.reverse(v, cond)   -> maps prior sample to VAE latent space
    #  - x_sample = ae.decode(z_sample, im_cond=cond) -> (B, 1, 128, 128)


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--out_dir", type=str, default="./cinn_training_run")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["devices"] = [int(x) for x in args_cli.devices.split(",") if x.strip()]
    cfg["out_dir"] = args_cli.out_dir
    cfg["cuda"] = torch.cuda.is_available()

    args = Namespace(**cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    print("device:", device)
    print("output directory:", args.out_dir)

    trainer = Trainer(args, device)
    trainer.fit()