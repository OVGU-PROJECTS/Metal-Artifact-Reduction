"""Per-term gradient shares of the configured objective at a saved checkpoint
- "tools.calibrate grad" measures each term unweighted.
- This probe applies the configured w_* instead, so it reports what actually drives the update:

    share_i = (w_i g_i . G) / ||G||^2,  G = sum_j w_j g_j.

    python -m tools.measure_gradient_shares \
      --config pconv \
      --checkpoint /path/to/last.pt \
      --split val --steps 50 --batch_size 1 \
      --fixed_case_seed 42 --output_json /path/to/epoch40_grad_shares.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

from data.dataset import build_loader
from model.pconv_model import PConvUNet
from training.terms import CALIB_KEY, WEIGHT_KEY
from tools.calibrate import resolve_dc_modes, term_losses
from training.config import load_config


def mean_sd(values):
    """mean and sample SD of one per-batch field"""
    avg = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": avg, "sample_sd": 0.0}
    var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return {"mean": avg, "sample_sd": math.sqrt(var)}

def spread(values):
    return {**mean_sd(values), "min": min(values), "max": max(values)}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def grad_vector(loss, parameters, *, retain_graph):
    """flat CPU gradient of one loss term over the trainable parameters."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), device=parameter.device))
        else:
            pieces.append(gradient.detach().reshape(-1))
    return torch.cat(pieces).to(device="cpu")

def weight_of(cfg, term):
    return float(getattr(cfg, WEIGHT_KEY["dc" if term.startswith("dc") else term]))

def scale_of(cfg, term):
    return float(getattr(cfg, CALIB_KEY["dc" if term.startswith("dc") else term]))



def batch_statistics(names, losses, vectors, cfg):
    """Loss and projected-gradient shraes for one batch. Each set sums to 1"""
    norms, weights = {}, {}
    for a in names:
        norms[a] = float(vectors[a].norm().item())
        weights[a] = weight_of(cfg, a)

    dots, cosines = {}, {}
    for a in names:
        dots[a], cosines[a] = {}, {}
        for b in names:
            dots[a][b] = float(torch.dot(vectors[a], vectors[b]).item())
            if norms[a] > 0.0 and norms[b] > 0.0:
                cosines[a][b] = max(-1.0, min(1.0, dots[a][b] / (norms[a] * norms[b])))
            else:
                cosines[a][b] = float("nan")

    # the squared length of the update direction G = sum_i w_i g_i
    total_grad_sq = 0.0
    for a in names:
        for b in names:
            total_grad_sq += weights[a] * weights[b] * dots[a][b]
    if total_grad_sq <= 0.0:
        raise RuntimeError(f"Non-positive configured total-gradient norm: {total_grad_sq}")

    weighted_losses = {}
    for a in names:
        weighted_losses[a] = weights[a] * losses[a]["calibrated_loss"]
    total_weighted_loss = sum(weighted_losses.values())
    if total_weighted_loss <= 0.0:
        raise RuntimeError(f"Non-positive configured weighted loss: {total_weighted_loss}")

    terms, grad_shares, loss_shares = {}, {}, {}
    for a in names:
        projected = 0.0
        for b in names:
            projected += weights[b] * dots[a][b]
        grad_shares[a] = weights[a] * projected / total_grad_sq
        loss_shares[a] = weighted_losses[a] / total_weighted_loss
        terms[a] = {**losses[a], "weight": weights[a], "grad_norm": norms[a],
                    "weighted_grad_norm": abs(weights[a]) * norms[a],
                    "loss_share": loss_shares[a],
                    "projected_gradient_share": grad_shares[a]}

    return {
        "terms": terms,
        "pairwise_gradient_dot": dots,
        "pairwise_gradient_cosine": cosines,
        "configured_total_gradient_norm": math.sqrt(total_grad_sq),
        "configured_weighted_loss_numerator": total_weighted_loss,
        "loss_share_sum": sum(loss_shares.values()),
        "projected_gradient_share_sum": sum(grad_shares.values()),
    }


def summarize(per_batch, names):
    """Mean, sample SD, min and max of every per-batch field, across all batches."""
    fields = ("raw_loss", "calibrated_loss", "grad_norm", "weighted_grad_norm",
              "loss_share", "projected_gradient_share")
    term_summary = {}
    for a in names:
        term_summary[a] = {}
        for field in fields:
            term_summary[a][field] = spread([row["terms"][a][field] for row in per_batch])
        term_summary[a]["weight"] = per_batch[0]["terms"][a]["weight"]

    cosine_summary = {}
    for a in names:
        cosine_summary[a] = {}
        for b in names:
            cosine_summary[a][b] = mean_sd([row["pairwise_gradient_cosine"][a][b]
                                            for row in per_batch])

    return {
        "terms": term_summary,
        "pairwise_gradient_cosine": cosine_summary,
        "configured_total_gradient_norm":
            spread([row["configured_total_gradient_norm"] for row in per_batch]),
        "mean_loss_share_sum":
            mean_sd([row["loss_share_sum"] for row in per_batch])["mean"],
        "mean_projected_gradient_share_sum":
            mean_sd([row["projected_gradient_share_sum"] for row in per_batch])["mean"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pconv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_case_seed", type=int, default=42,
                        help="Seed for deterministic one-time case-order shuffle.")
    parser.add_argument("--ordered", action="store_true",
                        help="Use lexicographic case order instead of a representative "
                             "fixed shuffle.")
    parser.add_argument("--dc_modes", default="")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch_size must be positive")

    cfg = load_config(args.config)
    args.dc_modes = resolve_dc_modes(args.dc_modes, cfg.dc_mode)
    if len(args.dc_modes) != 1:
        parser.error("Projected shares require exactly one active DC mode")

    auto = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(auto if args.device == "auto" else args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(cfg.stats_path) as handle:
        stats = json.load(handle)
    global_mean = (stats["head"]["mean"], stats["body"]["mean"])
    global_std = (stats["head"]["std"], stats["body"]["std"])

    model = PConvUNet(global_mean=global_mean, global_std=global_std).to(device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    loader, _ = build_loader(
        batch_size=args.batch_size, global_mean=global_mean, global_std=global_std,
        shuffle=False, num_workers=args.num_workers, fixed_case_shuffle=not args.ordered,
        fixed_case_seed=args.fixed_case_seed, distributed=False, rank=0, world_size=1,
        seed=args.seed, base_dir={"train": cfg.train_root, "val": cfg.val_root}[args.split])
    # zip() below would otherwise probe fewer batches and still write sample_count.
    if args.steps > len(loader):
        parser.error(f"--steps={args.steps} exceeds available batches={len(loader)}")

    selected_case_ids = [Path(p).name
                         for p in loader.dataset.case_paths[: args.steps * args.batch_size]]
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameter_count = sum(p.numel() for p in parameters)
    per_batch, names = [], None

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    print(f"device={device} checkpoint_epoch={epoch} split={args.split} "
          f"steps={args.steps} batch_size={args.batch_size} "
          f"fixed_case_shuffle={not args.ordered} fixed_case_seed={args.fixed_case_seed}",
          flush=True)

    for step, batch in zip(range(args.steps), loader):
        term_values = term_losses(model, batch, device, args.dc_modes)
        names = list(term_values)
        vectors, losses = {}, {}
        for index, name in enumerate(names):
            scale = scale_of(cfg, name)
            calibrated = term_values[name] / (scale + 1e-8)
            vectors[name] = grad_vector(calibrated, parameters,
                                        retain_graph=index < len(names) - 1)
            losses[name] = {"raw_loss": float(term_values[name].detach().item()),
                            "calibration_scale": scale,
                            "calibrated_loss": float(calibrated.detach().item())}

        row = batch_statistics(names, losses, vectors, cfg)
        row["batch_index"] = step
        start = step * args.batch_size
        row["case_ids"] = selected_case_ids[start : start + args.batch_size]
        per_batch.append(row)
        del vectors, term_values
        print(f"[gradient-share] {step + 1}/{args.steps}", flush=True)

    result = {
        "method": {
            "gradient_share": "(w_i*g_i dot G) / ||G||^2",
            "total_gradient": "G = sum_i w_i*g_i",
            "aggregation": "mean and sample SD of exact per-batch shares",
            "gradient_parameters": "all trainable model parameters, frozen VGG parameters excluded",
        },
        "probe": {
            "config": args.config,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_epoch": epoch,
            "checkpoint_loss_weights": (checkpoint.get("loss_weights")
                                        if isinstance(checkpoint, dict) else None),
            "split": args.split, "steps": args.steps, "batch_size": args.batch_size,
            "sample_count": len(selected_case_ids), "seed": args.seed,
            "fixed_case_shuffle": not args.ordered, "fixed_case_seed": args.fixed_case_seed,
            "case_ids": selected_case_ids, "device": str(device),
            "dc_modes": args.dc_modes, "trainable_parameter_count": parameter_count,
        },
        "weights": {name: weight_of(cfg, name) for name in names},
        "calibration": {name: scale_of(cfg, name) for name in names},
        "summary": summarize(per_batch, names),
        "per_batch": per_batch,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"saved_json={output_path}", flush=True)


if __name__ == "__main__":
    main()
