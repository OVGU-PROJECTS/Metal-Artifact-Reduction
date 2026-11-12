"""Loss calibration and gradient-strength probe.

Both modes run the model over N batches and report the six loss terms
(rmse, hole, dc, img, edge, perc). They differ in what they measure:

  calib : the mean magnitude of each term  -> the calib_* scales.
  grad  : the gradient norm and cosine-to-hole of each calibrated term -> the w_*.

stats_path, dataset roots, dc_mode and the calib_* baseline all come from
configs/<config>.yaml, the same file the trainer reads.

  python -m tools.calibrate calib --warmup_steps 500
  python -m tools.calibrate calib --weights last.pt
  python -m tools.calibrate grad  --weights last.pt --dc_modes all
"""
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_inconsistency import DC_MODES, dc_loss
from model.pconv_model import PConvUNet
from training.terms import CALIB_KEY, TERMS, WEIGHT_KEY
from data.dataset import build_loader
from training.config import load_config

# decimals to print each calib_* value with
PRECISION = {"rmse": 7, "hole": 7, "dc": 7, "img": 9, "edge": 8, "perc": 8}
# --w_* flags the grad mode recommends; rmse is not tuned that way
W_FLAG = {term: f"--{WEIGHT_KEY[term]}" for term in TERMS if term != "rmse"}


def make_loader(cfg, split, gm, gs, args, shuffle=False):
    loader, _ = build_loader(
        batch_size=args.batch_size, global_mean=gm, global_std=gs, shuffle=shuffle,
        num_workers=args.num_workers, distributed=False, rank=0, world_size=1,
        seed=args.seed, base_dir={"train": cfg.train_root, "val": cfg.val_root}[split])
    return loader


def forward(model, batch, device):
    """Move one batch to the device and run the model; returns output and target."""
    (proj, mask), y = batch
    proj, mask, y = (t.to(device, non_blocking=True) for t in (proj, mask, y))
    return model(proj, mask), y


def term_losses(model, batch, device, dc_modes):
    """Raw per-batch loss terms as mean scalars (graph intact), in report order."""
    out, y = forward(model, batch, device)
    yp, mo, yt, pack = out[:, 0:1], out[:, 1:2], y[:, 0:1], y[:, 1:2]
    fov = pack[:, 0, -1, -1]
    i = torch.round(fov).long()
    mu, sg = model.fov_mean[i].view(-1, 1, 1, 1), model.fov_std[i].view(-1, 1, 1, 1)
    ytd, ypd = yt * sg + mu, yp * sg + mu
    img_v, edge_v, perc_v = model.img_loss(pack[:, :, :256, :256], ypd,
                                           fov_codes=fov, proj_mask=mo, return_perc=True)
    t = {"hole": model.loss_hole(mo, yt, yp).mean(), "rmse": model.RMSE(yt, yp).mean()}
    for m in dc_modes:
        key = "dc" if len(dc_modes) == 1 else f"dc_{m}"
        t[key] = dc_loss(ytd, ypd, model.dc_pack, mode=m).mean()
    t["img"], t["edge"], t["perc"] = img_v.mean(), edge_v.mean(), perc_v.mean()
    return t


def grad_vec(model):
    """Flat detached gradient over every parameter (zeros where grad is None)."""
    return torch.cat([p.grad.detach().flatten() if p.grad is not None
                      else torch.zeros(p.numel(), device=p.device)
                      for p in model.parameters()])


def grad_stats(g, ref, refn):
    """Gradient norm plus alignment with the reference (hole) gradient `ref`.

    matched_w = |ref| / |g| is the weight that makes this term pull as hard as hole.
    """
    gn = g.norm().item()
    dot = float(g @ ref)
    return {"grad_norm": gn, "matched_w": refn / gn if gn > 0 else float("nan"),
            "cos": dot / (refn * gn) if refn * gn > 0 else float("nan"), "dot": dot}


def resolve_dc_modes(value, default_mode):
    """'' -> [default_mode], 'all' -> all DC_MODES, else a comma list. Validated."""
    if value == "all":
        modes = list(DC_MODES)
    else:
        modes = [m.strip() for m in value.split(",") if m.strip()] or [default_mode]
    bad = [m for m in modes if m not in DC_MODES]
    if bad:
        raise ValueError(f"unknown dc mode(s) {bad}; choose from {DC_MODES} or 'all'")
    return modes


# =============================================================================
# calib mode
# =============================================================================
def measure(model, loader, cfg, device, args, late):
    """Average the loss terms over --steps batches.

    Phase 1: measure fan-domain fidelity terms only (cheap; no recon/VGG/DC).
    Phase 2: measure DC and reconstruction-domain terms after warmup/checkpoint.
    """
    model.eval()
    acc = ({"dc": 0.0, "img": 0.0, "edge": 0.0, "perc": 0.0} if late
           else {"rmse": 0.0, "hole": 0.0})
    n = 0
    with torch.no_grad():
        for step, batch in zip(range(args.steps), loader):
            if late:
                vals = term_losses(model, batch, device, [cfg.dc_mode])
            else:
                out, y = forward(model, batch, device)
                yp, mo, yt = out[:, 0:1], out[:, 1:2], y[:, 0:1]
                vals = {"rmse": model.RMSE(yt, yp).mean(),
                        "hole": model.loss_hole(mo, yt, yp).mean()}
            for k in acc:
                acc[k] += float(vals[k])
            n += 1
            print(f"  [phase{2 if late else 1}] {step + 1}/{args.steps}", flush=True)
    return {k: acc[k] / max(1, n) for k in acc}


def warmup_rmse(model, cfg, gm, gs, device, args):
    """RMSE-only warmup before measuring late losses."""
    if args.weights or args.warmup_steps <= 0:
        return
    print(f"\nWarmup: {args.warmup_steps} RMSE-only steps")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step, batch in zip(range(args.warmup_steps),
                           make_loader(cfg, "train", gm, gs, args, shuffle=True)):
        out, y = forward(model, batch, device)
        loss = model.RMSE(y[:, 0:1], out[:, 0:1]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  {step + 1}/{args.warmup_steps} rmse={loss.item():.5f}", flush=True)


def run_calib(model, loader, cfg, gm, gs, device, args):
    """Two-phase calibration: rmse/hole before the warmup, then dc/img/edge/perc
    after it (or at the --weights checkpoint, when one is given)."""
    print("\nPhase 1: rmse + hole")
    meas = measure(model, loader, cfg, device, args, late=False)
    warmup_rmse(model, cfg, gm, gs, device, args)
    print("\nPhase 2: dc + img + edge + perc")
    meas.update(measure(model, loader, cfg, device, args, late=True))

    print("\n" + "=" * 58)
    print(f"  {'term':<6}{'measured':>16}{'config':>16}{'ratio':>10}")
    print("-" * 58)
    for k, attr in CALIB_KEY.items():
        old = getattr(cfg, attr)
        r = meas[k] / old if old else float("nan")
        flag = "LOW" if r > 2 else "HIGH" if r < 0.5 else "ok"
        print(f"  {k:<6}{meas[k]:>16.7f}{old:>16.7f}{r:>9.3f}x  {flag}")
    print("=" * 58)
    print("\nPaste into configs/<config>.yaml (calib_* block):")
    for k, attr in CALIB_KEY.items():
        print(f"  {attr}: {meas[k]:.{PRECISION[k]}f}")
    return meas

# =============================================================================
# grad mode
# =============================================================================
def run_grad(model, loader, cfg, device, args):
    """Gradient norm of each calibrated term against hole, plus the matching weights.

    Also builds the pairwise cosine matrix: cos-to-hole alone cannot say whether
    two auxiliary terms pull in the same direction as each other.
    """
    model.eval()
    rows, gram_sum = [], {}
    for step, batch in zip(range(args.steps), loader):
        t = term_losses(model, batch, device, args.dc_modes)
        names, ref, refn, per, units = list(t), None, 0.0, {}, {}
        for j, name in enumerate(names):
            scale = getattr(cfg, CALIB_KEY["dc" if name.startswith("dc") else name])
            model.zero_grad(set_to_none=True)
            scaled = t[name] / (scale + 1e-8)
            scaled.backward(retain_graph=j < len(names) - 1)
            g = grad_vec(model)
            if name == "hole":            # reference term: comes first, so set once
                ref, refn = g, g.norm().item()
            per[name] = {"loss": float(t[name]), "scale": scale,
                         "scaled_loss": float(scaled), **grad_stats(g, ref, refn)}
            gn = g.norm()
            units[name] = (g / gn.clamp_min(1e-12)).cpu()
        for a in names:
            for b in names:
                gram_sum[(a, b)] = gram_sum.get((a, b), 0.0) + float(units[a] @ units[b])
        rows.append(per)
        print(f"  [grad] {step + 1}/{args.steps}", flush=True)

    names = list(rows[0])
    avg = {n: {k: sum(r[n][k] for r in rows) / len(rows) for k in rows[0][n]}
           for n in names}
    gram = {a: {b: gram_sum[(a, b)] / len(rows) for b in names} for a in names}
    print("\npairwise gradient cosine (batch-averaged):")
    print("term," + ",".join(names))
    for a in names:
        print(a + "," + ",".join(f"{gram[a][b]:.4f}" for b in names))
    cols = ("loss", "scale", "scaled_loss", "grad_norm", "matched_w", "cos", "dot")
    print("\nterm," + ",".join(cols))
    for n in names:
        print(n + "," + ",".join(f"{avg[n][c]:.8f}" for c in cols))
    print("recommended --w_* to match hole gradient:")
    for n in names:
        flag = "--w_dc" if n.startswith("dc") else W_FLAG.get(n)
        if flag and n != "hole":
            print(f"  {flag} {avg[n]['matched_w']:.8f}  ({n})")
    for n in names:                     # carried in the JSON; gen_configs ignores it
        avg[n]["cos_to"] = gram[n]
    return avg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("calib", "grad"))
    ap.add_argument("--config", default="pconv",
                    help="configs/<config>.yaml: stats_path, dataset roots, dc_mode, calib_*.")
    ap.add_argument("--weights", default="", help="checkpoint ('model' key or raw state_dict).")
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--steps", type=int, default=None,
                    help="batches to average (default: 1000 calib / 1 grad).")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_steps", type=int, default=500,
                    help="calib only: RMSE warmup before measuring (ignored with --weights).")
    ap.add_argument("--dc_modes", default="",
                    help="grad only: comma list or 'all' (default: config dc_mode).")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()
    if args.steps is None:
        args.steps = 1000 if args.mode == "calib" else 1

    cfg = load_config(args.config)
    try:
        args.dc_modes = resolve_dc_modes(args.dc_modes, cfg.dc_mode)
    except ValueError as e:
        ap.error(str(e))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"mode={args.mode} device={device} split={args.split} steps={args.steps} "
          f"weights={args.weights or 'random'}")

    with open(cfg.stats_path) as f:
        s = json.load(f)
    gm = (s["head"]["mean"], s["body"]["mean"])
    gs = (s["head"]["std"], s["body"]["std"])

    model = PConvUNet(global_mean=gm, global_std=gs).to(device)
    if args.weights:
        st = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(st["model"] if isinstance(st, dict) and "model" in st else st)
    loader = make_loader(cfg, args.split, gm, gs, args)

    result = (run_calib(model, loader, cfg, gm, gs, device, args) if args.mode == "calib"
              else run_grad(model, loader, cfg, device, args))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"mode": args.mode, "config": args.config, "weights": args.weights,
                       "dc_modes": args.dc_modes, "results": result}, f, indent=2)
        print(f"saved_json={args.output_json}")


if __name__ == "__main__":
    main()
