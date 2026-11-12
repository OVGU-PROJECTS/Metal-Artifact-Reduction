"""Run the trained PConv-MAR on the test set (once for head and once for body)
- needs CUDA (the FDK recon)
- writes target/, mask/, pred/ and metrics_sino.csv

    python -m evaluation.run_inference --test_dir ../test_images_1000/head \
        --ckpt results/pconv_fullstack/checkpoints/epoch_175.pt \
        --out_dir evaluation/predictions/pconv_fullstack_bestckpt/head
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


from mar_common.cases import inference_fov, list_case_dirs
from mar_common.io import read_mha
from mar_common.units import mu2hu
from evaluation.common import load_model, predict_proj
from metrics.scoring.sino import score_sinogram
from recon import metric_fdk_setups, reconstruct
CT_MAR_ROOT = Path(__file__).resolve().parents[2]


def recon_img(recon_ctx, proj, case_fov, device):
    """FDK reconstruction of a mu-domain sinogram, returned in HU (512 x 512)."""
    cfg, hsp = recon_ctx["body" if case_fov == 1 else "head"]
    proj_t = torch.from_numpy(proj.astype(np.float32)).to(device)
    with torch.inference_mode():
        img_mu = reconstruct(cfg, proj_t, hsp.to(device))
    return mu2hu(img_mu.detach().cpu().numpy()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, required=True,
                    help="Path to testset folder containing case subfolders")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="path to the selected checkpoint, pass the thesis checkpoint explicitly")
    ap.add_argument("--stats_path", type=str,
                    default=str(CT_MAR_ROOT / "configs/head_body_valid_stats.json"),
                    help="Path to per-case or head/body normalization stats")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder to write predictions")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--head_last", type=int, default=136,
                    help="Case ids <= this are head FOV, 136 for test_images_1000, 1626 for training")
    ap.add_argument("--fov", choices=["auto", "head", "body"], default="auto",
                    help="Geometry: explicit FOV, then head/body folder, then case ID")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":  # recon/recon.py backprojects on CUDA only
        raise SystemExit(f"FDK reconstruction requires CUDA, got --device {args.device}")

    stats = json.load(open(args.stats_path, "r"))

    model = load_model(args.ckpt, device)
    recon_ctx = metric_fdk_setups()

    case_dirs = list_case_dirs(test_dir)
    print(f"[info] Found {len(case_dirs)} case folders in {test_dir}")

    target_path, mask_path, pred_path = (out_dir / n for n in ("target", "mask", "pred"))
    for d in (target_path, mask_path, pred_path):
        d.mkdir(parents=True, exist_ok=True)

    sino_rows = []
    n_ok, n_fail = 0, 0
    for case in tqdm(case_dirs):
        try:
            case_fov = inference_fov(case, head_last=args.head_last, fov=args.fov)
            norm = (stats["cases"][case.name] if "cases" in stats
                    else stats["body" if case_fov == 1 else "head"])
            mean = float(norm["mean"])
            std = float(norm["std"])
            pred, valid = predict_proj(model, case / "input_proj.mha",
                                       case / "input_metal_trace_proj.mha",
                                       mean, std, device)

            # Sinogram metrics, scored before reconstruction.
            gt_proj = read_mha(str(case / "output_proj.mha")).astype(np.float32)
            sino_row = {"case": case.name, **score_sinogram(gt_proj, pred, valid < 0.5)}

            gt_img = read_mha(str(case / "output_img.mha")).astype(np.float32)
            metal_mask = read_mha(str(case / "input_mask_dilated.mha")).astype(np.float32)
            pred = recon_img(recon_ctx, pred, case_fov, device)

            xx = str(n_ok + 1).zfill(4)  # numbered over saved cases, so it matches metrics_sino.csv
            img_fn = f"nometal_{xx}_512x512.npy"
            mask_name = f"metalonlymask_{xx}_512x512.npy"
            np.save(target_path / img_fn, gt_img)
            np.save(mask_path / mask_name, metal_mask)
            np.save(pred_path / img_fn, pred)
            sino_rows.append(sino_row)
            n_ok += 1
        except Exception as exc:  # a broken case must not stop the sweep
            n_fail += 1
            print(f"[warn] {case.name}: {type(exc).__name__}: {exc}")

    if not sino_rows:
        raise SystemExit(f"[done] ok=0, failed={n_fail}, no case succeeded in {test_dir}")
    keys = list(sino_rows[0].keys())
    with open(out_dir / "metrics_sino.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sino_rows)
    print("[sino] " + " ".join(
        f"{k}={np.mean([r[k] for r in sino_rows]):.5f}" for k in keys[1:]))
    print(f"[done] ok={n_ok}, failed={n_fail}, out={out_dir}")


if __name__ == "__main__":
    main()
    
    
    
    
