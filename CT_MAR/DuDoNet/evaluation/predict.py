"""Run the trained DuDoNet on the test set.
- writes the same target/, mask/, pred/ and metrics_sino.csv 

    python -m evaluation.predict --test_dir ../../test_images_1000/head \
        --ckpt results/final_run/checkpoints/best.pt \
        --out_dir evaluation/predictions/final_run_bestckpt/head
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


from mar_common.cases import fov_from_folder, list_case_dirs
from mar_common.io import read_mha, to_4d_tensor
from mar_common.units import mu2hu
from metrics.scoring.sino import score_sinogram


FOV_CODE = {"head": 0, "body": 1}


def load_model(ckpt_path, fov_head_mm, fov_body_mm, device):
    from models import DuDoNet2RIL  # CUDA dependencies are needed only for inference.

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = DuDoNet2RIL(fov_head_mm, fov_body_mm, img_size=512).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def fov_from_test_dir(test_dir, fov):
    """FOV code named on the command line or by the path; None means per case."""
    if fov in FOV_CODE:
        return FOV_CODE[fov]
    for fov_name in ("head", "body"):
        if fov_name in Path(test_dir).parts:
            return FOV_CODE[fov_name]
    return None


def predict_img(model, case, case_fov, device):
    """One case: image in HU (512x512), SE-Net sinogram, metal trace."""
    y_li = read_mha(str(case / "input_proj_li.mha")).astype(np.float32)
    mt = read_mha(str(case / "input_metal_trace_proj.mha")).astype(np.float32)

    fov_t = torch.tensor([case_fov], dtype=torch.long, device=device)
    y_li_t = to_4d_tensor(y_li).to(device)
    mt_t = to_4d_tensor(mt).to(device)
    with torch.inference_mode():
        outputs = model(y_li_t, mt_t, fov_t)

    img_mu = outputs["X_out"][0, 0].detach().cpu().numpy()
    y_out = outputs["Y_out"][0, 0].detach().cpu().numpy().astype(np.float32)
    return mu2hu(img_mu).astype(np.float32), y_out, mt > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, required=True,
                    help="Path to testset folder containing case subfolders")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to PyTorch checkpoint, e.g. .../checkpoints/best.pt")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder to write predictions")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fov", type=str, default="auto", choices=["head", "body", "auto"],
                    help="Recon geometry; auto reads it from --test_dir, else from the case id")
    ap.add_argument("--head_last", type=int, default=136,
                    help="Case ids <= this are head FOV; 136 for test_images_1000, 1773 for training")
    ap.add_argument("--fov_head_mm", type=float, default=220.16)
    ap.add_argument("--fov_body_mm", type=float, default=400.0)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":  # the RIL backprojects with a Numba CUDA kernel
        raise SystemExit(f"The Radon Inversion Layer requires CUDA, got --device {args.device}")

    model = load_model(args.ckpt, args.fov_head_mm, args.fov_body_mm, device)
    dir_fov = fov_from_test_dir(test_dir, args.fov)

    case_dirs = list_case_dirs(test_dir)
    print(f"[info] Found {len(case_dirs)} case folders in {test_dir}")

    target_path = out_dir / "target"
    mask_path = out_dir / "mask"
    pred_path = out_dir / "pred"
    target_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    pred_path.mkdir(parents=True, exist_ok=True)

    sino_rows = []
    fov_counts = {0: 0, 1: 0}
    n_ok, n_fail = 0, 0
    for case in tqdm(case_dirs):
        try:
            case_fov = dir_fov
            if case_fov is None:
                case_fov = int(fov_from_folder(case, head_last=args.head_last))

            pred, y_out, hole = predict_img(model, case, case_fov, device)

            # Sinogram metrics on the SE-Net output.
            gt_proj = read_mha(str(case / "output_proj.mha")).astype(np.float32)
            sino_row = {"case": case.name, **score_sinogram(gt_proj, y_out, hole)}

            gt_img = read_mha(str(case / "output_img.mha")).astype(np.float32)
            metal_mask = read_mha(str(case / "input_mask_dilated.mha")).astype(np.float32)

            xx = str(n_ok + 1).zfill(4)  # numbered over saved cases, so it matches metrics_sino.csv
            img_fn = f"nometal_{xx}_512x512.npy"
            mask_name = f"metalonlymask_{xx}_512x512.npy"
            np.save(target_path / img_fn, gt_img)
            np.save(mask_path / mask_name, metal_mask)
            np.save(pred_path / img_fn, pred)

            sino_rows.append(sino_row)
            fov_counts[case_fov] += 1
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
    print(f"[fov] head={fov_counts[0]} body={fov_counts[1]}")
    print(f"[done] ok={n_ok}, failed={n_fail}, out={out_dir}")


if __name__ == "__main__":
    main()
    
    
    
