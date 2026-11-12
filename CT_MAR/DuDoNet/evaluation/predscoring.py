"""AAPM submission for the 29 clinical cases with DuDoNet.
- FOV and metal HU of each case come from configs/aapm_eval_cases.json
- writes submission/01.raw ... 29.raw

    python -m evaluation.predscoring --data_dir ../scoring29_mix \
        --ckpt results/final_run/checkpoints/best.pt \
        --out_dir ../../example-main/AAPM_datachallenge/scoring/DuDoNet/bestckpt_scoring29_mix
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


from mar_common.aapm_cases import case_id, default_cases_meta, load_eval_cases
from mar_common.cases import list_case_dirs
from mar_common.io import read_mha, to_4d_tensor, write_raw
from mar_common.units import mu2hu


def load_model(state, fov_mm, device):
    from models import DuDoNet2RIL  # Keep CLI help usable without CuPy/CUDA.

    model = DuDoNet2RIL(fov_mm, fov_mm, img_size=512).to(device)
    model.load_state_dict(state, strict=True)  # SE/IE weights; the RIL has none
    model.eval()
    return model


def predict_img(model, proj_li, trace_p, device):
    """One clinical case in HU (512x512); no FOV is head/body, so all use slot 0."""
    y_li = read_mha(str(proj_li)).astype(np.float32)
    mt = read_mha(str(trace_p)).astype(np.float32)

    fov_t = torch.tensor([0], dtype=torch.long, device=device)
    y_li_t = to_4d_tensor(y_li).to(device)
    mt_t = to_4d_tensor(mt).to(device)
    with torch.inference_mode():
        outputs = model(y_li_t, mt_t, fov_t)

    img_mu = outputs["X_out"][0, 0]
    return mu2hu(img_mu.detach().cpu().numpy()).astype(np.float32)


def paste_metal(img, mask, metal_hu):
    """The scorer counts voxels strictly above its threshold as metal."""
    return metal_hu * mask + (1.0 - mask) * img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing the 29 clinical case subfolders")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to PyTorch checkpoint, e.g. .../checkpoints/best.pt")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder; the submission goes to <out_dir>/submission")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cases_meta", type=str, default=None,
                    help="Per-case table (default: configs/aapm_eval_cases.json)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":  # the RIL backprojects with a Numba CUDA kernel
        raise SystemExit(f"The Radon Inversion Layer requires CUDA, got --device {args.device}")
    submission = out_dir / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    if any(submission.glob("[0-9][0-9].raw")):
        raise FileExistsError(f"Use a fresh output directory; {submission} already contains raw files")

    meta_path = Path(args.cases_meta) if args.cases_meta else default_cases_meta()
    meta = load_eval_cases(meta_path)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state = checkpoint["model_state"]

    case_dirs = list_case_dirs(data_dir)
    print(f"[info] Found {len(case_dirs)} case folders in {data_dir}")

    models = {}
    n_ok, n_skip, n_fail = 0, 0, 0
    for case in tqdm(case_dirs):
        xx = case_id(case)
        if xx not in meta:
            print(f"[skip] no metadata for case {xx} in {case}")
            n_skip += 1
            continue

        proj_li = case / f"sino_li_{xx}.mha"
        trace_p = case / f"metal_trace_{xx}.mha"
        mask_p = case / f"mask_{xx}.mha"
        if not (proj_li.exists() and trace_p.exists() and mask_p.exists()):
            print(f"[skip] missing inputs in {case}")
            n_skip += 1
            continue

        try:
            fov_mm = float(meta[xx]["fov_mm"])
            metal_hu = float(meta[xx]["metal_hu"])
            if fov_mm not in models:
                models[fov_mm] = load_model(state, fov_mm, device)

            pred = predict_img(models[fov_mm], proj_li, trace_p, device)

            metal_mask = read_mha(str(mask_p)).astype(np.float32)
            pred = paste_metal(pred, metal_mask, metal_hu)

            write_raw(submission / f"{xx}.raw", pred)
            print(f"[{xx}] fov={fov_mm:.0f}mm metal={metal_hu:.0f}HU "
                  f"mean={pred.mean():.1f} min={pred.min():.1f} max={pred.max():.1f}")
            n_ok += 1
        except Exception as exc:  # a broken case must not stop the sweep
            n_fail += 1
            print(f"[warn] {xx}: {type(exc).__name__}: {exc}")

    if (n_ok, n_skip, n_fail) != (29, 0, 0):
        for raw in submission.glob("[0-9][0-9].raw"):
            raw.unlink()
        raise RuntimeError(f"Incomplete run: ok={n_ok}, skipped={n_skip}, failed={n_fail}")
    print(f"[done] ok=29, skipped=0, failed=0, out={submission}")


if __name__ == "__main__":
    main()
