"""AAPM submission for the 29 clinical cases
- writes submission/01.raw ... 29.raw
    python -m evaluation.aapm-score.predscoring --data_dir ../scoring29 \
        --ckpt results/pconv_fullstack/checkpoints/epoch_175.pt \
        --out_dir ../example-main/AAPM_datachallenge/scoring/pconv/fullstack_bestckpt
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


from ct_ops.fbp import reconstruct_fdk
from mar_common.aapm_cases import case_id, default_cases_meta, load_eval_cases
from mar_common.cases import list_case_dirs
from mar_common.io import read_mha, write_raw
from mar_common.scanner_config import recon_config
from evaluation.common import load_model, predict_proj

CT_MAR_ROOT = Path(__file__).resolve().parents[3]


def recon_img(cfg, proj):
    """returned in HU (512 x 512)"""
    return reconstruct_fdk(cfg, proj.astype(np.float32)).astype(np.float32)



def paste_metal(img, mask, metal_hu):
    """The scorer counts voxels above its threshold as metal"""
    return metal_hu * mask + (1.0 - mask) * img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    
                    help="containing the 29 clinical case subfolders")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to the selected thesis checkpoint")
    ap.add_argument("--stats_path", type=str,
                    default=str(CT_MAR_ROOT / "configs/aapm_norm_global_mean_case_std.json"),
                    help="path to per-case or head/body normalization stats")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder, <out_dir>/submission")
    ap.add_argument("--device", type=str, default="cpu",
                    help=" cuda to run on GPU")
    ap.add_argument("--cases_meta", type=str, default=None,
                    help="Per-case table (default: configs/aapm_eval_cases.json)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    submission = out_dir / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    if any(submission.glob("[0-9][0-9].raw")):
        raise FileExistsError(f"Use a fresh output directory; {submission} already contains raw files")

    meta_path = Path(args.cases_meta) if args.cases_meta else default_cases_meta()
    meta = load_eval_cases(meta_path)

    stats = json.loads(Path(args.stats_path).read_text())

    model = load_model(args.ckpt, device)

    case_dirs = list_case_dirs(data_dir)
    print(f"[info] Found {len(case_dirs)} case folders in {data_dir}")

    setups = {}
    n_ok, n_skip, n_fail = 0, 0, 0
    for case in tqdm(case_dirs):
        xx = case_id(case)
        if xx not in meta:
            print(f"[skip] no metadata for case {xx} in {case}")
            n_skip += 1
            continue

        proj_p = case / f"sino_{xx}.mha"
        trace_p = case / f"metal_trace_{xx}.mha"
        mask_p = case / f"mask_{xx}.mha"
        if not (proj_p.exists() and trace_p.exists() and mask_p.exists()):
            print(f"[skip] missing inputs in {case}")
            n_skip += 1
            continue

        try:
            fov_mm = float(meta[xx]["fov_mm"])
            metal_hu = float(meta[xx]["metal_hu"])
            norm = stats["cases"][xx] if "cases" in stats else stats[meta[xx]["norm"]]
            mean = float(norm["mean"])
            std = float(norm["std"])

            pred, _ = predict_proj(model, proj_p, trace_p, mean, std, device)

            if fov_mm not in setups:
                setups[fov_mm] = recon_config(fov_mm)
            pred = recon_img(setups[fov_mm], pred)

            metal_mask = read_mha(str(mask_p)).astype(np.float32)
            pred = paste_metal(pred, metal_mask, metal_hu)

            write_raw(submission / f"{xx}.raw", pred)
            print(f"[{xx}] fov={fov_mm:.0f}mm metal={metal_hu:.0f}HU "
                  f"norm=({mean:.6f}, {std:.6f}) "
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
    
    
