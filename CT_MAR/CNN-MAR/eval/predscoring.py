"""AAPM submission for the 29 clinical cases with CNN-MAR.
- the raws hold the post-processed image
- FOV and metal HU of each case come from configs/aapm_eval_cases.json
- writes submission/01.raw ... 29.raw
    python -m eval.predscoring --data_dir ../scoring29_mix \
        --model final_run_huda/saved/last_ep_weights/cnnmar_weights.weights.h5 \
        --out_dir ../../example-main/AAPM_datachallenge/scoring/cnn_mar/bestckpt_scoring29_mix
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data_pipline.preprocessing import prepare_inference_input, prediction_mu_to_hu
from mar_common.aapm_cases import case_id, load_eval_cases
from mar_common.cases import list_case_dirs
from mar_common.io import read_mha, write_raw
from mar_common.scanner_config import recon_config
from postprocessing import postprocess_cnnmar
from models import ZhangCNNMAR

CT_MAR_ROOT = Path(__file__).resolve().parents[2]


def load_model(model_path):
    """Load a saved-model folder or a weights-only file."""
    model_path = Path(model_path)
    if model_path.is_dir():
        return tf.keras.models.load_model(str(model_path), compile=False)

    model = ZhangCNNMAR(nMC=3, nFeature=32, nKernel=3)
    _ = model(tf.zeros((1, 64, 64, 3), tf.float32), training=False)
    model.load_weights(str(model_path))
    return model


def predict_img(model, raw_p, bhc_p, li_p, mask_p):
    """Network prediction for one clinical case, in HU (512 x 512)."""
    raw = read_mha(str(raw_p)).astype(np.float32)
    bhc = read_mha(str(bhc_p)).astype(np.float32)
    li = read_mha(str(li_p)).astype(np.float32)
    mask = read_mha(str(mask_p))

    x = prepare_inference_input(raw, bhc, li, mask)[None, ...]
    pred_mu = model(x, training=False).numpy()[0, ..., 0]
    return prediction_mu_to_hu(pred_mu).astype(np.float32)


def postprocess_img(cfg, cnn_hu, proj_p, trace_p, mask_p):
    """CNN-MAR post-processing of one prediction, in HU (512 x 512)."""
    result = postprocess_cnnmar(
        read_mha(str(proj_p)).astype(np.float32),
        cnn_hu,
        read_mha(str(mask_p)),
        cfg,
        metal_trace=read_mha(str(trace_p)),
    )
    return result.image_hu.astype(np.float32), result.prior_info


def paste_metal(img, mask, metal_hu):
    """The scorer counts voxels strictly above its threshold as metal."""
    return metal_hu * mask + (1.0 - mask) * img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Folder containing the 29 clinical case subfolders")
    ap.add_argument("--model", type=str, required=True,
                    help="Saved-model directory or weights file")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="output folder, the submission goes to <out_dir>/submission")
    ap.add_argument("--cases_meta", type=str, default=None,
                    help="Per-case table (default: configs/aapm_eval_cases.json)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    submission = out_dir / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    if any(submission.glob("[0-9][0-9].raw")):
        raise FileExistsError(f"Use a fresh output directory, {submission} already contains raw files")

    config_dir = Path(os.environ.get("MAR_CONFIG_DIR", CT_MAR_ROOT / "configs"))
    meta_path = Path(args.cases_meta) if args.cases_meta else config_dir / "aapm_eval_cases.json"
    meta = load_eval_cases(meta_path)

    case_dirs = list_case_dirs(data_dir)
    expected_ids = {f"{i:02d}" for i in range(1, 30)}
    if len(case_dirs) != 29 or {case_id(case) for case in case_dirs} != expected_ids:
        raise ValueError("Expected exactly one clinical case folder for each ID 01 through 29")
    if not expected_ids.issubset(meta):
        raise ValueError(f"Missing clinical case metadata in {meta_path}")
    print(f"[info] Found {len(case_dirs)} case folders in {data_dir}")

    model = load_model(args.model)

    setups = {}
    n_ok, n_skip, n_fail = 0, 0, 0
    for case in tqdm(case_dirs):
        xx = case_id(case)
        if xx not in meta:
            print(f"[skip] no metadata for case {xx} in {case}")
            n_skip += 1
            continue

        raw_p = case / f"img_{xx}.mha"
        bhc_p = case / f"img_bhc_{xx}.mha"
        li_p = case / f"img_li_{xx}.mha"
        proj_p = case / f"sino_{xx}.mha"
        trace_p = case / f"metal_trace_{xx}.mha"
        mask_p = case / f"mask_{xx}.mha"
        inputs = (raw_p, bhc_p, li_p, proj_p, trace_p, mask_p)
        if not all(p.exists() for p in inputs):
            print(f"[skip] missing inputs in {case}")
            n_skip += 1
            continue

        try:
            fov_mm = float(meta[xx]["fov_mm"])
            metal_hu = float(meta[xx]["metal_hu"])

            pred = predict_img(model, raw_p, bhc_p, li_p, mask_p)

            if fov_mm not in setups:
                setups[fov_mm] = recon_config(fov_mm)
            pred, prior_info = postprocess_img(
                setups[fov_mm], pred, proj_p, trace_p, mask_p
            )

            metal_mask = read_mha(str(mask_p)).astype(np.float32)
            pred = paste_metal(pred, metal_mask, metal_hu)
            if pred.shape != (512, 512) or not np.isfinite(pred).all():
                raise ValueError("Submission image must be a finite 512x512 array")

            write_raw(submission / f"{xx}.raw", pred)
            centers = ", ".join(f"{v:.1f}" for v in prior_info.cluster_centers_hu)
            print(f"[{xx}] fov={fov_mm:.0f}mm metal={metal_hu:.0f}HU centers=[{centers}] "
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



