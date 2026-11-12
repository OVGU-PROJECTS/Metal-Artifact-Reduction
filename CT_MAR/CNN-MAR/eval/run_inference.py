"""Run the trained CNN-MAR on the test set, with its post-processing.
- run once for head and once for body
- writes target/, mask/ and pred/ for score_images.py

    python -m eval.run_inference --test_dir ../../test_images_1000/head \
        --model final_run_huda/saved/last_ep_weights/cnnmar_weights.weights.h5 \
        --out_dir eval/predictions/final_run_huda_bestckpt/head
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_pipline.preprocessing import prepare_inference_input, prediction_mu_to_hu
from geometry.fanbeam import RECO_FOV
from mar_common.cases import inference_fov, list_case_dirs
from mar_common.io import read_mha
from mar_common.scanner_config import recon_config
from postprocessing import postprocess_cnnmar
from models import ZhangCNNMAR


def load_model(model_path):
    """Load a saved-model folder or a weights-only file."""
    model_path = Path(model_path)
    if model_path.is_dir():
        return tf.keras.models.load_model(str(model_path), compile=False)

    model = ZhangCNNMAR(nMC=3, nFeature=32, nKernel=3)
    _ = model(tf.zeros((1, 64, 64, 3), tf.float32), training=False)
    model.load_weights(str(model_path))
    return model


def predict_img(model, case):
    """Network prediction for one case, in HU (512 x 512)."""
    raw = read_mha(str(case / "input_img.mha")).astype(np.float32)
    bhc = read_mha(str(case / "img_bhc.mha")).astype(np.float32)
    li = read_mha(str(case / "img_li.mha")).astype(np.float32)
    mask = read_mha(str(case / "input_mask.mha"))

    x = prepare_inference_input(raw, bhc, li, mask)[None, ...]
    pred_mu = model(x, training=False).numpy()[0, ..., 0]
    return prediction_mu_to_hu(pred_mu).astype(np.float32)


def recon_setups():
    """recon_config per FOV, built once."""
    return {"head": recon_config(float(RECO_FOV["head"])),
            "body": recon_config(float(RECO_FOV["body"]))}



def postprocess_img(recon_ctx, cnn_hu, case, case_fov):
    """CNN-MAR post-processing of one prediction, in HU (512 x 512)."""
    cfg = recon_ctx["body" if case_fov == 1 else "head"]
    trace_path = case / "input_metal_trace_proj.mha"
    result = postprocess_cnnmar(
        read_mha(str(case / "input_proj.mha")).astype(np.float32),
        cnn_hu,
        read_mha(str(case / "input_mask.mha")),
        cfg,
        metal_trace=read_mha(str(trace_path)) if trace_path.exists() else None,
    )
    return result.image_hu.astype(np.float32)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, required=True,
                    help="Path to testset folder containing case subfolders")
    ap.add_argument("--model", type=str, required=True,
                    help="Saved-model directory or weights file")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder to write predictions")
    ap.add_argument("--head_last", type=int, default=136,
                    help="Case ids <= this are head FOV, 136 for test_images_1000, 1773 for training")
    ap.add_argument("--fov", choices=["auto", "head", "body"], default="auto",
                    help="Geometry: explicit FOV, then head/body folder, then case ID")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)

    case_dirs = list_case_dirs(test_dir)
    if not case_dirs:
        raise ValueError(f"No case folders in {test_dir}")
    print(f"[info] Found {len(case_dirs)} case folders in {test_dir}")

    target_path = out_dir / "target"
    mask_path = out_dir / "mask"
    pred_path = out_dir / "pred"
    for folder in (target_path, mask_path, pred_path):
        if any(folder.glob("*.npy")):
            raise FileExistsError(f"Use a fresh output directory, {folder} already contains arrays")

    model = load_model(args.model)
    recon_ctx = recon_setups()

    target_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    pred_path.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(tqdm(case_dirs), start=1):
        try:
            case_fov = inference_fov(case, head_last=args.head_last, fov=args.fov)
            pred = predict_img(model, case)

            gt_img = read_mha(str(case / "output_img.mha")).astype(np.float32)
            metal_mask = read_mha(str(case / "input_mask_dilated.mha")).astype(np.float32)
            pred = postprocess_img(recon_ctx, pred, case, case_fov)

            for name, array in (("target", gt_img), ("mask", metal_mask), ("prediction", pred)):
                if array.shape != (512, 512) or not np.isfinite(array).all():
                    raise ValueError(f"{name} must be a finite 512x512 array")

            xx = str(index).zfill(4)  # stable position in the sorted input cohort
            img_fn = f"nometal_{xx}_512x512.npy"
            mask_name = f"metalonlymask_{xx}_512x512.npy"
            np.save(target_path / img_fn, gt_img)
            np.save(mask_path / mask_name, metal_mask)
            np.save(pred_path / img_fn, pred)
        except Exception as exc:
            raise RuntimeError(
                f"Failed case {case}: {exc}. Output is incomplete, do not score it. "
                "Fix the input/problem and rerun into a fresh output directory."
            ) from exc

    print(f"[done] ok={len(case_dirs)}, failed=0, out={out_dir}")


if __name__ == "__main__":
    main()
    
    
