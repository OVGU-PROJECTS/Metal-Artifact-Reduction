"""Run CNN-MAR inference and write the prediction back into each case folder.

generated pred_cnnmar_img.mha, the direct CNN output used for the invariance analysis.
"""
import os, argparse, numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from training.trainer import build_model
from mar_common.io import read_mha, write_mha
from mar_common.cases import list_case_dirs
from data_pipline.preprocessing import prepare_inference_input, prediction_mu_to_hu

def load_models(model_path: Path):
    """Load either a full saved-model directory or a weights-only file into the model."""
    if model_path.is_dir():
        return tf.keras.models.load_model(str(model_path), compile= False)
    
    model = build_model(lr=1.66e-3)
    _= model(tf.zeros((1,64,64,3), tf.float32), training= False)
    model.load_weights(str(model_path))
    return model
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required= True,
                    help="Path to testset folder containing case subfolders")
    ap.add_argument("--model", type=str, required= True,
                    help="Path to weights file (e.g. .../saved/weights/best.weights.h5)")
    args = ap.parse_args()

    data_dir     = Path(args.data_dir)
    model_dir = Path(args.model)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"weights not found: {model_dir}")

    # Build + load weights (subclassed model => weights-only load is the robust way)
    model = load_models(model_dir)

    case_dirs = list_case_dirs(data_dir)
    print(f"[info] Found {len(case_dirs)} case folders in {data_dir}")

    n_ok, n_skip = 0, 0
    for case in tqdm(case_dirs):
        try:
            raw  = case / "input_img.mha"
            bhc  = case / "img_bhc.mha"
            li   = case / "img_li.mha"
            metal = case / "input_mask.mha"
            if not (raw.exists() and bhc.exists() and li.exists() and metal.exists()):
                print(f"[skip] missing file(s) in {case.name}")
                n_skip += 1
                continue
            raw = read_mha(raw)
            bhc = read_mha(bhc)
            li  = read_mha(li)
            metal = read_mha(metal)

            x = prepare_inference_input(raw, bhc, li, metal)[None, ...]
            y = prediction_mu_to_hu(model(x, training=False).numpy()[0, ..., 0])

            pred_path =case /"pred_cnnmar_img.mha"
            write_mha(y, str(pred_path))
            
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"done [{n_ok}]: /{case}")
        except Exception as e:
            print(f"[skip] {case.name}: {type(e).__name__}: {e}")
            n_skip += 1
    print(f"[done] ok={n_ok}, skipped={n_skip}")


if __name__ == "__main__":
    main()
'''
conda activate myenv
cd /mnt/public/bahaa/compare/CT_MAR/CNN-MAR

GPU=$(gpurequest)
echo "Using physical GPU: $GPU"

CUDA_VISIBLE_DEVICES="$GPU" \
python -u -m eval.pred_invariance \
  --data_dir /mnt/public/bahaa/dataset_training \
  --model "/mnt/public/bahaa/compare/CT_MAR/CNN-MAR/results/final_run/saved/best_tf"
'''