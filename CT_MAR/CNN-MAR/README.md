
# CNN-MAR

This directory contains the TensorFlow/Keras reimplementation of CNN-MAR used
in the thesis. It covers generation of the analytical LI and BHC inputs,
patch-based CNN training, direct-image inference, analytical CNN-MAR
post-processing, and stage-wise evaluation.

## Structure

- `configs/`: CNN-MAR training and preprocessing configuration.
- `data_pipline/`: patch selection, data generation, and image preparation.
- `data_prep/`: generation of LI and BHC input images.
- `models/`: CNN-MAR network definition.
- `training/`: trainer, callbacks, and checkpoint handling.
- `postprocessing/`: prior construction and projection-domain correction.
- `eval/`: direct inference, post-processing drivers, and evaluation scripts.
- `results/`: trained models, histories, metrics, and qualitative examples.

## Environment

Activate the shared CT-MAR environment and make the project and its shared
packages importable:

```bash
conda activate myenv
cd /path/to/CNN-MAR
export PYTHONPATH="$(dirname "$(dirname "$PWD")"):$PWD:${PYTHONPATH:-}"
```
## Configuration

Training and preprocessing use:

```text
configs/cnnmar.yaml
```

Important fields are:

- `train_root`, `val_root`, and `test_root`;
- `out_dir`;
- `bh_li_dir`, which selects the split processed by `BHC_LI.py`;
- `lr`, `batch_size`, `epochs`, and `seed`;
- `trainset` and `valset`;
- `patch`, `stride`, and `strong_ratio`.


## Generate LI and BHC inputs

Every case must contain `input_proj.mha` and `input_mask_dilated.mha`. Set
`bh_li_dir` in `configs/cnnmar.yaml` to the desired dataset split and run:

`data_prep/BHC_LI.py` writes `img_li.mha` and `img_bhc.mha` into every processed case folder.

## Train CNN-MAR

Training script is found in `training/trainer.py`

## Generate the direct CNN output

`eval/pred_invariance.py` applies the selected model and writes the direct CNN
output into every source case as `pred_cnnmar_img.mha`:

```bash
CUDA_VISIBLE_DEVICES="$GPU" \
python -u -m eval.pred_invariance \
  --data_dir /path/to/cases \
  --model /path/to/best_tf
```

This direct network output is the CNN-MAR conditioning image used by the invariance analysis.

## Analytical post-processing

After direct inference, construct the prior and final CNN-MAR reconstruction with `eval/postprocess.py`

For each case compute:

- `pred_cnnmar_prior.mha`: the three-tissue prior,
- `pred_cnnmar_final.mha`: the final corrected reconstruction.

## Stage-wise evaluation

Compute RMSE, SSIM, and MAE for the uncorrected image, LI, BHC, direct CNN
output, and final CNN-MAR reconstruction with `eval/stagewise_metrics.py`.

The evaluator compares every stage with `output_img.mha`, excludes the dilated
metal region, and uses a clipping range of `[-1000, 3000]` HU for RMSE and MAE. It writes:

```text
stagewise_per_case.csv
stagewise_summary.csv
stagewise_latex_rows.tex
```

## Qualitative stage-wise evaluation

Use `eval/postprocess_random_case.py` to randomly select an eligible validation case, apply the analytical post-processing, and export all intermediate CNN-MAR stages.

