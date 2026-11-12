# Metrics

The thesis evaluation metrics and training measurements used by the current code
Trace descriptors and debugging checks are identified separately below.

## Test set (1000 paired cases)

Image metrics come from `src/metrics/scoring/image.py`.
`Pconv_mar/evaluation/score_images.py` can score saved predictions from all three models.
They are measured inside a field-of-view (FOV) circle of 470 mm diameter excluding metal mask.

| Column | What it is |
| --- | --- |
| `global_rmse` | Root mean squared error in HU, emphasizes large errors |
| `global_mae` | Mean absolute error in HU |
| `global_psnr` | Peak signal-to-noise ratio in decibels, with a fixed 5000 HU range |
| `global_ssim` | Structural similarity, using Gaussian weights and a 5000 HU range |
| `global_streak_amp` | Gap between the means of the highest and lowest 5% of image errors, in the flattest quarter of the reference, in HU |
| `global_edge_ratio` | mean Sobel edge strength of the prediction divided by that of the reference |
| `global_noise_ratio` | estimated fine-scale noise of the prediction divided by that of the reference |

Sinogram quality metrics come from `src/metrics/scoring/sino.py`, for PConv-MAR and
DuDoNet (`metrics_sino.csv`). They measure errors inside or along the boundary of
the metal trace

| Column | What it is |
| --- | --- |
| `fill_mae` | MAE of the filled bins |
| `seam_gradient` | gradient error in a band on both sides of the trace border |
| `trace_gradient_err` | detector-axis gradient error inside the trace |

The same CSV also includes two trace descriptors, not image-quality scores:
- `trace_width`: mean number of trace bins per view, counting only views with metal.
- `hole_frac`: fraction of all sinogram bins inside the trace.

## CNN-MAR stages

`cnn_mar/eval/stagewise_metrics.py` gives RMSE, SSIM and MAE for each stage (uncorrected,
LI, BHC, network output, final CNN-MAR), in the patient region without the dilated metal.
RMSE and MAE use HU clipped to [-1000, 3000] by default, SSIM uses unclipped HU values.

## Clinical scoring (29 cases)

The evaluation scripts generate the 29 submission images. The official AAPM scoring
program is external to this repository and scores those images separately.
Its eight measures are CT-number accuracy (RMSE), noise, edge magnitude, streak
amplitude, SSIM, metal geometry, bone geometry, and proton range.
Each is mapped to a case-specific score from 0 to 4, then averaged across cases and
measures for the final score, lower is better. These clinical scores use their own
regions and definitions, not the paired-test metrics above.

## Training logs

- PConv-MAR, `logs/loss_breakdown_{train,val}.csv`: the six loss terms `rmse`, `hole`,
  `dc`, `img`, `edge`, `perc`, and `ssim_aapm`, `ssim_trace` (SSIM inside the dilated metal mask),
  `mae_near_metal` (HU error near metal) and `streak_edge_err` (edge-map error).
  `csv/train.csv` has the train
  and val loss, the learning rate and the loss weights.
- DuDoNet, `metrics.csv`: the total loss `L` and its parts `Gs`, `RC`, `Gi`, and the
  train and val `ssim_aapm`.
- CNN-MAR: the loss and the full-image train and val `ssim_aapm`.

The training measurement `streak_edge_err` compares edge maps. It is different from
the test-set measurement `global_streak_amp`, which measures the spread of image errors.

## Hyperparameter search

- PConv-MAR (`Pconv_mar/finetune/`): the default trial score is
  `hole + (1 - ssim_aapm) + 1.5e-5 * mae_near_metal`, taking the best-scoring validation epoch.
- DuDoNet (`DuDoNet/finetune/`): by default, 50 learning-rate trials of 12 epochs,
  ranked by their best validation loss.

## Loss-term and debugging tools (`Pconv_mar/tools/`)

- `calibrate.py`: the size of each loss term, used for the `*_scale` constants.
- `measure_gradient_shares.py`: how strong the gradient of each loss term is, and how it
  lines up with the trace loss.
- `check_backward.py`: a debugging check of the reconstruction's backward pass against
  its forward pass, not an image-quality metric. Requires CUDA.

























  
