# Results and experiment artifacts

This directory contains the trained checkpoints, training histories, generated sample banks, quantitative evaluation results, and qualitative figures used for the thesis invariance analysis.

The archived files are organized so that the reported results can be traced from model training through sample generation and final evaluation.

## Directory structure

```text
results/
├── README.md
├── runs/
│   ├── cvae/
│   ├── cinn/
│   └── evaluation/
├── sample_banks/
│   ├── dml_banks/
│   └── test_banks/
├── dml/
├── vae_md/
├── posterior_mean_reconstructions/
└── invariance_samples_selected_examples.png
```

## runs

The `runs/` directory contains the archived training outputs for the conditional VAEs, conditional invertible neural networks, and embedding models used for evaluation.

### Conditional-VAE runs

The method-specific conditional-VAE runs are stored under:

```text
results/runs/cvae/
├── cvae_pconv/
├── cvae_cnnmar/
└── cvae_dudonet/
```

Each run contains:

```text
config_used.yaml
initial_values.csv
losses.csv
checkpoints/best.pt
checkpoints/last.pt
```

- `config_used.yaml` is the frozen configuration used for the run.
- `initial_values.csv` records the metrics before the first optimization epoch.
- `losses.csv` contains the complete epoch-wise training and validation history.
- `best.pt` is the checkpoint selected according to the configured validation criterion.
- `last.pt` is the checkpoint from the final completed epoch.

The cVAE checkpoint for each MAR method must remain paired with the corresponding method-specific cINN and prediction images.

### Conditional-INN runs

The method-specific cINN runs are stored under:

```text
results/runs/cinn/
├── cinn_pconv/
├── cinn_cnnmar/
└── cinn_dudonet/
```

Each run contains:

```text
config_used.yaml
initial_cinn_metrics.csv
initial_cinn_metrics.json
Losses.csv
cinn_best.pt
cinn_latest.pt
```

- `initial_cinn_metrics.csv` and `initial_cinn_metrics.json` contain the untrained-model metrics.
- `Losses.csv` contains the complete cINN training and validation history.
- `cinn_best.pt` is the checkpoint with the best validation result.
- `cinn_latest.pt` is the most recently saved checkpoint.

The model pairings used in the final analysis are:

| MAR method | Conditional VAE | Conditional INN |
|---|---|---|
| PConv-MAR | `cvae/cvae_pconv` | `cinn/cinn_pconv` |
| CNN-MAR | `cvae/cvae_cnnmar` | `cinn/cinn_cnnmar` |
| DuDoNet | `cvae/cvae_dudonet` | `cinn/cinn_dudonet` |

### Evaluation-model runs

The models used to evaluate the generated invariance samples are stored under:

```text
results/runs/evaluation/
├── dml/
└── uncond_vae/
```

#### DML embedding

The archived DML run contains:

```text
results/runs/evaluation/dml/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── initial_metrics.csv
├── initial_metrics.json
├── losses.csv
└── run_meta.json
```

- `initial_metrics.csv` and `initial_metrics.json` contain the metrics of the untrained ImageNet-initialized embedding before training.
- `losses.csv` contains the metrics after each completed training epoch.
- `best.pt` is the checkpoint with the lowest validation triplet loss.
- `last.pt` is the checkpoint from the final completed epoch.
- `run_meta.json` records the resolved configuration and device.

#### Unconditional-VAE embedding

The archived unconditional-VAE run contains:

```text
results/runs/evaluation/uncond_vae/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── initial_values.csv
└── losses.csv
```

The selected unconditional-VAE checkpoint is used as a frozen embedding model during the final invariance evaluation.


## Sample banks

### DML training and validation banks

These are the sample banks used to train the DML embedding.

The training banks contain \(K=50\) generated invariance realizations per case:

```text
pconv_train_k50.pt
cnnmar_train_k50.pt
dudonet_train_k50.pt
```

The validation banks contain \(K=2\) fixed realizations per case:

```text
pconv_val_k2.pt
cnnmar_val_k2.pt
dudonet_val_k2.pt
```

### Final test sample banks

These are the final test banks with K=1000.

The six banks are:

```text
pconv_head_full_k1000_seed2026.pt
cnnmar_head_full_k1000_seed2026.pt
dudonet_head_full_k1000_seed2026.pt

pconv_body_full_k1000_seed2026.pt
cnnmar_body_full_k1000_seed2026.pt
dudonet_body_full_k1000_seed2026.pt
```

Each bank contains:

```text
x        Original corrupted patches
p        Method-specific MAR predictions
m        Metal masks
samples  K=1000 generated invariance realizations per patch
meta     Patch locations and source-case metadata
```

The `_seed2026` suffix identifies the fixed seed used for the final test-bank generation. The same selected test patches are used for all three MAR methods to support aligned cross-method comparisons.

The sample banks are large binary PyTorch files. They can not be uploaded to a conventional Git repository.

## DML evaluation results

The final DML evaluation results are stored under:

```text
results/dml/
```

This directory contains:

```text
final_dml_results.csv
sample_count_convergence.csv
summary.json

pconv_head_full_k1000_seed2026_per_patch_metrics.csv
cnnmar_head_full_k1000_seed2026_per_patch_metrics.csv
dudonet_head_full_k1000_seed2026_per_patch_metrics.csv

pconv_body_full_k1000_seed2026_per_patch_metrics.csv
cnnmar_body_full_k1000_seed2026_per_patch_metrics.csv
dudonet_body_full_k1000_seed2026_per_patch_metrics.csv
```

### `final_dml_results.csv`

Contains the model-level DML cosine-similarity score for each method and subset, together with the image-domain mean difference and uncertainty information.

### Per-patch metrics

The six `*_per_patch_metrics.csv` files contain the patch-wise results used to calculate:

- Mean cosine similarity
- Standard deviation
- Standard error
- Median
- Interquartile range
- Minimum
- Maximum
- Bootstrap confidence interval
- Image-domain mean difference in normalized units
- Image-domain mean difference in Hounsfield units

These patch-wise files are also the source data for the DML distribution and violin plots.

### `sample_count_convergence.csv`

Contains the evaluation repeated with increasing numbers of generated realizations per patch.

The principal thesis comparison uses:

```text
K' = 100, 200, 500, and 1000
```

The primary final result uses all \(K=1000\) generated samples.

### `summary.json`

Contains a summary of the evaluated banks, settings, and aggregate results.

## Unconditional-VAE and image-domain results

The unconditional-VAE and image-domain mean-difference results are stored under:

```text
results/vae_md/
```

A separate output directory is retained for each method and anatomical subset:

```text
pconv_head_full_k1000_seed2026_mu/
cnnmar_head_full_k1000_seed2026_mu/
dudonet_head_full_k1000_seed2026_mu/

pconv_body_full_k1000_seed2026_mu/
cnnmar_body_full_k1000_seed2026_mu/
dudonet_body_full_k1000_seed2026_mu/
```

The `_mu` suffix indicates that the unconditional-VAE posterior mean was used as the deterministic embedding.

Each directory contains:

```text
<bank_name>_per_patch_metrics.csv
final_unconditional_vae_results.csv
sample_count_convergence.csv
```

### `final_unconditional_vae_results.csv`

Contains the final unconditional-VAE cosine-similarity score and image-domain mean-difference results for the corresponding bank.

### Per-patch metrics

The per-patch CSV contains the values used to calculate the distribution summaries and create the unconditional-VAE and image-domain violin plots.

### `sample_count_convergence.csv`

Contains the results obtained with increasing numbers of samples per patch.

## Posterior-mean reconstruction figures

Qualitative reconstruction checks are stored under:

```text
results/posterior_mean_reconstructions/
```

The subdirectories are:

```text
pconv_cvae_reconstructions/
cnnmar_cvae_reconstructions/
dudonet_cvae_reconstructions/
uncond_vae_reconstructions/
```

The conditional-VAE figures compare validation patches with their deterministic posterior-mean reconstructions under the corresponding MAR condition.

The unconditional-VAE figures are qualitative checks of the independent embedding model used for the final VAE-based evaluation.

These figures are sanity checks of the learned latent representations and are not the final sampled-invariance figure.

## Final qualitative invariance figure

The final selected qualitative figure is:

```text
results/invariance_samples_selected_examples.png
```

It contains four selected examples:

| Column | Subset | Patch index | Sample seed |
|---:|---|---:|---:|
| 1 | Body | 717 | 1332 |
| 2 | Body | 589 | 1332 |
| 3 | Body | 666 | 2026 |
| 4 | Head | 65 | 26 |

Each column contains:

1. Original corrupted input.
2. Artifact-free reference with the metal pixels reinserted.
3. PConv-MAR sample.
4. CNN-MAR sample.
5. DuDoNet sample.