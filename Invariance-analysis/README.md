
# Invariance Analysis for Metal Artifact Reduction

This repository contains the code, configurations, trained checkpoints, generated sample banks, numerical results, and visualization utilities used for the invariance-analysis of MAR models.

The objective is to investigate what variations MAR models treat as irrelevant. A desirable MAR model should be insensitive to metal-induced artifacts and noise while remaining sensitive to anatomical structures that should be preserved.

The analysis compares three MAR methods:

- PConv-MAR
- CNN-MAR
- DuDoNet

The overall invariance-reconstruction approach follows:

- Rombach et al., who introduced the use of invertible neural networks for reconstructing and analyzing learned invariances.
- Eulig et al., who adapted the approach to low-dose CT denoising using conditional generative models.

The VAE and cINN components build on the implementation associated with:

```text
https://github.com/eeulig/ldct-invariances
```

## Method overview

For a corrupted CT image \(x\), the conditioning information is:

```text
c = [MAR prediction, metal mask]
```

The pipeline consists of the following stages:

1. Train a method-specific conditional VAE.
2. Train a method-specific conditional invertible neural network in the cVAE latent space.
3. Train the unconditional-VAE and DML embedding used for evaluation.
4. Generate alternative corrupted-image realizations under a fixed MAR condition.
5. Evaluate the generated samples in the image domain.
6. Evaluate their similarity in unconditional-VAE and DML embedding spaces.
7. Inspect selected samples qualitatively.

## Repository structure

```text
Invariance-analysis/
├── README.md
├── MAR_Inv/
│   ├── __init__.py
│   ├── vae/
│   ├── cinn/
│   ├── dml/
│   ├── evaluation/
│   └── utils/
├── data/
│   ├── __init__.py
│   ├── patches_generator.py
│   └── README.md
├── configs/
│   ├── uncond.yaml
│   ├── pconv.yaml
│   ├── cnnmar.yaml
│   ├── dudonet.yaml
│   ├── cinn.yaml
│   ├── dml.yaml
│   ├── head_body_stats.json
│   ├── head_body_stats_testset.json
│   └── README.md
└── results/
    ├── runs/
    ├── sample_banks/
    ├── dml/
    ├── vae_md/
    ├── posterior_mean_reconstructions/
    ├── invariance_samples_selected_examples.png
    └── README.md
```
## Main components

### Conditional and unconditional VAEs

The `MAR_Inv/vae/` directory contains the implementation code of the VAE, training script and reconstruction utilities.

### Conditional invertible neural network

The `MAR_Inv/cinn/` directory contains the conditional invertible neural network and its training code.

The cINN is trained after its corresponding conditional VAE. The conditional VAE remains frozen during cINN training.


### Deep metric learning embedding

The `MAR_Inv/dml/` directory contains the implementation code and training script of the DML embedding used for evaluation.

- The shared DML embedding network.
- Sample-bank datasets.
- Cosine-distance semi-hard triplet loss.
- DML training and validation code.

The shared embedding is trained using sample banks from all three MAR methods.

### Evaluation and visualization

The `MAR_Inv/evaluation/` directory contains:

- Invariance sample-bank generation.
- Image-domain mean-difference evaluation.
- Unconditional-VAE embedding evaluation.
- DML embedding evaluation.
- Qualitative figure generation.

### Shared utilities

The `MAR_Inv/utils/` directory contains reusable utilities for:

- Model and optimizer construction.
- Training support.
- Distribution handling.
- File metadata.
- Metrics and loss calculations.

### Data interface

The `data/patches_generator.py` module implements the patch-based data interface used by the VAE, cINN, sample-generation, and evaluation pipelines.

## Data organization

Each dataset directory should contain one subdirectory per CT case. A typical case directory contains:

```text
input_img.mha
output_img.mha
input_mask.mha
pred_pconv_img.mha
pred_cnnmar_img.mha
pred_dudonet_img.mha
```

The files have the following roles:

| File | Purpose |
|---|---|
| `input_img.mha` | Original metal-corrupted CT image |
| `output_img.mha` | Artifact-free reference used for patch selection and qualitative evaluation |
| `input_mask.mha` | Binary metal mask |
| `pred_pconv_img.mha` | PConv-MAR conditioning prediction |
| `pred_cnnmar_img.mha` | CNN-MAR conditioning prediction |
| `pred_dudonet_img.mha` | DuDoNet conditioning prediction |

The prediction filename can be selected through `pred_name` or supplied explicitly.

Before patch extraction, the original metal pixels are reinserted into the MAR prediction using the metal mask. This ensures that the conditioning image retains the implant while using the MAR prediction outside the metal region.

The default patch size is:

```text
128 x 128 pixels
```

Patch selection uses a mixture of:

- Artifact-aware sampling with probability 0.7.
- Random valid-location sampling with probability 0.3.

## Normalization

Separate head and body statistics are used for z-score normalization.

The configuration files are:

```text
configs/head_body_stats.json
configs/head_body_stats_testset.json
```

Use:

- `head_body_stats.json` for training and validation.
- `head_body_stats_testset.json` for the final test-set evaluation.

## Configurations

The `configs/` directory contains:

| Configuration | Purpose |
|---|---|
| `uncond.yaml` | Unconditional VAE |
| `pconv.yaml` | PConv-MAR conditional VAE |
| `cnnmar.yaml` | CNN-MAR conditional VAE |
| `dudonet.yaml` | DuDoNet conditional VAE |
| `cinn.yaml` | Conditional-INN template |
| `dml.yaml` | DML embedding training |

The generic cINN configuration is a template. Before training, set:

- The matching cVAE checkpoint.
- The appropriate `pred_name`.
- The intended output directory.

## Archived results

The `results/` directory contains:

- Frozen cVAE and cINN training runs.
- Unconditional-VAE and DML evaluation-model runs.
- DML training and validation sample banks.
- Final K=1000 test banks.
- DML evaluation results.
- Unconditional VAE results.
- Image domain results (MD).
- Posterior-mean reconstruction figures.
- Final qualitative invariance figure.
