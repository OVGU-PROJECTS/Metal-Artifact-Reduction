
# DML invariance evaluation

This directory contains the deep metric learning (DML) implementation used to evaluate invariance samples.

The implementation follows the evaluation idea described by Eulig et al., particularly the use of a ResNet-based embedding and triplet-based learning to compare original inputs with reconstructed invariance samples. The model is trained so that an original corrupted CT patch is close to invariance samples generated from the same case and farther from samples belonging to different cases.

## Files

- `dataset.py`: dataset classes for loading precomputed invariance sample banks.
- `network.py`: ResNet-based metric embedding network.
- `losses.py`: cosine distance and the semi-hard triplet loss. 
- `training.py`: the training and validation scripts.

## Input sample bank

The DML code expects a precomputed sample bank containing at least:

```text
x        [N,1,H,W]
samples  [N,K,1,H,W]
```

where `x` contains the original corrupted patches and `samples` contains `K` invariance reconstructions for each case.

Sample banks are generated with:

```bash
python -m MAR_Inv.evaluation.generate_samples --help
```

The training experiment requires three \(K=50\) training banks and three \(K=2\) validation banks, one for each MAR method.

Each bank must be generated using the cVAE and cINN checkpoints corresponding to the same MAR method.

## Training pairs

For each training item:

```text
anchor   = original corrupted patch
positive = invariance sample from the same case
negative = positive sample from another case in the batch
```

## Embedding network

The configuration uses a ResNet backbone, with an ImageNet-pretrained ResNet-50 as the default. The first convolution is adapted for single-channel CT images, and the original classification layer is replaced by a linear embedding head. The final feature vector is normalized to unit L2 norm. 

## Loss

Training uses a semi-hard triplet loss based on cosine distance.
The loss first searches for a semi-hard negative that is farther from the anchor than its positive but still violates the configured margin. If no semi-hard negative exists, the closest valid negative is used.

Cosine distance is used in the triplet loss.

## Training

`training.py` builds the training and validation datasets, trains the embedding network with Adam, and applies a StepLR learning-rate scheduler. 

The used configuration are saved in `configs/dml.yaml`

Training and validation statistics are written to `losses.csv`, including:

```text
loss
positive distance
negative distance
active triplet fraction
semi-hard negative fraction
```

The trained model is saved in: `results/runs/evaluation/dml`
