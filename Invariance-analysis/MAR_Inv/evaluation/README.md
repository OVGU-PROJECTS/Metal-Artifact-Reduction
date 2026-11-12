# Evaluation utilities

This directory contains the scripts used to generate invariance samples and evaluate the reconstructed invariance distributions.

The evaluation pipeline consists of three stages:

1. Generate method-specific invariance sample banks using the matched conditional VAE and conditional invertible neural network.
2. Evaluate the generated samples in the image domain and in learned embedding spaces.
3. Create qualitative figures from selected invariance samples.

## Files

- `generate_samples.py`:  generates invariance sample banks using a trained cVAE and cINN.

- `evaluate_vae_md.py`: computes the image-domain mean difference and cosine similarity in the unconditional-VAE embedding.

- `evaluate_dml.py`: computes the image-domain mean difference and cosine similarity in the trained DML embedding.

- `plot_selected_mixed_invariance_examples.py`: creates qualitative figures for selected head and body cases. The figure includes the corrupted input, artifact-free reference, and reconstructed invariance samples from PConv-MAR, CNN-MAR, and DuDoNet.

## Sample-bank generation

`generate_samples.py` loads a matched method-specific cVAE and cINN and generates alternative corrupted-image realizations under a fixed MAR condition.

The condition consists of:

- The prediction produced by the corresponding MAR method.
- The metal mask.

The sampled data flow is:

```text
Gaussian invariance sample
    -> inverse cINN
    -> conditional-VAE latent representation
    -> conditional-VAE decoder
    -> sampled corrupted-image realization
```

Each cVAE must be paired with the cINN and prediction image belonging to the same MAR method.

A generated bank is saved as a PyTorch file containing:

```text
x        Original corrupted patches
p        Corresponding MAR predictions
m        Metal masks
samples  Generated invariance realizations
meta     Patch and source-case metadata
```

The principal tensor shapes are:

```text
x:       [N, 1, H, W]
samples: [N, K, 1, H, W]
```

where:

- `N` is the number of selected patches.
- `K` is the number of generated realizations per patch.
- `H` and `W` are the patch dimensions.

The final evaluation uses \(K=1000\) generated realizations per test patch. Separate banks are generated for the head and body subsets and for each MAR method.

They are stored under: `results/sample_banks/test_banks/`

## Image-domain mean difference

Both evaluation scripts calculate the non-metal image-domain mean difference between an original corrupted patch and its generated samples.

For each patch, the difference is averaged over the non-metal pixels and the generated realizations. The final model-level result is obtained by averaging the patch-wise results.

The evaluator reports the image-domain difference in:

- Normalized intensity units.
- Hounsfield units when valid normalization statistics and patch metadata are available.

The patch-wise distribution is summarized by:

- Mean
- Standard deviation
- Standard error
- Median
- Interquartile range
- Minimum
- Maximum
- Bootstrap confidence interval

## Unconditional-VAE evaluation

`evaluate_vae_md.py` evaluates the samples using an independently trained unconditional VAE.

The posterior mean is used by default:

```text
--embedding_mode mean
```

A stochastic posterior sample can alternatively be selected with:

```text
--embedding_mode sample
```

For each original patch \(x_i\) and generated sample \(\tilde{x}_{i,k}\), cosine similarity is computed between their VAE latent representations. The sample-level similarities are first averaged across the \(K\) samples of each patch.

```text
S_VAE(i) = mean_k cosine(e_VAE(\(x_i\)), e_VAE(\(\tilde{x}_{i,k}\)))
```

The model-level score is the mean of the patch-level scores.

Use:

```bash
python -m MAR_Inv.evaluation.evaluate_vae_md --help
```

The thesis evaluation uses:

```text
results/runs/evaluation/uncond_vae/checkpoints/best.pt
```

with:

```bash
--embedding_mode mean
```
## DML evaluation

`evaluate_dml.py` evaluates each sample bank using the shared DML encoder trained on the PConv-MAR, CNN-MAR, and DuDoNet training banks.

For test patch \(i\), the DML score is:

```text
S_DML(i) = mean_k cosine(h(\(x_i\)), h(\(\tilde{x}_{i,k}\)))
```

where `h` is the frozen DML embedding network.

For each patch, cosine similarities between the original embedding and all corresponding invariance-sample embeddings are computed. Their average gives the patch-level DML similarity score, which is subsequently averaged across patches for the final result.

Use:

```bash
python -m MAR_Inv.evaluation.evaluate_dml --help
```

The thesis evaluation uses:

```text
results/runs/evaluation/dml/checkpoints/best.pt
```

## Statistical reporting

Both evaluation approaches report:

```text
cosine-similarity score
non-metal mean absolute difference
per-patch statistics
bootstrap confidence interval
sample-count convergence
```

For every patch, the distribution across the generated samples is summarized using measures including mean, standard deviation, Monte Carlo standard error, median, quantiles, minimum, maximum, and interquartile range.

A bootstrap confidence interval is calculated over the patch-level similarity scores.

The scripts also evaluate convergence with respect to the number of generated samples \(K\), allowing the final similarity score to be compared for different numbers of invariance samples.

## Qualitative visualization
`plot_selected_mixed_invariance_examples.py` creates the final mixed head/body qualitative figure.

Each column represents one selected test patch. The figure contains five rows:

1. Original corrupted input patch.
2. Artifact-free reference with metal pixels restored from the corrupted input.
3. PConv-MAR invariance sample.
4. CNN-MAR invariance sample.
5. DuDoNet invariance sample.

The artifact-free reference is constructed by loading `output_img.mha` and reinserting the original metal pixels using the input mask.
