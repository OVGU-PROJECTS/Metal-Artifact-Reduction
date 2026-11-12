# VAE and conditional-VAE experiments

This directory contains the VAE and conditional-VAE models used in the MAR invariance pipeline, together with the critic network, training code, and supporting network blocks.

The same basic encoder-decoder architecture is used for the unconditional VAE and for the conditional VAEs associated with DuDoNet, CNN-MAR, and PConv-MAR.

The encoder is ResNet-based and produces the parameters of a diagonal Gaussian latent distribution. The decoder follows a BigGAN-style generator architecture. During training, a latent sample is drawn from the posterior and decoded back to image space.

For the unconditional VAE, the encoder receives only the corrupted CT patch `x`.

For a conditional VAE, the condition is formed as `[p, m]`

The conditional VAE must always use the MAR prediction corresponding to its configured condition.

## Model components

The main components are:

* `network.py`: ResNet-based encoder, latent Gaussian representation, and BigGAN-style decoder.
* `blocks.py`: reusable generator, normalization, attention, and conditioning blocks.
* `critic.py`: convolutional critic used for adversarial training.
* `training.py`: VAE/cVAE training and validation loop.
* `plot_reconstructions_uncond_vae.py`: reconstruction visualization for the unconditional VAE.
* `plot_reconstruction_cvae.py`: reconstruction visualization for conditional VAEs.

The VAE network, critic, and supporting network blocks are adapted from the implementation provided in:

https://github.com/eeulig/ldct-invariances

## Configurations

The experiment-specific configurations are stored separately:

* `configs/uncond.yaml`: unconditional VAE.
* `configs/dudonet.yaml`: DuDoNet-conditioned VAE.
* `configs/cnnmar.yaml`: CNN-MAR-conditioned VAE.
* `configs/pconv.yaml`: PConv-MAR-conditioned VAE.


## Training objective

Training combines four components:

`L_G = alpha * L_rec + beta * L_perc + kl_weight * L_KL + L_adv`

## Training

Example for the DuDoNet-conditioned VAE:

```bash
GPU1=$(gpurequest)
GPU2=$(gpurequest)

CUDA_VISIBLE_DEVICES="$GPU1,$GPU2" python -u -m MAR_Inv.vae.training \
  --config configs/dudonet.yaml \
  --devices 0,1 \
  --out_dir results/runs/cvae/cvae_dudonet_new
```

For CNN-MAR, PConv-MAR, or the unconditional VAE, change the configuration file and output directory accordingly.

Training and validation losses are written to `losses.csv`.

The saved checkpoints contain the generator, critic, optimizer states, configuration, and, when enabled, learning-rate scheduler states.

## Plotting and reconstruction utilities

Two scripts are provided for qualitative reconstruction checks:

plot_reconstructions_uncond_vae.py: visualizes corrupted patches and their unconditional VAE reconstructions.

plot_reconstruction_cvae.py: visualizes corrupted patches, the MAR conditioning image, and the corresponding cVAE reconstruction.

The conditional script also supports random selection and also selectingv--indices so the same validation patches can be compared across DuDoNet, CNN-MAR, and PConv-MAR conditions.