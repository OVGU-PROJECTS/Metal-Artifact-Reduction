
# Data interface

`patches_generator.py` implements the patch-based dataset used for conditional VAE, cINN, unconditional VAE, and evaluation experiments. Each case folder contains the corrupted CT image (`input_img.mha`) and, for conditional experiments, a metal mask (`input_mask.mha`) and the MAR prediction used as conditioning. The artifact-free image (`output_img.mha`) is used only for artifact-aware patch sampling when available. It is not returned as a training target. If it is unavailable, the sampling strength is estimated from image gradients instead.

In conditional mode, the prediction filename is constructed as `pred_<pred_name>_img.mha`, where `pred_name` is specified by the configuration, for example `pred_dudonet_img.mha` or `pred_cnnmar_img.mha`. A prediction filename can also be supplied explicitly. For CNN-MAR, the selected prediction must correspond to the representation used when training the respective conditional VAE; when the cVAE was trained using the direct CNN output, this refers to the direct network prediction rather than the subsequent physics-based post-processing result.

Before patch extraction, the binary metal mask is applied to the MAR prediction and the original corrupted image values are reinserted inside the metal region. The resulting conditioning image therefore retains the implant while using the MAR prediction outside the metal mask.

Each case contributes a fixed number of patches per epoch. Patch locations are resampled between epochs. By default, 70% of patches are selected from artifact-relevant regions using the strongest values of the sampling map, while the remaining 30% are sampled randomly. The corrupted image and MAR prediction patches are clipped at 3000 HU and z-score normalized using separate head and body statistics. The metal mask remains binary.

The dataset returns the normalized corrupted patch `x` in all modes. In conditional mode, it additionally returns the binary metal-mask patch `m` and normalized MAR-conditioning patch `p`. Optional metadata contains the case identifier, patch location, epoch, field-of-view class, and sampling mode.
