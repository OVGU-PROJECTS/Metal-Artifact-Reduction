# callbacks.py
"""Custom training callbacks for CNN-MAR.

Extracted from trainer.py to keep the trainer lean. The core logic is unchanged.
"""
import numpy as np
import tensorflow as tf

from mar_common.io import read_mha
from metrics.numpy_ssim import ssim_aapm, ssim_1500, ssim_2000
from data_pipline.preprocessing import prepare_inference_input, prediction_mu_to_hu


class FullImageSSIMCallback(tf.keras.callbacks.Callback):
    """Report whole-image SSIM each epoch (training itself only sees 64x64 patches).

    Builds the 3-channel input (uncorrected/BHC/LI) for each case, predicts the
    full 512x512 image, and averages the three SSIMs,
    so cnn_mar's numbers are comparable to DuDoNet and Pconv_mar.
    """
    def __init__(self, train_cases, val_cases):
        super().__init__()
        self.case_groups = {"train": train_cases, "val": val_cases}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        for split, case_dirs in self.case_groups.items():
            sums = {"ssim_aapm_hu": 0.0, "ssim_hu_1000_1500": 0.0, "ssim_hu_1000_2000": 0.0}
            for case_dir in case_dirs:
                original_mask = read_mha(str(case_dir / "input_mask.mha")) > 0
                inputs = prepare_inference_input(
                    read_mha(str(case_dir / "input_img.mha")),
                    read_mha(str(case_dir / "img_bhc.mha")),
                    read_mha(str(case_dir / "img_li.mha")),
                    original_mask,
                )[None]
                reference = read_mha(str(case_dir / "output_img.mha"))
                dilated_mask = read_mha(str(case_dir / "input_mask_dilated.mha")) > 0
                prediction = prediction_mu_to_hu(self.model(inputs, training=False).numpy()[0, ..., 0])

                sums["ssim_aapm_hu"] += ssim_aapm_hu(reference, prediction, dilated_mask)
                sums["ssim_hu_1000_1500"] += ssim_hu_1000_1500(reference, prediction, original_mask)
                sums["ssim_hu_1000_2000"] += ssim_hu_1000_2000(reference, prediction, original_mask)

            count = float(len(case_dirs))
            if count == 0:
                raise ValueError(f"No {split} cases available for SSIM evaluation")
            prefix = "train_" if split == "train" else "val_"
            logs[prefix + "ssim_aapm_hu"] = sums["ssim_aapm_hu"] / count
            logs[prefix + "ssim_hu_1000_1500"] = sums["ssim_hu_1000_1500"] / count
            logs[prefix + "ssim_hu_1000_2000"] = sums["ssim_hu_1000_2000"] / count

        print(
            "\nFull-image SSIM | "
            f"train=({logs['train_ssim_aapm_hu']:.6f}, "
            f"{logs['train_ssim_hu_1000_1500']:.6f}, "
            f"{logs['train_ssim_hu_1000_2000']:.6f}) | "
            f"val=({logs['val_ssim_aapm_hu']:.6f}, "
            f"{logs['val_ssim_hu_1000_1500']:.6f}, "
            f"{logs['val_ssim_hu_1000_2000']:.6f})"
        )
