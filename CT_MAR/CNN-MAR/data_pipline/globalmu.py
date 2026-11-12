"""One-off utility: estimate the global attenuation (mu) value range of the dataset.

Samples reference pixels across many cases and takes the 1st/99th percentiles, giving
robust min/max bounds (ignoring outliers) for normalization. Running this module
prints the dataset-wide MU_MIN/MU_MAX; it is a calibration script, not part of training.
"""
import numpy as np
from .data_gen import gen_cnnmar

def estimate_mu_range_percentiles(split,
                                  max_examples,   # how many images to sample
                                  batch_size=100,
                                  seed=123,
                                  max_pixels=5_000_000):  # cap total pixels for memory
    gen = gen_cnnmar(
        data_t=split,
        max_examples=max_examples,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    samples = []
    total_pixels = 0

    for Xb, yb in gen:
        y_flat = yb.ravel()
        samples.append(y_flat)
        total_pixels += y_flat.size
        if total_pixels >= max_pixels:
            break

    all_vals = np.concatenate(samples)
    mu_min = float(np.quantile(all_vals, 0.01))   # 0.1 %
    mu_max = float(np.quantile(all_vals, 0.99))   # 99.9 %
    return mu_min, mu_max


mu_min_train, mu_max_train = estimate_mu_range_percentiles("train", 13000)
mu_min_val,   mu_max_val   = estimate_mu_range_percentiles("val", 1000)

MU_MIN = min(mu_min_train, mu_min_val)
MU_MAX = max(mu_max_train, mu_max_val)
mu_range = MU_MAX - MU_MIN
print(f"min: {MU_MIN}, max = {MU_MAX}, mu_range= {mu_range}" )
