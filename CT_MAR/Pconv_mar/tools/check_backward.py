"""small backward check,

    python tools/check_backward.py

- Uses the existing CUDA reconstruction, random numbers, and no trained model.
- For fixed geometry, reconstruction R is linear.
- Its backward function B should therefore satisfy <R(x), y> = <x, B(y)>.
"""
import math
import sys
import torch

from mar_common.scanner_config import config_downsampled
from recon import RECO_FOV, createFdkCache, load_createhsp, reconstruct

def main():

    # a practical float32 checks threshold, not a mathematical error bound
    tolerance = 1e-4
    worst_error = 0.0
    print("Checking the reconstruction backward function (no training).", flush=True)
    print("Relative error = |left - right| / max(|left|, |right|)")
    print("PASS requires error < 1e-4 for all six checks.", flush=True)

    for fov in ("head", "body"):
        # 1. Use exactly the geometry and filter setup used by PConv-MAR.
        cfg = config_downsampled(RECO_FOV[fov])
        cfg._fdk_cache = createFdkCache(cfg, target_views=500, view_block=50,
                                        raw_view_count=1000)
        ramp_filter = load_createhsp(fov).cuda()

        for seed in (0, 1, 2):
            # 2. Make a random sinogram and reconstruct it.
            torch.manual_seed(seed)
            x = torch.randn(1000, cfg._fdk_cache.YL, device="cuda",
                            dtype=torch.float32, requires_grad=True)
            image = reconstruct(cfg, x, ramp_filter)
            y = torch.randn_like(image)

            # 3. Ask PyTorch to run the actual registered backward function.
            backward_y = torch.autograd.grad(image, x, grad_outputs=y)[0]

            # 4. Compare the two inner products. only these sums use float64
            # the reconstruction and backward kernels stay in float32.
            left = (image.detach().double() * y.double()).sum().item()
            right = (x.detach().double() * backward_y.double()).sum().item()
            denominator = max(abs(left), abs(right))
            error = abs(left - right) / denominator if denominator > 0 else math.inf
            if not math.isfinite(error):
                error = math.inf
            worst_error = max(worst_error, error)
            status = "PASS" if error < tolerance else "FAIL"
            print(f"{fov:4s}, seed {seed}: {status}  relative error = {error:.6e}",
                  flush=True)

    print(f"\nMaximum relative error: {worst_error:.6e}")
    if worst_error >= tolerance:
        sys.exit("FAIL: At least one check exceeded the tolerance or was invalid.")
    print("PASS: Forward and backward agree within tolerance for these checks.")


if __name__ == "__main__":
    main()
    
    
    
    
