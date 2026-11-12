"""Linear-Interpolation MAR (LI) """
import os
from copy import deepcopy
from enum import Enum
import matplotlib.pylab as pylab
import numpy as np

from mar_common.scanner_config import recon_config
from mar_common.io import write_mha, read_mha
from mar_common.cases import get_dir_names
from geometry.fanbeam import RECO_FOV
from classical_mar.viz import params

from ct_ops.fbp import reconstruct_fdk

# Tracks whether the scan along a detector row is currently inside or outside
class ROIMAR(Enum):
    INSIDE_METAL = 0,
    OUTSIDE_METAL = 1


def mar_li(proj, metal_proj):
    """Metal Artifact Reduction using Linear Interpolation (LI) """
    num_view = proj.shape[0]
    num_det_bins = proj.shape[1]
    proj_mar = deepcopy(proj)

    for view_idx in range(0, num_view):  # Iterate through each row.
        roi_state = ROIMAR.OUTSIDE_METAL

        for bin_idx in range(0, num_det_bins):  # Iterate through each detector bin/column.
            mask_val = metal_proj[view_idx, bin_idx]
            metal_thr = 0.01

            if roi_state == ROIMAR.OUTSIDE_METAL:  # The previous state is outside metal.
                if mask_val > metal_thr:  # Metal region begins.
                    metal_start_bin_idx = bin_idx
                    # Get the last valid value for interpolation.
                    valid_val_start_bin_idx = bin_idx - 1
                    valid_val_start = proj[view_idx, valid_val_start_bin_idx]
                    # Update the state.
                    roi_state = ROIMAR.INSIDE_METAL

            else:  # The Previous state is inside the metal region.

                if mask_val <= metal_thr:  # Non-metal region begins.
                    metal_end_bin_idx = bin_idx - 1
                    valid_val_end_bin_idx = bin_idx  # for interpolation.
                    valid_val_end = proj[view_idx, valid_val_end_bin_idx]
                    val_incr = (valid_val_end - valid_val_start) / (valid_val_end_bin_idx - valid_val_start_bin_idx)
                    interp_val = valid_val_start

                    for def_pix_bin in range(metal_start_bin_idx, metal_end_bin_idx + 1):
                        interp_val = interp_val + val_incr
                        proj_mar[view_idx, def_pix_bin] = interp_val

                    roi_state = ROIMAR.OUTSIDE_METAL # Update the state.
    return proj_mar


def plot_img(img_input, img_mar, img_gt, disp_range_list, op_fig_name=''):
    """Side-by-side display of input / LI-corrected / ground-truth images."""
    pylab.rcParams.update(params)

    pylab.subplot(1, 3, 1)
    pylab.imshow(img_input, cmap='gray', vmin=min(disp_range_list), vmax=max(disp_range_list))
    pylab.axis('off')
    pylab.title('Input')

    pylab.subplot(1, 3, 2)
    pylab.imshow(img_mar, cmap='gray', vmin=min(disp_range_list), vmax=max(disp_range_list))
    pylab.axis('off')
    pylab.title('MAR: LI')

    pylab.subplot(1, 3, 3)
    pylab.imshow(img_gt, cmap='gray', vmin=min(disp_range_list), vmax=max(disp_range_list))
    pylab.axis('off')
    pylab.title('GT')

    if op_fig_name == '':
        pylab.show()
    else:
        pylab.savefig(op_fig_name)


if __name__ == '__main__':
    MAIN_DIR_NAME = os.path.join('/mnt/public/bahaa/dataset_training/00001')
    FOV_MM = RECO_FOV['head']
    RECON_CONFIG = recon_config(FOV_MM)
    PROJ_DIR_NAMES = get_dir_names(MAIN_DIR_NAME)
    for PROJ_DIR_NAME in PROJ_DIR_NAMES:
        print(PROJ_DIR_NAME)
        assert os.path.isdir(PROJ_DIR_NAME) is True
        PROJ_FILENAME = os.path.join(PROJ_DIR_NAME, 'input_proj.mha')
        PROJ = read_mha(PROJ_FILENAME)

        METAL_TRACE_FILENAME = os.path.join(PROJ_DIR_NAME, 'input_metal_trace_proj.mha')
        METAL_TRACE = read_mha(METAL_TRACE_FILENAME)

        assert PROJ.shape == METAL_TRACE.shape
        PROJ_MAR = mar_li(PROJ, METAL_TRACE)

        MASK_INV = np.invert(np.array(METAL_TRACE, dtype=bool))
        PROJ_NON_METAL = PROJ * MASK_INV
        PROJ_MAR_NON_METAL = PROJ_MAR * MASK_INV
        assert np.allclose(PROJ_NON_METAL, PROJ_MAR_NON_METAL) is True

        VOL = reconstruct_fdk(RECON_CONFIG, PROJ)
        VOL_MAR = reconstruct_fdk(RECON_CONFIG, PROJ_MAR)

        VOL_GT_FILENAME = os.path.join(PROJ_DIR_NAME, 'output_img.mha')
        VOL_GT = read_mha(VOL_GT_FILENAME)

        PROJ_GT_FILENAME = os.path.join(PROJ_DIR_NAME, 'output_proj.mha')
        PROJ_GT = read_mha(PROJ_GT_FILENAME)

        OP_DIR_NAME = '/mnt/public/bahaa/media/plots'
        IMG_FIG_FILENAME = os.path.join(OP_DIR_NAME, 'img_' + os.path.basename(PROJ_DIR_NAME) + '.png')
        plot_img(VOL, VOL_MAR, VOL_GT, [-150, 250], op_fig_name=IMG_FIG_FILENAME)

        PROJ_FIG_FILENAME = os.path.join(OP_DIR_NAME, 'proj_' + os.path.basename(PROJ_DIR_NAME) + '.png')
        plot_img(PROJ, PROJ_MAR, PROJ_GT, [0.0, 6.0], op_fig_name=PROJ_FIG_FILENAME)

        MAR_PROJ_FILENAME = os.path.join(PROJ_DIR_NAME, 'proj_mar.mha')
        write_mha(PROJ_MAR, MAR_PROJ_FILENAME)

        MAR_IMG_FILENAME = os.path.join(PROJ_DIR_NAME, 'img_mar.mha')
        write_mha(VOL_MAR, MAR_IMG_FILENAME)
