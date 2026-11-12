"""Generate the metal-trace sinogram and dilated metal mask for each case.

For every case folder:
1- reads the metal mask ("input_mask.mha")
2- dilates it
4- forward-projects it to get the binary "metal trace" 
5- both are saved back into the case folder for training.
"""
import os, numpy as np
import SimpleITK as sitk
from scipy import ndimage

from ct_ops.forward_proj import compute_sinogram
from mar_common.scanner_config import recon_config
from mar_common.cases import get_dir_names, get_filenames
from mar_common.io import read_img_raw


if __name__ == '__main__':
    main_train_dir_name = '/mnt/public/bahaa/test_images_1000/body'
    dir_names = get_dir_names(main_train_dir_name)
    recon_config = recon_config(fov_mm = 400.0)
    
    for train_sample_dir_name in dir_names:
        out_trace = os.path.join(train_sample_dir_name, 'input_metal_trace_proj.mha')
        out_dilat = os.path.join(train_sample_dir_name, 'input_mask_dilated.mha')

        #print(train_sample_dir_name)
        assert os.path.exists(train_sample_dir_name) is True

        mask_img_filename = os.path.join(train_sample_dir_name, 'input_mask.mha')
        assert os.path.exists(mask_img_filename) is True

        # Skip already-processed folders
        if os.path.exists(out_trace) and os.path.exists(out_dilat):
            print(f"✓ already processed → {train_sample_dir_name}")
            continue

        print(f"→ processing {train_sample_dir_name}")
        
        # Dilate the mask so the projected trace safely covers the metal borders.
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_img_filename))
        struct_elem = ndimage.generate_binary_structure(2, 5)
        mask_dilated = ndimage.binary_dilation(mask, struct_elem).astype(mask.dtype)

        # Compute metal trace projection by forward projection.
        metal_trace_proj = compute_sinogram(mask_dilated, recon_config)
        threshold = 0.001
        metal_trace_proj[metal_trace_proj >= threshold] = 1.0
        metal_trace_proj[metal_trace_proj < threshold] = 0.0

            
        sitk.WriteImage(sitk.GetImageFromArray(metal_trace_proj), os.path.join(train_sample_dir_name,
                                                                               'input_metal_trace_proj.mha'))
        sitk.WriteImage(sitk.GetImageFromArray(mask_dilated), os.path.join(train_sample_dir_name,
                                                                           'input_mask_dilated.mha'))
