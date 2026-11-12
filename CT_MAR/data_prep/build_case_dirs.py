"""Reorganize raw scan files into one folder per case.

The raw data is split across:
1- "Baseline/" (metal-affected projection + image)
2- "Mask/" (metal mask)
3- "Target/" (clean projection + image)
This script matches the files by case index and writes them, converted to ".mha", into a
numbered per-case.
"""

import os, numpy as np
from mar_common.cases import  get_filenames
from mar_common.io import read_img_raw, read_proj_raw, write_mha


def get_idx_from_proj_filename(proj_filename):
    """Returns index of projection file"""
    name = os.path.split(proj_filename)[-1]
    return int(name[name.find('_sino') + len('_sino'):name.find('_900x1000')])

def get_idx_from_img_filename(img_filename):
    """Returns index of image file (also for mask)"""
    name = os.path.split(img_filename)[-1]
    return int(name[name.find('_img') + len('_img'):name.find('_512x512x1')])


if __name__ == '__main__':
    main_dir_name = '/mnt/public/bahaa/test_images_1000/body1'
    op_dir_name = '/mnt/public/bahaa/test_images_1000/body'
    os.makedirs(op_dir_name, exist_ok=True)
    

    input_proj_img_dir_name = os.path.join(main_dir_name, 'Baseline')
    files_input_all = get_filenames(input_proj_img_dir_name)
    print(f"files_input_all: {files_input_all[0]}")
    files_input_proj = [file for file in files_input_all if 'sino' in file]
    files_input_img = [file for file in files_input_all if 'img' in file]
    assert len(files_input_proj) == len(files_input_img)
    assert files_input_proj != files_input_img

    input_mask_dir_name = os.path.join(main_dir_name, 'Mask')
    files_masks = get_filenames(input_mask_dir_name)
    files_input_mask_img = [file for file in files_masks if 'img' in file]
    assert len(files_input_proj) == len(files_input_mask_img)

    output_dir_name = os.path.join(main_dir_name, 'Target')
    files_output_all = get_filenames(output_dir_name)
    files_output_proj = [file for file in files_output_all if 'sino' in file]
    files_output_img = [file for file in files_output_all if 'img' in file]
    
    start_idx= 1
    for n, (proj_ip, img_ip, mask_ip, proj_op, img_op) in enumerate ( zip(files_input_proj,
                                                         files_input_img,
                                                         files_input_mask_img,
                                                         files_output_proj,
                                                         files_output_img),start=start_idx):
        dir_idx = get_idx_from_proj_filename(proj_ip)
        assert dir_idx == get_idx_from_proj_filename(proj_op)
        assert dir_idx == get_idx_from_img_filename(img_ip)
        assert dir_idx == get_idx_from_img_filename(img_op)
        assert dir_idx == get_idx_from_img_filename(mask_ip)

        print(f"proj_ip shape: {proj_ip}")


        op_sample_dir_name = os.path.join(op_dir_name, str(n).zfill(4))
        print(op_sample_dir_name)
        os.makedirs(op_sample_dir_name, exist_ok=True)

        # Save input datasets
        write_mha(read_proj_raw(proj_ip), os.path.join(op_sample_dir_name, 'input_proj.mha'))
        write_mha(read_img_raw(img_ip)  , os.path.join(op_sample_dir_name, 'input_img.mha'))
        write_mha(read_img_raw(mask_ip) , os.path.join(op_sample_dir_name, 'input_mask.mha'))
        write_mha(read_proj_raw(proj_op), os.path.join(op_sample_dir_name, 'output_proj.mha'))
        write_mha(read_img_raw(img_op)  , os.path.join(op_sample_dir_name, 'output_img.mha'))
     