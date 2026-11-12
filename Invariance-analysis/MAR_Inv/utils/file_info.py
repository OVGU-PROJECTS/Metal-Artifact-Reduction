import os
import pathlib
import SimpleITK as sitk
import gecatsim as xc
import numpy as np

def get_dir_names(main_dir_name):
    """Returns the list of directory names"""
    names = [f for f in pathlib.Path(main_dir_name).iterdir() if f.is_dir()]
    dir_list = [f.absolute().as_posix() for f in names]
    dir_list.sort()
    return dir_list


def get_filenames(dir_name):
    """Returns the list of filenames"""
    filenames = [f for f in pathlib.Path(dir_name).iterdir() if f.is_file()]
    file_list = [f.absolute().as_posix() for f in filenames]
    file_list.sort()
    return file_list


def read_raw(raw_filename, num_slices, num_cols, num_rows):
    """Reads CATSIM raw binary file"""
    return xc.rawread(raw_filename,
                      [num_slices, num_cols, num_rows],
                      'float')

def read_proj_raw(raw_filename):
    """Reads projections from CATSIM binary raw file"""
    proj = read_raw(raw_filename, 1000, 900, 1)
    return proj.squeeze()



def read_img_raw(raw_filename):
    """Reads images from CATSIM binary raw file"""
    img = read_raw(raw_filename, 1, 512, 512)
    img = img.squeeze()
    return img


def write_raw(raw_filename, arr):
    arr = np.asarray(arr, dtype=np.float32)     
    arr.tofile(raw_filename)   

def write_mha(data, mha_filename):
    """Write data to a mha file"""
    sitk.WriteImage(sitk.GetImageFromArray(data.squeeze()), mha_filename)


def read_mha(mha_filename):
    """Read mha file"""
    assert os.path.isfile(mha_filename) is True
    data = sitk.GetArrayFromImage(sitk.ReadImage(mha_filename))
    return np.squeeze(data)


def read_files(dir_name):
    """ get dictionary after reading imgs"""
    filenames_head = get_filenames(dir_name)
    
    img_labels = ["input_img",0, "input_mask",0,
              "input_mask_dilated",0, "input_metal_trace_proj",0, "input_metal_trace_proj_rebin",0,
              "input_proj",0, "input_proj_rebin",0, "output_img",0, "output_img_rebin",0,
              "output_proj",0, "output_proj_rebin",0]
    
    imgs= [sitk.GetArrayFromImage(sitk.ReadImage(file)) for file in filenames_head 
           if os.path.isfile(file) == True ]
    
    img_labels[1:len(img_labels):2]=imgs
    
    return {img_labels[i]:img_labels[i+1] for i in range(0, len(img_labels),2)}


def get_array2(dir_name, file_name):
    """read file through dir_name & file_name"""
    FILENAME = os.path.join(dir_name, file_name)
    assert os.path.isfile(FILENAME) == True
    img = sitk.GetArrayFromImage(sitk.ReadImage(FILENAME))
    assert img.size != 0
    return img

def get_array(file_dir):
    """read img from file dir"""
    assert os.path.isfile(file_dir) == True
    img = sitk.GetArrayFromImage(sitk.ReadImage(file_dir))
    assert img.size != 0
    return img