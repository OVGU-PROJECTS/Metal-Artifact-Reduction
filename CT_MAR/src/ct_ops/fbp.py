# Copyright 2020, General Electric Company. All rights reserved. See https://github.com/xcist/code/blob/master/LICENSE
"""Analytic FDK reconstruction: turn a sinogram back into an image"""
import ctypes as ct
import math
from math import ceil
from copy import deepcopy
import numpy as np

from gecatsim.reconstruction.pyfiles.createHSP import createHSP
from gecatsim.pyfiles.CommonTools import *
from gecatsim.reconstruction.pyfiles.fdk_equiAngle import load_C_recon_lib
import gecatsim as xc

from geometry.fanbeam import compute_fan_ang_incr_rad

# Init ctypes types
FLOAT = ct.c_float
PtrFLOAT = ct.POINTER(FLOAT)
PtrPtrFLOAT = ct.POINTER(PtrFLOAT)
PtrPtrPtrFLOAT = ct.POINTER(PtrPtrFLOAT)

class TestStruct(ct.Structure):
    _fields_ = [
        ("ScanR", ct.c_float),  # Radius of the scanning trajectory of x-ray source
        ("DecFanAng", ct.c_float),  # Fan angle coverage of the detector element along the horizontal diretion
        ("startangle", ct.c_float),  # recon startangle
        ("rotdir", ct.c_int),  # //rotation direction
        ("DecHeight", ct.c_float),  # Physical height of the detector along the vertical direction
        ("YL", ct.c_int),  # Detector cell number on each row along the horizontal direction
        ("ZL", ct.c_int),  # Detector cell number on each column along the vertical direction
        ("dectorYoffset", ct.c_float),  # Detector along the horizontal direction (pixel, e.g. quarter pixel)
        ("dectorZoffset", ct.c_float),  # Detector offset along the vertical direcion (pixel, e.g. quarter pixel)
        ("XOffSet", ct.c_float),  # recon offset along the x axis
        ("YOffSet", ct.c_float),  # recon offset along the y axis
        ("ZOffSet", ct.c_float),  # recon offset along the z axis
        ("phantomXOffSet", ct.c_float),  # phantom offset along the x axis
        ("phantomYOffSet", ct.c_float),  # phantom offset along the y axis
        ("phantomZOffSet", ct.c_float),  # phantom offset along the z axis
        ("AngleNumber", ct.c_int),  # Number of view samples on the scanning trajectory
        ("DistD", ct.c_float),  # Distance between the x-ray source and the detector
        ("Radius", ct.c_float),  # Radius of the phantom
        ("RecSize", ct.c_int),  # Reconstructed size
        ("sliceThickness", ct.c_float),  # Reconstructed thickness
        #                 ("centerX", ct.c_float),          # Reconstructed center on x axis
        #                 ("centerY", ct.c_float),          # Reconstructed center on y axis
        #                 ("centerZ", ct.c_float),          # Reconstructed center on z axis
        ("FOILength", ct.c_int),  # Reconstructed length on x axis
        ("FOIWidth", ct.c_int),  # Reconstructed length on y axis
        ("FOIHeight", ct.c_int),  # Reconstructed length on z axis
        ("GF", PtrPtrPtrFLOAT),  # Projection data/ Sinogram data
        ("RecIm", PtrPtrPtrFLOAT)  # Reconstructed 3D image data
    ]

def float3Darray2pointer(arr):
    # Converts a 3D numpy to ctypes 3D array.
    arr_dimx = FLOAT * arr.shape[2]
    arr_dimy = PtrFLOAT * arr.shape[1]
    arr_dimz = PtrPtrFLOAT * arr.shape[0]

    arr_ptr = arr_dimz()

    for i, row in enumerate(arr):
        arr_ptr[i] = arr_dimy()
        for j, col in enumerate(row):
            arr_ptr[i][j] = arr_dimx()
            for k, val in enumerate(col):
                arr_ptr[i][j][k] = val
    return arr_ptr

def float3Dpointer2array(ptr, n, m, o):
    # Converts ctypes 3D array into a 3D numpy array.
    arr = np.zeros(shape=(n, m, o), dtype=np.single)

    for i in range(n):
        for j in range(m):
            for k in range(o):
                arr[i, j, k] = ptr[i][j][k]

    return arr

def mapConfigVariablesToFDK(cfg):
    sid = cfg.scanner.sid
    sdd = cfg.scanner.sdd
    nMod = ceil(cfg.scanner.detectorColCount / cfg.scanner.detectorColsPerMod)
    rowSize = cfg.scanner.detectorRowSize
    modWidth = cfg.scanner.detectorColsPerMod * cfg.scanner.detectorColSize
    dectorYoffset = -cfg.scanner.detectorColOffset
    dectorZoffset = cfg.scanner.detectorRowOffset

    fov = cfg.recon.fov
    imageSize = cfg.recon.imageSize
    sliceCount = cfg.recon.sliceCount
    sliceThickness = cfg.recon.sliceThickness
    centerOffset = deepcopy(cfg.recon.centerOffset)
    # Pass desired X as Y
    centerOffset[1] = deepcopy(cfg.recon.centerOffset[0])
    # Pass desired Y as X; this minus sign is needed to get correct offset
    centerOffset[0] = -deepcopy(cfg.recon.centerOffset[1])

    # The FDK recon seems to be using a "start view" rather than a "start angele".
    # This is a hack until that gets fixed.
    startView_at_view_angle_equals_0 = cfg.protocol.viewCount / 2
    if cfg.recon.startAngle <= 180:
        startView = startView_at_view_angle_equals_0 + int(cfg.protocol.viewCount * cfg.recon.startAngle / 360)
    elif cfg.recon.startAngle > 180:
        startView = startView_at_view_angle_equals_0 - int(cfg.protocol.viewCount * cfg.recon.startAngle / 360)
    else:
        raise Exception("******** Error! Invalid start angle = {} specified. ********".format(cfg.recon.startAngle))

    rotdir = cfg.protocol.rotationDirection
    kernelType = cfg.recon.kernelType

    return sid, sdd, nMod, rowSize, modWidth, dectorYoffset, dectorZoffset, \
        fov, imageSize, sliceCount, sliceThickness, centerOffset, startView, rotdir, kernelType

def filter_proj(proj, fan_ang_incr_rad, num_det_rows, num_det_cols, num_views, kernel_type):
    nn = int(math.pow(2, (math.ceil(math.log2(abs(num_det_cols))) + 1)))
    filt_fft = createHSP(nn, kernel_type)
    fft_size = nn * 2
    proj_filtered = proj

    for view_idx in range(0, num_views):
        for row_idx in range(num_det_rows):
            proj_row = np.ones(num_det_cols)
            for k in range(num_det_cols):
                proj_row[k] = proj[k, row_idx, view_idx]
            proj_row_fft = np.fft.fft(proj_row, fft_size)
            proj_row = np.fft.ifft(proj_row_fft * filt_fft).imag
            for k in range(num_det_cols):
                proj_filtered[k, row_idx, view_idx] = -proj_row[k]

    proj_filtered = proj_filtered / fan_ang_incr_rad
    
    return proj_filtered

def cos_weighting(proj, num_det_cols, num_det_rows, sdd_mm,
                  det_center_col, det_center_row, fan_ang_incr_rad, det_pix_height_mm):
    for col_idx in range(num_det_cols):
        for row_idx in range(num_det_rows):
            in_plane_weight = math.cos((col_idx - det_center_col) * fan_ang_incr_rad)
            out_plane_weight = (sdd_mm / np.sqrt(sdd_mm ** 2 + ((row_idx - det_center_row) * det_pix_height_mm) ** 2))
            proj[col_idx, row_idx, :] *= (in_plane_weight * out_plane_weight)
    return proj

def fdk_fanbeam(cfg, prep):
    # scanner & recon geometry
    sid_mm, sdd_mm, num_modules, det_row_size_mm, module_width_mm, dectorYoffset, dectorZoffset, \
        fov_diameter_mm, recon_img_size, num_slices, slice_thickness_mm, img_center_offset, start_view, rot_dir, kernel_type \
        = mapConfigVariablesToFDK(cfg)

    # initialize parameters
    fov_radius_mm = fov_diameter_mm / 2
    proj_data = prep.transpose(2, 1, 0)
    num_views = cfg.protocol.viewCount
    fan_ang_incr_rad = compute_fan_ang_incr_rad(cfg)
    det_fan_ang_rad = num_modules * fan_ang_incr_rad
    proj = np.array(proj_data, dtype=np.float32)
    num_det_cols = int(cfg.scanner.detectorColCount)
    num_det_rows = int(cfg.scanner.detectorRowCount)

    det_center_col = (num_det_cols - 1) * 0.5
    det_center_row = (num_det_rows - 1) * 0.5
    det_height_mm = det_row_size_mm * num_det_rows

    det_pix_height_mm = det_height_mm / num_det_rows

    print(" Cosine weighting...")
    cos_weighting(proj, num_det_cols, num_det_rows, sdd_mm,
                  det_center_col, det_center_row, fan_ang_incr_rad, det_pix_height_mm)

    print("* Filtering...")
    proj_filtered = filter_proj(proj, fan_ang_incr_rad, num_det_rows, num_det_cols, num_views, kernel_type)
    print("* Running the backprojection...")
    # special case when ZL is 1, minimum two rows are needed for backprojection (bi-linear interpolation?)
    if num_det_rows == 1:
        proj_filtered = np.append(proj_filtered, proj_filtered, axis=1)
        num_det_rows = 2

    # Load the compiled library
    recon = load_C_recon_lib()

    # Define arguments of the C function
    recon.fbp.argtypes = [ct.POINTER(TestStruct)]

    # Define the return type of the C function
    recon.fbp.restype = None

    # init the struct
    t = TestStruct()

    t.ScanR = sid_mm
    t.DistD = sdd_mm
    t.DecFanAng = det_fan_ang_rad
    t.startangle = start_view  # Inconsistent C variable name - should be startAngle. Also, this seems to be the start view, not angle.
    t.rotdir = rot_dir
    t.DecHeight = det_height_mm
    t.YL = num_det_cols
    t.ZL = num_det_rows
    t.dectorYoffset = dectorYoffset
    t.dectorZoffset = dectorZoffset
    t.AngleNumber = num_views

    t.Radius = fov_radius_mm
    t.RecSize = recon_img_size
    t.sliceThickness = slice_thickness_mm
    t.FOILength = recon_img_size
    t.FOIWidth = recon_img_size
    t.FOIHeight = num_slices
    t.XOffSet = img_center_offset[0]
    t.YOffSet = img_center_offset[1]
    t.ZOffSet = img_center_offset[2]
    t.phantomXOffSet = 0
    t.phantomYOffSet = 0
    t.phantomZOffSet = 0

    if cfg.recon.printReconParameters:
        print("* Reconstruction parameters:")
        print("* SID: {} mm".format(t.ScanR))
        print("* SDD: {} mm".format(t.DistD))
        print("* Fan angle: {} degrees".format(t.DecFanAng))
        print("* Start view: {}".format(t.startangle))
        print("* Number of detector cols: {}".format(t.YL))
        print("* Number of detector rows: {}".format(t.ZL))
        print("* Detector height: {} mm".format(t.DecHeight))
        print("* Detector X offset: {} mm".format(t.dectorYoffset))
        print("* Detector Z offset: {} mm".format(t.dectorZoffset))
        print("* Scan number of views: {} ".format(t.AngleNumber))
        print("* Recon FOV: {} mm".format(2 * t.Radius))
        print("* Recon XY pixel size: {} mm".format(t.RecSize))
        print("* Recon Slice thickness: {} mm".format(t.sliceThickness))
        print("* Recon XY: {} pixels".format(t.FOIWidth))
        print("* Recon Z: {} slices".format(t.FOIHeight))
        print("* Recon X offset: {} mm".format(t.XOffSet))
        print("* Recon Y offset: {} mm".format(t.YOffSet))
        print("* Recon Z offset: {} mm".format(t.ZOffSet))

    #print("* Converting projection data from a numpy array to a C array...")
    GF_ptr = float3Darray2pointer(proj_filtered)
    t.GF = GF_ptr

    #print("* Allocating a C array for the recon results...")
    RecIm = np.zeros(shape=(t.FOILength, t.FOIWidth, t.FOIHeight), dtype=np.single)
    RecIm_ptr = float3Darray2pointer(RecIm)
    t.RecIm = RecIm_ptr

    #print("* In C...")
    recon.fbp(ct.byref(t))

    #print("* Converting the recon results from a C array to a numpy array...")
    RecA = float3Dpointer2array(RecIm_ptr, *RecIm.shape)
    RecA = RecA[:, :, ::-1]

    return RecA

def reconstruct_fdk(config, proj):
    proj = np.expand_dims(proj, axis=1)
    recon_img = fdk_fanbeam(config, proj)
    recon_img = scale_recon_img(config, recon_img)
    return np.squeeze(recon_img)

def scale_recon_img(config, recon_img):
    print('* Scaling recon data...')
    #recon_img= 50000*recon_img - 1000
    if config.recon.unit == 'HU':
        recon_img = recon_img * (1000 / (config.recon.mu)) + config.recon.huOffset
    elif config.recon.unit == '/mm':
        pass
    elif config.recon.unit == '/cm':
        recon_img = recon_img * 10
    else:
        raise Exception(
            '******** Error! An unsupported recon unit was specified: {:s}. ********'.format(config.recon.unit))
    return recon_img

