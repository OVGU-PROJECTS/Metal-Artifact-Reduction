# Source: https://github.com/xcist/example/blob/main/AAPM_datachallenge/proj_and_recon/AAPMRecon.py
from gecatsim.pyfiles.CommonTools import CFG


def recon_config(fov_mm):
    config = CFG()

    # protocol
    config.protocol.viewCount = 1000  # total number of views in scan
    config.protocol.rotationDirection = 1  # gantry rotation direction (1=CW, -1 CCW, seen from table foot-end)

    # Scanner
    config.scanner.sid = 550.0  # source-to-iso distance (in mm)
    config.scanner.sdd = 950.0  # source-to-detector distance (in mm)
    config.scanner.detectorColsPerMod = 1  # number of detector columns per module
    config.scanner.detectorRowsPerMod = 1  # number of detector rows per module
    config.scanner.detectorColOffset = -1.25  # detector column offset relative to centered position (in detector columns)
    config.scanner.detectorRowOffset = 0.0  # detector row offset relative to centered position (in detector rows)
    config.scanner.detectorColSize = 1.0  # detector column pitch or size (in mm)
    config.scanner.detectorRowSize = 1.0  # detector row pitch or size (in mm)
    config.scanner.detectorColCount = 900  # total number of detector columns
    config.scanner.detectorRowCount = config.scanner.detectorRowsPerMod  # total number of detector rows

    # recon
    config.recon.fov = fov_mm  # diameter of the reconstruction field-of-view (in mm)
    config.recon.imageSize = 512  # number of columns and rows to be reconstructed (square)
    config.recon.sliceCount = 1  # number of slices to reconstruct
    config.recon.sliceThickness = 0.579  # reconstruction slice thickness AND inter-slice interval (in mm)
    config.recon.centerOffset = [0.0, 0.0, 0.0]  # reconstruction offset relative to center of rotation (in mm)
    config.recon.reconType = 'fdk_equiAngle'  # Name of the recon function to call
    config.recon.kernelType = 'standard'  # 'R-L' for the Ramachandran-Lakshminarayanan (R-L) filter, rectangular window function
    config.recon.startAngle = 0  # in degrees; 0 is with the X-ray source at the top
    config.recon.unit = 'HU'  # '/mm', '/cm', or 'HU'
    config.recon.mu = 0.02  # in /mm; typically around 0.02/mm
    config.recon.huOffset = -1000  # unit is HU, -1000 HU by definition but sometimes something else is preferable
    config.recon.printReconParameters = False  # Flag to print the recon parameters

    return config


def config_downsampled(fov_mm):
    config = CFG()

    config.protocol.viewCount = 500
    config.protocol.rotationDirection = 1

    config.scanner.sid = 550.0
    config.scanner.sdd = 950.0
    config.scanner.detectorColsPerMod = 1
    config.scanner.detectorRowsPerMod = 1
    config.scanner.detectorColOffset = -1.25
    config.scanner.detectorRowOffset = 0.0
    config.scanner.detectorColSize = 1.0
    config.scanner.detectorRowSize = 1.0
    config.scanner.detectorColCount = 900
    config.scanner.detectorRowCount = config.scanner.detectorRowsPerMod

    config.recon.fov = fov_mm
    config.recon.imageSize = 256
    config.recon.sliceCount = 1
    config.recon.sliceThickness = 0.579
    config.recon.centerOffset = [0.0, 0.0, 0.0]
    config.recon.reconType = "fdk_equiAngle"
    config.recon.kernelType = "standard"
    config.recon.startAngle = 0
    config.recon.unit = "HU"
    config.recon.mu = 0.02
    config.recon.huOffset = -1000
    config.recon.printReconParameters = False

    return config