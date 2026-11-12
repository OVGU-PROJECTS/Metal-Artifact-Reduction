import os
import pathlib
import re

HEAD_LAST_ID = 1773  # case ids <= this are head (FOV code 0), > this are body (code 1)


def get_dir_names(main_dir_name):
    """Sorted list of absolute POSIX paths of the subdirectories of ``main_dir_name``."""
    names = [f for f in pathlib.Path(main_dir_name).iterdir() if f.is_dir()]
    dir_list = [f.absolute().as_posix() for f in names]
    dir_list.sort()
    return dir_list


def get_filenames(dir_name):
    """Sorted list of absolute POSIX paths of the files in ``dir_name``."""
    filenames = [f for f in pathlib.Path(dir_name).iterdir() if f.is_file()]
    file_list = [f.absolute().as_posix() for f in filenames]
    file_list.sort()
    return file_list


def list_case_dirs(root):
    """Sorted list of ``pathlib.Path`` case directories under ``root`` (sorted by name)."""
    root = pathlib.Path(root)
    cases = [p for p in root.iterdir() if p.is_dir()]
    cases.sort(key=lambda p: p.name)
    return cases



def inference_fov(case_path, head_last=HEAD_LAST_ID, fov="auto"):
    """Prefer an explicit FOV or head/body folder; otherwise use the case ID."""
    if fov not in ("auto", "head", "body"):
        raise ValueError(f"Unknown FOV: {fov}")
    if fov != "auto":
        return 0 if fov == "head" else 1
    for part in reversed(pathlib.Path(case_path).parts):
        if part in ("head", "body"):
            return 0 if part == "head" else 1
    return fov_from_folder(case_path, head_last=head_last)


def fov_from_folder(folder_path, head_last=HEAD_LAST_ID):
    """assign a fov codes (head=0, body=1) to each training example"""
    base = os.path.basename(folder_path)
    nums = re.findall(r"\d+", base)
    if not nums:
        parent = os.path.basename(os.path.dirname(folder_path))
        nums = re.findall(r"\d+", parent)
    if not nums:
        return 1  # conservative fallback: body
    case_id = int(nums[-1])
    return 0 if case_id <= head_last else 1
