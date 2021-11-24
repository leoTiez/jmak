import os
from pathlib import Path
import numpy as np


def validate_dir(rel_path=''):
    """
    Validates path and creates parent/child folders if path is not existent
    :param rel_path: Relative path from the current directory to the target directory
    :type rel_path: str
    :return: Path as a string
    """
    curr_dir = os.getcwd()
    Path('%s/%s/' % (curr_dir, rel_path)).mkdir(parents=True, exist_ok=True)
    return '%s/%s/' % (curr_dir, rel_path)


def frequency_bins(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))


