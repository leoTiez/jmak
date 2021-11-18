import os
from pathlib import Path
import argparse


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
