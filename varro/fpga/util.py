"""
This module contains FPGA utility functions.
"""

import os
from os.path import join

from varro.misc.util import make_path

FPGA_CONFIG_DIR = "data/config"


def get_config_dir():
    """Returns the directory containing the bitstream/config folders."""
    make_path(FPGA_CONFIG_DIR)
    return FPGA_CONFIG_DIR

def is_int(str):
    """Determines if a number is an integer."""
    try:
        int(str)
        return True
    except ValueError:
        return False

def get_max_id():
    """Finds the maximum id in the config folder."""
    max_id = 0
    for filename in os.listdir(get_config_dir()):
        if is_int(filename):
            file_id = int(filename)
            if file_id > max_id:
                max_id = file_id
    return max_id

def get_new_id():
    """Generates a new id for a new bitstream."""
    new_id = get_max_id() + 1
    make_path(join(get_config_dir(), str(new_id)))
    return new_id
