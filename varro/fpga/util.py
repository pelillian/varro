"""
This module contains FPGA utility functions.
"""

import os
from os.path import join

from varro.misc.util import make_path

BITSTREAM_DIR = "fpga/bitstreams"


def get_bitstream_dir():
    """Returns the directory containing the bitstream folders."""
    make_path(BITSTREAM_DIR)
    return BITSTREAM_DIR

def is_int(str):
    """Determines if a number is an integer."""
    try: 
        int(str)
        return True
    except ValueError:
        return False

def get_max_id():
    """Finds the maximum id in the bitstream."""
    max_id = 0
    for filename in os.listdir(get_bitstream_dir()):
        if is_int(filename):
            file_id = int(filename)
            if file_id > max_id:
                max_id = file_id
    return max_id

def get_new_id():
    """Generates a new id for a new bitstream."""
    new_id = get_max_id() + 1
    make_path(join(get_bitstream_dir(), str(new_id)))
    return new_id
