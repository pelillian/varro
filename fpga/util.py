"""
This module contains FPGA utility functions.
"""

import os

BITSTREAM_DIR = "fpga/bitstreams"


def make_path(dir):
    os.makedirs(dir, exist_ok=True)

def get_bitstream_dir():
    make_path(BITSTREAM_DIR)
    return BITSTREAM_DIR

def is_int(str):
    try: 
        int(str)
        return True
    except ValueError:
        return False

def get_max_id():
    max_id = 0
    for filename in os.listdir(get_bitstream_dir()):
        if is_int(filename) and int(filename) > max_id:
            max_id = filename
    return max_id

def get_new_id():
    return get_max_id() + 1
