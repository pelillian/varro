"""
This module contains FPGA utility functions.
"""

import os
from os.path import join
import numpy as np
import pytrellis

from varro.misc.util import make_path
from varro.misc.variables import FPGA_CONFIG_DIR
from varro.misc.variables import FPGA_BITSTREAM_SHAPE


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

def bit_to_cram(filename):
    """Take a .bit file and return the CRAM array."""
    chip = load_bit_to_chip(filename)
    return chip_to_cram(chip)

def load_bit_to_chip(filename):
    """Take a .bit file and return the deserialized chip."""
    pytrellis.load_database("../prjtrellis-db")
    bs = pytrellis.Bitstream.read_bit(filename)
    return bs.deserialise_chip()

def chip_to_cram(chip):
    """Take a pytrellis chip object and return the CRAM array."""
    cram_bits = np.empty(FPGA_BITSTREAM_SHAPE, dtype=np.bool)
    for i in range(chip.cram.frames()):
        for j in range(chip.cram.bits()):
            cram_bits[i][j] = chip.cram.bit(i, j)

    return cram_bits
