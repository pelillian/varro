"""
This module handles communication of data to the FPGA
"""

from fpga.util import make_path, get_new_id, get_bitstream_dir

import os
from os.path import join
import pytrellis

pytrellis.load_database("../prjtrellis-db")


def put_array(data):
    """Loads input data (in numpy array or other iterable form) onto the FPGA."""
    pass


def flash_from_file(filename): 
    """Given a bitstream stored in a file (*.bit), flashes the FPGA with the file contents."""
    # Call the Makefile containing the compile/flash statements
    os.sys("make prog")


class Bitstream:
    def __init__(self):
        """This class handles flashing and evaluating the FGPA bitstream"""
        self.chip = pytrellis.Chip("LFE5U-85F")
        self.id = get_new_id()

    def get_dir(self):
        """Return this bitstream's directory."""
        return join(get_bitstream_dir(), str(self.id))

    def get_config(self):
        """Return this bitstream's config file."""
        return join(self.get_dir(), str(self.id) + ".config")

    def flash(self, data):
        """Flash a 2d array of configuration data to the FPGA"""
        # TODO: Speed this loop up using C++
        for i in range(self.chip.cram.frames()):
            for j in range(self.chip.cram.bits()):
                self.chip.cram.set_bit(i, j, bool(data[i,j]))

        with open(self.get_config(), "w") as f:
            for tile in self.chip.get_all_tiles():
                config = tile.dump_config()
                if len(config.strip()) > 0:
                    f.write(config)

    def evaluate(self, data):
        return None

