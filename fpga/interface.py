"""
This module handles communication of data to the FPGA
"""

from fpga.util import make_path, get_new_id, get_bitstream_dir

import os
from os.path import join
import pytrellis

pytrellis.load_database("../prjtrellis-db")


class Bitstream:
    def __init__(self):
        """This class handles flashing and evaluating the FGPA bitstream"""
        self.chip = pytrellis.Chip("LFE5U-85F")
        self.id = get_new_id()

    def get_dir(self):
        """Returns this bitstream's directory."""
        return join(get_bitstream_dir(), str(self.id))

    def get_config(self):
        """Returns this bitstream's config file."""
        return join(self.get_dir(), str(self.id) + ".config")

    def flash(self, data):
        """Flashes a 2d array of configuration data to the FPGA"""
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
        """Evaluates given data on the FPGA."""
        return None

