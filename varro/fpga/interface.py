"""
This module handles communication of data to the FPGA
"""

import os
from os.path import join
import pytrellis

from varro.fpga.util import make_path, get_new_id, get_config_dir
from varro.fpga.flash import flash_config_file

pytrellis.load_database("../prjtrellis-db")


class FpgaConfig:
    def __init__(self, config_data=None):
        """This class handles flashing and evaluating the FGPA bitstream"""
        self.chip = pytrellis.Chip("LFE5U-85F")
        self.id = get_new_id()
        if config_data is not None:
            self.load_cram(config_data)

    def get_dir(self):
        """Returns this bitstream's directory."""
        return join(get_config_dir(), str(self.id))

    def this_config_dir(self):
        """Returns this bitstream's config file."""
        return join(self.get_dir(), str(self.id) + ".config")

    def load_cram(self, config_data):
        # TODO: Speed this loop up using C++
        for i in range(self.chip.cram.frames()):
            for j in range(self.chip.cram.bits()):
                self.chip.cram.set_bit(i, j, bool(config_data[i,j]))

    def write_config_file(self):
        with open(self.this_config_dir(), "w") as f:
            print(".device {}".format(self.chip.info.name), file=f)
            print("", file=f)
            for meta in self.chip.metadata:
                print(".comment {}".format(meta), file=f)
            print("", file=f)

            for tile in self.chip.get_all_tiles():
                config = tile.dump_config()
                if len(config.strip()) > 0:
                    print(".tile {}".format(tile.info.name), file=f)
                    print(config, file=f)

    def load_fpga(self, config_data):
        """Loads a 2d array of configuration data onto to the FPGA"""
        self.load_cram(config_data)
        self.write_config_file()
        flash_config_file(self.this_config_dir())

    def evaluate(self, data):
        """Evaluates given data on the FPGA."""
        return [0] * len(data)

