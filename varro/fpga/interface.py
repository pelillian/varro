"""
This module handles communication of data to the FPGA
"""

import os
from os.path import join
import pytrellis

from varro.fpga.util import make_path, get_new_id, get_config_dir
from varro.fpga.flash import flash_config_file
import varro.fpga.arduino as arduino

pytrellis.load_database("../prjtrellis-db")
arduino_connection = arduino.initialize_connection()


class FpgaConfig:
    def __init__(self, config_data=None):
        """This class handles flashing and evaluating the FGPA bitstream"""
        self.chip = pytrellis.Chip("LFE5U-85F")
        self.id = get_new_id()
        if config_data is not None:
            self.load_cram(config_data)

    @property
    def basedir(self):
        """Returns this bitstream's directory."""
        return join(get_config_dir(), str(self.id))

    @property
    def base_file_name(self):
        """Returns this bitstream's base file name"""
        return join(self.basedir, str(self.id))

    @property
    def config_file(self):
        """Returns this bitstream's base file name"""
        return self.base_file_name + ".config"

    def load_cram(self, config_data):
        # TODO: Speed this loop up using C++
        for i in range(self.chip.cram.frames()):
            for j in range(self.chip.cram.bits()):
                self.chip.cram.set_bit(i, j, bool(config_data[i,j]))

    def write_config_file(self):
        with open(self.config_file, "w") as f:
            print(".device {}".format(self.chip.info.name), file=f)
            print("", file=f)
            for meta in self.chip.metadata:
                print(".comment {}".format(meta), file=f)
            print("", file=f)

            from varro.fpga.tiles import TILES
            for tile in self.chip.get_all_tiles():
                if not tile.info.name in TILES:
                    continue
                config = tile.dump_config()
                config = os.linesep.join([line for line in config.splitlines() if "unknown" not in line])
                if len(config.strip()) > 0:
                    print(".tile {}".format(tile.info.name), file=f)
                    print(config, file=f)
                    print("", file=f)

    def load_fpga(self, config_data=None):
        """Loads a 2d array of configuration data onto to the FPGA"""

        self.load_cram(config_data)
        self.write_config_file()
        flash_config_file(self.base_file_name)

    def evaluate(self, data):
        """Evaluates given data on the FPGA."""

        results = []
        for datum in data:
            # Send the data and recieves a string back
            retval = arduino.send_and_recieve(arduino_connection, datum, 0.2)

            # Parse the correct value from the string
            retval = retval.decode("utf-8").split("Read value: ", 1)[1][0]

            results.append(int(retval))

        return results

