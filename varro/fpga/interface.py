"""
This module handles communication of data to the FPGA
"""

import os
from os.path import join
import pytrellis
from time import sleep

from varro.cython.fast_cram import load_cram_fast
from varro.misc.variables import PRJTRELLIS_DATABASE, CHIP_NAME, CHIP_COMMENT
from varro.fpga.util import make_path, get_new_id, get_config_dir
from varro.fpga.flash import flash_config_file
from varro.arduino.communication import initialize_connection, send, receive

pytrellis.load_database(PRJTRELLIS_DATABASE)
arduino_connection = initialize_connection()


class FpgaConfig:
    def __init__(self, config_data=None):
        """This class handles flashing and evaluating the FGPA bitstream"""
        self.chip = pytrellis.Chip(CHIP_NAME)
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
        """Returns this bitstream's config file name"""
        return self.base_file_name + ".config"

    def load_cram(self, config_data):
        load_cram_fast(self.chip.cram, config_data)

    def write_config_file(self):
        with open(self.config_file, "w") as f:
            print(".device {}".format(self.chip.info.name), file=f)
            print("", file=f)
#            for meta in self.chip.metadata:
#                print(".comment {}".format(meta), file=f)
            print(CHIP_COMMENT, file=f)
            print("", file=f)

            from varro.fpga.tiles import SIMPLE_STEP_TILES, SIMPLE_STEP_CFG
            for tile in self.chip.get_all_tiles():
                if not tile.info.name in SIMPLE_STEP_TILES:
                    continue
                config = tile.dump_config()
#                config = os.linesep.join([line for line in config.splitlines() if "unknown" not in line])
                if len(config.strip()) > 0:
                    print(".tile {}".format(tile.info.name), file=f)
                    print(config, file=f)
                    print("", file=f)
            print(SIMPLE_STEP_CFG, file=f)

    def load_fpga(self, config_data):
        """Loads a 2d array of configuration data onto to the FPGA"""

        self.load_cram(config_data)
        self.write_config_file()
        flash_config_file(self.base_file_name)

    def evaluate(self, data):
        """Evaluates given data on the FPGA."""

        results = []
        for datum in data:
            # Format data to be written to digital pins on Arduino
            # For now, just send either all ones or all zero
            value = str(data[0])
            msg = "".join([value] * 12)

            # Send and receive formatted data 
            send(arduino_connection, msg)
            sleep(0.96)
            return_value = receive(arduino_connection) 

            # convert data into format usable for evaluation
            data = return_value.decode("utf-8")
            data = data.split(",")
            for num in data: 
                num = int(num)
                num /= 1024

            results.append(data)

        return results
