"""
This module handles communication of data to the FPGA
"""

import os
from os.path import join
import pytrellis
from dowel import logger
import numpy as np

from varro.cython.fast_cram import load_cram_fast
from varro.util.variables import PRJTRELLIS_DATABASE, CHIP_NAME, CHIP_COMMENT
from varro.fpga.config import get_new_id, get_config_dir, clean_config_dir
from varro.fpga.flash import flash_config_file
from varro.arduino.communication import evaluate_arduino

pytrellis.load_database(PRJTRELLIS_DATABASE)


class FpgaConfig:
    def __init__(self, config_data=None):
        """This class handles flashing and evaluating the FPGA bitstream"""
        self.chip = pytrellis.Chip(CHIP_NAME)
        clean_config_dir()
        self.id = get_new_id()
        if config_data is not None:
            self.load_fpga(config_data)

    @property
    def basedir(self):
        """Returns this bitstream's directory."""
        return join(get_config_dir(), str(self.id))

    @property
    def base_file_name(self):
        """Returns this bitstream's base file name, without a file extension."""
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
        logger.start_timer() 
        self.load_cram(config_data)
        self.write_config_file()
        flash_config_file(self.base_file_name)
        logger.stop_timer('INTERFACE.PY load_fpga')

    def evaluate(self, data):
        """Evaluates given data on the FPGA."""
        logger.start_timer()
        results = []
        for datum in data:
            pred = None
            while pred is None:
                try:
                    pred = evaluate_arduino(datum)
                except (UnicodeDecodeError, ValueError):
                    pass
            results.append(pred)    
        logger.stop_timer('INTERFACE.PY Evaluation complete')
        return results
