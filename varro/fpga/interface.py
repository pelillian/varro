"""
This module handles communication of data to the FPGA
"""

import os
from os.path import join
import numpy as np
import pytrellis
from dowel import logger

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

            from varro.fpga.tiles import FIXED_TILES, FIXED_TYPES, CONFIG
            for tile in self.chip.get_all_tiles():
                if tile.info.name in FIXED_TILES or any(ftype in tile.info.name for ftype in FIXED_TYPES):
                    continue
                row = tile.info.get_row_col().first
                col = tile.info.get_row_col().second
                if row < 81 or col < 113 or row >= 94 or col >= 125:
                    continue
                config = tile.dump_config()
#                config = os.linesep.join([line for line in config.splitlines() if "unknown" not in line])
                if len(config.strip()) > 0:
                    print(".tile {}".format(tile.info.name), file=f)
                    print(config, file=f)
                    print("", file=f)
            print(CONFIG, file=f)

    def load_fpga(self, config_data):
        """Loads a 2d array of configuration data onto to the FPGA"""
        logger.start_timer() 
        self.load_cram(config_data)
        self.write_config_file()
        flash_config_file(self.base_file_name)
        logger.stop_timer('INTERFACE.PY load_fpga')

    def evaluate(self, data, datatype=int):
        """Evaluates given data on the FPGA."""
        logger.start_timer()
        results = []
        for datum in data:
            pred = None
            attempts = 0
            while pred is None:
                attempts += 1
                if attempts > 10:
                    raise ValueError('Tried 10 times to evaluate_arduino')
                try:
                    pred = evaluate_arduino(datum, send_type=datatype, return_type=datatype)
                except (UnicodeDecodeError, ValueError):
                    pass
            results.append(pred)    
        if logger._print_time:
            d = np.array(data)
            r = np.array(results)
            arr = np.column_stack((d,r))
            print(arr)
        logger.stop_timer('INTERFACE.PY Evaluation complete')
        return results
