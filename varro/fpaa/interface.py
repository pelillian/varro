"""
This module handles communication of data to the FPAA
"""

import os
from os.path import join
from dowel import logger
from time import sleep


class FpaaConfig:
    def __init__(self, config_data=None):
        """This class handles flashing and evaluating the FPAA bitstream"""
        clean_config_dir()
        self.id = get_new_id()
        if config_data is not None:
            self.load_fpaa(config_data)

    @property
    def basedir(self):
        """Returns this bitstream's directory."""
        return join(get_config_dir(), str(self.id))

    def load_fpaa(self, config_data):
        """Loads a 2d array of configuration data onto to the FPAA"""
        logger.start_timer() 
        raise NotImplementedError
        logger.stop_timer('INTERFACE.PY load_fpaa')

    def evaluate(self, data):
        """Evaluates given data on the FPAA."""
        logger.start_timer()
        results = []
        for datum in data:
            pred = None
            while pred is None:
                try:
                    raise NotImplementedError
                except (UnicodeDecodeError, ValueError):
                    pass
            results.append(pred)    
        logger.stop_timer('INTERFACE.PY Evaluation complete')
        return results
