"""
This module handles communication of data to the FPGA once it has already been flashed.
"""

import os
import flash
import pytrellis

pytrellis.load_database("~/Git/prjtrellis-db")


def put_array(data):
    """Loads input data (in numpy array or other iterable form) onto the FPGA."""
    pass


def flash_from_file(filename): 
    """Given a bitstream stored in a file (*.bit), flashes the FPGA with the file contents."""
    # Call the Makefile containing the compile/flash statements
    os.sys("make prog")


class Bitstream:
    def __init__(self):
        self.chip = pytrellis.Chip("LFE5U-85F")
    def flash(self, data):
        #self.chip.cram
        pass
    def evaluate(self, data):
        pass

