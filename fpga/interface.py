"""
This module handles communication of data to the FPGA once it has already been flashed.
"""
import os

def put_array(data):
    """Loads input data (in numpy array or other iterable form) onto the FPGA."""
    pass

"""
Given a bitstream stored in a file (*.bit), flashes the FPGA with the file contents
"""
def flash_from_file(filename): 
    # Call the Makefile containing the compile/flash statements
    os.sys("make prog")

