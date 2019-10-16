"""
This module contains code for interfacing betwee the evolutionary algorithm with FPGA bitstreams
"""

from fpga.interface import Bitstream

def evaluate_fpga(individual):
    bits = Bitstream()
    bits.flash(individual)
    return bits.evaluate(None)

