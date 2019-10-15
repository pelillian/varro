"""
This module contains code for interfacing the evolutionary algorithm with FPGA bitstreams
"""

from fpga.interface import Bitstream

def evaluate_fpga(individual):
    bits = Bitstream()
    bits.flash(None)
    return bits.evaluate(None)

