"""
This module contains code for interfacing the evolutionary algorithm with FPGA bitstreams
"""

from fpga.interface import FPGA

def evaluate_fpga(individual):
    fpga = FPGA()
    fpga.flash(None)
    return fpga.evaluate(None)

