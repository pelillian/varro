"""
This module defines the mating protocol to be used for FPGA evolutionary algos
"""

import pytrellis

from varro.fpga.interface import FpgaConfig

FPGA_BITSTREAM_SHAPE = (13294, 1136)


def cross_over(ind1, ind2):
    """Performing cross-overs that preserve wire configs
    """
    # Load individuals into FpgaConfig
    ind1_config, ind2_config = FpgaConfig(config_data=ind1.reshape(FPGA_BITSTREAM_SHAPE)), FpgaConfig(config_data=ind2.reshape(FPGA_BITSTREAM_SHAPE))

    # Cross-over
    import pdb; pdb.set_trace()
    for tile in ind1_config.chip.get_all_tiles():
        pass

    return child1, child2
