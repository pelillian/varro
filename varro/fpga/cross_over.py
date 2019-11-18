"""
This module defines the mating protocol to be used for FPGA evolutionary algos
"""

import pytrellis

from varro.fpga.interface import FpgaConfig


def cross_over(ind1, ind2):
    """Performing cross-overs that preserve wire configs
    """
    # Load individuals into FpgaConfig
    ind1_config, ind2_config = FpgaConfig(config_data=ind1), FpgaConfig(config_data=ind2)

    # Cross-over
    import pdb; pdb.set_trace()
    for tile in ind1_config.chip.get_all_tiles():
        pass

    return child1, child2
