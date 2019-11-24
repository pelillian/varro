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
    point_1_idx, point_2_idx = sorted(np.random.choice(np.arange(0, len(ind1_config.chip.get_all_tiles())), size=2, replace=False))
    for idx in range(point_1_idx, point_2_idx):
        ind1_config.chip.swap_tiles(ind1_config.chip.get_all_tiles()[idx].name, ind2_config.chip)

    return child1, child2
