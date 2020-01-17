"""
This module defines the mating protocol to be used for FPGA evolutionary algos
"""

import numpy as np
import pytrellis

from varro.fpga.interface import FpgaConfig
from varro.misc.variables import FPGA_BITSTREAM_SHAPE


def cross_over(ind1, ind2):
    """Performing cross-overs that preserve wire configs
    """
    def replace_ind_values(ind, ind_chip):
        """Convert ind list values with new cram from the altered ind_chip

        Args:
            ind (toolbox.Individual): Individual to be modified
            ind_chip (Chip): Chip that had gone through cross_over

        Returns:
            Individual with new values from CRAM after cross over
        """
        ind_bits = np.empty(FPGA_BITSTREAM_SHAPE)
        for i in range(ind_chip.cram.frames()):
            for j in range(ind_chip.cram.bits()):
                ind_bits[i][j] = ind_chip.cram.bit(i, j)

        ind_bits = ind_bits.flatten()
        for idx in range(len(ind)):
            ind[idx] = ind_bits[idx]

        return ind

    # Load individuals into FpgaConfig
    ind1_config, ind2_config = FpgaConfig(config_data=ind1.reshape(FPGA_BITSTREAM_SHAPE)), FpgaConfig(config_data=ind2.reshape(FPGA_BITSTREAM_SHAPE))

    # Cross-over
    point_1_idx, point_2_idx = sorted(np.random.choice(np.arange(0, len(ind1_config.chip.get_all_tiles())), size=2, replace=False))
    for idx in range(point_1_idx, point_2_idx):
        ind1_config.chip.swap_tiles(ind1_config.chip.get_all_tiles()[int(idx)].info.name, ind2_config.chip)

    # Convert the cross-overed chips to individuals
    child1, child2 = replace_ind_values(ind1, ind1_config.chip), replace_ind_values(ind2, ind2_config.chip)

    return child1, child2
