import unittest
import numpy as np

from varro.fpga.interface import FpgaConfig, write_config


class TestInterface(unittest.TestCase):
    def test_generate_config():
        config_data = np.random.choice(a=[False, True], size=(13294, 1136))
        config = FpgaConfig(config_data)
        config.write_config_file()

if __name__ == '__main__':
    unittest.main()

