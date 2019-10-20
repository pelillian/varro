import unittest
import numpy as np
import os

from varro.fpga.interface import FpgaConfig


class TestInterface(unittest.TestCase):
    def test_generate_config(self):
        config_data = np.random.choice(a=[False, True], size=(13294, 1136))
        config = FpgaConfig(config_data)
        config.write_config_file()
        assert os.path.isfile(config.this_config_dir()) == 1

if __name__ == '__main__':
    unittest.main()

