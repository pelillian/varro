"""
This module contains classes for defining each type of model.
"""

from varro.algo.models import Model


class ModelFPGA(Model):
    FPGA_BITSTREAM_SHAPE = (13294, 1136)

    def __init__(self):
        """FPGA architecture wrapper class"""
        pass

    def load_weights(self, weights):
        """Loads an array of weights into this model.

        Args:
            weights (np.ndarray of floats): The new values for the weights
                - e.g. [[0, 1, 0, 1, ..., 0],
                        [1, 0, 1, 1, ..., 1],
                        ...
                        [0, 0, 1, 0, ..., 0]]

        """
        from varro.fpga.interface import FpgaConfig
        self.config = FpgaConfig(weights)
        self.config.load_fpga()

    def predict(self, X):
        """Evaluates the model on given data."""
        return self.config.evaluate(X)

    @property
    def weights_shape(self):
        return ModelFPGA.FPGA_BITSTREAM_SHAPE

