"""
This module contains the FPGA model class.
"""

import numbers
import numpy as np

from varro.algo.models import Model


class ModelFPGA(Model):
    FPGA_BITSTREAM_SHAPE = (13294, 1136)

    def __init__(self, problem):
        """FPGA architecture wrapper class"""
        self.name = 'fpga'

    def load_parameters(self, parameters):
        """Loads an array of parameters into this model.

        Args:
            parameters (np.ndarray of floats): The new values for the parameters
                - e.g. [[0, 1, 0, 1, ..., 0],
                        [1, 0, 1, 1, ..., 1],
                        ...
                        [0, 0, 1, 0, ..., 0]]

        """
        from varro.fpga.interface import FpgaConfig
        reshaped_parameters = parameters.reshape(self.FPGA_BITSTREAM_SHAPE)
        self.config = FpgaConfig(reshaped_parameters)

    def predict(self, X, problem=None):
        """Evaluates the model on given data."""
        X = np.asarray(X)
        if isinstance(X[0], numbers.Real) and not isinstance(X[0], numbers.Integral):
            if problem is not None:
                X -= problem.minimum
                X *= 255.0 / (problem.maximum - problem.minimum)
            X = X.astype(int)

        y = self.config.evaluate(X)
        #TODO: scale y if necessary
        #if isinstance(X[0], numbers.Real) and not isinstance(X[0], numbers.Integral):
        return y


    @property
    def parameters_shape(self):
        return self.FPGA_BITSTREAM_SHAPE
