"""
This module contains the FPAA model class.
"""

import numbers
import numpy as np

from varro.algo.models import Model


class ModelFPAA(Model):
    FPAA_BITSTREAM_SHAPE = (None, None)

    def __init__(self, problem):
        """FPAA architecture wrapper class"""
        self.name = 'fpaa'

    def load_parameters(self, parameters):
        """Loads an array of parameters into this model.

        Args:
            parameters (np.ndarray of floats): The new values for the parameters
                - e.g. [[0, 1, 0, 1, ..., 0],
                        [1, 0, 1, 1, ..., 1],
                        ...
                        [0, 0, 1, 0, ..., 0]]

        """
        from varro.fpaa.interface import FpaaConfig
        self.config = FpaaConfig(parameters.reshape(self.FPAA_BITSTREAM_SHAPE))
        self.config.load_fpaa(parameters.reshape(self.FPAA_BITSTREAM_SHAPE))

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
        return self.FPAA_BITSTREAM_SHAPE
