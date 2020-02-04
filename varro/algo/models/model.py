"""
This module contains classes for defining each type of model.
"""


class Model:
    def __init__(self, problem):
        """Wrapper class for different types of models."""
        self.name = 'Model'

    def load_parameters(self, parameters):
        """Loads an array of parameters into this model.

        Args:
            parameters (np.ndarray): The new values for the parameters
                - e.g. [0.93, 0.85, 0.24, ..., 0.19]

        """
        pass

    def predict(self, X, problem=None):
        """Evaluates the model on given data."""
        pass

    @property
    def parameters_shape(self):
        pass
