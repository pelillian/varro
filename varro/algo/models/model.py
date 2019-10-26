"""
This module contains classes for defining each type of model.
"""


class Model:
    def __init__(self):
        """Wrapper class for different types of models."""
        pass

    def load_weights(self, weights):
        """Loads an array of parameters into this model.

        Args:
            parameters (np.ndarray): The new values for the parameters
                - e.g. [0.93, 0.85, 0.24, ..., 0.19]

        """
        pass

    def predict(self, X):
        """Evaluates the model on given data."""
        pass

    @property
    def weights_shape(self):
        pass

