"""
This module contains a function that returns the training set for a given problem
"""

from abc import ABC, abstractmethod

class Problem(ABC):
    CLASSIFICATION = 0
    REGRESSION = 1

    def __init__(self):
        """This class defines a problem & dataset for a model to solve."""
        self._input_dim = None
        self._output_dim = None
        self._approx_type = None
        self._name = None

    def training_set(self):
        return None, None

    @abstractmethod
    def reset_train_set(self):
        pass

    @property
    def name(self):
        """Name of Problem we're trying to optimize"""
        return self._name

    @property
    def approx_type(self):
        """Classification or Regression"""
        return self._approx_type

    @property
    def input_dim(self):
        """Dimension of input vector"""
        return self._input_dim

    @property
    def output_dim(self):
        """Dimension of output vector"""
        return self._output_dim
