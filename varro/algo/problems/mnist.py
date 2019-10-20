"""
This module contains a function that returns the training set for mnist
"""

from keras.datasets import mnist
import numpy as np

from varro.algo.problems import Problem


class ProblemMNIST(Problem):
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self._input_dim = np.prod(self.X_train[0].shape)
        self._output_dim = len(np.unique(self.y_train))
        self._approx_type = Problem.CLASSIFICATION

