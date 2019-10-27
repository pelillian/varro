"""
This module contains a function that returns the training set for mnist
"""

import numpy as np

from varro.algo.problems import Problem


class ProblemMNIST(Problem):
    def __init__(self):
        from keras.datasets import mnist
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # Flatten the MNIST images into a 784 dimension vector
        self.X_train = np.array([x.flatten() for x in self.X_train])
        self.X_test = np.array([x.flatten() for x in self.X_test])

        self._input_dim = np.prod(self.X_train[0].shape)
        self._output_dim = len(np.unique(self.y_train))
        self._approx_type = Problem.CLASSIFICATION
        self.minimum = None
        self.maximum = None

