"""
This module contains a function that returns the training set for mnist
"""

import numpy as np
import random

from varro.algo.problems import Problem

# Percentage of total mnist train data
# we will use for a single generation training
TRAIN_SIZE = 0.2

class ProblemMNIST(Problem):
    def __init__(self):
        # Set seed
        random.seed(100)

        self._approx_type = Problem.CLASSIFICATION
        self._name = 'mnist'
        self.minimum = None
        self.maximum = None

        # Set the X_train and y_train for function to approximate
        from keras.datasets import mnist

        # Load the MNIST dataset
        (self.full_X_train, self.full_y_train), (self.X_test, self.y_test) = mnist.load_data()

        # Flatten the MNIST images into a 784 dimension vector
        self.full_X_train = np.array([x.flatten() for x in self.full_X_train])
        self.X_test = np.array([x.flatten() for x in self.X_test])
        self.reset_train_set()

        # Set the input output dimensions for NN
        self._input_dim = np.prod(self.X_train[0].shape)
        self._output_dim = len(np.unique(self.y_train))

    def reset_train_set(self):
        """Sets the ground truth training input X_train and output y_train
        for the function specified to approximate

        """
        # Get a random set of training
        # set indices from mnist training data
        train_idxs = np.random.choice(np.arange(len(self.full_X_train)), size=int(len(self.full_X_train)*TRAIN_SIZE))

        self.X_train, self.y_train = self.full_X_train[train_idxs], self.full_y_train[train_idxs]
