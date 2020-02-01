"""
This module contains a function that returns training set for functions to approximate
"""

import random
import numpy as np

from varro.algo.problems import Problem


def rastrigin(x):
    """Rastrigin function

    Args:
        x (list): Input list

    Returns:
        Outputs of the rastrigin function given the inputs
    """
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + np.sum(x**2 - 10 * np.cos( 2 * np.pi * x))

def rosenbrock(x):
    """Rosenbrock function

    Args:
        x (list): Input list

    Returns:
        Outputs of the rosenbrock function given the inputs
    """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (np.sum( (1 - x0) **2 ) + 100 * np.sum( (x1 - x0**2) **2 ))

class ProblemFuncApprox(Problem):
    def __init__(self, func):
        # Set seed
        random.seed(100)

        # Choose classification or regression
        self._approx_type = Problem.REGRESSION
        self._name = func
        self._input_dim = 1
        self._output_dim = 1

        # Set the X_train and y_train for function to approximate
        self.reset_train_set()

    def sample_float(self, start, end, step, size=500):
        """Gets a random list of floats from a range of floats

        Args:
            start (float): The lower bound of our range
            end (float): The upper bound of our range
            step (float): The precision of our floats to be sampled
            size (int): Number of floats to sample from list

        Returns:
            A random sample of floats from the list
        """
        self.minimum = start
        self.maximum = end
        return random.sample(list(np.arange(start, end, step)), k=size)

    def sample_int(self, start, end, size=500):
        """Gets a random list of ints from a range of ints

        Args:
            start (int): The lower bound of our range
            end (int): The upper bound of our range
            size (int): Number of ints to sample from list

        Returns:
            A random sample of ints from the list
        """
        self.minimum = start
        self.maximum = end
        return np.random.randint(start, end + 1, size=size)

    def sample_bool(self, size=500):
        """Gets a random list of bools, with the same number of 1's and 0's

        Args:
            size (int): Number of bools to sample

        Returns:
            A random sample of bools from the list
        """
        self.minimum = 0
        self.maximum = 1
        sample = np.concatenate((np.zeros(size//2, dtype=np.int8), np.ones(size//2, dtype=np.int8)))
        np.random.shuffle(sample)
        return sample

    def reset_train_set(self):
        """Sets the ground truth training input X_train and output y_train
        for the function specified to approximate

        """
        func = self._name
        if func == 'sinx':
            self.X_train = self.sample_float(-2*np.pi, 2*np.pi, 0.001)
            self.y_train = np.sin(self.X_train)
        elif func == 'cosx':
            self.X_train = self.sample_float(-2*np.pi, 2*np.pi, 0.001)
            self.y_train = np.cos(self.X_train)
        elif func == 'tanx':
            self.X_train = self.sample_float(-2*np.pi, 2*np.pi, 0.001)
            self.y_train = np.tan(self.X_train)
        elif func == 'x':
            self.X_train = self.sample_float(-10, 10, 0.001)
            self.y_train = self.X_train
        elif func == 'ras':
            self.X_train = self.sample_float(-5.12, 5.12, 0.01)
            self.y_train = rastrigin(self.X_train)
        elif func == 'rosen':
            self.X_train = self.sample_float(-10, 10, 0.001)
            self.y_train = rosenbrock(self.X_train)
        elif func == 'step':
            self.X_train = self.sample_float(-10, 10, 0.001)
            self.y_train = (np.array(self.X_train) > 0).astype(float)
        elif func == 'simple_step':
            self.X_train = self.sample_bool(size=40)
            self.y_train = self.X_train
        else:
            raise ValueError('Problem \'' + str(func) + '\' not recognised')
