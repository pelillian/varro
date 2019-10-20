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
    def __init__(self, func='sinx'):
        # Set seed
        random.seed(100)

        # Get the function to approximate
        if func == 'sinx':
            self.X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
            self.y_train = np.sin(self.X_train)
        elif func == 'cosx':
            self.X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
            self.y_train = np.cos(self.X_train)
        elif func == 'tanx':
            self.X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
            self.y_train = np.tan(self.X_train)
        elif func == 'x':
            self.X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
            self.y_train = self.X_train
        elif func == 'ras':
            self.X_train = random.sample(list(np.arange(-5.12, 5.12, 0.01)), k=500)
            self.y_train = rastrigin(self.X_train)
        elif func == 'rosen':
            self.X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
            self.y_train = rosenbrock(self.X_train)
        else:
            raise ValueError('Problem \'' + str(func) + '\' not recognised')

        self._input_dim = 1
        self._output_dim = 1
        self._approx_type = Problem.REGRESSION
