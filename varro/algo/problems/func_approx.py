"""
This module contains a function that returns training set for functions to approximate
"""

import re
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
    raise NotImplementedError
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
    raise NotImplementedError
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (np.sum( (1 - x0) **2 ) + 100 * np.sum( (x1 - x0**2) **2 ))

def step_function(x):
    return (np.array(x) > 0).astype(float)

class ProblemFuncApprox(Problem):
    func_dict = dict(
            sin=np.sin,
            cos=np.cos,
            tan=np.tan,
            x=np.copy,
            ras=rastrigin,
            rosen=rosenbrock,
            step=step_function,
        )
    range_dict = dict(
            sin=(-1,1),
            cos=(-1,1),
            tan=(-np.inf, np.inf),
            x=(-np.inf, np.inf),
            step=(0,1),
        )
    def __init__(self, name):
        # Set seed
        random.seed(100)

        # Choose classification or regression
        self._approx_type = Problem.REGRESSION
        self._name = name
        self._input_dim = 1
        self._output_dim = 1

        split = re.split(':', name)
        self.func = split[0]
        self.datatype = None
        if len(split) > 1:
            self.datatype = split[1]
        
        if self.func not in self.func_dict.keys():
            raise ValueError('Problem \'' + self.func + '\' not recognised')

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
            start (int): The lower bound of our range (inclusive)
            end (int): The upper bound of our range (exclusive)
            size (int): Number of ints to sample from list

        Returns:
            A random sample of ints from the list
        """
        self.minimum = start
        self.maximum = end
        return np.random.randint(start, end, size=size)

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

    @property
    def range(self):
        return self.range_dict[self.func]

    def unscale_y(self, data, values=64449):
        ymin, ymax = self.range
        return self.unscale(data, unscaled_min=ymin, unscaled_max=ymax, values=values)

    def unscale(self, data, unscaled_min, unscaled_max, values=4096):
        assert np.max(data) <= values
        unscaled = data.astype(float)
        unscaled *= (unscaled_max - unscaled_min) / float(values)
        unscaled += unscaled_min
        return unscaled

    def apply_func(self, X):
        return self.func_dict[self.func](X)

    def reset_train_set(self, xmin=-2*np.pi, xmax=2*np.pi):
        """Sets the ground truth training input X_train and output y_train
        for the function specified to approximate

        """
        
        if 'float' in self.datatype:
            self.X_train = self.sample_float(xmin, xmax, 0.001, size=400)
            self.y_train = self.apply_func(self.X_train)
        elif 'uint' in self.datatype:
            values = 2 ** int(re.sub('\D', '', self.datatype)) # For example, sin:uint12 has 4096 values
            self.X_train = self.sample_int(0, values, size=40) # X_train is scaled
            X_unscaled = self.unscale(self.X_train, values=values, unscaled_min=xmin, unscaled_max=xmax)
            self.y_train = self.apply_func(X_unscaled) # y_train is unscaled
        elif 'bool' in self.datatype:
            self.X_train = self.sample_bool(size=40)
            self.y_train = self.apply_func(self.X_train.astype(float))
        else:
            raise ValueError('Problem Datatype \'' + self.datatype + '\' not recognised')
