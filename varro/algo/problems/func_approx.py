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

    def get_func_dtype(self):
        split = re.split(':', self._name)
        func = split[0]
        datatype = None
        if len(split) > 1:
            datatype = split[1]
        return func, datatype

    def reset_train_set(self, minimum=-2*np.pi, maximum=2*np.pi):
        """Sets the ground truth training input X_train and output y_train
        for the function specified to approximate

        """
        func, datatype = self.get_func_dtype() # Example: sin:int12 becomes func=sin, datatype=int12
        
        func_dict = dict(
                sin=np.sin,
                cos=np.cos,
                tan=np.tan,
                x=np.copy,
                ras=rastrigin,
                rosen=rosenbrock,
                step=step_function,
            )
        if func not in func_dict.keys():
            raise ValueError('Problem \'' + func + '\' not recognised')
        
        if 'float' in datatype:
            self.X_train = self.sample_float(minimum, maximum, 0.001, size=400)
            self.y_train = func_dict[func](self.X_train)
        elif 'int' in datatype:
            values = 2 ** int(re.sub('\D', '', datatype)) # For example, sin:int12 has 4096 values
            self.X_train = self.sample_int(0, values, size=40)
            X_unscaled = self.X_train.astype(float)
            X_unscaled *= (maximum - minimum) / float(values)
            X_unscaled += minimum
            self.y_train = func_dict[func](X_unscaled)
        elif 'bool' in datatype:
            self.X_train = self.sample_bool(size=40)
            self.y_train = func_dict[func](self.X_train.astype(float))
        else:
            raise ValueError('Problem Datatype \'' + datatype + '\' not recognised')
