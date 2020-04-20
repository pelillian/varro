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

def notstep_function(x):
    return (np.array(x) <= 0).astype(float)

class ProblemFuncApprox(Problem):
    func_dict = dict(
            sin=np.sin,
            cos=np.cos,
            tan=np.tan,
            x=np.copy,
            ras=rastrigin,
            rosen=rosenbrock,
            step=step_function,
            notstep=notstep_function,
        )
    range_dict = dict(
            sin=(-1,1),
            cos=(-1,1),
            tan=(-np.inf, np.inf),
            x=(-np.inf, np.inf),
            step=(0,1),
        )
    datatype_dict = dict(
            int=int,
            bool=bool,
            float=float,
        )
    def __init__(self, name, sample_size=500):
        # Set seed
        random.seed(100)

        # Choose classification or regression
        self._approx_type = Problem.REGRESSION
        self._name = name
        self._input_dim = 1
        self._output_dim = 1
        self._sample_size = sample_size

        split = re.split(':', name)
        self.func = split[0]
        self.datatype = None
        if len(split) > 1:
            self.datatype = split[1]
            self.values = 2 ** int(re.sub('\D', '', self.datatype)) # For example, sin:uint12 has 4096 values
            for key in self.datatype_dict.keys():
                if key in self.datatype:
                    self.datatype = self.datatype_dict[key]
                    break
            if isinstance(self.datatype, str):
                raise ValueError('Problem Datatype \'' + self.datatype + '\' not recognised')
        
        if self.func not in self.func_dict.keys():
            raise ValueError('Problem \'' + self.func + '\' not recognised')

        # Set the X_train and y_train for function to approximate
        self.reset_train_set()

    def sample_float(self, start, end, step):
        """Gets a random list of floats from a range of floats

        Args:
            start (float): The lower bound of our range
            end (float): The upper bound of our range
            step (float): The precision of our floats to be sampled

        Returns:
            A random sample of floats from the list
        """
        self.minimum = start
        self.maximum = end
        return random.sample(list(np.arange(start, end, step)), k=self._sample_size)

    def sample_int(self, start, end):
        """Gets a random list of ints from a range of ints

        Args:
            start (int): The lower bound of our range (inclusive)
            end (int): The upper bound of our range (exclusive)

        Returns:
            A random sample of ints from the list
        """
        self.minimum = start
        self.maximum = end
        return np.random.randint(start, end, size=self._sample_size)

    def sample_bool(self):
        """Gets a random list of bools, with the same number of 1's and 0's

        Returns:
            A random sample of bools from the list
        """
        self.minimum = 0
        self.maximum = 1
        sample = np.concatenate((np.zeros(self._sample_size//2, dtype=np.int8), np.ones(self._sample_size//2, dtype=np.int8)))
        np.random.shuffle(sample)
        return sample

    @property
    def range(self):
        return self.range_dict[self.func]

    def scale_y(self, data, values=64449):
        ymin, ymax = self.range
        return self.scale(data, unscaled_min=ymin, unscaled_max=ymax, values=values)

    def scale(self, data, unscaled_min, unscaled_max, values=64449):
        assert np.max(data) <= unscaled_max and np.min(data) >= unscaled_min
        scaled = data.astype(float)
        scaled -= unscaled_min
        scaled *= float(values) / (unscaled_max - unscaled_min)
        scaled = scaled.astype(int)
        return scaled

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
        
        if self.datatype is float:
            self.X_train = self.sample_float(xmin, xmax, 0.001)
            self.y_train = self.apply_func(self.X_train)
        elif self.datatype is int:
            # The scaled space is comprised of integers from 0 to self.values
            # The unscaled space is comprised of floats from xmin to xmax
            # Each integer in the scaled space represents a different value in the unscaled space
            # We have to unscale it to apply the function, and then we have to rescale the output of the function
            # The scaled data is what is sent to the FPGA
            self.X_train = self.sample_int(0, self.values) # X_train is scaled
            X_unscaled = self.unscale(self.X_train, values=self.values, unscaled_min=xmin, unscaled_max=xmax)
            y_unscaled = self.apply_func(X_unscaled)
            self.y_train = self.scale_y(y_unscaled, values=64449) # y_train is scaled
        elif self.datatype is bool:
            self.X_train = self.sample_bool()
            self.y_train = self.apply_func(self.X_train.astype(float))
        else:
            raise NotImplementedError
