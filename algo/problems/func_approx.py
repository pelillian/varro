"""
This module contains a function that returns training set for functions to approximate
"""

import random
import numpy as np


def rastrigin(x):
	"""Rastrigin function

	Args:
		x (list): Input list

	Returns:
		Outputs of the rastrigin function given the inputs
	"""
	x = np.asarray_chkfinite(x)
	n = len(x)
	return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))


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
    return (sum( (1 - x0) **2 ) + 100 * sum( (x1 - x0**2) **2 ))


def training_set(problem):
	"""Loads in random training data and appropriate labes, 
	returns tuple of numpy arrays  

	Args:
		problem (str): String representing the function to approximate

	Returns:
		Tuple of the ground truth dataset (X_train: features, y_train: labels)
	"""

	# Set seed
	random.seed(100)

	# Get the function to approximate
	if problem == 'sinx':
	    X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
	    y_train = np.sin(X_train)
	elif problem == 'cosx':
		X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
		y_train = np.cos(X_train)
	elif problem == 'tanx':
		X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
		y_train = np.tan(X_train)
	elif problem == 'x':
		X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
		y_train = X_train
	elif problem == 'ras':
		X_train = random.sample(list(np.arange(-5.12, 5.12, 0.1)), k=500)
		y_train = rastrigin(X_train)
	elif problem == 'rosen':
		X_train = random.sample(list(np.arange(-100, 100, 0.1)), k=500)
		_train = rosenbrock(X_train)
	else:
		pass

	return X_train, y_train