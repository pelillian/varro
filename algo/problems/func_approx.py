"""
This module contains a function that returns training set for functions to approximate
"""

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
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))

def training_set(func_to_approx):
	"""Loads in random training data and appropriate labes, 
	returns tuple of numpy arrays  

	Args:
		func_to_approx (str): String representing the function to approximate

	Returns:
		Tuple of the ground truth dataset (X_train: features, y_train: labels)
	"""

	# Get the function to approximate
	if func_to_approx == 'sinx':
		# Apply function to each of the inputs
	    X_train = np.random.randint(-1000, 1000, 500) 
	    y_train = np.array(list(map(np.sin, X)))
	elif func_to_approx == 'cosx':
		func = np.cos
	elif func_to_approx == 'tanx':
		func = np.tan
	elif func_to_approx == 'x':
		func = lambda x: x
	elif func_to_approx == 'ras':
		func = rastrigin
	elif func_to_approx == 'rosen':
		func = rosenbrock
	else:
		pass
    
    return X_train, y_train