"""
This module contains a function that returns the training set for mnist
"""

from keras.datasets import mnist


def training_set():
    """Loading MNIST training data from keras in two np.arrays

    Args:

    Returns:
    	Tuple of the ground truth dataset (X_train: features, y_train: labels)
    """
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    return X_train, y_train



