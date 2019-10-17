"""
This module contains a function that returns the training set for mnist
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import keras 
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf


def mnistTraining():
    
    """
    Loading MNIST training data from keras in two np.arrays
    """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_total = np.array(x_train + x_test)
    y_total = np.array(y_train + y_total)
    
    return (x_total, y_total)



