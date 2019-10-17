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
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return ((x_train, y_train), (x_test, y_test))


