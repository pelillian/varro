"""
This module contains a function that returns training set for functions to approximate
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import keras 
from keras.layers import Dense, Activation
from keras.models import Sequential

def funcTraining(function):
   
    training_set = np.random.randint(-1000, 1000, 500) 
    y_true = np.array(list(map(function, training_set)))
    
    return (training_set, y_true) 