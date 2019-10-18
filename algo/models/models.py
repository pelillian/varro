"""
This module contains the class for defining a Neural Net model
"""
import numpy as np
import keras 
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf

def get_model(problem):
    """Creates the neural network architecture specific to the
    problem to optimize

    Args:
        problem (str): String specifying the type of problem 
        we're dealing with

    Returns:
        A keras model with architecture specified for the problem
    """
    if problem == 'mnist':

        # Basic Neural net model for MNIST
        model = Sequential() 
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    
    else:

        # Basic Neural net model for approximating functions
        model = Sequential() 
        model.add(Dense(1, input_dim=1, activation='relu'))
        model.add(Dense(3, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    return model
    