"""
This module stores the utility functions to be used for the evolutionary algorithms.
"""

#############
# LIBRARIES #
#############
import keras 
import random 
from keras.layers import Dense, Activation
from keras.models import Sequential
import argparse
import numpy as np
import random
import functools
from tqdm import tqdm
from deap import base, creator, tools
from array import array # Use this if speed is an issue
from collections import defaultdict

def get_args():
    '''
    Function:
    ---------
    Utility function to read in arguments when running
    experiment to evolve weights of neural
    network to approximate a function
    
    Parameters:
    -----------
    None.
    
    Returns:
    --------
    (Type: Namespace) that keeps all the attributes parsed
    '''
    parser = argparse.ArgumentParser(
        description='Evolves weights of neural network to approximate a function'
    )
    parser.add_argument('--cxpb',
                        default=0.5,
                        const=0.5,
                        nargs='?',
                        metavar='CROSSOVER-PROBABILITY', 
                        action='store', 
                        help='Set the Cross-over probability for offspring', 
                        type=float)
    parser.add_argument('--mutpb',
                        default=0.2,
                        const=0.2,
                        nargs='?',
                        metavar='MUTATION-PROBABILITY', 
                        action='store', 
                        help='Set the Mutation probability', 
                        type=float)
    parser.add_argument('--ngen', 
                        default=40,
                        const=40,
                        nargs='?',
                        metavar='NUMBER-OF-GENERATIONS', 
                        action='store', 
                        help='Set the Number of Generations to evolve the weights of neural net', 
                        type=float)
    parser.add_argument('--func', 
                        default='sinx',
                        const='sinx',
                        nargs='?',
                        metavar='FUNCTION-TO-APPROXIMATE', 
                        action='store', 
                        choices=['x', 'sinx', 'cosx', 'tanx'], 
                        help='Set function to approximate using evolutionary strategy on neural network weights')
    settings = parser.parse_args()
    
    return settings

def evaluate(individual, function=np.sin):
    '''
    Function:
    ---------
    Loads an individual (list) as the weights
    of neural net, computes the Mean Squared Error of
    neural net with the given weights in approximating 
    function provided
    
    Parameters:
    -----------
    individual: An individual (represented by list of floats) 
        - e.g. [0.93, 0.85, 0.24, ..., 0.19], ...}
    function: Function to be approximated by neural net
    
    Returns:
    --------
    A single scalar of the Mean Squared Error, representing fitness of the individual
    '''
    ###################################
    # TODO: Chris taking care of this #
    ###################################
    
    # FUTURE:
    # flash_ecp5(None)
    model = Sequential() 
    model.add(Dense(1, input_dim=1, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.set_weights(individual)
    neural_func = model.predict(training_set)
    mse = (np.square(correct_outputs - neural_func)).mean(axis=ax)
    
    return (mse)