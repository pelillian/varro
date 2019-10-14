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

def evaluate_neural_network(individual, function=np.sin):
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

    #Our neural net
    model = Sequential() 
    model.add(Dense(1, input_dim=1, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    #TODO: build a suitable training set of examples for the neural net to test on 
    training_set = np.random.randint(-1000, 1000, 500) 
    correct_outputs = [function(number) for number in training_set]
    
    def load_weights(ind, model):
        ind_idx = 0
        result = []
        for idx, x in enumerate(res):
            if idx % 2:
                result.append(x)
            else: 
                num_weights_taken = x.shape[0]*x.shape[1]
                result.append(ind[ind_idx:ind_idx+num_weights_taken].reshape(x.shape))
                ind_idx += num_weights_taken
        
        # Set Weights using individual
        model.set_weights(result)
        neural_func = model.predict(training_set)
        mse = np.square(correct_outputs - neural_func).mean()
               
        return mse

    return load_weights(individual, model)

def evaluate_fpga(): 
    #TODO: evaluate FPGA board 