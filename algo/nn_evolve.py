"""
This module contains code for testing the evolutionary algorithm on a neural network.
"""

import numpy as np
import keras 
from keras.layers import Dense, Activation
from keras.models import Sequential


def evaluate_neural_network(individual, function=np.sin):
    """Loads an individual (list) as the weights of neural net and computes the
    Mean Squared Error of the neural net with the given weights and provided
    approximating function
    
    Args:
        individual: An individual (represented by list of floats) 
            - e.g. [0.93, 0.85, 0.24, ..., 0.19], ...}
        function: Function to be approximated by neural net
    
    Returns:
        A single scalar of the Mean Squared Error, representing fitness of the individual

    """
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
