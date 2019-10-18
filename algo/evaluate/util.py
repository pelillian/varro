"""
This module contains the utility functions needed for the evaluation functions in evaluate.py
"""

import numpy as np


def load_weights(individual, model):
    """Reshapes individual as weights of the neural net architecture
    prespecified

    Args:
        individual: An individual (represented by an np.ndarray of floats) 
            - e.g. [0.93, 0.85, 0.24, ..., 0.19], ...}

        function: Reshapes individuals to weights of a neural net

    Returns:
        An np.ndarray of the fitness score(s) of the individual
            - e.g. [Mean Squared Error]
    """

    # Pull out the numbers from the individual and
    # load them as the shape from the model's weights
    ind_idx = 0
    new_weights = []
    for idx, x in enumerate(model.get_weights()):
        if idx % 2:
            new_weights.append(x)
        else: 
            # Number of weights we'll take from the individual for this layer
            num_weights_taken = np.prod(x.shape)
            new_weights.append(individual[ind_idx:ind_idx+num_weights_taken].reshape(x.shape))
            ind_idx += num_weights_taken

    # Set Weights using individual
    model.set_weights(new_weights)

    return model