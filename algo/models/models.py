"""
This module contains the class for defining a Neural Net model
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import keras 
from keras.layers import Dense, Activation
from keras.models import Sequential

class Model(model): 
    
    def __init__(self, model):
        self.model = model 
        self.isize = get_i_size() 
    """
    Figure out how many weights there are and return them in a numpy array 
    """
    def get_i_size(): 
        num_weights = 0
        for idx, x in enumerate(self.model.get_weights()):
            if idx % 2:
                pass 
            else: 
                # Number of weights we'll take from the individual for this layer
                num_weights += np.prod(x.shape)
        return num_weights
    
    