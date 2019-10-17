"""
This module contains the class for defining a Neural Net model
"""

class Model(model): 
    
    def __init__(self, model):
        self.model = model 
        
    """
    Figure out how many weights there are and return them in a numpy array 
    """
    def return_weights(): 
        for idx, x in enumerate(model.get_weights()):
            if idx % 2:
                pass 
            else: 
                # Number of weights we'll take from the individual for this layer
                num_weights_taken += np.prod(x.shape)
        return num_weights_shape