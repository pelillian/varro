"""
This module contains code for testing the evolutionary algorithm on a neural network.
"""

import numpy as np
import keras 
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras.backend as K


def evaluate_nn_mnist(individual):
    """Loads an individual (list) as the weights of neural net and computes the
    Negative Categorical Accuracy of the individual
    
    Args:
        individual: An individual (represented by list of floats) 
            - e.g. [0.93, 0.85, 0.24, ..., 0.19], ...}
    
    Returns:
        An np.ndarray of the fitness score(s) of the individual

    """
    # Load mnist data from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = len(np.unique(y_train)) # Get number of classes for mnist (10)
    input_dim = np.prod(x_train[0].shape) # Get input dimension of the flattened mnist image

    # Flatten the MNIST images into a 784 dimension vector
    flattened_x_train = np.array([x.flatten() for x in x_train])

    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(y=y_train, num_classes=num_classes)

    # Basic Neural net model
    model = Sequential() 
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

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
        y_pred = np.array(model.predict(flattened_x_train))

        # Calculate the categorical accuracy
        categorical_accuracy = tf.Session().run(\
            K.mean(K.equal(K.argmax(one_hot_labels, axis=-1), K.argmax(y_pred, axis=-1))))
               
        return np.array([-categorical_accuracy])

    return load_weights(individual, model)
