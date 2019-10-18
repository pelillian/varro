"""
This module contains the class for Evaluate, with one constructor for FPGA and one for Neural Networks
"""

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

    
        
    # Set Weights using individual
    model.set_weights(new_weights)
    y_pred = np.array(model.predict(flattened_x_train))

    # Calculate the categorical accuracy
    categorical_accuracy = tf.Session().run(\
        K.mean(K.equal(K.argmax(one_hot_labels, axis=-1), K.argmax(y_pred, axis=-1))))
           
    return np.array([-categorical_accuracy])