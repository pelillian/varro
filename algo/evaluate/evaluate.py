"""
This module contains the class for Evaluate, with one constructor for FPGA and one for Neural Networks
"""

from algo.evaluate.util import load_weights


def evaluate_mnist_nn(population, model, X, y):
    """Evaluates an entire population on the mnist dataset on the neural net
    architecture specified by the model, and calculates the negative categorical
    accuracy of each individual
    
    Args:
        population: A list of np.ndarrays that represent the individuals
            - e.g. [np.array([0.93, 0.85, 0.24, ..., 0.19]),
            		np.array([0.93, 0.85, 0.24, ..., 0.19]),
            		...,
            		np.array([0.93, 0.85, 0.24, ..., 0.19])]
        model (keras model): Model to be used for the neural network
        X: Training Input for mnist
        y: Training Labels for mnist

    
    Returns:
        An np.ndarray of the fitness scores of the individuals

    """
    # Initialize list to keep the fitness scores
    fitness_scores = []

    # Get number of classes for mnist (10)
    num_classes = len(np.unique(y)) 

    # Get input dimension of the flattened mnist image
    input_dim = np.prod(X[0].shape) 

    # Flatten the MNIST images into a 784 dimension vector
    flattened_x_train = np.array([x.flatten() for x in x_train])

    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(y=y, num_classes=num_classes)
    
    # Get fitness score for each individual in population
    for individual in population:
    	
    	# Load Weights into model using individual
    	model = load_weights(individual, model)
    	
    	# Predict labels
    	y_pred = np.array(model.predict(flattened_x_train))

	    # Calculate the categorical accuracy
	    categorical_accuracy = tf.Session().run(\
	        K.mean(K.equal(K.argmax(one_hot_labels, axis=-1), K.argmax(y_pred, axis=-1))))

	    fitness_scores.append(-categorical_accuracy)
           
    return np.array(fitness_scores)


def evaluate_func_approx_nn(population, model, X, y):
	"""Evaluates an entire population on X on the neural net
    architecture specified by the model, and calculates the mean 
    squared error of each individual
    
    Args:
        population: A list of np.ndarrays that represent the individuals
            - e.g. [np.array([0.93, 0.85, 0.24, ..., 0.19]),
            		np.array([0.93, 0.85, 0.24, ..., 0.19]),
            		...,
            		np.array([0.93, 0.85, 0.24, ..., 0.19])]
        model (keras model): Model to be used for the neural network
        X: Training Input for the neural network
        y: Training Labels for the neural network

    
    Returns:
        An np.ndarray of the fitness scores of the individuals

    """
	# Initialize list to keep the fitness scores
    fitness_scores = []

	# Get fitness score for each individual in population
    for individual in population:
    	
    	# Load Weights into model using individual
    	model = load_weights(individual, model)
    	
    	# Predict labels
    	y_pred = np.array(model.predict(X))

	    # Get the mean squared error of the 
	    # individual
	    mse = np.mean(np.square(y - y_pred))

	    fitness_scores.append(mse)
           
    return np.array(fitness_scores)


def evaluate_mnist_fpga(population, X, y):
	"""Evaluates an entire population on the mnist dataset on the neural net
    architecture specified by the model, and calculates the negative categorical
    accuracy of each individual
    
    Args:
        population: A list of boolean np.ndarrays that represent the individuals
            - e.g. [np.array([0, 1, 0, ..., 0]),
            		np.array([1, 0, 0, ..., 1]),
            		...,
            		np.array([1, 1, 0, ..., 0])]
        X: Training Input for mnist
        y: Training Labels for mnist

    
    Returns:
        An np.ndarray of the fitness scores of the individuals

    """
    # Initialize list to keep the fitness scores
    fitness_scores = []

    # Get number of classes for mnist (10)
    num_classes = len(np.unique(y)) 

    # Get input dimension of the flattened mnist image
    input_dim = np.prod(X[0].shape) 

    # Flatten the MNIST images into a 784 dimension vector
    flattened_x_train = np.array([x.flatten() for x in x_train])

    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(y=y, num_classes=num_classes)
    
    # TODO
    # Get fitness score for each individual in population
    fitness_scores = flash_fpga(problem='mnist', flattened_x_train)
           
    return np.array(fitness_scores)


def evaluate_func_approx_fpga(population, X, y):
	"""Evaluates an entire population on X on the neural net
    architecture specified by the model, and calculates the mean 
    squared error of each individual
    
    Args:
        population: A list of np.ndarrays that represent the individuals
            - e.g. [np.array([0.93, 0.85, 0.24, ..., 0.19]),
            		np.array([0.93, 0.85, 0.24, ..., 0.19]),
            		...,
            		np.array([0.93, 0.85, 0.24, ..., 0.19])]
        model (keras model): Model to be used for the neural network
        X: Training Input for the neural network
        y: Training Labels for the neural network

    
    Returns:
        An np.ndarray of the fitness scores of the individuals

    """
	# Initialize list to keep the fitness scores
    fitness_scores = []

	# TODO
    # Get fitness score for each individual in population
    fitness_scores = flash_fpga(problem='func_approx', flattened_x_train)
           
    return np.array(fitness_scores)