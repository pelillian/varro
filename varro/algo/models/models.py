"""
This module contains the class for defining a Neural Net model
"""
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

def get_nn_model(problem, input_dim, output_dim):
	"""Creates the neural network architecture specific to the
	problem to optimize

	Args:
		problem (str): String specifying the type of problem 
			we're dealing with
		input_dim (int): The input dimension required
		output_dim (int): The output dimension required

	Returns:
		A keras model with architecture specified for the problem,
		and the number of weights we can alter in the neural network
	"""
	if problem == 'mnist':

		# Basic Neural net model for MNIST
		model = Sequential() 
		model.add(Dense(128, input_dim=input_dim, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(output_dim, activation='softmax'))
	
	else:

		# Basic Neural net model for approximating functions
		model = Sequential() 
		model.add(Dense(1, input_dim=input_dim, activation='relu'))
		model.add(Dense(3, activation='relu'))
		model.add(Dense(2, activation='relu'))
		model.add(Dense(output_dim, activation='sigmoid'))

	# Get the number of weights we can alter in
	# the neural net because this will be the
	# size of each individual
	num_weights_alterable = np.sum([np.prod(layer.shape) for layer in model.get_weights()[::2]])

	return model, num_weights_alterable
	
