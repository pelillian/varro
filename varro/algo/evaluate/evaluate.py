"""
This module contains the class for Evaluate, with one constructor for FPGA and one for Neural Networks
"""

import numpy as np
import keras
from sklearn.metrics import accuracy_score

from varro.algo.problems import Problem
from varro.fpga.interface import FpgaConfig


def evaluate(population, model, X, y, approx_type):
	"""Evaluates an entire population on a dataset on the neural net
	architecture specified by the model, and calculates the negative categorical
	accuracy of each individual.

	Args:
		population: An iterable of np.ndarrays that represent the individuals
		model: Model to be used for evaluation
		X: Training Input
		y: Training Labels

	Returns:
		(np.ndarray): the fitness scores (list) of the individuals

	"""
	# Initialize list to keep the fitness scores
	fitness_scores = []

	# Flatten the MNIST images into a 784 dimension vector
	flattened_X = np.array([x.flatten() for x in X])

	# Get fitness score for each individual in population
	for individual in population:

		# Load Weights into model using individual
		model.load_weights(individual)

		# Predict labels
		y_pred = np.array(model.predict(flattened_X))

		if approx_type == Problem.CLASSIFICATION:
			categorical_accuracy = accuracy_score(y_true=y, y_pred=np.argmax(y_pred, axis=-1))
			fitness_scores.append([-categorical_accuracy])
		elif approx_type == Problem.REGRESSION:
			mse = np.mean(np.square(y - y_pred))
			fitness_scores.append([mse])
		else:
			raise ValueError('Unknown approximation type ' + str(problem.approx_type))


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
		An np.ndarray of the fitness scores (list) of the individuals

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

		fitness_scores.append([mse])

	return np.array(fitness_scores)


def evaluate_mnist_fpga(population, X, y):
	"""Evaluates an entire population on the mnist dataset on the FPGA
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
		An np.ndarray of the fitness scores (list) of the individuals

	"""
	# Initialize list to keep the fitness scores
	fitness_scores = []

	# Get number of classes for mnist (10)
	num_classes = len(np.unique(y))

	# Get input dimension of the flattened mnist image
	input_dim = np.prod(X[0].shape)

	# Flatten the MNIST images into a 784 dimension vector
	flattened_X = np.array([x.flatten() for x in X])

	# Get fitness score for each individual in population
	for individual in population:
		config = FpgaConfig(individual)
		config.flash()

		y_pred = config.evaluate(flattened_X)

		categorical_accuracy = accuracy_score(y_true=y, y_pred=np.argmax(y_pred, axis=-1))
		fitness_scores.append([-categorical_accuracy])

	return np.array(fitness_scores)


def evaluate_func_approx_fpga(population, X, y):
	"""Evaluates an entire population on X on the FPGA
	architecture specified by the model, and calculates the mean
	squared error of each individual

	Args:
		population: A list of np.ndarrays that represent the individuals
			- e.g. [np.array([0, 1, 0, ..., 0]),
					np.array([1, 0, 0, ..., 1]),
					...,
					np.array([1, 1, 0, ..., 0])]
		model (keras model): Model to be used for the neural network
		X: Training Input
		y: Training Labels


	Returns:
		An np.ndarray of the fitness scores (list) of the individuals

	"""
	# Initialize list to keep the fitness scores
	fitness_scores = []

	# Get fitness score for each individual in population
	for individual in population:
		config = FpgaConfig(individual)
		config.flash()

		y_pred = config.evaluate(X)

		mse = np.mean(np.square(y - y_pred))
		fitness_scores.append([mse])

	return np.array(fitness_scores)

