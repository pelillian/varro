"""
This module contains the utility functions to run the experiment.py
"""

import os
import argparse
from functools import partial
import numpy as np

from algo.problems import func_approx, mnist
from algo.models.models import get_model
from algo.strategies.ea.evolve import evolve
from algo.strategies.ea.toolbox import nn_toolbox, fpga_toolbox
from algo.evaluate.evaluate import evaluate_mnist_nn, evaluate_func_approx_nn, evaluate_mnist_fpga, evaluate_func_approx_fpga

FPGA_BITSTREAM_SHAPE = (13294, 1136)


def mkdir(path):
	"""Creates new folder

	"""
	try: 
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise
		else:
			print("(%s) already exists" % (path))


def get_args():
	"""Reads command-line arguments.
	
	Returns:
		(Namespace) Parsed attributes

	"""
	parser = argparse.ArgumentParser(
		description='Runs an evolutionary algorithm to optimize \
			the weights of a neural network or circuit configuration of an FPGA'
	)

	##########################
	# 1. FPGA or Neural Net? #
	##########################
	parser.add_argument('--target', 
						default='nn',
						const='nn',
						nargs='?',
						metavar='TARGET-TO-OPTIMIZE', 
						action='store', 
						choices=['fpga', 'nn'], 
						help='The target platform that the parameters are evaluated on')

	######################################################
	# 2. What problem are we trying to solve / optimize? #
	######################################################
	parser.add_argument('--problem', 
						default='sinx',
						const='sinx',
						nargs='?',
						metavar='PROBLEM-TO-TACKLE', 
						action='store', 
						choices=['x', 'sinx', 'cosx', 'tanx', 'ras', 'rosen', 'mnist'], 
						help='The problem to solve / optimize using an evolutionary strategy')

	##########################################################
	# 3. Which strategy should we use to solve this problem? #
	##########################################################
	parser.add_argument('--strategy', 
						default='ea',
						const='ea',
						nargs='?',
						metavar='OPTIMIZATION-STRATEGY', 
						action='store', 
						choices=['ea', 'cma-es', 'ns'], 
						help='The optimization strategy chosen to solve the problem specified')

	##########################################################################
	# 3a. What cross-over probability do you want for the evolutionary algo? #
	##########################################################################
	parser.add_argument('--cxpb',
						default=0.5,
						const=0.5,
						nargs='?',
						metavar='CROSSOVER-PROBABILITY', 
						action='store', 
						help='Set the Cross-over probability for offspring', 
						type=float)

	########################################################################
	# 3b. What mutation probability do you want for the evolutionary algo? #
	########################################################################
	parser.add_argument('--mutpb',
						default=0.2,
						const=0.2,
						nargs='?',
						metavar='MUTATION-PROBABILITY', 
						action='store', 
						help='Set the Mutation probability', 
						type=float)

	#########################################
	# 3c. What population size do you want? #
	#########################################
	parser.add_argument('--popsize', 
						default=10,
						const=10,
						nargs='?',
						metavar='POPULATION-SIZE', 
						action='store', 
						help='Set number of individuals in population', 
						type=int)

	################################################################################
	# 3d. What number of generations do you want to run the evolutionary algo for? #
	################################################################################
	parser.add_argument('--ngen', 
						default=100,
						const=100,
						nargs='?',
						metavar='NUMBER-OF-GENERATIONS', 
						action='store', 
						help='Set the number of generations to evolve', 
						type=int)

	settings = parser.parse_args()
	
	return settings


def optimize(target, 
			 problem, 
			 strategy, 
			 cxpb=None, 
			 mutpb=None, 
			 popsize=None,
			 ngen=None):
	"""Control center to call other modules to execute the optimization

	Args:
		target (str): A string specifying whether we're optimizing on a neural network
			or field programmable gate array
		problem (str): A string specifying what type of problem we're trying to optimize
		strategy (str): A string specifying what type of optimization algorithm to use
		cxpb (float): Cross-over probability for evolutionary algorithm
		mutpb (float): Mutation probability for evolutionary algorithm
		ngen (int): Number of generations to run an evolutionary algorithm

	Returns:
		None.
	"""
	# 1. Choose Target Platform
	# Neural Network
	if target == 'nn':

		# 2. Choose Problem and get the specific evaluation function 
		# for that problem
		if problem == 'mnist':

			# Get training set for MNIST
			# and set the evaluation function
			# for the population
			X_train, y_train = mnist.training_set()

			num_classes = len(np.unique(y_train)) # Get number of classes for mnist (10)
			input_dim = np.prod(X_train[0].shape) # Get input dimension of the flattened mnist image

			# Get the neural net architecture
			model, num_weights = get_model(problem, input_dim=input_dim, output_dim=num_classes)

			evaluate_population = partial(evaluate_mnist_nn, 
										  model=model, 
										  X=X_train, 
										  y=y_train)

		else:

			# Get training set for function approximation
			# and set the evaluation function
			# for the population
			X_train, y_train = func_approx.training_set(problem=problem)

			# Get the neural net architecture
			model, num_weights = get_model(problem, input_dim=1, output_dim=1)

			evaluate_population = partial(evaluate_func_approx_nn, 
										  model=model, 
										  X=X_train, 
										  y=y_train)

		# Set the individual size to the number of weights
		# we can alter in the neural network architecture specified
		toolbox = nn_toolbox(i_size=num_weights,
							 evaluate_population=evaluate_population)

	# FPGA
	else:

		# 2. Choose Problem and get the specific evaluation function 
		# for that problem
		if problem == 'mnist':

			# Get training set for MNIST
			# and set the evaluation function
			# for the population
			X_train, y_train = mnist.training_set()
			evaluate_population = partial(evaluate_mnist_fpga, 
										  X=X_train, 
										  y=y_train)

		else:

			# Get training set for function approximation
			# and set the evaluation function
			# for the population
			X_train, y_train = func_approx.training_set(problem=problem)
			evaluate_population = partial(evaluate_func_approx_fpga, 
										  X=X_train, 
										  y=y_train)

		# Set the individual according to the FPGA Bitstream shape
		toolbox = fpga_toolbox(i_shape=FPGA_BITSTREAM_SHAPE,
							   evaluate_population=evaluate_population)

	# 3. Choose Strategy
	if strategy == 'ea':
	  pop, avg_fitness_scores = evolve(toolbox=toolbox,
									   crossover_prob=cxpb,
									   mutation_prob=mutpb,
									   pop_size=popsize,
									   num_generations=ngen)
	elif strategy == 'cma-es':
	  pass
	elif strategy == 'cma-es':
	  pass
	else:
	  pass

