"""
This module contains the main function we'll use to run the 
experiment to solve the problem using a specified 
evolutionary algorithm
"""

from functools import partial
import numpy as np

from varro.algo.util import get_args, mkdir
from varro.algo.problems import func_approx, ProblemMNIST
from varro.algo.models.models import get_nn_model
from varro.algo.strategies.ea.evolve import evolve
from varro.algo.strategies.ea.toolbox import nn_toolbox, fpga_toolbox
from varro.algo.evaluate.evaluate import evaluate_mnist_nn, evaluate_func_approx_nn, evaluate_mnist_fpga, evaluate_func_approx_fpga

FPGA_BITSTREAM_SHAPE = (13294, 1136)


def optimize(model, 
			 problem, 
			 strategy, 
			 cxpb=None, 
			 mutpb=None, 
			 popsize=None,
			 ngen=None):
	"""Control center to call other modules to execute the optimization

	Args:
		model (str): A string specifying whether we're optimizing on a neural network
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
	if model == 'nn':

		# 2. Choose Problem and get the specific evaluation function 
		# for that problem
		if problem == 'mnist':

			# Get training set for MNIST
			# and set the evaluation function
			# for the population
			mnist = ProblemMNIST()
			X_train, y_train = mnist.training_set()

			# Get the neural net architecture
			model, num_weights = get_nn_model(problem, input_dim=mnist.input_dim, output_dim=mnist.output_dim)

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
			model, num_weights = get_nn_model(problem, input_dim=1, output_dim=1)

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
		pop, avg_fitness_scores = evolve(problem=problem,
										 toolbox=toolbox,
										 crossover_prob=cxpb,
										 mutation_prob=mutpb,
										 pop_size=popsize,
										 num_generations=ngen)
	elif strategy == 'cma-es':
		pass
	elif strategy == 'ns':
		pass
	else:
		pass


def main():
	# Create Logs folder if not created
	mkdir('./algo/logs/')

	# Get the Arguments parsed from file execution
	args = get_args()

	# Start Optimization
	optimize(model=args.model, 
			 problem=args.problem, 
			 strategy=args.strategy, 
			 cxpb=args.cxpb, 
			 mutpb=args.mutpb, 
			 popsize=args.popsize,
			 ngen=args.ngen)

if __name__ == "__main__":
	main()

