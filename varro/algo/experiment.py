"""
This module contains the main function we'll use to run the 
experiment to solve the problem using a specified 
evolutionary algorithm
"""

from functools import partial
import numpy as np

from varro.misc.util import make_path
from varro.algo.util import get_args
from varro.algo.problems import ProblemFuncApprox, ProblemMNIST
from varro.algo.strategies.ea.evolve import evolve
from varro.algo.strategies.ea.toolbox import ea_toolbox
from varro.algo.evaluate import evaluate

LOGS_PATH = '../logs/algo/'

def fit(model_type,
        problem_type,
        strategy,
        cxpb=None,
        mutpb=None,
        popsize=None,
        ngen=None):
    """Control center to call other modules to execute the optimization

    Args:
        model_type (str): A string specifying whether we're optimizing on a neural network
            or field programmable gate array
        problem_type (str): A string specifying what type of problem we're trying to optimize
        strategy (str): A string specifying what type of optimization algorithm to use
        cxpb (float): Cross-over probability for evolutionary algorithm
        mutpb (float): Mutation probability for evolutionary algorithm
        ngen (int): Number of generations to run an evolutionary algorithm

    """
    # 1. Choose Problem and get the specific evaluation function 
    # for that problem
    if problem_type == 'mnist':
        problem = ProblemMNIST()
    else:
        problem = ProblemFuncApprox(problem_type)

    # 2. Choose Target Platform
    # Neural Network
    if model_type == 'nn':
        from varro.algo.models import ModelNN  # Import here so we don't load tensorflow if not needed
        model = ModelNN(problem)
    elif model_type == 'fpga':
        from varro.algo.models import ModelFPGA
        model = ModelFPGA()

    evaluate_population = partial(evaluate, model=model, X=problem.X_train, y=problem.y_train, approx_type=problem.approx_type)

    # Set the individual size to the number of weights
    # we can alter in the neural network architecture specified
    toolbox = ea_toolbox(i_size=model.weights_shape,
                            evaluate_population=evaluate_population,
                            model_type=model_type)

    # 3. Choose Strategy
    if strategy == 'ea':
        pop, avg_fitness_scores = evolve(problem=problem_type,
                                         toolbox=toolbox,
                                         crossover_prob=cxpb,
                                         mutation_prob=mutpb,
                                         pop_size=popsize,
                                         num_generations=ngen)
    elif strategy == 'cma-es':
        raise NotImplementedError
    elif strategy == 'ns':
        raise NotImplementedError
    else:
        raise NotImplementedError


def predict(model_type, problem_type):
    raise NotImplementedError


def main():
    # Create Logs folder if not created
    make_path(LOGS_PATH)

    # Get the Arguments parsed from file execution
    args = get_args()

    # Start Optimization
    fit(model_type=args.model_type,
        problem_type=args.problem_type,
        strategy=args.strategy,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        popsize=args.popsize,
        ngen=args.ngen)

if __name__ == "__main__":
    main()

