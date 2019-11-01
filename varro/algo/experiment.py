"""
This module contains the main function we'll use to run the
experiment to solve the problem using a specified
evolutionary algorithm
"""

import os
from functools import partial
import numpy as np
import pickle
import logging
from deap import base, creator

from varro.misc.util import make_path
from varro.misc.variables import ABS_ALGO_EXP_LOGS_PATH, ABS_ALGO_HYPERPARAMS_PATH
from varro.algo.util import get_args
from varro.algo.problems import Problem, ProblemFuncApprox, ProblemMNIST
from varro.algo.strategies.ea.evolve import evolve
from varro.algo.strategies.ea.toolbox import ea_toolbox
from varro.algo.evaluate import evaluate


def fit(model_type,
        problem_type,
        strategy,
        cxpb=None,
        mutpb=None,
        imutpb=None,
        imutmu=None,
        imutsigma=None,
        popsize=None,
        elitesize=None,
        ngen=None,
        ckpt=None):
    """Control center to call other modules to execute the optimization

    Args:
        model_type (str): A string specifying whether we're optimizing on a neural network
            or field programmable gate array
        problem_type (str): A string specifying what type of problem we're trying to optimize
        strategy (str): A string specifying what type of optimization algorithm to use
        cxpb (float): Cross-over probability for evolutionary algorithm
        mutpb (float): Mutation probability for evolutionary algorithm
        imutpb (float): Mutation probability for each individual's attribute
        imutmu (float): Mean parameter for the Gaussian Distribution we're mutating an attribute from
        imutsigma (float): Sigma parameter for the Gaussian Distribution we're mutating an attribute from
        popsize (int): Number of individuals to keep in each Population
        elitesize (float): Percentage of fittest individuals to pass on to next generation
        ngen (int): Number of generations to run an evolutionary algorithm
        ckpt (str): Location of checkpoint to load the population

    """
    # 1. Choose Problem and get the specific evaluation function
    # for that problem
    if problem_type == 'mnist':
        problem = ProblemMNIST()
    else:
        problem = ProblemFuncApprox(func=problem_type)

    # 2. Choose Target Platform
    # Neural Network
    if model_type == 'nn':
        from varro.algo.models import ModelNN  # Import here so we don't load tensorflow if not needed
        model = ModelNN(problem)
    elif model_type == 'fpga':
        from varro.algo.models import ModelFPGA
        model = ModelFPGA()

    # Evaluate Population
    evaluate_population = partial(evaluate,
                                  model=model,
                                  problem=problem)

    # Set the individual size to the number of weights
    # we can alter in the neural network architecture specified
    toolbox = ea_toolbox(i_shape=model.weights_shape,
                         evaluate_population=evaluate_population,
                         model_type=model_type,
                         imutpb=imutpb,
                         imutmu=imutmu,
                         imutsigma=imutsigma)

    # 3. Choose Strategy
    if strategy == 'ea':
        pop, avg_fitness_scores = evolve(problem=problem,
                                         toolbox=toolbox,
                                         crossover_prob=cxpb,
                                         mutation_prob=mutpb,
                                         pop_size=popsize,
                                         elite_size=elitesize,
                                         num_generations=ngen,
                                         imutpb=imutpb,
                                         imutmu=imutmu,
                                         imutsigma=imutsigma,
                                         checkpoint=ckpt)
    elif strategy == 'cma-es':
        raise NotImplementedError
    elif strategy == 'ns':
        raise NotImplementedError
    else:
        raise NotImplementedError


def predict(model_type,
            problem_type,
            X,
            ckpt):
    """Predicts the output from loading the model saved in checkpoint
    and saves y_pred into same path as X but with a _y_pred in the name

    Args:
        model_type (str): A string specifying whether we're optimizing on a neural network
            or field programmable gate array
        problem_type (str): A string specifying what type of problem we're trying to optimize
        X (str): Path to the .npy that stores the np.ndarray to use as Input data for model
        ckpt (str): Location of checkpoint to load the population

    """
    # Get logger
    logger = logging.getLogger(__name__)

    # 1. Choose Problem and get the specific evaluation function
    # for that problem
    if problem_type == 'mnist':
        problem = ProblemMNIST()
    else:
        problem = ProblemFuncApprox(func=problem_type)

    # 1. Choose Target Platform
    # Neural Network
    if model_type == 'nn':
        from varro.algo.models import ModelNN  # Import here so we don't load tensorflow if not needed
        model = ModelNN(problem)
    elif model_type == 'fpga':
        from varro.algo.models import ModelFPGA
        model = ModelFPGA()

    # Load data from pickle file
    with open(ckpt, "rb") as cp_file:
        # Define objective, individuals, population, and evaluation
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        cp = pickle.load(cp_file)
    halloffame = cp["halloffame"] # Load the best individual in the population

    # Load Weights into model using individual
    model.load_weights(halloffame)

    # Predict labels using np array in X
    y_pred = np.array(model.predict(np.load(X)))

    # Save the y_pred into a file
    np.save(os.path.join(X[:-4] + '_y_pred.npy'), y_pred)
    logger.info('Predictions saved in {}!'.format(os.path.join(X[:-4] + '_y_pred.npy')))


def main():
    # Create Logs folder if not created
    make_path(ABS_ALGO_EXP_LOGS_PATH)
    make_path(ABS_ALGO_HYPERPARAMS_PATH)

    # Get the Arguments parsed from file execution
    args = get_args()

    # Check if we're fitting or predicting
    if args.purpose == 'fit':

        # Start Optimization
        fit(model_type=args.model_type,
            problem_type=args.problem_type,
            strategy=args.strategy,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            imutpb=args.imutpb,
            imutmu=args.imutmu,
            imutsigma=args.imutsigma,
            popsize=args.popsize,
            elitesize=args.elitesize,
            ngen=args.ngen,
            ckpt=args.ckpt)

    else:

        # Make prediction
        predict(model_type=args.model_type,
                problem_type=args.problem_type,
                X=args.X,
                ckpt=args.ckpt)

if __name__ == "__main__":
    main()
