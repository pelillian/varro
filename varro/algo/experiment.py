"""
This module contains the main function we'll use to run the
experiment to solve the problem using a specified
evolutionary algorithm
"""

import pickle
from os import listdir
from os.path import isfile, join
from functools import partial
from tqdm import tqdm
import numpy as np
import logging
from deap import base, creator

from varro.misc.util import make_path
from varro.misc.variables import ABS_ALGO_EXP_LOGS_PATH, ABS_ALGO_HYPERPARAMS_PATH, ABS_ALGO_PREDICTIONS_PATH
from varro.algo.util import get_args
from varro.algo.problems import Problem, ProblemFuncApprox, ProblemMNIST
from varro.algo.strategies.es.evolve import evolve
from varro.algo.strategies.sga import StrategySGA
from varro.algo.strategies.ns_es import StrategyNSES
from varro.algo.strategies.nsr_es import StrategyNSRES


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
        ckpt=None,
        novelty_metric=None,
        halloffamesize=None,
        grid_search=False):
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
        novelty_metric (str): The distance metric to be used to measure an Individual's novelty
        halloffamesize (float): Percentage of individuals in population we store in the HallOfFame / Archive
        grid_search (bool): Whether grid search will be in effect

    Returns:
        fittest_ind_score: Scalar of the best individual in the population's fitness score

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
        if grid_search:
            model = ModelNN(problem, tensorboard_logs=False)
        else:
            model = ModelNN(problem)
    elif model_type == 'fpga':
        from varro.algo.models import ModelFPGA
        model = ModelFPGA()

    # 3. Set Strategy
    if strategy == 'sga':
        strategy = StrategySGA(model=model,
                               problem=problem,
                               cxpb=cxpb,
                               mutpb=mutpb,
                               popsize=popsize,
                               elitesize=elitesize,
                               ngen=ngen,
                               imutpb=imutpb,
                               imutmu=imutmu,
                               imutsigma=imutsigma,
                               ckpt=ckpt,
                               halloffamesize=halloffamesize)
    elif strategy == 'ns-es':
        strategy = StrategyNSES(novelty_metric=novelty_metric,
                                model=model,
                                problem=problem,
                                cxpb=cxpb,
                                mutpb=mutpb,
                                popsize=popsize,
                                elitesize=elitesize,
                                ngen=ngen,
                                imutpb=imutpb,
                                imutmu=imutmu,
                                imutsigma=imutsigma,
                                ckpt=ckpt,
                                halloffamesize=halloffamesize)
    elif strategy == 'nsr-es':
        strategy = StrategyNSRES(novelty_metric=novelty_metric,
                                 model=model,
                                 problem=problem,
                                 cxpb=cxpb,
                                 mutpb=mutpb,
                                 popsize=popsize,
                                 elitesize=elitesize,
                                 ngen=ngen,
                                 imutpb=imutpb,
                                 imutmu=imutmu,
                                 imutsigma=imutsigma,
                                 ckpt=ckpt,
                                 halloffamesize=halloffamesize)
    elif strategy == 'cma-es':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # 4. Evolve
    pop, avg_fitness_scores, fittest_ind_score = evolve(strategy=strategy,
                                                        grid_search=grid_search)

    return fittest_ind_score


def predict(model_type,
            problem_type,
            strategy,
            X,
            ckpt,
            save_dir):
    """Predicts the output from loading the model saved in checkpoint
    and saves y_pred into same path as X but with a _y_pred in the name

    Args:
        model_type (str): A string specifying whether we're optimizing on a neural network
            or field programmable gate array
        problem_type (str): A string specifying what type of problem we're trying to optimize
        strategy (str): A string specifying what type of optimization algorithm to use
        X (str): Path to the .npy that stores the np.ndarray to use as Input data for model
        ckpt (str): Location of checkpoint to load the population
        save_dir (str): Location of where to store the predictions

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
    # The hall of fame contains the best individual
    # that ever lived in the population during the
    # evolution. It is lexicographically sorted at all
    # time so that the first element of the hall of fame
    # is the individual that has the best first fitness value
    # ever seen, according to the weights provided to the fitness at creation time.
    with open(ckpt, "rb") as cp_file:
        if strategy == 'sga':
            StrategySGA.init_fitness_and_inds()
        elif strategy == 'ns-es':
            StrategyNSES.init_fitness_and_inds()
        elif strategy == 'nsr-es':
            StrategyNSRES.init_fitness_and_inds()
        elif strategy == 'cma-es':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Initialize individual based on strategy
        cp = pickle.load(cp_file)
        best_ind = cp["halloffame"][0]

    # Load Weights into model using individual
    model.load_parameters(best_ind)

    # Predict labels using np array in X
    y_pred = np.array(model.predict(np.load(X)))

    # Save the y_pred into a file
    y_pred_path = join(save_dir, ckpt.split('_')[-1][:-4] + '_' + X[:-4].split('/')[-1] + '_y_pred.npy')
    np.save(y_pred_path, y_pred)
    logger.info('Predictions saved in {}!'.format(y_pred_path))


def main():
    # Create Logs folder if not created
    make_path(ABS_ALGO_EXP_LOGS_PATH)
    make_path(ABS_ALGO_HYPERPARAMS_PATH)
    make_path(ABS_ALGO_PREDICTIONS_PATH)

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
            ckpt=args.ckpt,
            novelty_metric=args.novelty_metric,
            halloffamesize=args.halloffamesize)

    else:

        if args.ckptfolder:
            # Make predictions using the best
            # individual from each generation
            # in ckptfolder
            save_dir = join(ABS_ALGO_PREDICTIONS_PATH, args.ckptfolder.split('/')[-1])
            make_path(save_dir)
            ckpt_files = [join(args.ckptfolder, f) for f in listdir(args.ckptfolder) if isfile(join(args.ckptfolder, f))]
            for ckpt in tqdm(ckpt_files):
                predict(model_type=args.model_type,
                        problem_type=args.problem_type,
                        strategy=args.strategy,
                        X=args.X,
                        ckpt=ckpt,
                        save_dir=save_dir)
        else:
            # Make a single prediction
            save_dir = join(ABS_ALGO_PREDICTIONS_PATH, args.ckpt.split('/')[-2])
            make_path(save_dir)
            predict(model_type=args.model_type,
                    problem_type=args.problem_type,
                    strategy=args.strategy,
                    X=args.X,
                    ckpt=args.ckpt,
                    save_dir=save_dir)

if __name__ == "__main__":
    main()
