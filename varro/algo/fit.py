import pickle
import numpy as np
from deap import base, creator
from dowel import logger
import time

from varro.algo.problems import Problem, ProblemFuncApprox, ProblemMNIST
from varro.algo.strategies.es.evolve import evolve
from varro.algo.strategies.sga import StrategySGA
from varro.algo.strategies.moga import StrategyMOGA
from varro.algo.strategies.ns_es import StrategyNSES
from varro.algo.strategies.nsr_es import StrategyNSRES


def fit(model_type,
        problem_type,
        strategy,
        cxpb=None,
        mutpb=None,
        imutpb=None,
        imutpb_decay=None,
        imutmu=None,
        imutsigma=None,
        popsize=None,
        elitesize=None,
        ngen=None,
        ckpt=None,
        ckpt_freq=10,
        novelty_metric=None,
        halloffamesize=None,
        earlystop=False,
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
	imutpb (float): Generational decay rate for imutpb
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
    # 1. Choose Problem and get the specific evaluation function for that problem

    logger.start_timer()
    logger.log("Loading problem...")
    if problem_type == 'mnist':
        problem = ProblemMNIST()
    else:
        problem = ProblemFuncApprox(func=problem_type)

    logger.stop_timer('FIT.PY Choosing problem and getting specific evaluation function')
    logger.start_timer()

    # 2. Choose Target Platform
    logger.log("Loading target platform...")
    if model_type == 'nn':
        from varro.algo.models import ModelNN as Model  # Import here so we don't load tensorflow if not needed
    elif model_type == 'fpga':
        from varro.algo.models import ModelFPGA as Model
    model = Model(problem)

    logger.stop_timer('FIT.PY Loading target platform')
    logger.start_timer()

    strategy_args = {
	    'novelty_metric' : novelty_metric,
            'model' : model,
            'problem' : problem,
            'cxpb' : cxpb,
            'mutpb' : mutpb,
            'popsize' : popsize,
            'elitesize' : elitesize,
            'ngen' : ngen,
            'imutpb' : imutpb,
            'imutpb_decay' : imutpb_decay,
            'imutmu' : imutmu,
            'imutsigma' : imutsigma,
            'ckpt' : ckpt,
            'halloffamesize' : halloffamesize,
            'earlystop' : earlystop,
	    'novelty_metric' : novelty_metric,
    }

    # 3. Set Strategy
    logger.log("Loading strategy...")
    if strategy == 'sga':
        strategy = StrategySGA(**strategy_args)
    elif strategy == 'moga':
        strategy = StrategyMOGA(**strategy_argsp)
    elif strategy == 'ns-es':
        strategy = StrategyNSES(**strategy_args)
    elif strategy == 'nsr-es':
        strategy = StrategyNSRES(**strategy_args)
    elif strategy == 'cma-es':
        strategy = StrategyCMAES(**strategy_args)
    else:
        raise NotImplementedError

    logger.start_timer()
    # 4. Evolve
    pop, avg_fitness_scores, fittest_ind_score = evolve(strategy=strategy, grid_search=grid_search, ckpt_freq=ckpt_freq)

    logger.stop_timer('FIT.PY Evolving')
    logger.start_timer()


    return fittest_ind_score
