"""
This module contains the evolutionary algorithm logic
"""

import os
import json
import pickle
import random
import logging
import numpy as np
import functools
from tqdm import tqdm
from deap import base, creator, tools
from datetime import datetime

from varro.misc.util import make_path
from varro.misc.variables import ABS_ALGO_EXP_LOGS_PATH, EXPERIMENT_CHECKPOINTS_PATH, GRID_SEARCH_CHECKPOINTS_PATH, FREQ
from varro.algo.problems import Problem


def mate(pop, cxpb, toolbox):
    """Mates individuals in the population using the scheme
    defined in toolbox in-place

    Args:
        pop (list: Individual): List of individuals to be mated
        cxpb (float): Crossover probability from 0-1
        toolbox (deap.ToolBox): DEAP's configured toolbox
    """
    # Apply crossover on the population by
    # choosing alternate individuals
    # e.g. if pop = [ind1, ind2, ind3, ind4],
    # we are doing 2-point crossover between
    # ind1, ind3 and ind2, ind4
    for child1, child2 in zip(pop[::2], pop[1::2]):
        if random.random() < cxpb:

            # In-place Crossover
            toolbox.mate(child1, child2)

            # Delete fitness values after crossover
            # because the individuals are changed
            # and will have different fitness values
            del child1.fitness.values
            del child2.fitness.values


def mutate(pop, mutpb, toolbox):
    """Mutates individuals in the population using the scheme
    defined in toolbox in-place

    Args:
        pop (list: Individual): List of individuals to be mutated
        mutpb (float): Mutation probability from 0-1
        toolbox (deap.ToolBox): DEAP's configured toolbox
    """
    # Apply mutation
    for mutant in pop:
        if random.random() < mutpb:

            # In-place Mutation
            toolbox.mutate(mutant)

            # Delete fitness values after crossover
            # because the individuals are changed
            # and will have different fitness values
            del mutant.fitness.values


def evolve(strategy,
           logs_path=ABS_ALGO_EXP_LOGS_PATH,
           ckpts_path=EXPERIMENT_CHECKPOINTS_PATH,
           grid_search=False):
    """Evolves parameters to train a model on a dataset.

    Args:
        strategy (Strategy): The strategy to be used for evolving, Simple Genetic Algorithm (sga) / Novelty Search (ns) / Covariance-Matrix Adaptation (cma-es)
        logs_path (str): Path to the experiment logs directory
        ckpts_path (str): Path to the checkpoints directory
        grid_search (bool): Whether grid search will be in effect

    Returns:
        pop: Population of the fittest individuals so far
        avg_fitness_scores: A list of the average fitness scores for each generation

    """
    ########################################################
    # 1. SET UP LOGGER, FOLDERS, AND FILES TO SAVE DATA TO #
    ########################################################

    # Set log and checkpoint dirs
    if grid_search:
        experiment_checkpoints_dir = os.path.join(GRID_SEARCH_CHECKPOINTS_PATH, 'tmp')
    else:
        experiment_checkpoints_dir = os.path.join(EXPERIMENT_CHECKPOINTS_PATH, strategy.problem.name + '_' + datetime.now().strftime("%b-%d-%Y-%H:%M:%S"))
    experiment_logs_file = os.path.join(ABS_ALGO_EXP_LOGS_PATH, strategy.problem.name + '_' + datetime.now().strftime("%b-%d-%Y-%H:%M:%S") + '.log')

    # Create experiment folder to store
    # snapshots of population
    make_path(ABS_ALGO_EXP_LOGS_PATH)
    make_path(experiment_checkpoints_dir)

    # Set Logging configuration
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=experiment_logs_file,
                        level=logging.INFO,
                        format=log_fmt)

    # Get logger
    logger = logging.getLogger(__name__)
    logger.info('Start Evolution ...')
    logger.info('strategy: {}'.format(strategy.name))
    logger.info('problem_type: {}'.format(strategy.problem.name))
    logger.info('cxpb: {}'.format(strategy.cxpb))
    logger.info('mutpb: {}'.format(strategy.mutpb))
    logger.info('popsize: {}'.format(strategy.popsize))
    logger.info('elitesize: {}'.format(strategy.elitesize))
    logger.info('ngen: {}'.format(strategy.ngen))
    logger.info('imutpb: {}'.format(strategy.imutpb))
    logger.info('imutmu: {}'.format(strategy.imutmu))
    logger.info('imutsigma: {}'.format(strategy.imutsigma))
    logger.info('halloffamesize: {}'.format(strategy.halloffamesize))

    # Set additional logging information about experiment
    # if not simple genetic algorithm strategy
    if strategy.name == 'ns-es' or strategy.name == 'nsr-es':
        logger.info('novelty_metric: {}'.format(strategy.novelty_metric))

    ###############################
    # 2. CURRENT POPULATION STATS #
    ###############################
    # Track the Average fitness scores
    avg_fitness_scores = []

    # Evaluate the entire population
    avg_fitness_score = strategy.toolbox.evaluate(pop=strategy.pop)
    avg_fitness_scores.append(avg_fitness_score)

    #################################
    # 4. EVOLVE THROUGH GENERATIONS #
    #################################
    # Iterate for generations
    start_gen = strategy.curr_gen
    for g in tqdm(range(start_gen, strategy.ngen)):

        # Select the next generation individuals
        non_alterable, alterable = strategy.generate_offspring()

        # Mate offspring
        mate(alterable, strategy.cxpb, strategy.toolbox)

        # Mutate offspring
        mutate(alterable, strategy.mutpb, strategy.toolbox)

        # Recombine Non-alterable offspring with the
        # ones that have been mutated / cross-overed
        offspring = non_alterable + alterable

        # Evaluate the entire population
        strategy.curr_gen = g # Set the current generation
        avg_fitness_score = strategy.toolbox.evaluate(pop=offspring)
        avg_fitness_scores.append(avg_fitness_score)

        # Save snapshot of population (offspring)
        if g % FREQ == 0:

            # Save the checkpoint
            strategy.save_ckpt(exp_ckpt_dir=experiment_checkpoints_dir)

        # Best individual's fitness / novelty score,
        # whichever is the first element of the fitness
        # values tuple because:
        # The hall of fame contains the best individual
        # that ever lived in the population during the
        # evolution. It is lexicographically sorted at all
        # time so that the first element of the hall of fame
        # is the individual that has the best first fitness value
        # ever seen, according to the weights provided to the fitness at creation time.
        if strategy.name == 'sga' or strategy.name == 'nsr_es':
            fittest_ind_score = strategy.halloffame[0].fitness.values.fitness_score
        elif strategy.name == 'ns-es':
            fittest_ind_score = strategy.halloffame[0].fitness.values.novelty_score
        else:
            raise NotImplementedError

        # Log Average score of population
        logger.info('Generation {} Avg. Fitness Score: {} | Fittest Individual Score: {}'\
                        .format(g,
                                avg_fitness_score,
                                fittest_ind_score))

        # Early Stopping if average fitness
        # score is close to the minimum possible,
        # or if stuck at local optima (average fitness score
        # hasnt changed for past 10 rounds)
        if strategy.name == 'sga' or strategy.name == 'nsr-es':
            if strategy.problem.approx_type == Problem.CLASSIFICATION:
                if round(-fittest_ind_score, 4) > 0.95:
                    logger.info('Early Stopping activated because Accuracy > 95%.')
                    break;
                if len(avg_fitness_scores) > 10 and len(set(avg_fitness_scores[-10:])) == 1:
                    logger.info('Early Stopping activated because fitness scores have converged.')
                    break;
            else:
                if round(fittest_ind_score, 4) < 0.01:
                    logger.info('Early Stopping activated because MSE < 0.01.')
                    break;
                if len(avg_fitness_scores) > 10 and len(set(avg_fitness_scores[-10:])) == 1:
                    logger.info('Early Stopping activated because fitness scores have converged.')
                    break;


    return pop, avg_fitness_scores
