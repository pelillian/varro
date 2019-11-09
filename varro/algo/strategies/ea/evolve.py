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


def evolve(strategy,
           problem,
           toolbox,
           crossover_prob,
           mutation_prob,
           pop_size,
           elite_size,
           num_generations,
           imutpb,
           imutmu,
           imutsigma,
           checkpoint=None,
           logs_path=ABS_ALGO_EXP_LOGS_PATH,
           ckpts_path=EXPERIMENT_CHECKPOINTS_PATH,
           grid_search=False):
    """Evolves parameters to train a model on a dataset.

    Args:
        strategy (str): The strategy to be used for evolving, Simple Genetic Algorithm (sga) / Novelty Search (ns) / Covariance-Matrix Adaptation (cma-es)
        problem (object): A Problem object that includes the type of problem we're trying to optimize
        toolbox (deap.ToolBox): DEAP's configured toolbox
        crossover_prob (float): Crossover probability from 0-1
        mutation_prob (float): Mutation probability from 0-1
        pop_size (int): Number of individuals to keep in each Population
        elite_size (float): Percentage of fittest individuals to pass on to next generation
        num_generations (int): Number of generations to run algorithm
        imutpb (float): Mutation probability for each individual's attribute
        imutmu (float): Mean parameter for the Gaussian Distribution we're mutating an attribute from
        imutsigma (float): Sigma parameter for the Gaussian Distribution we're mutating an attribute from
        checkpoint (str): String to specify the checkpoint file to load the population

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
        experiment_checkpoints_dir = os.path.join(EXPERIMENT_CHECKPOINTS_PATH, problem.name + '_' + datetime.now().strftime("%b-%d-%Y-%H:%M:%S"))
    experiment_logs_file = os.path.join(ABS_ALGO_EXP_LOGS_PATH, problem.name + '_' + datetime.now().strftime("%b-%d-%Y-%H:%M:%S") + '.log')

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
    logger.info('problem_type: {}'.format(problem.name))
    logger.info('cxpb: {}'.format(crossover_prob))
    logger.info('mutpb: {}'.format(mutation_prob))
    logger.info('popsize: {}'.format(pop_size))
    logger.info('elitesize: {}'.format(elite_size))
    logger.info('ngen: {}'.format(num_generations))
    logger.info('imutpb: {}'.format(imutpb))
    logger.info('imutmu: {}'.format(imutmu))
    logger.info('imutsigma: {}'.format(imutsigma))

    ##################################
    # 2. LOAD CHECKPOINT IF PROVIDED #
    ##################################
    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            # Define objective, individuals, population, and evaluation
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
            cp = pickle.load(cp_file)
        random.seed(cp["rndstate"])
        pop = cp["population"]
        start_gen = int(cp["generation"])
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]

    else:
        # Start a new evolution
        random.seed(100) # Set seed
        pop = toolbox.population(n=pop_size)
        start_gen = 0
        halloffame = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()


    ###############################
    # 3. CURRENT POPULATION STATS #
    ###############################
    # Track the Average fitness scores
    avg_fitness_scores = []

    # Evaluate the entire population
    avg_fitness_score, num_ind_evaluated = toolbox.evaluate_population(population=pop)
    avg_fitness_scores.append(avg_fitness_score)

    # Save statistics about our current population loaded
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    #################################
    # 4. EVOLVE THROUGH GENERATIONS #
    #################################
    # Iterate for generations
    for g in tqdm(range(start_gen, num_generations)):

        # Select the next generation individuals
        offspring = toolbox.select(pop, k=len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Keep the elite individuals for next generation
        # without mutation
        elite_num = int(elite_size*pop_size) + 1
        elite = offspring[:elite_num]
        non_elite = offspring[elite_num:]

        # Apply crossover on the non-elite offspring by
        # choosing alternate offsprings
        # e.g. if pop = [ind1, ind2, ind3, ind4],
        # we are doing 2-point crossover between
        # ind1, ind3 and ind2, ind4
        for child1, child2 in zip(non_elite[::2], non_elite[1::2]):
            if random.random() < crossover_prob:

                # Crossover
                toolbox.mate(child1, child2)

                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the non-elite offspring
        for mutant in non_elite:
            if random.random() < mutation_prob:

                # Mutate
                toolbox.mutate(mutant)

                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                del mutant.fitness.values

        # Recombine Elites with non-elites
        offspring = elite + non_elite

        # Evaluate the entire population
        avg_fitness_score, num_ind_evaluated = toolbox.evaluate_population(population=offspring)
        avg_fitness_scores.append(avg_fitness_score)

        # The population is entirely replaced by the
        # evaluated offspring
        pop[:] = offspring

        # Update population statistics
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(num_ind_evaluated), **record)

        # Save snapshot of population (offspring)
        if g % FREQ == 0:

            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=pop,
                      generation=g,
                      halloffame=halloffame,
                      logbook=logbook,
                      rndstate=random.getstate())

            with open(os.path.join(experiment_checkpoints_dir, 'checkpoint_gen{}.pkl'.format(g)), "wb") as cp_file:
                pickle.dump(cp, cp_file)

        # Log Average score of population
        logger.info('Generation {} Avg. Fitness Score: {} | Fittest Individual Score: {}'\
                        .format(g,
                                avg_fitness_score,
                                halloffame.fitness.values.fitness_score))

        # Early Stopping if average fitness
        # score is close to the minimum possible,
        # or if stuck at local optima (average fitness score
        # hasnt changed for past 10 rounds)
        if problem.approx_type == Problem.CLASSIFICATION:
            if round(-halloffame.fitness.values.fitness_score, 4) > 0.95:
                logger.info('Early Stopping activated because Accuracy > 95%.')
                break;
            if len(avg_fitness_scores) > 10 and len(set(avg_fitness_scores[-10:])) == 1:
                logger.info('Early Stopping activated because fitness scores have converged.')
                break;
        else:
            if round(halloffame.fitness.values.fitness_score, 4) < 0.01:
                logger.info('Early Stopping activated because MSE < 0.01.')
                break;
            if len(avg_fitness_scores) > 10 and len(set(avg_fitness_scores[-10:])) == 1:
                logger.info('Early Stopping activated because fitness scores have converged.')
                break;


    return pop, avg_fitness_scores
