"""
This module contains the evolutionary algorithm logic
"""

import os
import pickle
import random
import logging
import numpy as np
import functools
from tqdm import tqdm
from deap import tools

from varro.misc.util import make_path
from varro.misc.variables import ABSOLUTE_ALGO_LOGS_PATH, EXPERIMENT_CHECKPOINTS_PATH, FREQ


def evolve(problem, toolbox, crossover_prob, mutation_prob, pop_size, num_generations, checkpoint=None):
    """Evolves weights to train a model on a dataset.
    
    Args:
        problem (str): A string specifying what type of problem we're trying to optimize
        toolbox (deap.ToolBox): DEAP's configured toolbox
        crossover_prob (float): Crossover probability from 0-1
        mutation_prob (float): Mutation probability from 0-1
        num_generations (int): Number of generations to run algorithm
        checkpoint (str): String to specify the checkpoint file to load the population
    
    Returns:
        pop: Population of the fittest individuals so far
        avg_fitness_scores: A list of the average fitness scores for each generation

    """
    ########################################################
    # 1. SET UP LOGGER, FOLDERS, AND FILES TO SAVE DATA TO #
    ########################################################
    # Set experiment name
    experiment_name = 'experiment-{}-popsize{}-ngen{}-cxpb{}-mutpb{}'.format(problem, pop_size, num_generations, crossover_prob, mutation_prob)
    experiment_checkpoints_dir = os.path.join(EXPERIMENT_CHECKPOINTS_PATH, experiment_name)
    experiment_logs_file = os.path.join(ABSOLUTE_ALGO_LOGS_PATH, experiment_name + '.log')

    # Create experiment folder to store
    # snapshots of population
    make_path(experiment_snapshots_dir)

    # Set Logging configuration
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=experiment_logs_file,
                        level=logging.INFO,
                        format=log_fmt)

    # Get logger
    logger = logging.getLogger(__name__)
    logger.info('Start Evolution ...')


    ##################################
    # 2. LOAD CHECKPOINT IF PROVIDED #
    ##################################
    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        random.seed(cp["rndstate"])
        pop = cp["population"]
        start_gen = cp["generation"]
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
    fitness_scores_population = toolbox.evaluate_population(pop)

    # WARNING: BE CAREFUL HERE WHEN WE HAVE MUTLIPLE DISTINCT
    # FITNESS SCORES IN THE FUTURE
    avg_fitness_scores.append(np.mean([fitness_score \
                                        for fitness_scores_ind in fitness_scores_population \
                                        for fitness_score in fitness_scores_ind]))
    for ind, fitness_scores_ind in zip(pop, fitness_scores_population):
        ind.fitness.values = fitness_scores_ind 

    # Save statistics about our current population loaded
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)


    #################################
    # 4. EVOLVE THROUGH GENERATIONS #
    #################################
    # Iterate for generations
    for g in tqdm(range(start_gen, num_generations)):
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover on the offspring by
        # choosing alternate offsprings
        # e.g. if pop = [ind1, ind2, ind3, ind4],
        # we are doing 2-point crossover between
        # ind1, ind3 and ind2, ind4
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                
                # Crossover
                toolbox.mate(child1, child2)
                
                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < mutation_prob:
                
                # Mutate
                toolbox.mutate(mutant)
                
                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        # (These are the individuals that have been mutated
        # or the offspring after crossover with fitness deleted)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness_scores_population = toolbox.evaluate_population(invalid_ind)
        for ind, fitness_scores_ind in zip(invalid_ind, fitness_scores_population):
            ind.fitness.values = fitness_scores_ind
        
        # Compute Average fitness score of generation
        valid_ind = [ind for ind in offspring if ind.fitness.valid]
        avg_fitness_score = np.mean([fitness_score \
                                        for fitness_scores_ind in list(fitness_scores_population) + list(toolbox.evaluate_population(valid_ind)) \
                                        for fitness_score in fitness_scores_ind])
        avg_fitness_scores.append(avg_fitness_score)

        # Log Average score of population
        logger.info('Generation {} Avg. Fitness Score: {}'.format(g, avg_fitness_score))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Update population statistics
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)

        # Save snapshot of population (offspring)
        if gen % FREQ == 0:

            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=pop, generation=g, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())

            with open(os.path.join(experiment_checkpoints_dir, 'checkpoint_gen{}.pkl'.format(g)), "wb") as cp_file:
                pickle.dump(cp, cp_file)
        
        
    return pop, avg_fitness_scores

