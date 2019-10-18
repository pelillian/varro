"""
This module contains the evolutionary algorithm logic
"""

import logging
import numpy as np
import functools
from tqdm import tqdm


def evolve(toolbox, crossover_prob, mutation_prob, num_generations):
    """Evolves weights of neural network to train classifier for MNIST
    
    Args:
        crossover_prob (float): Crossover probability from 0-1
        mutation_prob (float): Mutation probability from 0-1
        num_generations (int): Number of generations to run algorithm
    
    Returns:
        pop: Population of the fittest individuals so far
        avg_fitness_scores: A list of the average fitness scores for each generation

    """

    # Set Logging configuration
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='./algo/logs/evolve_mnist.log',
                        level=logging.INFO,
                        format=log_fmt)

    # Get logger
    logger = logging.getLogger(__name__)
    logger.info('Start Evolution ...')

    # Initialize random population
    pop = toolbox.population(n=100)
    
    # Track the Average fitness scores
    avg_fitness_scores = []

    # Evaluate the entire population
    fitnesses = toolbox.evaluate(pop)
    avg_fitness_scores.append(np.mean([fitness_score for fitness in fitnesses for fitness_score in fitness]))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Iterate for generations
    for g in tqdm(range(num_generations)):
        
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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Compute Average fitness score of generation
        valid_ind = [ind for ind in offspring if ind.fitness.valid]
        avg_fitness_score = np.mean([fitness_score \
                                        for fitness in list(fitnesses) + list(map(toolbox.evaluate, valid_ind)) \
                                        for fitness_score in fitness])
        avg_fitness_scores.append(avg_fitness_score)
        logger.info('Generation {} Avg. Fitness Score: {}'.format(g, avg_fitness_score))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        
    return pop, avg_fitness_scores