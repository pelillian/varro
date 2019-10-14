"""
This module implements an evolutionary strategies algorithm.
"""

#############
# LIBRARIES #
#############

# from fpga.flash import flash_ecp5
from algo.util import get_args, evaluate_neural_network

import argparse
import numpy as np
import random
import functools
from tqdm import tqdm
from deap import base, creator, tools
from array import array # Use this if speed is an issue
from collections import defaultdict

##################
# INITIALIZATION #
##################

# Set seed
random.seed(100)

# Define a Class called FitnessMin to
# define the fitness objective 
# for the Selection of individuals
# to become offspring
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Define a Class called Individual to inherit from
# list and has a fitness attribute
# = creator.FitnessMin
creator.create("Individual", list, fitness=creator.FitnessMin)

# Size of each individual
IND_SIZE = 12

# Initialzie Toolbox
toolbox = base.Toolbox()

# Define an attribute variable
toolbox.register("attribute", random.random)

# Define an individual that has 
toolbox.register("individual", 
                 tools.initRepeat, 
                 creator.Individual,
                 toolbox.attribute, 
                 n=IND_SIZE)

# Define a population of individuals
toolbox.register("population", 
                 tools.initRepeat, 
                 list, 
                 toolbox.individual)

# Defines a mating function that takes in 
# 2 tuples (2 individuals) and performs 2 point cross over
toolbox.register("mate", 
                 tools.cxTwoPoint)

# Defines a mutation function that takes in
# a single tuple (an individual) and for each
# entry in the tuple, we have a different probability
# of mutation given by indpb, and parameters for
# how much to mutate each entry by, using a gaussian
# distribution
toolbox.register("mutate", 
                 tools.mutGaussian, 
                 mu=0, 
                 sigma=1, 
                 indpb=0.1)
                 
# Defines the selection method for the mating
# pool / offspring 
toolbox.register("select", 
                 tools.selTournament, 
                 tournsize=3)
                 
# Defines the evaluation function
# we will use for calculating the fitness of
# an individual
toolbox.register("evaluate", 
                 evaluate_neural_network)

#################
# MAIN FUNCTION #
#################
def main():
    '''
    Function:
    ---------
    Evolves weights of neural network to approximate
    a function
    
    Parameters:
    -----------
    None.
    
    Returns:
    --------
    Population of the fittest individuals so far and a list
    of the average fitness scores for each generation
    '''
    
    # Get the Arguments parsed from file execution
    args = get_args()
    
    # Initialize random population
    pop = toolbox.population(n=50)
    
    # Initialize Cross-over probability 
    # for offspring, mutation probability,
    # and number of generations to run algo
    CXPB, MUTPB, NGEN = args.cxpb, args.mutpb, args.ngen
      
    # Function to approximate
    FUNC_TO_APPROX = args.func
    
    # Track the Average fitness scores
    avg_fitness_scores = []

    # Evaluate the entire population
    import pdb; pdb.set_trace()
    fitnesses = map(functools.partial(toolbox.evaluate, function=FUNC_TO_APPROX), pop)
    avg_fitness_scores.append(np.mean([fitness_score for fitness in fitnesses for fitness_score in fitness]))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Iterate for generations
    for g in tqdm(range(NGEN)):
        
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
            if random.random() < CXPB:
                
                # Crossover
                toolbox.mate(child1, child2)
                
                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                
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
        avg_fitness_scores.append(np.mean([fitness_score for fitness in list(fitnesses) + list(map(toolbox.evaluate, valid_ind)) for fitness_score in fitness]))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
     
    # Print Average Fitness Scores
    for idx, avg_fitness_score in enumerate(avg_fitness_scores):
        print('Generation {} Avg. Fitness Score: {}'.format(idx, avg_fitness_score))
        
    return pop, avg_fitness_scores

if __name__ == "__main__":
    main()

