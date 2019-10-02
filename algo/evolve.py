"""
This module implements an evolutionary strategies algorithm.
"""

#############
# LIBRARIES #
#############

from fpga.flash import flash_ecp5
from utilities import get_args, evaluate

import argparse
import numpy as np
import random
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
                 evaluate)

#################
# MAIN FUNCTION #
#################
def main():
    
    # Initialize random population
    pop = toolbox.population(n=50)
    
    # Initialize Cross-over probability 
    # for offspring, mutation probability,
    # and number of generations to run algo
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Iterate for generations
    for g in range(NGEN):
        
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
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

if __name__ == "__main__":
    main()

