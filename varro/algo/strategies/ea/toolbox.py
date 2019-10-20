"""
This module contains functions to configure the toolbox 
for neural net / fpga
"""

import random
import numpy as np
from deap import base, creator, tools


def ea_toolbox(i_size, evaluate_population, model_type, p=0.5):
    """Initializes and configures the DEAP toolbox for evolving the weights of a model.

    Args:
        i_size (int): Size of an individual in the population (array length)
        evaluate_population (function): Function to evaluate an entire population
        p: Probability that random bit in each individual is 0 / 1

    Returns:
        toolbox (deap.base.Toolbox): Configured DEAP Toolbox for the algorithm.

    """
    # Set seed
    random.seed(100)

    # Initialzie Toolbox
    toolbox = base.Toolbox()

    # Define objective, individuals, population, and evaluation
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    if model_type == 'nn':
        toolbox.register("attribute", random.random)
        toolbox.register("individual", 
                         tools.initRepeat, 
                         creator.Individual,
                         toolbox.attribute, 
                         n=i_size)
        toolbox.register("mutate", 
                         tools.mutGaussian, 
                         mu=0, 
                         sigma=1, 
                         indpb=0.1)
    elif model_type == 'fpga':
        toolbox.register("individual", 
                         np.random.choice(a=[False, True], size=i_shape, p=[p, 1-p]))
        toolbox.register("mutate", 
                         tools.mutFlipBit,
                         indpb=0.1)

    toolbox.register("population", 
                     tools.initRepeat, 
                     list, 
                     toolbox.individual)
    toolbox.register("mate", 
                     tools.cxTwoPoint)
    toolbox.register("select", 
                     tools.selTournament, 
                     tournsize=3)
    toolbox.register("evaluate_population", 
                     evaluate_population)
    
    return toolbox

