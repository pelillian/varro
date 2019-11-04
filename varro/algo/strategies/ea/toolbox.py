"""
This module contains functions to configure the toolbox
for neural net / fpga
"""

import random
import numpy as np
from deap import base, creator, tools


def ea_toolbox(i_shape,
               evaluate_population,
               model_type,
               imutpb=None,
               imutmu=None,
               imutsigma=None):
    """Initializes and configures the DEAP toolbox for evolving the parameters of a model.

    Args:
        i_shape (int or tuple): Size or shape of an individual in the population
        evaluate_population (function): Function to evaluate an entire population
        imutpb (float): Mutation probability for each individual's attribute
        imutmu (float): Mean parameter for the Gaussian Distribution we're mutating an attribute from
        imutsigma (float): Sigma parameter for the Gaussian Distribution we're mutating an attribute from
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

    # Defines Individual
    if model_type == "nn":
        toolbox.register("attribute", random.random)
        toolbox.register("individual",
                         tools.initRepeat,
                         creator.Individual,
                         toolbox.attribute,
                         n=i_shape)
        toolbox.register("mutate",
                         tools.mutGaussian,
                         mu=imutmu,
                         sigma=imutsigma,
                         indpb=imutpb)
    elif model_type == "fpga":
        toolbox.register("attribute", np.random.choice, [False, True])
        size = np.prod(i_shape)

        def init_individual(ind_class):
            return ind_class(np.random.choice([False, True], size=size))
        toolbox.register("individual",
                         init_individual,
                         creator.Individual)

        def mutate_individual(ind):
            idx = np.argwhere(np.random.choice([False, True], size, p=[0.9, 0.1]))
            ind[idx] = np.invert(ind[idx])
            return ind
        toolbox.register("mutate", mutate_individual)

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
