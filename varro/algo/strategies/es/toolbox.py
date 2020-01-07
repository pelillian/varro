"""
This module contains functions to configure the toolbox
for neural net / fpga
"""

import random
import numpy as np
from dowel import logger
from deap import base, creator, tools


def es_toolbox(strategy_name,
               i_shape,
               evaluate,
               model_type,
               imutpb=None,
               imutmu=None,
               imutsigma=None):
    """Initializes and configures the DEAP toolbox for evolving the parameters of a model.

    Args:
        strategy_name (str): The strategy that is being used for evolution
        i_shape (int or tuple): Size or shape of an individual in the population
        evaluate (function): Function to evaluate an entire population
        model_type (str): A string specifying whether we're optimizing on a neural network
            or field programmable gate array
        imutpb (float): Mutation probability for each individual's attribute
        imutmu (float): Mean parameter for the Gaussian Distribution we're mutating an attribute from
        imutsigma (float): Sigma parameter for the Gaussian Distribution we're mutating an attribute from

    Returns:
        toolbox (deap.base.Toolbox): Configured DEAP Toolbox for the algorithm.

    """
    logger.log("Initializing toolbox...")
    # Set seed
    random.seed(100)

    # Initialzie Toolbox
    toolbox = base.Toolbox()

    # Defining tools specific to model
    if model_type == "nn":

        # ATTRIBUTE
        toolbox.register("attribute", random.random)

        # INDIVIDUAL
        toolbox.register("individual",
                         getattr(tools, 'initRepeat'),
                         getattr(creator, 'Individual'),
                         getattr(toolbox, 'attribute'),
                         n=i_shape)

        # MUTATION
        toolbox.register("mutate",
                         getattr(tools, 'mutGaussian'),
                         mu=imutmu,
                         sigma=imutsigma,
                         indpb=imutpb)

        # POPULATION
        toolbox.register("population",
                         getattr(tools, 'initRepeat'),
                         list,
                         getattr(toolbox, 'individual'))

        # MATING
        toolbox.register("mate",
                         getattr(tools, 'cxTwoPoint'))

    elif model_type == "fpga":

        # ATTRIBUTE
        toolbox.register("attribute", np.random.choice, [False, True])
        size = np.prod(i_shape)

        # INDIVIDUAL
        def init_individual(ind_class):
            return ind_class(np.random.choice([False, True], size=size))
        toolbox.register("individual",
                         init_individual,
                         getattr(creator, 'Individual'))

        # MUTATION
        def mutate_individual(ind):
            idx = np.argwhere(np.random.choice([False, True], size, p=[0.9, 0.1]))
            ind[idx] = np.invert(ind[idx])
            return ind
        toolbox.register("mutate", mutate_individual)

        # POPULATION
        toolbox.register("population",
                         getattr(tools, 'initRepeat'),
                         list,
                         getattr(toolbox, 'individual'))

        # MATING
        from varro.fpga.cross_over import cross_over
        toolbox.register("mate", cross_over)

    # SELECTION METHOD
    if strategy_name == 'nsr-es':
        toolbox.register("select_elite",
                         getattr(tools, 'selSPEA2')) # Use Multi-objective selection method
        toolbox.register("select",
                         getattr(tools, 'selRandom'))
    else:
        toolbox.register("select_elite",
                         getattr(tools, 'selTournament'),
                         tournsize=3)
        toolbox.register("select",
                         getattr(tools, 'selRandom'))

    # EVALUATE
    toolbox.register("evaluate",
                     evaluate)

    return toolbox
