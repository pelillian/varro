"""
This module contains functions to configure the toolbox
for neural net / fpga
"""

import random
import numpy as np
import time
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
    seed = int(time.time())
    random.seed(seed)

    logger.log('TOOLBOX.PY random seed is {}'.format(seed))
    
    logger.start_timer()

    # Initialize Toolbox
    toolbox = base.Toolbox()

    logger.stop_timer('TOOLBOX.PY Initializing toolbox')

    logger.start_timer()
    # Defining tools specific to model
    if model_type == "nn":

        # ATTRIBUTE
        toolbox.register("attribute", random.random)

        logger.stop_timer('TOOLBOX.PY register("attribute")')

        # INDIVIDUAL
        logger.start_timer()
        toolbox.register("individual",
                         getattr(tools, 'initRepeat'),
                         getattr(creator, 'Individual'),
                         getattr(toolbox, 'attribute'),
                         n=i_shape)

        logger.stop_timer('TOOLBOX.PY register("individual")')

        # MUTATION
        logger.start_timer()
        toolbox.register("mutate",
                         getattr(tools, 'mutGaussian'),
                         mu=imutmu,
                         sigma=imutsigma,
                         indpb=imutpb)

        logger.stop_timer('TOOLBOX.PY register("mutate")')

        # POPULATION
        logger.start_timer()
        def init_population(ind_class, n):
            pop = np.random.uniform(low=-1, high=1, size=(n, size))
            return [ind_class(ind) for ind in pop]
        toolbox.register("population",
                         init_population,
                         getattr(creator, 'Individual'))

        logger.stop_timer('TOOLBOX.PY register("population")')

        # MATING
        logger.start_timer()
        toolbox.register("mate",
                         getattr(tools, 'cxTwoPoint'))

        logger.stop_timer('TOOLBOX.PY register("mate")')


    elif model_type == "fpga":

        # ATTRIBUTE
        logger.start_timer()
        toolbox.register("attribute", np.random.choice, [False, True])
        size = np.prod(i_shape)

        logger.stop_timer('TOOLBOX.PY register("attribute")')

        # MUTATION
        logger.start_timer()
        def mutate_individual(ind):
            idx = np.argwhere(np.random.choice([False, True], size, p=[1 - imutpb, imutpb]))
            ind[idx] = np.invert(ind[idx])
            return ind
        toolbox.register("mutate", mutate_individual)

        logger.stop_timer('TOOLBOX.PY register("mutate")')

        # POPULATION
        logger.start_timer()
        def init_population(ind_class, n):
            pop = np.random.choice([False, True], size=(n, size))
            return [ind_class(ind) for ind in pop]
        toolbox.register("population",
                         init_population,
                         getattr(creator, 'Individual'))

        logger.stop_timer('TOOLBOX.PY register("population")')

        # MATING
        logger.start_timer()
        from varro.fpga.cross_over import cross_over
        toolbox.register("mate", cross_over)

        logger.stop_timer('TOOLBOX.PY register("mate")')


    # SELECTION METHOD
    logger.start_timer()
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

    logger.stop_timer('TOOLBOX.PY register("select")')


    # EVALUATE
    logger.start_timer()
    toolbox.register("evaluate",
                     evaluate)
    logger.stop_timer('TOOLBOX.PY register("evaluate")')
    
    return toolbox
