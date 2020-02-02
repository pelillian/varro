"""
This module contains functions to configure the toolbox
for neural net / fpga
"""

import random
import numpy as np
from dowel import logger
from deap import base, creator, tools
import time

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
    
    timer = time.time()

    # Initialize Toolbox
    toolbox = base.Toolbox()

    timer = time.time() - timer
    logger.log('TOOLBOX.PY Initializing toolbox took {}s'.format(timer))
    timer = time.time()

    # Defining tools specific to model
    if model_type == "nn":

        # ATTRIBUTE
        toolbox.register("attribute", random.random)

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("attribute") took {}s'.format(timer))
        timer = time.time()


        # INDIVIDUAL
        toolbox.register("individual",
                         getattr(tools, 'initRepeat'),
                         getattr(creator, 'Individual'),
                         getattr(toolbox, 'attribute'),
                         n=i_shape)

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("individual") took {}s'.format(timer))
        timer = time.time()

        # MUTATION
        toolbox.register("mutate",
                         getattr(tools, 'mutGaussian'),
                         mu=imutmu,
                         sigma=imutsigma,
                         indpb=imutpb)

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("mutate") took {}s'.format(timer))
        timer = time.time()


        # POPULATION
        toolbox.register("population",
                         getattr(tools, 'initRepeat'),
                         list,
                         getattr(toolbox, 'individual'))

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("population") took {}s'.format(timer))
        timer = time.time()

        # MATING
        toolbox.register("mate",
                         getattr(tools, 'cxTwoPoint'))

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("mate") took {}s'.format(timer))
        timer = time.time()


    elif model_type == "fpga":

        timer = time.time()

        # ATTRIBUTE
        toolbox.register("attribute", np.random.choice, [False, True])
        size = np.prod(i_shape)

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("attribute") took {}s'.format(timer))
        timer = time.time()


        # MUTATION
        def mutate_individual(ind):
            idx = np.argwhere(np.random.choice([False, True], size, p=[0.9, 0.1]))
            ind[idx] = np.invert(ind[idx])
            return ind
        toolbox.register("mutate", mutate_individual)

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("mutate") took {}s'.format(timer))
        timer = time.time()


        # POPULATION
        def init_population(ind_class, n):
            pop = np.random.choice([False, True], size=(n, size))
            return [ind_class(ind) for ind in pop]
        toolbox.register("population",
                         init_population,
                         getattr(creator, 'Individual'))

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("population") took {}s'.format(timer))
        timer = time.time()


        # MATING
        from varro.fpga.cross_over import cross_over
        toolbox.register("mate", cross_over)

        timer = time.time() - timer
        logger.log('TOOLBOX.PY register("mate") took {}s'.format(timer))
        timer = time.time()


    # SELECTION METHOD
    timer = time.time()
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

    timer = time.time() - timer
    logger.log('TOOLBOX.PY register("select") took {}s'.format(timer))
    timer = time.time()


    # EVALUATE
    toolbox.register("evaluate",
                     evaluate)

    timer = time.time() - timer
    logger.log('TOOLBOX.PY register("evaluate") took {}s'.format(timer))
    timer = time.time()
    
    return toolbox
