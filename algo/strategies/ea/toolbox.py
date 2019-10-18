"""
This module contains functions to configure the toolbox 
for neural net / fpga
"""

from deap import base, creator, tools


def fpga_toolbox(i_shape, 
				 evaluate_population, 
				 p=0.5):
	"""Initializes and configures the DEAP toolboxfor evolving bitstream to fpga.

    Args:
        i_shape (tuple(int, int)): Shape of an individual in the population
        evaluate_population (function): The function we'll use to evaluate an entire population
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
    toolbox.register("individual", 
                     np.random.choice(a=[False, True], size=i_shape, p=[p, 1-p]))
    toolbox.register("population", 
                     tools.initRepeat, 
                     list, 
                     toolbox.individual)
    toolbox.register("mate", 
                     tools.cxTwoPoint)
    toolbox.register("mutate", 
                     tools.mutFlipBit,
                     indpb=0.1)
    toolbox.register("select", 
                     tools.selTournament, 
                     tournsize=3)
    toolbox.register("evaluate_population", 
                     evaluate_population)

    return toolbox


def nn_toolbox(i_size, 
			   evaluate_population):
	"""Initializes and configures the DEAP toolbox for evolving weights of neural network.

    Args:
        i_size (int): Size of an individual in the population (array length)
        evaluate_population (function): The function we'll use to evaluate an entire population

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
    toolbox.register("attribute", random.random)
    toolbox.register("individual", 
                     tools.initRepeat, 
                     creator.Individual,
                     toolbox.attribute, 
                     n=i_size)
    toolbox.register("population", 
                     tools.initRepeat, 
                     list, 
                     toolbox.individual)
    toolbox.register("mate", 
                     tools.cxTwoPoint)
    toolbox.register("mutate", 
                     tools.mutGaussian, 
                     mu=0, 
                     sigma=1, 
                     indpb=0.1)
    toolbox.register("select", 
                     tools.selTournament, 
                     tournsize=3)
    toolbox.register("evaluate_population", 
                     evaluate)

    return toolbox