"""
This module contains the class for Simple Genetic Algorithm strategy
"""

import numpy as np
import random

from varro.algo.strategies.strategy import Strategy
from varro.algo.strategies.ns_es import StrategyNSES


class StrategyNSRES(StrategyNSES):

    #############
    # VARIABLES #
    #############
    @property
    def novelty_metric(self):
        """The distance metric to be used to measure an Individual's novelty"""
        return self._novelty_metric

    #############
    # FUNCTIONS #
    #############
    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        # Define specific Fitness and Individual for Novelty Search
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0)) # Both Fitness and Novelty
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

        # Configure the rest of the toolbox that is independent
        # of which evolutionary strategy
        super().config_toolbox()


    def evaluate(self, pop):
        """Evaluates an entire population on a dataset on the neural net / fpga
        architecture specified by the model, and calculates the fitness scores for
        each individual, sorting the entire population by fitness scores in-place

        Args:
            pop (list): An iterable of np.ndarrays that represent the individuals

        Returns:
            Tuple of (Average fitness score of population, \
                Number of individuals with invalid fitness scores that have been evaluated)

        """
        # Re-generates the training set for the problem (if possible) to prevent overfitting
        self.problem.reset_train_set()

        # Define fitness
        Fitness = namedtuple('Fitness', ['fitness_score', 'novelty_score'])

        # Compute all fitness for population
        super(StrategyNSES, self).compute_fitness(pop, Fitness)

        # Calculate the Novelty scores for population
        super().compute_novelty(pop, Fitness)

        return np.mean([ind.fitness.values.fitness_score for ind in pop]), len(pop)


    ########
    # INIT #
    ########
    def __init__(self, novelty_metric, **kwargs):

        # Call Strategy constructor
        Strategy.__init__(name='nsr-es', **kwargs)

        # Set Novelty metric
        # Supported novelty metrics:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
        self._novelty_metric = novelty_metric

        # If we have a multiobjective strategy,
        # we also need to keep the Pareto Fronts
        self.paretofront = cp["paretofront"] if self.ckpt else tools.ParetoFront()
