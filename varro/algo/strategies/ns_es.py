"""
This module contains the class for Novelty Search Evolutionary Strategy
"""

from scipy.stats import wasserstein_distance
import numpy as np
import random
from sklearn.neighbors import BallTree
from deap import base, creator, tools
from collections import namedtuple

from varro.algo.strategies.sga import StrategySGA


class StrategyNSES(StrategySGA):
    def __init__(self, **kwargs):
        super().__init__(name='ns-es', **kwargs)

    @staticmethod
    def init_fitness_and_inds():
        """Initializes the novelty and definition of individuals"""

        class Novelty(base.Fitness):
            def __init__(self):
                super().__init__()
                self.__novelty_score = None

            @property
            def novelty_score(self):
                return self.values[0]

            @novelty_score.setter
            def novelty_score(self, novelty_score):
                self.__novelty_score = novelty_score
                if novelty_score:
                    # WARNING:
                    # Setting values breaks alot of things:
                    # self.__novelty_score is reset to None
                    # after setting values, so you should only
                    # set values after all the scores you require are set
                    self.values = (novelty_score,)

            @novelty_score.deleter
            def novelty_score(self):
                if hasattr(self, '__novelty_score'):
                    del self.__novelty_score

            def delValues(self):
                super().delValues()
                if hasattr(self, '__novelty_score'):
                    del self.__novelty_score

        creator.create("NoveltyMax", Novelty, weights=(1.0,)) # Just Novelty
        creator.create("Individual", np.ndarray, fitness=creator.NoveltyMax)


    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        # Define specific Fitness and Individual for Novelty Search
        self.init_fitness_and_inds()

        # Configure the rest of the toolbox that is independent
        # of which evolutionary strategy
        super().config_toolbox()


    def compute_novelty(self, pop, k=5):
        """Calculates the novelty scores for each individual in the
        population using average distance between k nearest neighbors approach according to
        http://eplex.cs.ucf.edu/noveltysearch/userspage/#howtoimplement

        Args:
            pop (list): An iterable of Individual(np.ndarrays) that represent the individuals
            k: The nearest k neighbors will be used for novelty calculation
        """
        # Init BallTree to find k-Nearest Neighbors
        if self.novelty_metric == 'wasserstein':
            tree = BallTree(np.asarray(pop), metric='pyfunc', func=wasserstein_distance)
        else:
            tree = BallTree(np.asarray(pop), metric=self.novelty_metric)

        for ind in pop:

            # Get the k-nearest neighbors of
            # the individual
            dist, ind_idxs = tree.query(ind.reshape(1, -1), k=k)

            # Ignore first value as it'll be 0 since
            # there's an instance of the same vector in
            # population
            ind.fitness.novelty_score = np.mean(dist.flatten()[1:])


    def evaluate(self, pop):
        """Evaluates an entire population on a dataset on the neural net / fpga
        architecture specified by the model, and calculates the fitness scores for
        each individual, sorting the entire population by fitness scores in-place

        Args:
            pop (list): An iterable of np.ndarrays that represent the individuals

        Returns:
            Average novelty score of population

        """
        # Re-generates the training set for the problem (if possible) to prevent overfitting
        self.problem.reset_train_set()

        # Calculate the Novelty scores for population
        self.compute_novelty(pop)

        # The population is entirely replaced by the
        # evaluated offspring
        self.pop[:] = pop

        # Update population statistics
        self.halloffame.update(self.pop)
        # record = self.stats.compile(self.pop)
        # self.logbook.record(gen=self.curr_gen, evals=len(self.pop), **record)

        return np.mean([ind.fitness.novelty_score for ind in pop])
