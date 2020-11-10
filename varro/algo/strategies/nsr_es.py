"""
This module contains the class for Novelty Search with Reward Evolutionary Strategy
-> Unique Individuals are benefitted
"""

import os
import pickle
import numpy as np
import random
from deap import base, creator, tools
from collections import namedtuple

from varro.algo.strategies.strategy import Strategy
from varro.algo.strategies.ns_es import StrategyNSES


class StrategyNSRES(StrategyNSES):
    def __init__(self, **kwargs):
        if "name" not in kwargs.keys():
            kwargs["name"] = "nsr-es"
        super().__init__(**kwargs)

    @staticmethod
    def init_fitness_and_inds():
        """Initializes the fitness and definition of individuals"""

        class Fitness(base.Fitness):
            def __init__(self):
                super().__init__()
                self.__fitness_score = None
                self.__novelty_score = None

            @property
            def fitness_score(self):
                return self.__fitness_score if self.__fitness_score else self.values[0]

            @fitness_score.setter
            def fitness_score(self, fitness_score):
                self.__fitness_score = fitness_score

            @fitness_score.deleter
            def fitness_score(self):
                if hasattr(self, '__fitness_score'):
                    del self.__fitness_score

            @property
            def novelty_score(self):
                return self.__novelty_score if self.__novelty_score else self.values[1]

            @novelty_score.setter
            def novelty_score(self, novelty_score):
                self.__novelty_score = novelty_score
                if novelty_score:
                    # WARNING:
                    # Setting values breaks alot of things:
                    # self.__novelty_score is reset to None
                    # after setting values, so you should only
                    # set values after all the scores you require are set
                    self.values = (self.fitness_score, novelty_score,)

            @novelty_score.deleter
            def novelty_score(self):
                if hasattr(self, '__novelty_score'):
                    del self.__novelty_score

            def delValues(self):
                super().delValues()
                if hasattr(self, '__fitness_score'):
                    del self.__fitness_score
                if hasattr(self, '__novelty_score'):
                    del self.__novelty_score

        creator.create("FitnessMulti", Fitness, weights=(-1.0, 1.0,)) # Both Fitness and Novelty
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)


    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        # Define specific Fitness and Individual for Novelty Search and Reward (Fitness)
        self.init_fitness_and_inds()

        # Configure the rest of the toolbox that is independent
        # of which evolutionary strategy
        super().config_toolbox()


    def load_es_vars(self):
        """Loads the evolutionary strategy variables from checkpoint given after
        creating the fitness and individual templates for DEAP evolution or initializes them
        """
        super().load_es_vars()

        # If we have a multiobjective strategy,
        # we also need to keep the Pareto Fronts
        self.paretofront = cp["paretofront"] if self.ckpt else tools.ParetoFront(similar=np.array_equal)


    def evaluate(self, pop):
        """Evaluates an entire population on a dataset on the neural net / fpga
        architecture specified by the model, and calculates the fitness scores for
        each individual, sorting the entire population by fitness scores in-place

        Args:
            pop (list): An iterable of np.ndarrays that represent the individuals

        Returns:
            Average fitness score of population

        """
        # Re-generates the training set for the problem (if possible) to prevent overfitting
        self.problem.reset_train_set()

        # Compute all fitness for population
        num_invalid_inds = self.compute_fitness(pop)

        # Calculate the Novelty scores for population
        self.compute_novelty(pop)

        # The population is entirely replaced by the
        # evaluated offspring
        self.pop[:] = pop

        # Update population statistics
        self.halloffame.update(self.pop)
        self.paretofront.update(self.pop)
        # record = self.stats.compile(self.pop)
        # self.logbook.record(gen=self.curr_gen, evals=num_invalid_inds, **record)

        return np.mean([ind.fitness.fitness_score for ind in pop])
