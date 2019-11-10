"""
This module contains the class for Simple Genetic Algorithm strategy
"""

import numpy as np
import random
from sklearn.neighbors import BallTree
from deap import base, creator, tools

from varro.algo.strategies.sga import StrategySGA


class StrategyNSES(StrategySGA):

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
    def init_fitness_and_inds(self):
        """Initializes the fitness and definition of individuals"""

        creator.create("FitnessMax", base.Fitness, weights=(1.0)) # Just Novelty
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        # Define specific Fitness and Individual for Novelty Search
        self.init_fitness_and_inds()

        # Configure the rest of the toolbox that is independent
        # of which evolutionary strategy
        super().config_toolbox()


    def load_es_vars(self):
        """Loads the evolutionary strategy variables from checkpoint given after
        creating the fitness and individual templates for DEAP evolution or initializes them
        """
        super().load_es_vars()


    def save_ckpt(self, exp_ckpt_dir):
        """Saves the checkpoint of the current generation of Population
        and some other information

        Args:
            exp_ckpt_dir (str): The experiment's checkpointing directory
        """
        super().save_ckpt()


    def compute_novelty(self, pop, Fitness, k=5):
        """Calculates the novelty scores for each individual in the
        population using average distance between k nearest neighbors approach according to
        http://eplex.cs.ucf.edu/noveltysearch/userspage/#howtoimplement

        Args:
            pop (list): An iterable of Individual(np.ndarrays) that represent the individuals
            Fitness (collections.namedtuple): A namedtuple that initializes what type of scores are in Fitness
        """
        # Init BallTree to find k-Nearest Neighbors
        tree = BallTree(population, metric=self.novelty_metric)

        for ind in pop:

            # Get the k-nearest neighbors of
            # the individual
            dist, ind_idxs = tree.query(ind, k=k)

            # Ignore first value as it'll be 0 since
            # there's an instance of the same vector in
            # population
            ind.fitness.values = Fitness(novelty_score=np.mean(dist[1:]))


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

        # Define fitness
        Fitness = namedtuple('Fitness', ['novelty_score'])

        # Calculate the Novelty scores for population
        self.compute_novelty(pop, Fitness)

        # The population is entirely replaced by the
        # evaluated offspring
        self.pop[:] = pop

        # Update population statistics
        self.halloffame.update(self.pop)
        self.record = stats.compile(self.pop)
        self.logbook.record(gen=self.curr_gen, evals=len(self.pop), **record)

        return np.mean([ind.fitness.values.novelty_score for ind in pop])


    def generate_offspring(self):
        """Generates new offspring using a combination of the selection methods
        specified to choose fittest individuals and custom preference

        Returns:
            A Tuple of (Non-alterable offspring, Alterable offspring)

        """
        return super().generate_offspring()


    ########
    # INIT #
    ########
    def __init__(self, novelty_metric, **kwargs):

        # Call Strategy constructor
        super(StrategySGA, self).__init__(name='ns-es', **kwargs)

        # Set Novelty metric
        # Supported novelty metrics:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
        self._novelty_metric = novelty_metric