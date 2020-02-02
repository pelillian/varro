"""
This module contains the class for Simple Genetic Algorithm strategy
"""

import os
import pickle
import numpy as np
import random
from deap import base, creator, tools
from collections import namedtuple
from functools import partial
from dowel import logger
import time

from varro.algo.strategies.strategy import Strategy


class StrategySGA(Strategy):
    def __init__(self, **kwargs):
        super().__init__(name='sga', **kwargs)

    #############
    # FUNCTIONS #
    #############
    @staticmethod
    def init_fitness_and_inds():
        timer = time.time()
        """Initializes the fitness and definition of individuals"""

        class Fitness(base.Fitness):
            def __init__(self):
                super().__init__()
                self.__fitness_score = None

            @property
            def fitness_score(self):
                return self.values[0]

            @fitness_score.setter
            def fitness_score(self, fitness_score):
                self.__fitness_score = fitness_score
                if fitness_score:
                    # WARNING:
                    # Setting values breaks alot of things:
                    # self.__fitness_score is reset to None
                    # after setting values, so you should only
                    # set values after all the scores you require are set
                    self.values = (fitness_score,)

            @fitness_score.deleter
            def fitness_score(self):
                if hasattr(self, '__fitness_score'):
                    del self.__fitness_score

            def delValues(self):
                super().delValues()
                if hasattr(self, '__fitness_score'):
                    del self.__fitness_score

        creator.create("FitnessMin", Fitness, weights=(-1.0,)) # Just Fitness
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        timer = time.time() - timer
        logger.log('SGA.PY Initializing fitness and individuals took {}s'.format(timer))
        timer = time.time()


    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        # Define specific Fitness and Individual for SGA
        timer = time.time()
        self.init_fitness_and_inds()

        # Configure the rest of the toolbox that is independent
        # of which evolutionary strategy
        super().config_toolbox()

        timer = time.time() - timer
        logger.log('SGA.PY Initializing toolboox took {}s'.format(timer))


    def load_es_vars(self):
        """Loads the evolutionary strategy variables from checkpoint given after
        creating the fitness and individual templates for DEAP evolution or initializes them
        """

        timer = time.time()

        if self.ckpt:
            # A file name has been given, then load the data from the file
            # Load data from pickle file
            with open(self.ckpt, "rb") as cp_file:
                cp = pickle.load(cp_file)

            self.rndstate = random.seed(cp["rndstate"])
            self.pop = cp["pop"]
            self.curr_gen = int(cp["curr_gen"])
            self.halloffame = cp["halloffame"]
            self.logbook = cp["logbook"]

        else:
            # Start a new evolution
            self.rndstate = random.seed(100) # Set seed
            self.pop = self.toolbox.population(n=self.popsize)
            self.curr_gen = 0
            self.halloffame = tools.HallOfFame(maxsize=int(self.halloffamesize*self.popsize), similar=np.array_equal)
            self.logbook = tools.Logbook()

        timer = time.time() - timer
        logger.log('SGA.PY Loading ES Vars took {}s'.format(timer))


    def save_ckpt(self, exp_ckpt_dir):
        """Saves the checkpoint of the current generation of Population
        and some other information

        Args:
            exp_ckpt_dir (str): The experiment's checkpointing directory
        """

        timer = time.time()

        # Fill the dictionary using the dict(key=value[, ...]) constructor
        cp = dict(pop=self.pop,
                  strategy=self.name,
                  curr_gen=self.curr_gen,
                  halloffame=self.halloffame,
                  logbook=self.logbook,
                  rndstate=self.rndstate)

        with open(os.path.join(exp_ckpt_dir, '{}.pkl'.format(self.curr_gen)), "wb") as cp_file:
            pickle.dump(cp, cp_file)

        timer = time.time() - timer
        logger.log('SGA.PY Saving checkpoint took {}s'.format(timer))


    def compute_fitness(self, pop):
        """Calculates the fitness scores for the entire Population

        Args:
            pop (list): An iterable of Individual(np.ndarrays) that represent the individuals

        Returns:
            Number of individuals with invalid fitness scores we updated
        """

        timer = time.time()

        # Evaluate the individuals with an invalid fitness or if we are at the start
        # of the evolutionary algo, AKA curr_gen == 0
        # (These are the individuals that have not been evaluated before -
        # individuals at the start of the evolutionary algorithm - or those
        # that have been mutated / the offspring after crossover with fitness deleted)
        invalid_inds = [ind for ind in pop if not ind.fitness.valid or self.curr_gen == 0]

        # Get fitness score for each individual with
        # invalid fitness score in population
        for ind in invalid_inds:

            # Load Weights into model using individual
            self.model.load_parameters(ind)

            # Calculate the Fitness score of the individual
            ind.fitness.fitness_score = self.fitness_score()

        timer = time.time() - timer
        logger.log('SGA.PY Computing fitness took {}s'.format(timer))

        return len(invalid_inds)


    def evaluate(self, pop):

        timer = time.time()

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

        timer = time.time() - timer
        logger.log('SGA.PY Regenerate the training set took {}s'.format(timer))
        timer = time.time()

        # Compute all fitness for population
        num_invalid_inds = self.compute_fitness(pop)
        timer = time.time() - timer
        logger.log('SGA.PY Computing all fitness for population took {}s'.format(timer))
        timer = time.time()

        # The population is entirely replaced by the
        # evaluated offspring
        self.pop[:] = pop
        timer = time.time() - timer
        logger.log('SGA.PY Replace populatin with evaluated offspring {}s'.format(timer))
        timer = time.time()

        # Update population statistics
        self.halloffame.update(self.pop)
        # record = self.stats.compile(self.pop)
        # self.logbook.record(gen=self.curr_gen, evals=num_invalid_inds, **record)
        timer = time.time() - timer
        logger.log('SGA.PY Updating population statistics took {}s'.format(timer))

        return np.mean([ind.fitness.fitness_score for ind in pop])


    def generate_offspring(self):

        timer = time.time()
        """Generates new offspring using a combination of the selection methods
        specified to choose fittest individuals and custom preference

        Returns:
            A Tuple of (Non-alterable offspring, Alterable offspring)

        """
        # Keep the elite individuals for next generation
        # without mutation or cross over
        elite_num = int(self.elitesize*self.popsize)
        elite = self.toolbox.select_elite(self.pop, k=elite_num)

        timer = time.time() - timer
        logger.log('SGA.PY Keeping elite individuals took {}s'.format(timer))
        timer = time.time()

        # Clone the selected individuals
        non_alterable_elite_offspring = list(map(self.toolbox.clone, elite))
        timer = time.time() - timer
        logger.log('SGA.PY Cloning the selected individuals took {}s'.format(timer))
        timer = time.time()

        # Choose the rest of the individuals
        # to be altered
        random_inds = self.toolbox.select(self.pop, k=self.popsize-elite_num)
        alterable_offspring = list(map(self.toolbox.clone, random_inds))
        timer = time.time() - timer
        logger.log('SGA.PY Choosing the rest of the individuals to be altered {}'.format(timer))

        return non_alterable_elite_offspring, alterable_offspring
