"""
This module contains the class for Simple Genetic Algorithm strategy
"""

import numpy as np
import random

from varro.algo.strategies.strategy import Strategy
from deap import base, creator, tools


class StrategySGA(Strategy):

    #############
    # FUNCTIONS #
    #############
    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        # Define specific Fitness and Individual for SGA
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Just Fitness
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        # Configure the rest of the toolbox that is independent
        # of which evolutionary strategy
        super().config_toolbox()


    def load_es_vars(self):
        """Loads the evolutionary strategy variables from checkpoint given after
        creating the fitness and individual templates for DEAP evolution or initializes them
        """
        if self.ckpt:
            # A file name has been given, then load the data from the file
            # Load data from pickle file
            with open(self.ckpt, "rb") as cp_file:
                cp = pickle.load(cp_file)

            self.rndstate = random.seed(cp["rndstate"])
            self.pop = cp["population"]
            self.curr_gen = int(cp["curr_gen"])
            self.halloffame = cp["halloffame"]
            self.logbook = cp["logbook"]

        else:
            # Start a new evolution
            self.rndstate = random.seed(100) # Set seed
            self.pop = toolbox.population(n=self.popsize)
            self.curr_gen = 0
            self.halloffame = tools.HallOfFame(maxsize=int(self.halloffamesize*self.popsize))
            self.logbook = tools.Logbook()


    def save_ckpt(self, exp_ckpt_dir):
        """Saves the checkpoint of the current generation of Population
        and some other information

        Args:
            exp_ckpt_dir (str): The experiment's checkpointing directory
        """
        # Fill the dictionary using the dict(key=value[, ...]) constructor
        cp = dict(pop=self.pop,
                  curr_gen=self.curr_gen,
                  halloffame=self.halloffame,
                  logbook=self.logbook,
                  rndstate=self.rndstate)

        with open(os.path.join(exp_ckpt_dir, 'checkpoint_gen{}.pkl'.format(g)), "wb") as cp_file:
            pickle.dump(cp, cp_file)


    def compute_fitness(self, pop, Fitness):
        """Calculates the fitness scores for the entire Population

        Args:
            pop (list): An iterable of Individual(np.ndarrays) that represent the individuals
            Fitness (collections.namedtuple): A namedtuple that initializes what type of scores are in Fitness
        """
        # Evaluate the individuals with an invalid fitness
        # (These are the individuals that have not been evaluated before -
        # individuals at the start of the evolutionary algorithm - or those
        # that have been mutated / the offspring after crossover with fitness deleted)
        invalid_inds = [ind for ind in pop if not ind.fitness.valid]

        # Get fitness score for each individual with
        # invalid fitness score in population
        for ind in invalid_inds:

            # Load Weights into model using individual
            self.model.load_parameters(ind)

            # Calculate the Fitness score of the individual
            ind.fitness.values = Fitness(fitness_score=super().fitness_score())


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
        Fitness = namedtuple('Fitness', ['fitness_score'])

        # Compute all fitness for population
        self.compute_fitness(pop, Fitness)

        # Update population statistics
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(num_ind_evaluated), **record)

        return np.mean([ind.fitness.values.fitness_score for ind in pop]), len(invalid_inds)


    ########
    # INIT #
    ########
    def __init__(self, **kwargs):

        # Call Strategy constructor
        super().__init__(name='sga', **kwargs)
