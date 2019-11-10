"""
This module contains an abstract class for Strategy
"""

import random
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from deap import base, creator, tools

from varro.algo.problems import Problem
from varro.algo.strategies.es.toolbox import es_toolbox


class Strategy(ABC):

    #############
    # VARIABLES #
    #############
    @property
    def name(self):
        """Name of Strategy we're using to optimize"""
        return self._name

    @property
    def cxpb(self):
        """Crossover probability from 0-1"""
        return self._cxpb

    @property
    def mutpb(self):
        """Mutation probability from 0-1"""
        return self._mutpb

    @property
    def popsize(self):
        """Number of individuals to keep in each Population"""
        return self._popsize

    @property
    def elitesize(self):
        """Percentage of fittest individuals to pass on to next generation"""
        return self._elitesize

    @property
    def ngen(self):
        """Number of generations to run algorithm"""
        return self._ngen

    @property
    def imutpb(self):
        """Mutation probability for each individual's attribute"""
        return self._imutpb

    @property
    def imutmu(self):
        """Mean parameter for the Gaussian Distribution we're mutating an attribute from"""
        return self._imutmu

    @property
    def imutsigma(self):
        """Sigma parameter for the Gaussian Distribution we're mutating an attribute from"""
        return self._imutsigma

    @property
    def ckpt(self):
        """String to specify an existing checkpoint file to load the population"""
        return self._ckpt

    @property
    def halloffamesize(self):
        """Percentage of individuals in population we store in the HallOfFame / Archive"""
        return self._halloffamesize

    #############
    # FUNCTIONS #
    #############
    @abstractmethod
    def init_fitness_and_inds(self):
        """Initializes the fitness and definition of individuals"""
        pass

    @abstractmethod
    def init_toolbox(self):
        """Initializes the toolbox according to strategy"""
        pass


    @abstractmethod
    def load_es_vars(self):
        """Loads the evolutionary strategy variables from checkpoint given after
        creating the fitness and individual templates for DEAP evolution or initializes them
        """
        pass


    @abstractmethod
    def save_ckpt(self, exp_ckpt_dir):
        """Saves the checkpoint of the current generation of Population
        and some other information

        Args:
            exp_ckpt_dir (str): The experiment's checkpointing directory
        """
        pass


    @abstractmethod
    def evaluate(self, pop):
        """Evaluates an entire population on a dataset on the neural net / fpga
        architecture specified by the model, and calculates the fitness scores for
        each individual, sorting the entire population by fitness scores in-place

        Args:
            pop (list): An iterable of np.ndarrays that represent the individuals

        Returns:
            Average fitness score of population

        """
        pass


    @abstractmethod
    def generate_offspring(self):
        """Generates new offspring using a combination of the selection methods
        specified to choose fittest individuals and custom preference

        Returns:
            A Tuple of (Non-alterable offspring, Alterable offspring)

        """
        pass


    def config_toolbox(self):
        """Configure ToolBox with experiment-specific params"""
        # Set the individual size to the number of parameters
        # we can alter in the neural network architecture specified,
        # and initialize the fitness metrics needed to evaluate an individual
        self.toolbox = es_toolbox(strategy_name=self.name,
                                  i_shape=self.model.parameters_shape,
                                  evaluate=self.evaluate,
                                  model_type='nn' if type(self.model).__name__ == 'ModelNN' else 'fpga',
                                  imutpb=self.imutpb,
                                  imutmu=self.imutmu,
                                  imutsigma=self.imutsigma)


    def fitness_score(self):
        """Calculates the fitness score for a particular
        model configuration (after loading parameters in the model) on the problem specified

        Returns:
            Returns the fitness score of the model w.r.t. the problem specified
        """
        # Predict labels
        y_pred = np.array(self.model.predict(self.problem.X_train, problem=self.problem))

        if self.problem.approx_type == Problem.CLASSIFICATION:
            if self.problem.name == 'mnist':
                categorical_accuracy = accuracy_score(y_true=self.problem.y_train,
                                                      y_pred=np.argmax(y_pred, axis=-1))
            else:
                categorical_accuracy = accuracy_score(y_true=self.problem.y_train,
                                                      y_pred=(np.array(y_pred) > 0.5).astype(float))
            return -categorical_accuracy

        elif self.problem.approx_type == Problem.REGRESSION:
            rmse = sqrt(mean_squared_error(self.problem.X_train, y_pred))
            return rmse

        else:
            raise ValueError('Unknown approximation type ' + str(self.problem.approx_type))


    ########
    # INIT #
    ########
    def __init__(self,
                 name,
                 model,
                 problem,
                 cxpb,
                 mutpb,
                 popsize,
                 elitesize,
                 ngen,
                 imutpb,
                 imutmu,
                 imutsigma,
                 ckpt,
                 halloffamesize):
        """This class defines the strategy and the methods that come with that strategy."""
        self._name = name
        self._cxpb = cxpb
        self._mutpb = mutpb
        self._popsize = popsize
        self._elitesize = elitesize
        self._ngen = ngen
        self._imutpb = imutpb
        self._imutmu = imutmu
        self._imutsigma = imutsigma
        self._ckpt = ckpt
        self._halloffamesize = halloffamesize

        # Storing model and problem
        self.model = model
        self.problem = problem

        # Initialize Toolbox
        self.init_toolbox()

        # Load evolutionary strategy variables
        self.load_es_vars()
