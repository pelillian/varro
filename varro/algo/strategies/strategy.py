"""
This module contains an abstract class for Strategy
"""

import random
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
from math import sqrt
from deap import base, creator, tools
from dowel import logger

from varro.algo.problems import Problem
from varro.algo.strategies.es.toolbox import es_toolbox


class Strategy(ABC):
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
                 imutpb_decay,
                 imutmu,
                 imutsigma,
                 ckpt,
                 halloffamesize,
                 novelty_metric,
                 earlystop, 
                 lambda_penalty):
        """This class defines the strategy and the methods that come with that strategy."""
        self.name = name
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.popsize = popsize
        self.elitesize = elitesize
        self.ngen = ngen
        self.imutpb = imutpb
        self.imutpb_decay = imutpb_decay
        self.imutmu = imutmu
        self.imutsigma = imutsigma
        self.ckpt = ckpt
        self.halloffamesize = halloffamesize
        self.earlystop = earlystop
        self.novelty_metric = novelty_metric
        self.lambda_penalty = lambda_penalty

        # Storing model and problem
        self.model = model
        self.problem = problem

        # Initialize Toolbox
        self.init_toolbox()

        # Load evolutionary strategy variables
        self.load_es_vars()

        # Initialize stats we care about for population
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("max", np.max)

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
                                  imutpb_decay=self.imutpb_decay,
                                  imutmu=self.imutmu,
                                  imutsigma=self.imutsigma)
    
    @staticmethod
    def log_loss_with_abs_diff_penalty(y_true, y_pred, λ=100):
        y_true_1_idxs = np.argwhere(y_true.astype(int) == 1).flatten()
        y_true_0_idxs = np.argwhere(y_true.astype(int) == 0).flatten()
        return log_loss(y_true=y_true, y_pred=y_pred) - λ*np.abs(np.mean(np.abs(y_pred[y_true_1_idxs])) - np.mean(np.abs(y_pred[y_true_0_idxs])))

    def fitness_score(self, reg_metric='rmse'):
        """Calculates the fitness score for a particular
        model configuration (after loading parameters in the model) on the problem specified

        Args:
            reg_metric (str): The regression metric to be used to measure how fit a model is [Minimization Objective]

        Returns:
            Returns the fitness score of the model w.r.t. the problem specified
            CLASSIFICATION fitness score: Accuracy
            REGRESSION fitness score: Root Mean Squared Error
        """
        # Predict labels
        y_pred = np.array(self.model.predict(self.problem.X_train, problem=self.problem))

        if self.problem.approx_type == Problem.CLASSIFICATION:
            if self.problem.name == 'mnist':
                categorical_accuracy = accuracy_score(y_true=self.problem.y_train,
                                                      y_pred=np.argmax(y_pred, axis=-1))
            elif self.problem.name == 'simple_step':
                loss = Strategy.log_loss_with_abs_diff_penalty(y_true=self.problem.y_train, y_pred=y_pred, λ=self.lambda_penalty) 
                if self.model.name == "fpga":
                    logger.log('{')
                    logger.log('Data:\n' + str(np.column_stack((y_pred, self.problem.y_train))))
                    logger.log('Loss: ' + str(loss))
                    logger.log('}')
                return loss
            else:
                categorical_accuracy = accuracy_score(y_true=self.problem.y_train,
                                                      y_pred=(np.array(y_pred) > 0.5).astype(float))
            return -categorical_accuracy

        elif self.problem.approx_type == Problem.REGRESSION:
            if reg_metric == 'rmse':
                return sqrt(mean_squared_error(self.problem.y_train, y_pred))
            elif reg_metric == 'mae':
                return mean_absolute_error(self.problem.y_train, y_pred)
            elif reg_metric == 'wasserstein':
                return wasserstein_distance(self.problem.y_train, y_pred)
            else:
                raise ValueError('Unknown reg metric ' + str(reg_metric))

        else:
            raise ValueError('Unknown approximation type ' + str(self.problem.approx_type))


    def mate(self, pop):
        """Mates individuals in the population using the scheme
        defined in toolbox in-place

        Args:
            pop (list: Individual): List of individuals to be mated
        """
        # Apply crossover on the population by
        # choosing alternate individuals
        # e.g. if pop = [ind1, ind2, ind3, ind4],
        # we are doing 2-point crossover between
        # ind1, ind3 and ind2, ind4
        for child1, child2 in zip(pop[::2], pop[1::2]):
            if random.random() < self.cxpb:

                # In-place Crossover
                self.toolbox.mate(child1, child2)

                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                child1.fitness.delValues()
                child2.fitness.delValues()


    def mutate(self, pop):
        """Mutates individuals in the population using the scheme
        defined in toolbox in-place

        Args:
            pop (list: Individual): List of individuals to be mutated
        """
        # Apply mutation
        for mutant in pop:
            if random.random() < self.mutpb:

                # In-place Mutation
                self.toolbox.mutate(mutant)

                # Delete fitness values after crossover
                # because the individuals are changed
                # and will have different fitness values
                mutant.fitness.delValues()
