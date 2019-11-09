"""
This module contains the class for Evaluate, with one constructor for FPGA and one for Neural Networks
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
from math import sqrt
from deap import tools

from varro.algo.problems import Problem


def fitness_score(model,
                  problem):
    """Calculates the fitness score for a particular
    model configuration on the problem specified

    Args:
        model (Model): A Model object that will be used for evaluation with loaded parameters
        problem (Problem): A Problem object that contains information about the training data

    Returns:
        Returns the fitness score of the model w.r.t. the problem specified
    """
    # Predict labels
    y_pred = np.array(model.predict(problem.X_train, problem=problem))

    if problem.approx_type == Problem.CLASSIFICATION:
        if problem.name == 'mnist':
            categorical_accuracy = accuracy_score(y_true=problem.y_train,
                                                  y_pred=np.argmax(y_pred, axis=-1))
        else:
            categorical_accuracy = accuracy_score(y_true=problem.y_train,
                                                  y_pred=(np.array(y_pred) > 0.5).astype(float))
        return -categorical_accuracy

    elif problem.approx_type == Problem.REGRESSION:
        rmse = sqrt(mean_squared_error(problem.X_train, y_pred))
        return rmse

    else:
        raise ValueError('Unknown approximation type ' + str(problem.approx_type))


def compute_novelty(population,
                    similarity_metric='euclidean',
                    k=5):
    """Calculates the novelty scores for each individual in the
    population using a nearest neighbors approach according to
    http://eplex.cs.ucf.edu/noveltysearch/userspage/#howtoimplement

    Args:
        population (list): An iterable of np.ndarrays that represent the individuals
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
    """
    tree = BallTree(population, metric=similarity_metric)

    for ind in population:

        # Get the k-nearest neighbors of
        # the individual
        dist, ind_idxs = tree.query(ind, k=k)

        # Ignore first value as it'll be 0 since
        # there's an instance of the same vector in
        # population
        ind.fitness.values.novelty_score = np.mean(dist[1:])


def compute_fitness(population,
                    model,
                    problem,
                    strategy,
                    similarity_metric=None):
    """Calculates the fitness scores for the entire Population

    Args:
        population (list): An iterable of np.ndarrays that represent the individuals
        model (Model): A Model object that will be used for evaluation
        problem (Problem): A Problem object that contains information about the training data
        strategy (str): The strategy to be used for evolving, Simple Genetic Algorithm (sga) / Novelty Search (ns) / Covariance-Matrix Adaptation (cma-es)
        similarity_metric (str): Optional similarity_metric to be used for novelty search

    Returns:
        Tuple of (Average fitness score of population, \
            Number of individuals with invalid fitness scores that have been evaluated)
    """
    if strategy == 'sga':
        Fitness = namedtuple('Fitness', ['fitness_score'])
    elif strategy == 'ns-es':
        Fitness = namedtuple('Fitness', ['novelty_score'])
    elif strategy == 'nsr-es':
        Fitness = namedtuple('Fitness', ['fitness_score', 'novelty_score'])
    elif strategy == 'cma-es':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Get fitness scores if either Simple
    # genetic or Novelty Search Reward
    if strategy == 'sga' or strategy == 'nsr-es':

        # Evaluate the individuals with an invalid fitness
        # (These are the individuals that have not been evaluated before -
        # individuals at the start of the evolutionary algorithm - or those
        # that have been mutated / the offspring after crossover with fitness deleted)
        invalid_inds = [ind for ind in population if not ind.fitness.valid]

        # Get fitness score for each individual with
        # invalid fitness score in population
        for ind in invalid_inds:

            # Load Weights into model using individual
            model.load_parameters(ind)

            # Calculate the Fitness score of the individual
            ind.fitness.values = Fitness(fitness_score=fitness_score(model, problem))

    # Get fitness scores if either Simple
    # genetic or Novelty Search Reward
    if strategy == 'ns-es' or strategy == 'nsr-es':
        # Calculate the Novelty scores for all individuals
        compute_novelty(population, similarity_metric)

        tools.sortNondominated(individuals, k, first_front_only=False)

    # In-place sorting of the Population
    # with all its fitness scores computed
    # TODO: Sort by fitness score,
    # Create archive for the individuals that are novel / paretofront

    return np.mean([ind.fitness.values.fitness_score for ind in population]), len(invalid_inds)


def evaluate(population,
             model,
             problem,
             strategy,
             similarity_metric=None):
    """Evaluates an entire population on a dataset on the neural net / fpga
    architecture specified by the model, and calculates the fitness scores for
    each individual, sorting the entire population by fitness scores in-place

    Args:
        population (list): An iterable of np.ndarrays that represent the individuals
        model (Model): A Model object that will be used for evaluation
        problem (Problem): A Problem object that contains information about the training data
        strategy (str): The strategy to be used for evolving, Simple Genetic Algorithm (sga) / Novelty Search (ns) / Covariance-Matrix Adaptation (cma-es)
        similarity_metric (str): Optional similarity_metric to be used for novelty search

    Returns:
        Tuple of (Average fitness score of population, \
            Number of individuals with invalid fitness scores that have been evaluated)

    """
    # Re-generates the training set for the problem (if possible) to prevent overfitting
    problem.reset_train_set()

    return compute_fitness(population, model, problem, strategy)
