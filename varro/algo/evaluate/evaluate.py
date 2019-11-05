"""
This module contains the class for Evaluate, with one constructor for FPGA and one for Neural Networks
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

from varro.algo.problems import Problem


def evaluate(population, model, problem):
    """Evaluates an entire population on a dataset on the neural net
    architecture specified by the model, and calculates the negative categorical
    accuracy of each individual.

    Args:
        population: An iterable of np.ndarrays that represent the individuals
        model: Model to be used for evaluation
        X: Training Input
        y: Training Labels

    Returns:
        (np.ndarray): the fitness scores (list) of the individuals

    """
    # Re-generates the training set for the problem (if possible) to prevent overfitting
    problem.reset_train_set()

    # loads training set
    X = problem.X_train
    y = problem.y_train
    approx_type = problem.approx_type

    # Initialize list to keep the fitness scores
    fitness_scores = []

    # Get fitness score for each individual in population
    for individual in population:

        # Load Weights into model using individual
        model.load_parameters(individual)

        # Predict labels
        y_pred = np.array(model.predict(X, problem=problem))

        if approx_type == Problem.CLASSIFICATION:
            if problem.name == 'mnist'
                categorical_accuracy = accuracy_score(y_true=y, y_pred=np.argmax(y_pred, axis=-1))
            else:
                categorical_accuracy = accuracy_score(y_true=y, y_pred=(np.array(y_pred) > 0.5).astype(float))
            fitness_scores.append([-categorical_accuracy])
        elif approx_type == Problem.REGRESSION:
            rmse = sqrt(mean_squared_error(y, y_pred))
            fitness_scores.append([rmse])
        else:
            raise ValueError('Unknown approximation type ' + str(problem.approx_type))

    return np.array(fitness_scores)
