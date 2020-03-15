"""
This module contains bayesian optimization for hyperparameter tuning
- http://krasserm.github.io/2018/03/21/bayesian-optimization/
"""

import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization

from varro.algo.experiment import fit


# STATIC VARS
MODEL_TYPE = 'nn'

# Hyperparameter Bounds
def bounds(model_type=MODEL_TYPE):
    """Returns the Bounds for each hyperparameter we're tuning for Bayesian optimization
    """
    return \
    [
        {'name': 'cxpb', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'mutpb', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'imutpb', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'imutmu', 'type': 'continuous', 'domain': (-1, 1)},
        {'name': 'imutsigma', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'elitesize', 'type': 'continuous', 'domain': (0, 0.5)}
    ] \
    if MODEL_TYPE == 'nn' else \
    [
        {'name': 'cxpb', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'mutpb', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'imutpb', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'elitesize', 'type': 'continuous', 'domain': (0, 0.5)}
    ]


# Optimization objective
def best_fitness_score(parameters, model_type=MODEL_TYPE):
    """Runs experiment with configurations specified and returns the best
    individual's fitness score

    Args:
        parameters (list): List of parameters
        model_type (str): Which model, neural network or fpga

    Returns:
        best individual's fitness score
    """
    parameters = parameters[0]
    if MODEL_TYPE == 'nn':
        score = np.array(
            fit(model_type=MODEL_TYPE,
                problem_type='sin',
                strategy='nsr-es',
                cxpb=parameters[0],
                mutpb=parameters[1],
                imutpb=parameters[2],
                imutmu=parameters[3],
                imutsigma=parameters[4],
                popsize=100,
                elitesize=parameters[5],
                ngen=100,
                ckpt=None,
                novelty_metric='wasserstein',
                halloffamesize=0.01)
            )
    else:
        score = np.array(
            fit(model_type=MODEL_TYPE,
                problem_type='sin',
                strategy='nsr-es',
                cxpb=parameters[0],
                mutpb=parameters[1],
                imutpb=parameters[2],
                imutmu=None,
                imutsigma=None,
                popsize=10,
                elitesize=parameters[3],
                ngen=100,
                ckpt=None,
                novelty_metric='wasserstein',
                halloffamesize=0.1)
            )

    return score


def bayesian_opt():
    optimizer = BayesianOptimization(f=best_fitness_score,
                                     domain=bounds(),
                                     model_type='GP',
                                     acquisition_type ='EI',
                                     acquisition_jitter = 0.05,
                                     exact_feval=True,
                                     maximize=False)

    # Only 20 iterations because we have 5 initial random points
    optimizer.run_optimization(max_iter=20)
    print(np.maximum.accumulate(-optimizer.Y).ravel())
