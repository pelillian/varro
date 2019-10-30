'''

*** UNTESTED ON SERVER, WORK IN PROGRESS :) ***

This module contains the utility functions to run grid search for experiment.py

# TODO: 
- Create directory structure for logs, pkls
    - Proposed:

        logs/
            varro/
                grid_search/
                            trial_x/ <- experment_path
                                    /log.log
                                    /fittest.pkl

        checkpoints/
            varro/
                grid_search/
                            trial_x/
                                    /log.log
                                    /fittest.pkl

- Parallelize

'''

import argparse
from itertools import product
import logging
import numpy as np
import os
import pickle
from shutil import rmtree
from varro.misc.variables import GRID_SEARCH_LOGS_PATH, GRID_SEARCH_CHECKPOINTS_PATH
from varro.algo.experiment import fit
from varro.misc.util import make_path

# Create Logs folder if not created
make_path(GRID_SEARCH_LOGS_PATH)
make_path(GRID_SEARCH_CHECKPOINTS_PATH)

def grid_search(problem_type, 
                strategy):
    """ Pickles the best permutation of lists of hyperparameters

    Args:
        problem_type(list): 
        strategy(list): 

    """

    params = {}
    params['problem_type'] = problem_type if isinstance(problem_type, list) else [problem_type]
    params['strategy'] = strategy if isinstance(strategy, list) else [strategy]
    params['cxpb'] = [0.1, 0.3, 0.6]
    params['mutpb'] = [0.1, 0.4]
    params['popsize'] = [10, 100, 1000]
    params['elitesize'] = [0.05, 0.1, 0.3]
    params['ngen'] = [100]
    fittest = {'fitness': 42069, 'args': {}, 'weights': None}
    all_runs = []

    # fit() for each argument permutation
    for aperm in product(*[*params.values()]):
        args = {'problem_type': aperm[0],
                'strategy': aperm[1],
                'cxpb': aperm[2],
                'mutpb': aperm[3],
                'popsize': aperm[4],
                'elitesize': aperm[5],
                'ngen': aperm[6]}

        fit(model_type='nn',
            problem_type=args['problem_type'],
            strategy=args['strategy'],
            cxpb=args['cxpb'],
            mutpb=args['mutpb'],
            popsize=args['popsize'],
            elitesize=args['elitesize'],
            ngen=args['ngen'])

        # Create temp folder to house checkpoints
        experiment_name = '{}-'\
                            'ps{}-'\
                            'es{}-'\
                            'ng{}-'\
                            'cx{}-'\
                            'mp{}'.format(args['problem_type'],\
                                            args['popsize'],\
                                            args['elitesize'],\
                                            args['ngen'],\
                                            args['cxpb'],\
                                            args['mutpb'])

        experiment_path = os.path.join(GRID_SEARCH_CHECKPOINTS_PATH, experiment_name)
        if not os.path.exists(experment_path):
            mkdir(experiment_path)

        pkl_path = os.path.join(experiment_path, 'checkpoint_gen{}.pkl'.format(args['ngen']-1))

        with open(pkl_path, 'rb') as cp_file:
            cp = pickle.load(cp_file)

        halloffame = cp['halloffame']
        fitness = halloffame.fitness.values[0]

        if fitness < fittest['fitness']:
            fittest['args'] = args
            fittest['fitness'] = fitness
            fittest['weights'] = halloffame

        args['fitness'] = fitness
        all_runs.append(args)

        rmtree(experiment_path)

    fittest_pkl_path = os.path.join(experiment_path, 'fittest.pkl')
    with open(fittest_pkl_path, 'w') as fittest_file:
        pickle.dump([fittest, all_runs], fittest_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Pickles the best permutation of lists of hyperparameters')

    parser.add_argument('--problem_type', 
                        default='x',
                        const='x',
                        nargs='?',
                        metavar='PROBLEM-TO-TACKLE', 
                        action='store', 
                        choices=['x', 'sinx', 'cosx', 'tanx', 'ras', 'rosen', 'step', 'mnist'], 
                        help='The problem to solve / optimize using an evolutionary strategy')

    parser.add_argument('--strategy', 
                        default='ea',
                        const='ea',
                        nargs='?',
                        metavar='OPTIMIZATION-STRATEGY', 
                        action='store', 
                        choices=['ea', 'cma-es', 'ns'], 
                        help='The optimization strategy chosen to solve the problem specified')

    args = parser.parse_args()

    grid_search(args.problem_type, args.strategy)