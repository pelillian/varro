'''This module contains the utility functions to run grid search for experiment.py

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
import numpy as np
import os
import pickle
from shutil import rmtree
from tests.varro.algo.grid_search.config import HYPERPARAM_DICT
from varro.misc.variables import GRID_SEARCH_CHECKPOINTS_PATH
from varro.algo.experiment import fit
from varro.misc.util import make_path

# Ensure checkpoints dir is on local
make_path(GRID_SEARCH_CHECKPOINTS_PATH)

def grid_search():
    """ Pickles the best permutation for given ranges of hyperparameters

    """

    params = HYPERPARAM_DICT

    fittest = {'fitness': 42069, 'args': {}, 'weights': None}
    all_runs = []

    # fit for each argument permutation
    # IF HYPERPARAMETERS ARE ADDED:
    #    note that aperm indexes hyperparams alphabetically
    for aperm in product(*[*params.values()]):
        args = {'cxpb': aperm[0],
                'elitesize': aperm[1],
                'imutpb': aperm[2],
                'imutsigma': aperm[3],
                'mutpb': aperm[4],
                'ngen': aperm[5],
                'popsize': aperm[6],
                'problem_type': aperm[7],
                'strategy': aperm[8]}

        fit(model_type='nn',
            problem_type=args['problem_type'],
            strategy=args['strategy'],
            cxpb=args['cxpb'],
            mutpb=args['mutpb'],
            imutpb=args['imutpb'],
            imutmu=0,
            imutsigma=args['imutsigma'],
            popsize=args['popsize'],
            elitesize=args['elitesize'],
            ngen=args['ngen'],
            grid_search=True)

        # Create temp folder to house checkpoints
        experiment_path = os.path.join(GRID_SEARCH_CHECKPOINTS_PATH, 'tmp')
        last_gen = max([int(f.split('_')[-1].split('.')[0][3:]) for f in os.listdir(experiment_path)])
        pkl_path = os.path.join(experiment_path, 'checkpoint_gen{}.pkl'.format(last_gen))

        with open(pkl_path, 'rb') as cp_file:
            cp = pickle.load(cp_file)

        halloffame = cp['halloffame']
        fitness = halloffame.fitness.values[0]

        if fitness < fittest['fitness']:
            fittest['args'] = args
            fittest['fitness'] = fitness
            fittest['parameters'] = halloffame

        args['fitness'] = fitness
        all_runs.append(args)

        rmtree(experiment_path)

    fittest_pkl_path = os.path.join(experiment_path, 'fittest.pkl')
    with open(fittest_pkl_path, 'w') as fittest_file:
        pickle.dump([fittest, all_runs], fittest_file)

if __name__ == '__main__':

    grid_search()
