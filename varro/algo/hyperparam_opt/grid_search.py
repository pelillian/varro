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

from varro.util.variables import GRID_SEARCH_CHECKPOINTS_PATH
from varro.util.util import make_path
from varro.algo.fit import fit

HYPERPARAM_DICT = {}
HYPERPARAM_DICT['model_type'] = ['fpga']
HYPERPARAM_DICT['imutpb'] = [1e-4, 5e-5, 1e-5]
HYPERPARAM_DICT['ngen'] = [50]
HYPERPARAM_DICT['popsize'] = [100, 30, 5]
HYPERPARAM_DICT['strategy'] = ['sga', 'nsr-es']

# Ensure checkpoints dir is on local
make_path(GRID_SEARCH_CHECKPOINTS_PATH)

def grid_search():
    """ Pickles the best permutation for given ranges of hyperparameters

    """
    import pdb;pdb.set_trace()
    print('o')
    # fit for each argument permutation
    # IF HYPERPARAMETERS ARE ADDED:
    #    note that aperm indexes hyperparams alphabetically

#    for aperm in product(*[*HYPERPARAM_DICT.values()]):
#        print('python experiment.py --model_type={} --imputpb={} --ngen={} --popsize={} strategy={} --novelty_metric={} --grid_search=True'.format(aperm[1], aperm[0], aperm[2], aperm[4], aperm[3], 'hamming' if aperm[4] == 'nsr-es' else None))
