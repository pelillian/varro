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

HYPERPARAM_DICT = {}
HYPERPARAM_DICT['model_type'] = ['fpga']
HYPERPARAM_DICT['imutpb'] = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
HYPERPARAM_DICT['ngen'] = [50]
HYPERPARAM_DICT['popsize'] = [100, 30, 5]
HYPERPARAM_DICT['strategy'] = ['sga', 'nsr-es']
HYPERPARAM_DICT['problem_type'] = ['simple_step']

def main():
    # fit for each argument permutation
    # IF HYPERPARAMETERS ARE ADDED:
    #    note that aperm indexes hyperparams alphabetically
    for idx, aperm in enumerate(product(*[*HYPERPARAM_DICT.values()])):
        print('Experiment ', idx)
        os.system('python -m varro.algo.experiment --purpose=fit --mutpb 1.0 --imutpb_decay=0.95 --model_type=\'{}\' --imutpb={} --ngen={} --popsize={} --strategy=\'{}\' --problem_type=\'{}\' --novelty_metric=\'{}\''.format(aperm[0], aperm[1], aperm[2], aperm[3], aperm[4], aperm[5], 'hamming' if aperm[4] == 'nsr-es' else None))

if __name__ == '__main__':
    main()
