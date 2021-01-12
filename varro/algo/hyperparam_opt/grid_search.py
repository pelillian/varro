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
HYPERPARAM_DICT['imutpb'] = [1e-3, 1e-4, 1e-5]
HYPERPARAM_DICT['ngen'] = [25]
HYPERPARAM_DICT['popsize'] = [30, 5]
HYPERPARAM_DICT['strategy'] = ['sga', 'nsr-es']
HYPERPARAM_DICT['problem_type'] = ['simple_step']
HYPERPARAM_DICT['imutpb_decay'] = [0.8, 0.9, 0.95, 0.99]
HYPERPARAM_DICT['lambda_penalty'] = [10, 100, 1000]

def main():
    # fit for each argument permutation
    # IF HYPERPARAMETERS ARE ADDED:
    #    note that aperm indexes hyperparams alphabetically
    for idx, aperm in enumerate(product(*[*HYPERPARAM_DICT.values()])):
        print('Experiment ', idx)
        os.system('python -m varro.algo.experiment --ckpt_freq=1000 --purpose=fit --mutpb 1.0 --model_type=\'{}\' --imutpb={} --ngen={} --popsize={} --strategy=\'{}\' --problem_type=\'{}\' --imutpb_decay={} --novelty_metric=\'{}\' --lambda_penalty=\'{}\' --halloffamesize=0.2'.format(aperm[0], aperm[1], aperm[2], aperm[3], aperm[4], aperm[5], aperm[6], 'hamming' if aperm[4] == 'nsr-es' else None, aperm[7]))

if __name__ == '__main__':
    main()
