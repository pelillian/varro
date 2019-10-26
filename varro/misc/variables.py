"""
This module stores the static variables we'll need for the project
"""

import os


# This is the Project Root
ROOT_DIR = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])

# This is the path to algo folder in varro
ABSOLUTE_ALGO_LOGS_PATH = os.path.join(ROOT_DIR, 'logs/varro/algo')

# This is the folder that keeps snapshots
# of population in each generation of experiment
EXPERIMENT_CHECKPOINTS_PATH = os.path.join(ROOT_DIR, 'checkpoints/varro/algo')

# Whole population will be written in a pickled
# dictionary every FREQ generations
FREQ = 1
