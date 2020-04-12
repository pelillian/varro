"""
This module stores the static variables we'll need for the project
"""

import os


# This is the Project Root
ROOT_DIR = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])

# This is the path to experiments folder in varro/algo
ABS_ALGO_EXP_LOGS_PATH = os.path.join(ROOT_DIR, 'logs/varro/algo/experiments')

# This is the path to hyperparameters folder in varro/algo
ABS_ALGO_HYPERPARAMS_PATH = os.path.join(ROOT_DIR, 'logs/varro/algo/hyperparams')

# This is the path to y_preds folder in varro/algo
ABS_ALGO_PREDICTIONS_PATH = os.path.join(ROOT_DIR, 'logs/varro/algo/predictions')

# This is the path to tensorboard logs folder in varro/algo
ABS_ALGO_TENSORBOARD_PATH = os.path.join(ROOT_DIR, 'logs/varro/algo/tensorboard')

# This is the folder that keeps snapshots
# of population in each generation of experiment
EXPERIMENT_CHECKPOINTS_PATH = os.path.join(ROOT_DIR, 'checkpoints/varro/algo')

DATE_NAME_FORMAT = "%Y-%b-%d-%H:%M:%S"

# Grid Search

# This is the folder that houses checkpoints
# for grid search
GRID_SEARCH_CHECKPOINTS_PATH = os.path.join(ROOT_DIR, 'checkpoints/varro/algo/grid_search')

# This is the folder that houses the config for grid search
GRID_SEARCH_CONFIG_PATH = os.path.join(ROOT_DIR, 'tests/varro/algo/grid_search')

# Interface
FPGA_BITSTREAM_SHAPE = (13294, 1136)
PRJTRELLIS_DATABASE = "../prjtrellis-db"
CHIP_NAME = "LFE5UM5G-85F"
CHIP_COMMENT = ".comment Part: LFE5UM5G-85F-8CABGA381"
ARDUINO_PORT = "/dev/ttyACM0"
FPGA_CONFIG_DIR = "data/config"
SLEEP_TIME = 0.005


