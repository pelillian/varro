import pickle
import numpy as np
from dowel import logger
from os.path import join
import time

from varro.algo.problems import Problem, ProblemFuncApprox, ProblemMNIST
from varro.algo.strategies.sga import StrategySGA
from varro.algo.strategies.moga import StrategyMOGA
from varro.algo.strategies.ns_es import StrategyNSES
from varro.algo.strategies.nsr_es import StrategyNSRES

def predict(model_type,
            problem_type,
            strategy,
            X,
            ckpt,
            save_dir):
    """Predicts the output from loading the model saved in checkpoint
    and saves y_pred into same path as X but with a _y_pred in the name

    Args:
        model_type (str): A string specifying whether we're optimizing on a neural network
            or field programmable gate array
        problem_type (str): A string specifying what type of problem we're trying to optimize
        strategy (str): A string specifying what type of optimization algorithm to use
        X (str): Path to the .npy that stores the np.ndarray to use as Input data for model
        ckpt (str): Location of checkpoint to load the population
        save_dir (str): Location of where to store the predictions

    """

    # 1. Choose Problem and get the specific evaluation function
    # for that problem

    logger.start_timer()
    logger.log("Loading problem...")
    if problem_type == 'mnist':
        problem = ProblemMNIST()
    else:
        problem = ProblemFuncApprox(func=problem_type)

    logger.stop_timer('PREDICT.PY Choosing evaluation function for problem')
    logger.start_timer()

    # 1. Choose Target Platform
    logger.log("Loading target platform...")
    if model_type == 'nn':
        from varro.algo.models import ModelNN as Model  # Import here so we don't load tensorflow if not needed
    elif model_type == 'fpga':
        from varro.algo.models import ModelFPGA as Model
    model = Model(problem)

    logger.stop_timer('PREDICT.PY Choosing target platform')
    logger.start_timer()


    if ckpt.endswith(".bit"):
        logger.log("Loading data from bit file...")
        from varro.fpga.util import bit_to_cram

        logger.start_timer()
        predict_ind = bit_to_cram(ckpt)
        parameters = bit_to_cram(ckpt)
        
        logger.stop_timer('PREDICT.PY Loading data from bit file')
        logger.start_timer()

    elif ckpt.endswith(".pkl"):
        logger.log("Loading data from pickle file...")

        logger.start_timer()
        with open(ckpt, "rb") as cp_file:
            if strategy == 'sga':
                StrategySGA.init_fitness_and_inds()
            elif strategy == 'moga':
                StrategyMOGA.init_fitness_and_inds()
            elif strategy == 'ns-es':
                StrategyNSES.init_fitness_and_inds()
            elif strategy == 'nsr-es':
                StrategyNSRES.init_fitness_and_inds()
            elif strategy == 'cma-es':
                raise NotImplementedError
            else:
                raise NotImplementedError

            # Initialize individual based on strategy
            cp = pickle.load(cp_file)
            predict_ind = cp["halloffame"][0]
            logger.stop_timer('PREDICT.PY Loading data from pickle file')
            logger.start_timer()
            parameters = cp["halloffame"][0]

    else:
        raise ValueError("Checkpoint file has unrecognised extension.")

    logger.start_timer()
    # Load Weights into model using individual
    model.load_parameters(parameters)

    logger.stop_timer('PREDICT.PY Loading weights into model')
    logger.start_timer()

    # Predict labels using np array in X
    logger.log("Running model.predict")
    y_pred = np.array(model.predict(np.load(X)))
    logger.log(str(y_pred))
    logger.stop_timer('PREDICT.PY Predicting labels using np array')

    # Save the y_pred into a file
    y_pred_path = join(save_dir, ckpt.split('_')[-1][:-4] + '_' + X[:-4].split('/')[-1] + '_y_pred.npy')
    np.save(y_pred_path, y_pred)
