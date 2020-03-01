"""
This module contains the main function we'll use to run the
experiment to solve the problem using a specified
evolutionary algorithm
"""

import datetime
from dowel import logger, TextOutput, StdOutput
from os import listdir
from os.path import isfile, join
import time

from varro.algo.fit import fit
from varro.algo.predict import predict
from varro.util.util import make_path
from varro.util.variables import ABS_ALGO_EXP_LOGS_PATH, ABS_ALGO_HYPERPARAMS_PATH, ABS_ALGO_PREDICTIONS_PATH, DATE_NAME_FORMAT
from varro.util.args import get_args
from varro.algo.hyperparam_opt.grid_search import grid_search


def main():
    # Create Logs folder if not created
    make_path(ABS_ALGO_EXP_LOGS_PATH)
    make_path(ABS_ALGO_HYPERPARAMS_PATH)
    make_path(ABS_ALGO_PREDICTIONS_PATH)

    # Get the Arguments parsed from file execution
    args = get_args()

    if args.hyper_opt is not None:
        if args.hyper_opt == 'grid_search':
            grid_search()
        elif args.hyper_opt == 'bayesian_opt':
            raise NotImplementedError
        else:
            raise ValueError("Unknown hyperparameter optimization method.")
        return


    # Init Loggers
    log_path = join(ABS_ALGO_EXP_LOGS_PATH, "{}_{}.log".format(args.problem_type, datetime.datetime.now().strftime(DATE_NAME_FORMAT)))

    logger.add_output(StdOutput())
    logger.add_output(TextOutput(log_path))
    logger.log("Running Project Varro")
    logger.log("Purpose: " + args.purpose)

    if args.use_timer is True: 
        logger.set_timer(True)

    # Check if we're fitting or predicting
    if args.purpose == 'fit':
        # Start Optimization

        logger.start_timer()
        fit(model_type=args.model_type,
            problem_type=args.problem_type,
            strategy=args.strategy,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            imutpb=args.imutpb,
            imutmu=args.imutmu,
            imutsigma=args.imutsigma,
            popsize=args.popsize,
            elitesize=args.elitesize,
            ngen=args.ngen,
            ckpt=args.ckpt,
            ckpt_freq=args.ckpt_freq,
            novelty_metric=args.novelty_metric,
            halloffamesize=args.halloffamesize,
            earlystop=args.earlystop)
        logger.stop_timer('EXPERIMENT.PY Fitting complete')

    else:
        if args.ckptfolder:
            # Make predictions using the best
            # individual from each generation
            # in ckptfolder

            logger.start_timer()
            save_dir = join(ABS_ALGO_PREDICTIONS_PATH, args.ckptfolder.split('/')[-1])
            make_path(save_dir)
            ckpt_files = [join(args.ckptfolder, f) for f in listdir(args.ckptfolder) if isfile(join(args.ckptfolder, f))]
            for ckpt in ckpt_files:
                predict(model_type=args.model_type,
                        problem_type=args.problem_type,
                        strategy=args.strategy,
                        input_data=args.input_data,
                        ckpt=ckpt,
                        save_dir=save_dir)

            logger.stop_timer('EXPERIMENT.PY Making predictions using the best individual from each generation')

        else:
            # Make a single prediction

            logger.start_timer()
            save_dir = join(ABS_ALGO_PREDICTIONS_PATH, args.ckpt.split('/')[-2])
            make_path(save_dir)
            predict(model_type=args.model_type,
                    problem_type=args.problem_type,
                    strategy=args.strategy,
                    input_data=args.input_data,
                    ckpt=args.ckpt,
                    save_dir=save_dir)

            logger.stop_timer('EXPERIMENT.PY Making a single prediction')


if __name__ == "__main__":
    main()
