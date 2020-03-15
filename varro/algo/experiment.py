"""
This module contains the main function we'll use to run the
experiment to solve the problem using a specified
evolutionary algorithm
"""

from datetime import datetime
from dowel import logger, TextOutput, StdOutput
from os import listdir
from os.path import isfile, join

from varro.algo.fit import fit
from varro.algo.predict import predict
from varro.util.util import make_path
from varro.util.variables import ABS_ALGO_EXP_LOGS_PATH, ABS_ALGO_HYPERPARAMS_PATH, ABS_ALGO_PREDICTIONS_PATH, DATE_NAME_FORMAT, EXPERIMENT_CHECKPOINTS_PATH, GRID_SEARCH_CHECKPOINTS_PATH
from varro.util.args import get_args


def main():
    # Create Logs folder if not created
    make_path(ABS_ALGO_EXP_LOGS_PATH)
    make_path(ABS_ALGO_HYPERPARAMS_PATH)
    make_path(ABS_ALGO_PREDICTIONS_PATH)

    # Get the Arguments parsed from file execution
    args = get_args()

    experiment_name = args.model_type + '_' + args.problem_type + '_' + datetime.now().strftime(DATE_NAME_FORMAT)

    # Init Loggers
    log_path = join(ABS_ALGO_EXP_LOGS_PATH, experiment_name + '.log')

    logger.add_output(StdOutput())
    logger.add_output(TextOutput(log_path))
    logger.log("Running Project Varro")
    logger.log("Purpose: " + args.purpose)

    if args.use_timer: 
        logger.set_timer(True)

    if args.hyper_opt is not None:
        if args.hyper_opt == 'grid_search':
            from varro.algo.hyperparam_opt.grid_search import grid_search
            checkpoint_dir = join(GRID_SEARCH_CHECKPOINTS_PATH, 'tmp')
            make_path(checkpoint_dir)
            grid_search()
        elif args.hyper_opt == 'bayesian_opt':
            raise NotImplementedError
        else:
            raise ValueError("Unknown hyperparameter optimization method.")
        return
    else:
        checkpoint_dir = join(EXPERIMENT_CHECKPOINTS_PATH, experiment_name)
        make_path(checkpoint_dir)


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
            earlystop=args.earlystop,
            ckpt_dir=checkpoint_dir)
        logger.stop_timer('EXPERIMENT.PY Fitting complete')

    else:
        if args.ckptfolder:
            # Make predictions using the best individual from each generation in ckptfolder

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
