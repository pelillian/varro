"""
This module contains the code for parsing arguments used by experiment.py
"""

import os
import argparse


def get_args():
    """Reads command-line arguments.

    Returns:
        (Namespace) Parsed attributes

    """
    parser = argparse.ArgumentParser(
        description='Runs an evolutionary algorithm to optimize \
            the parameters of a neural network or circuit configuration of an FPGA'
    )

    #####################
    # 1. Fit or Predict #
    #####################
    parser.add_argument('--purpose',
                        default='fit',
                        const='fit',
                        nargs='?',
                        metavar='PURPOSE',
                        action='store',
                        choices=['fit', 'predict'],
                        help='The purpose of running experiment')

    #########################################
    # 2. Checkpoint File to load population #
    #########################################
    parser.add_argument('--ckpt',
                        default=None,
                        const=None,
                        nargs='?',
                        metavar='CHECKPOINT-FILE-TO-LOAD-POPULATION',
                        action='store',
                        help='The checkpoint file to be used for prediction')

    ###################################################################################
    # 3. Checkpoint Folder to predict y_pred using best individual of each generation #
    ###################################################################################
    parser.add_argument('--ckptfolder',
                        default=None,
                        const=None,
                        nargs='?',
                        metavar='CHECKPOINT-FOLDER-TO-PREDICT',
                        action='store',
                        help='The checkpoint folder that contains the population of each generation')

    #######################################################
    # 4. .npy file to specify input data (features for prediction) #
    #######################################################
    parser.add_argument('--input_data',
                        nargs='?',
                        metavar='INPUT-DATA',
                        action='store',
                        help='The features to be used for predict')

    ######################################
    # 5. .npy file to specify labels (labels) #
    ######################################
    parser.add_argument('--labels',
                        nargs='?',
                        metavar='LABELS',
                        action='store',
                        help='The labels to be used for predict')

    ##########################
    # 6. FPGA or Neural Net? #
    ##########################
    parser.add_argument('--model_type',
                        default='nn',
                        const='nn',
                        nargs='?',
                        metavar='MODEL-TO-OPTIMIZE',
                        action='store',
                        choices=['fpga', 'nn'],
                        help='The target platform that the parameters are evaluated on')

    ######################################################
    # 7. What problem are we trying to solve / optimize? #
    ######################################################
    parser.add_argument('--problem_type',
                        default='sin',
                        const='sin',
                        nargs='?',
                        metavar='PROBLEM-TO-TACKLE',
                        action='store',
                        help='The problem to solve / optimize using an evolutionary strategy')

    ##########################################################
    # 8. Which strategy should we use to solve this problem? #
    ##########################################################
    parser.add_argument('--strategy',
                        default='sga',
                        const='sga',
                        nargs='?',
                        metavar='OPTIMIZATION-STRATEGY',
                        action='store',
                        choices=['sga', 'ns-es', 'nsr-es', 'cma-es'],
                        help='The optimization strategy chosen to solve the problem specified')

    #########################################################################
    # 9. What cross-over probability do you want for the evolutionary algo? #
    #########################################################################
    parser.add_argument('--cxpb',
                        default=0.5,
                        const=0.5,
                        nargs='?',
                        metavar='CROSSOVER-PROBABILITY',
                        action='store',
                        help='Set the Cross-over probability for offspring',
                        type=float)

    ########################################################################
    # 10. What mutation probability do you want for the evolutionary algo? #
    ########################################################################
    parser.add_argument('--mutpb',
                        default=0.01,
                        const=0.01,
                        nargs='?',
                        metavar='MUTATION-PROBABILITY',
                        action='store',
                        help='Set the Mutation probability',
                        type=float)

    #############################################################
    # 11. What is the individual attribute mutation probability #
    #############################################################
    parser.add_argument('--imutpb',
                        default=0.5,
                        const=0.5,
                        nargs='?',
                        metavar='IND-MUTATION-PROBABILITY',
                        action='store',
                        help='Set the Mutation probability for each attribute in the individual',
                        type=float)

    ##############################################################################
    # 12. What mean for gaussian distribution to pull the mutant attribute from? #
    ##############################################################################
    parser.add_argument('--imutmu',
                        default=0,
                        const=0,
                        nargs='?',
                        metavar='IND-MUTATION-MU',
                        action='store',
                        help='Set the mean for mutation probability distribution',
                        type=float)

    ############################################################################################
    # 13. What standard deviation for gaussian distribution to pull the mutant attribute from? #
    ############################################################################################
    parser.add_argument('--imutsigma',
                        default=0.1,
                        const=0.1,
                        nargs='?',
                        metavar='IND-MUTATION-SIGMA',
                        action='store',
                        help='Set the standard deviation for mutation probability distribution',
                        type=float)

    #########################################
    # 14. What population size do you want? #
    #########################################
    parser.add_argument('--popsize',
                        default=100,
                        const=100,
                        nargs='?',
                        metavar='POPULATION-SIZE',
                        action='store',
                        help='Set number of individuals in population',
                        type=int,
                        choices=range(2, 10000))

    ########################################################
    # 15. What elite size do you want? (Percentage of      #
    # best fitness individuals do you not want to change?) #
    ########################################################
    parser.add_argument('--elitesize',
                        default=0.2,
                        const=0.2,
                        nargs='?',
                        metavar='ELITE-SIZE',
                        action='store',
                        help='Percentage of fittest individuals to pass to next generation',
                        type=float)

    ################################################################################
    # 16. What number of generations do you want to run the evolutionary algo for? #
    ################################################################################
    parser.add_argument('--ngen',
                        default=100,
                        const=100,
                        nargs='?',
                        metavar='NUMBER-OF-GENERATIONS',
                        action='store',
                        help='Set the number of generations to evolve',
                        type=int)

    ###############################################################################
    # 17. What Distance metric will we use for measuring an individual's novelty? #
    ###############################################################################
    # Supported novelty metrics:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
    parser.add_argument('--novelty_metric',
                        default='euclidean',
                        const='euclidean',
                        nargs='?',
                        metavar='NOVELTY-METRIC',
                        action='store',
                        help='Set the distance metric to be used for measuring Novelty from this list: \
                            https://scikit-learn.org/stable/modules/generated/\
                                sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric',
                        type=str)

    ######################################################################################
    # 18. The size of HallOfFame which will store the best (fittest / novel) individuals #
    ######################################################################################
    # The hall of fame contains the best individuals of all time in the population. It is
    # lexicographically sorted at all times so that the first element of the hall of fame
    # is the individual that has the best first fitness value ever seen, according to the
    # weights provided to the fitness at creation time.
    parser.add_argument('--halloffamesize',
                        default=0.01,
                        const=0.01,
                        nargs='?',
                        metavar='HALLOFFAME-SIZE',
                        action='store',
                        help='Set the size of HallOfFame which will store the best (fittest / novel) individuals',
                        type=float)

    ######################################################################################
    # 19. Whether or not to stop the algorithm early when accuracy converges
    ######################################################################################
    parser.add_argument('--earlystop',
                        default=False,
                        const=False,
                        nargs='?',
                        metavar='EARLY-STOPPING',
                        action='store',
                        help='Set whether or not to stop the algorithm early when accuracy converges',
                        type=bool)

    ######################################################################################
    # 20. Whether or not to run hyperparameter optimization and the type to run
    ######################################################################################
    parser.add_argument('--hyper_opt',
                        default=None,
                        const=None,
                        nargs='?',
                        metavar='HYPERPARAMETER-OPTIMIZATION-TYPE',
                        action='store',
                        choices=[None, 'grid_search', 'bayesian_opt'],
                        help='The type of hyperparameter optimization to run')

    ######################################################################################
    # 21. Frequency to store checkpoints
    ######################################################################################
    parser.add_argument('--ckpt_freq',
                        default=10,
                        type=int,
                        nargs='?',
                        metavar='CHECKPOINT-FREQUENCY',
                        action='store',
                        help='Determines how many generations before a training run checkpoints')


    ######################################################################################
    # 22. Determine whether timing messages are logged
    ######################################################################################

    parser.add_argument('--use_timer',
                        default=False,
                        const=False,
                        nargs='?',
                        action='store',
                        help='Determine whether timing messages are logged',
                        type=bool)

    settings = parser.parse_args()

    # If we are predicting, we need to specify a
    # checkpoint file or a checkpoint folder, else
    # we can fit the NN or FPGA from scratch / evolve
    # from the checpoint
    if settings.purpose == 'predict' \
            and ((settings.ckpt is None and settings.ckptfolder is None) or settings.input_data is None):
        parser.error("--purpose='predict' requires --input_data and either --ckpt or --ckptfolder to be specified.")

    # Check that input_data and y are .npy files
    if settings.input_data and settings.input_data[-3:] != 'npy':
        parser.error("--input_data needs to be a .npy file.")
    if settings.labels and settings.labels[-3:] != 'npy':
        parser.error("--labels needs to be a .npy file.")

    # Check that sigma for the gaussian distribution were
    # mutating attribute of individual from is positive
    if settings.imutsigma < 0:
        parser.error("--imutsigma needs to be positive.")

    # Check that halloffame size is smaller than elitesize
    if settings.halloffamesize > settings.elitesize:
        parser.error("--halloffamesize must be smaller than --elitesize")

    # Check that elite size will be more than equal to 1
    if int(settings.elitesize*settings.popsize) < 1:
        parser.error("--elitesize too small")

    # Check that halloffame size will be more than equal to 1
    if int(settings.halloffamesize*settings.popsize) < 1:
        parser.error("--halloffamesize too small")

    return settings
