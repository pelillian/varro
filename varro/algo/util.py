"""
This module contains the utility functions to run the experiment.py
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
            the weights of a neural network or circuit configuration of an FPGA'
    )

    ######################
    # 0A. Fit or Predict #
    ######################
    parser.add_argument('--purpose',
                        default='fit',
                        const='fit',
                        nargs='?',
                        metavar='PURPOSE',
                        action='store',
                        choices=['fit', 'predict'],
                        help='The purpose of running experiment')

    ##########################################
    # 0B. Checkpoint File to load population #
    ##########################################
    parser.add_argument('--ckpt',
                        nargs='?',
                        metavar='CHECKPOINT-FILE-TO-LOAD-POPULATION',
                        action='store',
                        help='The checkpoint file to be used for prediction')

    ########################################################
    # 0C. .npy file to specify X (features for prediction) #
    ########################################################
    parser.add_argument('--X',
                        nargs='?',
                        metavar='FEATURES',
                        action='store',
                        help='The features to be used for predict')

    #######################################
    # 0D. .npy file to specify y (labels) #
    #######################################
    parser.add_argument('--y',
                        nargs='?',
                        metavar='LABELS',
                        action='store',
                        help='The labels to be used for predict')

    ##########################
    # 1. FPGA or Neural Net? #
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
    # 2. What problem are we trying to solve / optimize? #
    ######################################################
    parser.add_argument('--problem_type',
                        default='sinx',
                        const='sinx',
                        nargs='?',
                        metavar='PROBLEM-TO-TACKLE',
                        action='store',
                        choices=['x', 'sinx', 'cosx', 'tanx', 'ras', 'rosen', 'step', 'mnist'],
                        help='The problem to solve / optimize using an evolutionary strategy')

    ##########################################################
    # 3. Which strategy should we use to solve this problem? #
    ##########################################################
    parser.add_argument('--strategy',
                        default='ea',
                        const='ea',
                        nargs='?',
                        metavar='OPTIMIZATION-STRATEGY',
                        action='store',
                        choices=['ea', 'cma-es', 'ns'],
                        help='The optimization strategy chosen to solve the problem specified')

    ##########################################################################
    # 3a. What cross-over probability do you want for the evolutionary algo? #
    ##########################################################################
    parser.add_argument('--cxpb',
                        default=0.5,
                        const=0.5,
                        nargs='?',
                        metavar='CROSSOVER-PROBABILITY',
                        action='store',
                        help='Set the Cross-over probability for offspring',
                        type=float)

    ########################################################################
    # 3b. What mutation probability do you want for the evolutionary algo? #
    ########################################################################
    parser.add_argument('--mutpb',
                        default=0.01,
                        const=0.01,
                        nargs='?',
                        metavar='MUTATION-PROBABILITY',
                        action='store',
                        help='Set the Mutation probability',
                        type=float)

    #########################################
    # 3c. What population size do you want? #
    #########################################
    parser.add_argument('--popsize',
                        default=10,
                        const=10,
                        nargs='?',
                        metavar='POPULATION-SIZE',
                        action='store',
                        help='Set number of individuals in population',
                        type=int)

    ########################################################
    # 3d. What elite size do you want? (Percentage of      #
    # best fitness individuals do you not want to change?) #
    ########################################################
    parser.add_argument('--elitesize',
                        default=0.1,
                        const=0.1,
                        nargs='?',
                        metavar='ELITE-SIZE',
                        action='store',
                        help='Percentage of fittest individuals to pass to next generation',
                        type=float)

    ################################################################################
    # 3e. What number of generations do you want to run the evolutionary algo for? #
    ################################################################################
    parser.add_argument('--ngen',
                        default=100,
                        const=100,
                        nargs='?',
                        metavar='NUMBER-OF-GENERATIONS',
                        action='store',
                        help='Set the number of generations to evolve',
                        type=int)

    settings = parser.parse_args()

    # If we are predicting, we need to specify a
    # checkpoint file, else we can fit the NN or
    # FPGA from scratch / evolve from the checpoint
    if settings.purpose == 'predict' \
        and settings.ckpt is None \
        and settings.X is None:
        parser.error("--purpose == 'predict' requires --ckpt and --X to be specified.")

    # Check that X and y are .npy files
    if settings.X and settings.X[-3:] != 'npy':
        parser.error("--X needs to be a .npy file.")
    if settings.y and settings.y[-3:] != 'npy':
        parser.error("--y needs to be a .npy file.")

    return settings
