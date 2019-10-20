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
                        choices=['x', 'sinx', 'cosx', 'tanx', 'ras', 'rosen', 'mnist'], 
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
                        default=0.2,
                        const=0.2,
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

    ################################################################################
    # 3d. What number of generations do you want to run the evolutionary algo for? #
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
    
    return settings

