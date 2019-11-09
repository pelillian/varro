"""
This module contains the utility functions to run the experiment.py
"""

import pickle
import os
import argparse
from deap import base, creator, tools


def get_args():
    """Reads command-line arguments.

    Returns:
        (Namespace) Parsed attributes

    """
    parser = argparse.ArgumentParser(
        description='Runs an evolutionary algorithm to optimize \
            the parameters of a neural network or circuit configuration of an FPGA'
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
                        default=None,
                        const=None,
                        nargs='?',
                        metavar='CHECKPOINT-FILE-TO-LOAD-POPULATION',
                        action='store',
                        help='The checkpoint file to be used for prediction')

    ####################################################################################
    # 0C. Checkpoint Folder to predict y_pred using best individual of each generation #
    ####################################################################################
    parser.add_argument('--ckptfolder',
                        default=None,
                        const=None,
                        nargs='?',
                        metavar='CHECKPOINT-FOLDER-TO-PREDICT',
                        action='store',
                        help='The checkpoint folder that contains the population of each generation')

    ########################################################
    # 0D. .npy file to specify X (features for prediction) #
    ########################################################
    parser.add_argument('--X',
                        nargs='?',
                        metavar='FEATURES',
                        action='store',
                        help='The features to be used for predict')

    #######################################
    # 0E. .npy file to specify y (labels) #
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
                        help='The problem to solve / optimize using an evolutionary strategy')

    ##########################################################
    # 3. Which strategy should we use to solve this problem? #
    ##########################################################
    parser.add_argument('--strategy',
                        default='sga',
                        const='sga',
                        nargs='?',
                        metavar='OPTIMIZATION-STRATEGY',
                        action='store',
                        choices=['sga', 'ns-es', 'nsr-es', 'cma-es'],
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

    #############################################################
    # 3c. What is the individual attribute mutation probability #
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
    # 3d. What mean for gaussian distribution to pull the mutant attribute from? #
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
    # 3e. What standard deviation for gaussian distribution to pull the mutant attribute from? #
    ############################################################################################
    parser.add_argument('--imutsigma',
                        default=1,
                        const=1,
                        nargs='?',
                        metavar='IND-MUTATION-SIGMA',
                        action='store',
                        help='Set the standard deviation for mutation probability distribution',
                        type=float)

    #########################################
    # 3f. What population size do you want? #
    #########################################
    parser.add_argument('--popsize',
                        default=10,
                        const=10,
                        nargs='?',
                        metavar='POPULATION-SIZE',
                        action='store',
                        help='Set number of individuals in population',
                        type=int,
                        choices=range(2, 10000))

    ########################################################
    # 3g. What elite size do you want? (Percentage of      #
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
    # 3h. What number of generations do you want to run the evolutionary algo for? #
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
    # checkpoint file or a checkpoint folder, else
    # we can fit the NN or FPGA from scratch / evolve
    # from the checpoint
    if settings.purpose == 'predict' \
        and settings.ckpt is None \
        and settings.ckptfolder is None \
        and settings.X is None:
        parser.error("--purpose == 'predict' requires --ckpt or --ckptfolder and --X to be specified.")

    # Check that X and y are .npy files
    if settings.X and settings.X[-3:] != 'npy':
        parser.error("--X needs to be a .npy file.")
    if settings.y and settings.y[-3:] != 'npy':
        parser.error("--y needs to be a .npy file.")

    # Check that sigma for the gaussian distribution were
    # mutating attribute of individual from is positive
    if settings.imutsigma < 0:
        parser.error("--imutsigma needs to be positive.")

    return settings


def init_ind_fitness(strategy):
    """Creates the fitness and individual templates required for DEAP
    evolutionary strategies

    Args:
        strategy (str): The evolutionary strategy to be used
    """
    # Create fitness and individuals based on the strategy
    # chosen
    if strategy == 'sga':
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Just Fitness
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    elif strategy == 'ns-es':
        creator.create("FitnessMax", base.Fitness, weights=(1.0)) # Just Novelty
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    elif strategy == 'nsr-es':
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0)) # Both Fitness and Novelty
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
    elif strategy == 'cma-es':
        raise NotImplementedError
    else:
        raise NotImplementedError


def load_ckpt(strategy, ckpt):
    """Loads the checkpoint given after creating the fitness and individual
    templates for DEAP evolution

    Args:
        strategy (str): The evolutionary strategy to be used
        ckpt (str): The checkpoint file path that stores the population and other information of a generation

    Returns:
        A dictionary containing key features of the generation
    """
    with open(ckpt, "rb") as cp_file:
        init_ind_fitness(strategy)
        cp = pickle.load(cp_file)

    return cp

def save_ckpt(population,
              generation,
              halloffame,
              logbook,
              rndstate,
              exp_ckpt_dir):
    """Saves the checkpoint of the current generation of Population
    and some other information

    Args:
        population 
        generation
        halloffame
        logbook
        rndstate
        exp_ckpt_dir
    """
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(population=population,
              generation=generation,
              halloffame=halloffame,
              logbook=logbook,
              rndstate=rndstate)

    with open(os.path.join(exp_ckpt_dir, 'checkpoint_gen{}.pkl'.format(g)), "wb") as cp_file:
        pickle.dump(cp, cp_file)
