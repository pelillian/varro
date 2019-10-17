"""
This module contains evolutionary algorithm utility functions.
"""

import os
import argparse


def mkdir(path):
    """Creates new folder

    """
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
        else:
            print("(%s) already exists" % (path))

def get_args():
    """Reads command-line arguments.
    
    Returns:
        (Namespace) Parsed attributes

    """
    parser = argparse.ArgumentParser(
        description='Evolves weights of neural network to approximate a function'
    )
    parser.add_argument('--cxpb',
                        default=0.5,
                        const=0.5,
                        nargs='?',
                        metavar='CROSSOVER-PROBABILITY', 
                        action='store', 
                        help='Set the Cross-over probability for offspring', 
                        type=float)
    parser.add_argument('--mutpb',
                        default=0.2,
                        const=0.2,
                        nargs='?',
                        metavar='MUTATION-PROBABILITY', 
                        action='store', 
                        help='Set the Mutation probability', 
                        type=float)
    parser.add_argument('--ngen', 
                        default=100,
                        const=100,
                        nargs='?',
                        metavar='NUMBER-OF-GENERATIONS', 
                        action='store', 
                        help='Set the Number of Generations to evolve the weights of neural net', 
                        type=int)
    parser.add_argument('--isize', 
                        default=12,
                        const=12,
                        nargs='?',
                        metavar='INDIVIDUAL-SIZE', 
                        action='store', 
                        help='Size of each individual in the population', 
                        type=int)
    parser.add_argument('--func', 
                        default='sinx',
                        const='sinx',
                        nargs='?',
                        metavar='FUNCTION-TO-APPROXIMATE', 
                        action='store', 
                        choices=['x', 'sinx', 'cosx', 'tanx'], 
                        help='Set function to approximate using evolutionary strategy on neural network weights')
    parser.add_argument('--target', 
                        default='nn',
                        const='nn',
                        nargs='?',
                        metavar='TARGET-TO-OPTIMIZE', 
                        action='store', 
                        choices=['fpga', 'nn'], 
                        help='The target platform that the parameters are evaluated on')
    settings = parser.parse_args()
    
    return settings


