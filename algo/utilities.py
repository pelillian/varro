"""
This module stores the utility functions to be used for the evolutionary algorithms.
"""

def get_args():
    '''
    Function:
    ---------
    Utility function to read in arguments when running
    experiment to evolve weights of neural
    network to approximate a function
    
    Parameters:
    -----------
    None.
    
    Returns:
    --------
    (Type: Namespace) that keeps all the attributes parsed
    '''
    parser = argparse.ArgumentParser(
        description='Evolves weights of neural network to approximate a function'
    )
    parser.add_argument('--cxpb',
                        metavar='CROSSOVER-PROBABILITY', 
                        action='store', 
                        help='Set the Cross-over probability for offspring', 
                        type=float)
    parser.add_argument('--mutpb', 
                        metavar='MUTATION-PROBABILITY', 
                        action='store', 
                        help='Set the Mutation probability', 
                        type=float)
    parser.add_argument('--ngen', 
                        metavar='NUMBER-OF-GENERATIONS', 
                        action='store', 
                        help='Set the Number of Generations to evolve the weights of neural net', 
                        type=float)
    parser.add_argument('--func', 
                        metavar='FUNCTION-TO-APPROXIMATE', 
                        action='store', 
                        choices=['x', 'sinx', 'cosx', 'tanx'], 
                        help='Set function to approximate using evolutionary strategy on neural network weights')
    settings = parser.parse_args()
    
    return settings

def evaluate(individual, function=np.sin):
    '''
    Function:
    ---------
    Loads an individual (list) as the weights
    of neural net, computes the Mean Squared Error of
    neural net with the given weights in approximating 
    function provided
    
    Parameters:
    -----------
    individual: An individual (represented by list of floats) 
        - e.g. [0.93, 0.85, 0.24, ..., 0.19], ...}
    function: Function to be approximated by neural net
    
    Returns:
    --------
    A single scalar of the Mean Squared Error, representing fitness of the individual
    '''
    ###################################
    # TODO: Chris taking care of this #
    ###################################
    
    # FUTURE:
    # flash_ecp5(None)
    
    fitness = np.sum(individual)
    return (fitness,)