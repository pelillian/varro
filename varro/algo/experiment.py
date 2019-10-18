"""
This module contains the main function we'll use to run the 
experiment to solve the problem using a specified 
evolutionary algorithm
"""

from varro.algo.util import get_args, optimize, mkdir


#################
# MAIN FUNCTION #
#################
def main():
	"""Main Function
	"""
	# Get the Arguments parsed from file execution
	args = get_args()

	# Start Optimization
	optimize(target=args.target, 
			 problem=args.problem, 
			 strategy=args.strategy, 
			 cxpb=args.cxpb, 
			 mutpb=args.mutpb, 
			 popsize=args.popsize,
			 ngen=args.ngen)

if __name__ == "__main__":

	# Create Logs folder if not created
	mkdir('./algo/logs/')

	main()
