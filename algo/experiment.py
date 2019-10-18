"""
This module contains the main function we'll use to run the 
experiment to solve the problem using a specified 
evolutionary algorithm
"""

from pathlib import Path

from algo.util import get_args, optimize, mkdir


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

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	main()