"""
This module contains the main function we'll use to run the 
experiment to solve the problem using a specified 
evolutionary algorithm
"""

from functools import partial

from algo.problems import func_approx, mnist
from algo.evaluate.evaluate import 
from algo.models.model import Model
from algo.strategies.ea.evolve import evolve
from algo.strategies.ea.toolbox import nn_toolbox, fpga_toolbox

FPGA_BITSTREAM_SHAPE = (13294, 1136)


#################
# MAIN FUNCTION #
#################
def main():
	"""Main Function
	"""
	# Initialize logger
	logger = logging.getLogger(__name__)

	# Get the Arguments parsed from file execution
	args = get_args()

	# 1. Choose Target Platform
	if args.target == 'nn':

	  # 2. Choose Problem and get the specific evaluation function 
	  # for that problem
	  if args.problem == 'mnist':
		evaluate_population = 
	  else:
		evaluate_population =
	  
	  toolbox = nn_toolbox(i_size=i_size,
						   evaluate_population=evaluate_population)
	else:
	  toolbox = fpga_toolbox(i_shape=i_shape,
							 evaluate_population=evaluate_population)

	# 3. Choose Strategy
	if args.strategy == 'ea':
	  evolve(toolbox=toolbox,
			 crossover_prob=args.cxpb,
			 mutation_prob=args.mutpb,
			 num_generations=args.ngen)
	elif args.strategy == 'cma-es':
	  pass
	elif args.strategy == 'cma-es':
	  pass
	else:
	  pass

if __name__ == "__main__":

	# Create Logs folder if not created
	mkdir('./algo/logs/')

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	main()