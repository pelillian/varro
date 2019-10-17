#################
# MAIN FUNCTION #
#################
def main():

    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Get the Arguments parsed from file execution
    args = get_args()

    toolbox = init(args.isize)

    logger.info('Start Evolution ...')
    evolve(toolbox=toolbox,
           crossover_prob=args.cxpb,
           mutation_prob=args.mutpb,
           num_generations=args.ngen,
           func=args.func)

if __name__ == "__main__":

	# Create Logs folder if not created
    mkdir('./algo/logs/')

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()