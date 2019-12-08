"""
This module contains bayesian optimization for hyperparameter tuning
- http://krasserm.github.io/2018/03/21/bayesian-optimization/
"""

import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization

from varro.algo.experiment import fit


# Hyperparameter Bounds
bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
       {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
       {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
       {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
       {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]

# Optimization objective
def best_fitness_score(parameters):
    parameters = parameters[0]
    score = np.array(
        fit(learning_rate=parameters[0],
            gamma=int(parameters[1]),
            max_depth=int(parameters[2]),
            n_estimators=int(parameters[3]),
            min_child_weight = parameters[4])
        )
    return score

optimizer = BayesianOptimization(f=best_fitness_score,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True)

# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=20)
