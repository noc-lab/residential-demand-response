from energy.code.solvers.distributed.optimizeQ_dist import OptimizeQ_dist
from energy.code.solvers.centralized.optimizeQ_central import OptimizeQ_central
from energy.code.solvers.distributed.customQ_dist import CustomQ_dist
from energy.code.solvers.centralized.customQ_central import CustomQ_central


"""Interface to solvers."""


def init_solver(solver_type, inputs_dict, models, neighborhood, price):
    """Initializes algorithm."""



    if solver_type == 'optQ_central':
        solver = OptimizeQ_central(solver_type, models, neighborhood, inputs_dict, price)
    elif solver_type == 'optQ_dist':
        solver = OptimizeQ_dist(solver_type, models, neighborhood, inputs_dict, price)
    elif solver_type == 'customQ_central':
        solver = CustomQ_central(solver_type, models, neighborhood, inputs_dict, price)
    elif solver_type == 'customQ_dist':
        solver = CustomQ_dist(solver_type, models, neighborhood, inputs_dict, price)

    
    else:
        raise ValueError('invalid solver_type')
        
    return solver