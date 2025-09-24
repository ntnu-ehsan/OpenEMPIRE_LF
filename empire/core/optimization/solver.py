from pyomo.environ import SolverFactory
import sys


def set_solver(solver_name, logger):
    """Set the solver for the optimization problem and set the parameters (solver dependent). 
    Solver parameters are currently hardcoded!
    Solver options:
    - CPLEX
    - Xpress
    - Gurobi
    - GLPK

    Args:
        solver_name (str): The name of the solver to use.

    Returns:
        SolverFactoryClass: The solver instance.
    """
    
    if solver_name == "CPLEX":
        opt = SolverFactory("cplex", Verbose=True)
        opt.options["lpmethod"] = 4
        opt.options["solutiontype"] = 2
        logger.info("CPLEX solver is being used.")
        #instance.display('outputs_cplex.txt')
    elif solver_name == "Xpress":
        opt = SolverFactory("xpress") #Verbose=True
        opt.options["defaultAlg"] = 4
        opt.options["crossover"] = 0
        opt.options["lpLog"] = 1
        opt.options["Trace"] = 1
        logger.info("Xpress solver is being used.")
        #instance.display('outputs_xpress.txt')
    elif solver_name == "Gurobi":
        opt = SolverFactory('gurobi', Verbose=True)
        opt.options["Crossover"]=0
        # opt.options["Method"]=2


        opt.options["Method"] = 1   
        opt.options["FeasibilityTol"] = 1e-9
        opt.options["OptimalityTol"] = 1e-9
        # Increase numerical focus (0 = default, 3 = maximum)
        opt.options["NumericFocus"] = 3
        logger.info("Gurobi solver is being used.")
    elif solver_name == "GLPK":
        opt = SolverFactory("glpk", Verbose=True)
        logger.info("GLPK solver is being used.")
    else:
        sys.exit(f"ERROR! Invalid solver_name: {solver_name} Options: CPLEX, Xpress, Gurobi, GLPK")
    return opt