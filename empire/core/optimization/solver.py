from pyomo.environ import SolverFactory



def set_solver(solver):
    """Set the solver for the optimization problem and set the parameters (solver dependent). 
    Solver parameters are currently hardcoded!
    Solver options:
    - CPLEX
    - Xpress
    - Gurobi
    - GLPK

    Args:
        solver (str): The name of the solver to use.

    Returns:
        SolverFactoryClass: The solver instance.
    """
    if solver == "CPLEX":
        opt = SolverFactory("cplex", Verbose=True)
        opt.options["lpmethod"] = 4
        opt.options["solutiontype"] = 2
        #instance.display('outputs_cplex.txt')
    if solver == "Xpress":
        opt = SolverFactory("xpress") #Verbose=True
        opt.options["defaultAlg"] = 4
        opt.options["crossover"] = 0
        opt.options["lpLog"] = 1
        opt.options["Trace"] = 1
        #instance.display('outputs_xpress.txt')
    if solver == "Gurobi":
        opt = SolverFactory('gurobi', Verbose=True)
        opt.options["Crossover"]=0
        opt.options["Method"]=2
    if solver == "GLPK":
        opt = SolverFactory("glpk", Verbose=True)
    return opt