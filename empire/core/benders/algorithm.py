import logging
import numpy as np
from pathlib import Path
from pyomo.environ import AbstractModel, ConstraintList
from itertools import product

from empire.core.config import OperationalInputParams
from .master_problem import create_master_problem_instance, solve_master_problem
from .subproblem import create_subproblem_model, solve_subproblem, create_subproblem_instance

from empire.core.optimization.operational import prep_stochastic_parameters

logger = logging.getLogger(__name__)

def run_benders(
    flags,
    run_config,
    investment_params: dict,
    discountrate: float,
    LeapYearsInvestment: list,
    periods: list[int],
    wacc: float,
    lopf_kwargs: dict,
    solver_name: str,
    benders_cuts: list,
    sample_file_path: Path | None = None,
    temp_dir: Path | None = None,
    max_iterations: int = 50
) -> tuple[AbstractModel | None, float | None]:
    """
    Function to create and solve the Benders subproblem.

    Parameters
    ----------
    flags : EmpireFlags
        Configuration flags for the model run.
    run_config : EmpireRunConfiguration
        Configuration for the current model run.
    operational_input_params : OperationalInputParams
        Operational parameters and sets for the model.
    investment_params : dict
        Investment parameters as a dictionary.
    discountrate : float
        Discount rate for financial calculations.
    LeapYearsInvestment : list
        List of years considered for leap year investments.
    wacc : float
        Weighted Average Cost of Capital.
    lopf_kwargs : dict
        Additional keyword arguments for LOPF configuration.
    solver_name : str
        Name of the solver to be used (e.g., 'gurobi', 'cplex').
    benders_cuts : list
        List of Benders cuts to be applied in the subproblem.
    sample_file_path : Path, optional
        Path to the sample file for out-of-sample analysis, by default None.

    Returns
    -------
    AbstractModel
        The solved Pyomo model instance representing the Benders subproblem.
    """
    operational_input_params = OperationalInputParams()
    mp_instance = create_master_problem_instance(run_config, solver_name, run_config.temporary_directory, periods, operational_input_params, benders_cuts, discountrate, wacc, LeapYearsInvestment, flags, sample_file_path)
    logger.info("Creating Benders subproblem model...")

    sp_model, data = create_subproblem_model(run_config, solver_name, run_config.temporary_directory, periods, operational_input_params, investment_params, discountrate, LeapYearsInvestment, flags, sample_file_path, lopf_method=flags.lopf_method, lopf_kwargs=lopf_kwargs)
    sp_instance = create_subproblem_instance(sp_model, data)
    solve_master_problem(mp_instance, solver_name, flags, run_config, temp_dir, save_flag=False)


    scenario_data = load_scenario_data(data, operational_input_params.scenarios)
    # should iterate until convergence 
    sp_cuts = []
    last_mp_obj = -1

    for iteration in range(max_iterations):
        for i, w in product(periods, operational_input_params.scenarios):
            set_stochastic_input_subproblem(sp_instance, scenario_data[w], i)
            prep_stochastic_parameters(model, operational_input_params)
            operational_input_params.scenarios = [w]
            sp_cut = solve_subproblem(sp_instance, solver_name, run_config, investment_params)
            sp_cuts.append(sp_cut)
        add_cuts_to_mp(mp_instance, sp_cuts)
        mp_obj = solve_master_problem(mp_instance, solver_name, flags, run_config, temp_dir, save_flag=False)
        if np.isclose(mp_obj, last_mp_obj):
            logger.info("Benders converged.")
            return mp_instance, mp_obj
        last_mp_obj = mp_obj
 
    logger.info("Benders did not converge.")
    return None, None
        


def load_scenario_data(data, scenario) -> dict:
    pass


def filter_scenario_init(model, *idx):
    # Assume your param is indexed like (scenario, something else)
    scenario, other_idx = idx[0], idx[1:]
    if scenario == model.active_scenario:
        return model.maxRegHydroGenRaw[idx]
    return None  # or skip

def set_stochastic_input_subproblem(instance, scenario_data):


model.maxRegHydroGen = Param(
    model.maxRegHydroGenRaw.index_set(),
    initialize=filter_scenario_init,
    within=NonNegativeReals,
    default=0
)
    pass


def add_cuts_to_mp(model, benders_cuts):
    model.cut_constraints = ConstraintList()
    for cut in benders_cuts:
        model.cut_constraints.add(expr=cut)