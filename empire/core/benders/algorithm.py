import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pyomo.environ import AbstractModel, ConstraintList, value, DataPortal

from empire.core.config import OperationalInputParams, EmpireConfiguration, EmpireRunConfiguration
from .master_problem import create_master_problem_instance, solve_master_problem, extract_capacity_params, define_initial_capacity_params
from .subproblem import create_subproblem_model, solve_subproblem, create_subproblem_instance, load_data
from .cuts_v2 import define_cut_structure, CapacityVariableHandler
from empire.core.optimization.operational import load_stochastic_input, derive_stochastic_parameters, prep_operational_parameters, define_operational_parameters, define_period_and_scenario_dependent_parameters, define_stochastic_input
from empire.core.optimization.shared_data import define_shared_parameters
logger = logging.getLogger(__name__)

def run_benders(
    run_config: EmpireRunConfiguration,
    empire_config: EmpireConfiguration,
    operational_input_params: OperationalInputParams,
    periods: list[int],
    capacity_params_init: None | dict = None,
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

    if capacity_params_init is None:
        capacity_params = define_initial_capacity_params()
    else:
        capacity_params = capacity_params_init
    # del(mp_instance)
    mp_instance = create_master_problem_instance(run_config, empire_config, capacity_params=capacity_params)
    # solve_master_problem(mp_instance, empire_config.optimization_solver, flags, run_config, empire_config.temporary_directory, save_flag=False)
    logger.info("Creating Benders subproblem model...")







    # scenario_data = load_scenario_data(data, operational_input_params.scenarios)
    # should iterate until convergence 
    
    last_mp_obj = -1
    mp_instance.cut_constraints = ConstraintList()
    mp_objs = []
    for iteration in range(empire_config.max_benders_iterations):
        logger.info("Benders iteration %d", iteration + 1)
        for i in periods:
            
            cut = create_cut(
                mp_instance, 
                capacity_params, 
                i,
                operational_input_params.scenarios,
                empire_config,
                run_config,
                operational_input_params,
            )

            mp_instance.cut_constraints.add(expr=cut)
        breakpoint()
        opt_mp, mp_objective = solve_master_problem(mp_instance, empire_config, run_config, save_flag=False)
        capacity_params = extract_capacity_params(mp_instance)
        mp_objs.append(mp_objective)
        if np.isclose(mp_objective, last_mp_obj):
            logger.info("Benders converged.")
            for i, mp_obj in enumerate(mp_objs):
                print(f"Iteration {i+1}: Master problem objective = {mp_obj:.2e}")
            return opt_mp, mp_objective
        last_mp_obj = mp_objective

    logger.info("Benders did not converge.")
    return None, None
        


def create_cut(
        master_instance, 
        capacity_params: dict,
        period_active: int,
        scenarios: list[str],
        empire_config: EmpireConfiguration,
        run_config: EmpireRunConfiguration,
        operational_input_params: OperationalInputParams,
        ):

    expr = 0 
    scenario_objectives = {}
    cut_data = {}
    for w in scenarios:
        cut_data[w] = {}
        sp_model = create_subproblem_model(run_config, empire_config, operational_input_params)
        

        # 
        # define_shared_parameters(sp_model, empire_config.discount_rate, empire_config.leap_years_investment)  # should not be needed preferably. 
        # define_stochastic_input(sp_model)
        data = load_data(sp_model, run_config, empire_config, period_active, w, capacity_params, out_of_sample_flag=False)  # DUPLICATE? 

        sp_instance, opt = solve_sp(
            sp_model,
            data,
            capacity_params,
            empire_config,
            run_config
            )
        cut_structure: list[CapacityVariableHandler] = define_cut_structure(sp_instance, period_active, w)

        for capacity_variable_handler in cut_structure:
            cut_data[w][capacity_variable_handler.capacity_var_name] = capacity_variable_handler.extract_data(sp_instance)

        scenario_objectives[w] = value(sp_instance.Obj)

    # can pickle capacity data and cut structure here if needed
    expr = sum(scenario_objectives[w] for w in scenarios)
    for w in scenarios:
        for capacity_variable_name, dual_and_coeff_total in cut_data[w].items():

            # duals_var = duals.groupby(indices_to_keep).sum()
            # coefficients_var = coefficients.groupby(indices_to_keep).sum()
            # breakpoint()
            # 
            for inds, multiplier in dual_and_coeff_total.items():  # loops over all tuples of constraint indices
                if capacity_variable_name == "transmissionInstalledCap":
                    # transmission capacity variable can have n1, n2 switched around
                    if ((inds[0], inds[1]), inds[2]) in capacity_params.get(capacity_variable_name)[period_active]:
                        inds = ((inds[0], inds[1]), inds[2])
                    elif ((inds[1], inds[0]), inds[2]) in capacity_params.get(capacity_variable_name)[period_active]:
                        inds = ((inds[1], inds[0]), inds[2])
                    else:
                        raise ValueError("Transmission capacity indices not found in old capacities.")

                expr += multiplier * (
                    getattr(master_instance, capacity_variable_name)[inds]
                    -
                    value(capacity_params.get(capacity_variable_name)[period_active][inds])
                )
            # breakpoint()
    return master_instance.theta[period_active] >= expr


def solve_sp(
    sp_model: AbstractModel,
    data: DataPortal,
    capacity_params: dict,
    empire_config: EmpireConfiguration,
    run_config: EmpireRunConfiguration
    ):


    # set_stochastic_input_subproblem(sp_instance, scenario_data[w], i)
    
    # load_stochastic_input(sp_model, data, tab_file_path)  # DUPLICATE? 
    
    sp_instance = create_subproblem_instance(sp_model, data)

    # set_investment_values(sp_instance, capacity_params, period_active)
    derive_stochastic_parameters(sp_instance)

    opt = solve_subproblem(sp_instance, empire_config.optimization_solver, run_config, capacity_params)
    return sp_instance, opt


def load_scenario_data(data, scenario) -> dict:
    pass


def filter_scenario_init(model, *idx):
    # Assume your param is indexed like (scenario, something else)
    scenario, other_idx = idx[0], idx[1:]
    if scenario == model.active_scenario:
        return model.maxRegHydroGenRaw[idx]
    return None  # or skip

def set_stochastic_input_subproblem(instance, scenario_data):
    pass

# model.maxRegHydroGen = Param(
#     model.maxRegHydroGenRaw.index_set(),
#     initialize=filter_scenario_init,
#     within=NonNegativeReals,
#     default=0
# )
#     pass



