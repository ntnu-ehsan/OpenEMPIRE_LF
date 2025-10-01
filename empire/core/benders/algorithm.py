import logging
import numpy as np
from pyomo.environ import AbstractModel, ConstraintList, value

from empire.core.config import OperationalInputParams, EmpireConfiguration, EmpireRunConfiguration
from .master_problem import create_master_problem_instance, solve_master_problem, extract_capacity_params, define_initial_capacity_params
from .subproblem import create_subproblem_model, exe_subproblem_routine, load_data
from .cuts_v2 import define_cut_structure, CapacityVariableHandler
from empire.core.optimization.loading_utils import filter_data

logger = logging.getLogger(__name__)

def run_benders(
    run_config: EmpireRunConfiguration,
    empire_config: EmpireConfiguration,
    operational_input_params: OperationalInputParams,
    periods_active: list[int],
    capacity_params_init: None | dict = None,
) -> tuple[AbstractModel | None, float | None]:
    """
    Function to create and solve and EMPIRE instance using Benders decomposition.

    Parameters
    ----------
    run_config : EmpireRunConfiguration
        Configuration for the current model run.
    empire_config : EmpireConfiguration
        General configuration for the EMPIRE model.
    operational_input_params : OperationalInputParams
        Operational input parameters. 
    periods : list[int]
        List of periods to consider in the model. (shorter than full planning horizon?)
    capacity_params_init : dict, optional
        Initial capacity parameters to start the Benders iterations, by default None.

    Returns
    -------
    tuple[ConcreteModel | None, float | None]
        Returns the final master problem instance and its objective value if converged, otherwise (None, None).
    
    """
    mp_instance = create_master_problem_instance(run_config, empire_config, periods=periods_active)
    # capacity_params = extract_capacity_params(mp_instance)
    if capacity_params_init is None:
        capacity_params = define_initial_capacity_params(mp_instance)
    else:
        capacity_params = capacity_params_init

    # solve_master_problem(mp_instance, empire_config.optimization_solver, flags, run_config, empire_config.temporary_directory, save_flag=False)
    logger.info("Creating Benders subproblem model...")






    # scenario_data = load_scenario_data(data, operational_input_params.scenarios)
    # should iterate until convergence 
    
    last_mp_obj = -1.
    mp_instance.cut_constraints = ConstraintList()
    mp_objs = []
    
    for iteration in range(empire_config.max_benders_iterations):
        logger.info("Benders iteration %d", iteration + 1)
        for i in periods_active:
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

        mp_objective = solve_master_problem(mp_instance, empire_config, run_config, save_flag=False)
        capacity_params = extract_capacity_params(mp_instance)
        mp_objs.append(mp_objective)
        if np.isclose(mp_objective, last_mp_obj, rtol=1-8):
            logger.info("Benders converged.")
            for i, mp_obj in enumerate(mp_objs):
                print(f"Iteration {i+1}: Master problem objective = {mp_obj:.6e}")
            breakpoint()
            return mp_objective, mp_instance
        last_mp_obj = mp_objective

    logger.info("Benders did not converge.")
    return None, None
        


def create_cut(
        master_instance, 
        capacity_params: dict[str, dict[tuple]],
        period_active: int,
        scenarios: list[str],
        empire_config: EmpireConfiguration,
        run_config: EmpireRunConfiguration,
        operational_input_params: OperationalInputParams,
        ):

    expr = 0 
    scenario_objectives = {}
    cut_data: dict[str, dict[str, dict[tuple, float]]] = {}
    
    for scenario in scenarios:
        cut_data[scenario] = {}

        sp_instance, opt = exe_subproblem_routine(
            capacity_params,
            period_active,
            scenario,
            empire_config,
            run_config,
            operational_input_params,
            )

        cut_structure: list[CapacityVariableHandler] = define_cut_structure(sp_instance, period_active, scenario)

        for capacity_variable_handler in cut_structure:
            cut_data[scenario][(capacity_variable_handler.constraint_name, capacity_variable_handler.capacity_var_name)] = capacity_variable_handler.extract_data(sp_instance)

        scenario_objectives[scenario] = value(sp_instance.Obj)

    # can pickle capacity data and cut structure here if needed
    expr = sum(scenario_objectives[scenario] for scenario in scenarios)
    for scenario in scenarios:
        for (constraint_name, capacity_var_name), dual_and_coeff_total in cut_data[scenario].items():

            # duals_var = duals.groupby(indices_to_keep).sum()
            # coefficients_var = coefficients.groupby(indices_to_keep).sum()
            # breakpoint()
            # 
            for inds, multiplier in dual_and_coeff_total.items():  # loops over all tuples of constraint indices
                if capacity_var_name == "transmissionInstalledCap":
                    # transmission capacity variable can have n1, n2 switched around
                    if (inds[0], inds[1], inds[2]) in capacity_params[capacity_var_name]:
                        inds = (inds[0], inds[1], inds[2])
                    elif (inds[1], inds[0], inds[2]) in capacity_params[capacity_var_name]:
                        inds = (inds[1], inds[0], inds[2])
                    else:
                        breakpoint()
                        raise ValueError("Transmission capacity indices not found in old capacities.")

        # breakpoint()
                if abs(multiplier) > 1e-10:
                    expr += multiplier * (
                        getattr(master_instance, capacity_var_name)[inds]
                        -
                        value(capacity_params[capacity_var_name][inds])
                    )


    return master_instance.theta[period_active] >= expr



