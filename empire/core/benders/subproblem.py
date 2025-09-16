

from __future__ import division

import logging
import time
from pathlib import Path


from pyomo.environ import (
    DataPortal,
    AbstractModel,
    Suffix, 
    Set,
)
from empire.core.optimization.objective import define_objective
from empire.core.optimization.operational import define_operational_sets, define_operational_constraints, prep_operational_parameters, define_operational_variables, define_operational_parameters, load_operational_parameters, derive_stochastic_parameters, load_stochastic_input
from empire.core.optimization.shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from empire.core.optimization.out_of_sample_functions import set_investments_as_parameters
from empire.core.optimization.lopf_module import LOPFMethod, load_line_parameters
from empire.core.optimization.results import write_results, run_operational_model, write_operational_results, write_pre_solve
from empire.core.optimization.solver import set_solver
from empire.core.optimization.helpers import pickle_instance, log_problem_statistics, prepare_results_dir, prepare_temp_dir
from empire.core.config import EmpireRunConfiguration, OperationalInputParams, EmpireConfiguration

logger = logging.getLogger(__name__)


def create_subproblem_model(
        run_config: EmpireRunConfiguration,
        empire_config: EmpireConfiguration,
        solver_name: str, 
        temp_dir: Path, 
        period: int,
        operational_input_params: OperationalInputParams,
        investment_params: dict,

        sample_file_path: Path | None = None,
        out_of_sample_flag: bool = False,
        ) -> None | float:

    prepare_temp_dir(empire_config.use_temporary_directory, temp_dir=empire_config.temporary_directory)
    prepare_results_dir(run_config)
    
    model = AbstractModel()

    
    ########
    ##SETS##
    ########

    define_shared_sets(model, [period], empire_config.north_sea_flag)
    define_operational_sets(model, operational_input_params)


    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")

    define_shared_parameters(model, empire_config.discount_rate, empire_config.leap_years_investment)
    # define_investment_parameters(model, wacc)
    define_operational_parameters(model, operational_input_params, empire_config.emission_cap_flag, empire_config.load_change_module_flag)

    #Load the data

    data = DataPortal()
    load_shared_sets(model, data, run_config.tab_file_path, empire_config.north_sea_flag)
    load_shared_parameters(model, data, run_config.tab_file_path)
    load_operational_parameters(model, data, run_config.tab_file_path, empire_config.emission_cap_flag, empire_config.load_change_module_flag, out_of_sample_flag, sample_file_path=sample_file_path, scenario_data_path=run_config.scenario_data_path)
    load_stochastic_input(model, data, run_config.tab_file_path)

    # load_investment_parameters(model, data, run_config.tab_file_path)

    # Load electrical data for LOPF if requested (need to split investment and operations!)
    if empire_config.lopf_flag:
        load_line_parameters(model, run_config.tab_file_path, data, empire_config.lopf_kwargs, logger)


    logger.info("Sets and parameters declared and read...")

    #############
    ##VARIABLES##
    #############

    logger.info("Declaring variables...")

    set_investments_as_parameters(model)
    set_investment_values(model, investment_params)
    define_operational_variables(model)


    # model parameter preparations
    prep_operational_parameters(model, empire_config.load_change_module_flag)
    
    ###############
    ##CONSTRAINTS##
    ###############

    # constraint defintions
    define_operational_constraints(model, logger, empire_config.emission_cap_flag, include_hydro_node_limit_constraint_flag=True)

    if empire_config.lopf_flag:
        logger.info("LOPF constraints activated using method: %s", empire_config.lopf_method)
        from .lopf_module import add_lopf_constraints
        kw = {} if empire_config.lopf_kwargs is None else dict(empire_config.lopf_kwargs)
        add_lopf_constraints(model, method=empire_config.lopf_method, **kw)
    else:
        logger.warning("LOPF constraints not activated: %s", empire_config.lopf_method)


    define_objective(model)


    #################################################################

    #######
    ##RUN##
    #######

    logger.info("Model created")

    return model, data

def create_subproblem_instance(model, data):
    start = time.time()
    instance = model.create_instance(data) #, report_timing=True)
    instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)

    end = time.time()
    logger.info("Building instance took [sec]: %d", end - start)
    
    return instance 


def set_investment_values(
        sp_instance, 
        investment_params: dict,
        period_active   : int
        ):
    for param_name, capacities in investment_params.items():
        capacity_period = capacities[period_active]
        param = getattr(sp_instance, param_name)  # e.g. instance.genInvCap
        param[period_active] = capacity_period


def solve_subproblem(instance, solver_name, run_config, investment_params):
    set_investment_values(instance, investment_params)
    instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)
    opt = set_solver(solver_name, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=run_config.results_path / f"logfile_{run_config.run_name}.log")#, keepfiles=True, symbolic_solver_labels=True)
    return opt



INVESTMENT_VARS = [
    'genInvCap',
    'transmissionInvCap',
    'storPWInvCap',
    'storENInvCap',
    'genInstalledCap',
    'transmissionInstalledCap',
    'storPWInstalledCap',
    'storENInstalledCap'
]

def set_investment_values(model, investment_params: dict):
    """Set investment decision variables to fixed values from previous optimization run.
    """
    for var in INVESTMENT_VARS:
        setattr(model, var, investment_params[var])
    return 


def set_scenario_and_period_as_parameter(subproblem_model, scenario, period):
    """Fix scenario for Benders.
    Need to set parameters like sceProbab to have an index corresponding to the scenario"""
    subproblem_model.scenarios = Set(initialize=[scenario])
    subproblem_model.periods = Set(initialize=[period])
    # subproblem_model.genCapAvailStochRaw[n,g,h,s,i]
    return 

