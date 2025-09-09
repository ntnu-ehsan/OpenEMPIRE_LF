

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
from empire.core.optimization.operational import define_operational_sets, define_operational_constraints, prep_operational_parameters, define_operational_variables, define_operational_parameters, load_operational_parameters
from empire.core.optimization.shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from empire.core.optimization.out_of_sample_functions import set_investments_as_parameters
from .lopf_module import LOPFMethod, load_line_parameters
from .results import write_results, run_operational_model, write_operational_results, write_pre_solve
from .solver import set_solver
from empire.core.optimization.helpers import pickle_instance, log_problem_statistics, prepare_results_dir, prepare_temp_dir
from empire.core.config import EmpireRunConfiguration, OperationalInputParams, Flags

logger = logging.getLogger(__name__)


def define_subproblem(run_config: EmpireRunConfiguration,
               solver_name: str, 
               temp_dir: Path, 
               periods: list[int], 
               operational_params: OperationalInputParams,
               investment_params: dict,
               discountrate: float, 
               LeapYearsInvestment: float, 
               flags: Flags,
               sample_file_path: Path | None = None,
               lopf_method: str = LOPFMethod.KIRCHHOFF, 
               lopf_kwargs: dict | None = None
               ) -> None | float:

    prepare_temp_dir(flags, temp_dir, run_config)
    prepare_results_dir(flags, run_config)
    
    model = AbstractModel()

    
    ########
    ##SETS##
    ########

    define_shared_sets(model, periods, flags.north_sea_flag)
    define_operational_sets(model, operational_params)


    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")
    
    define_shared_parameters(model, discountrate, LeapYearsInvestment)
    # define_investment_parameters(model, wacc)
    define_operational_parameters(model, operational_params, flags.emission_cap_flag, flags.load_change_module_flag)

    #Load the data

    data = DataPortal()
    load_shared_sets(model, data, run_config.tab_file_path, flags.north_sea_flag)
    load_shared_parameters(model, data, run_config.tab_file_path)
    load_operational_parameters(model, data, run_config.tab_file_path, flags.emission_cap_flag, flags.load_change_module_flag, flags.out_of_sample_flag, sample_file_path=sample_file_path, scenario_data_path=run_config.scenario_data_path)
    # load_investment_parameters(model, data, run_config.tab_file_path)

    # Load electrical data for LOPF if requested (need to split investment and operations!)
    if flags.lopf_flag:
        load_line_parameters(model, run_config.tab_file_path, data, lopf_kwargs, logger)


    logger.info("Sets and parameters declared and read...")

    #############
    ##VARIABLES##
    #############

    logger.info("Declaring variables...")

    set_investments_as_parameters(model)
    set_investment_values(model, investment_params)
    define_operational_variables(model)


    # model parameter preparations
    prep_operational_parameters(model, flags.load_change_module_flag)

    ###############
    ##CONSTRAINTS##
    ###############

    # constraint defintions
    define_operational_constraints(model, logger, flags.emission_cap_flag, include_hydro_node_limit_constraint_flag=True)

    if flags.lopf_flag:
        logger.info("LOPF constraints activated using method: %s", lopf_method)
        from .lopf_module import add_lopf_constraints
        kw = {} if lopf_kwargs is None else dict(lopf_kwargs)
        add_lopf_constraints(model, method=lopf_method, **kw)
    else:
        logger.warning("LOPF constraints not activated: %s", lopf_method)


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
        instance, 
        investment_params: dict
        ):
    for param_name, new_values in investment_params.items():
        for period, capacity in new_values.items():
            param = getattr(instance, param_name)  # e.g. instance.genInvCap
            if period not in param.keys():
                raise ValueError(f"Period {period} not in parameter {param_name} keys.")
            param[period] = capacity

def solve_subproblem(instance, solver_name, run_config, investment_params):
    set_investment_values(instance, investment_params)
    opt = set_solver(solver_name, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=run_config.results_path / f"logfile_{run_config.run_name}.log")#, keepfiles=True, symbolic_solver_labels=True)

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


def set_scenario_as_parameter(subproblem_model):
    """Fix scenario for Benders.
    Need to set parameters like sceProbab to have an index corresponding to the scenario"""
    sname = "_"
    subproblem_model.scenarios = Set(initialize=[sname])
    # subproblem_model.genCapAvailStochRaw[n,g,h,s,i]
    return 