from __future__ import division

import logging
import os
import time
from pathlib import Path



from pyomo.environ import (
    DataPortal,
    AbstractModel,
    Suffix, 
    Var,
    NonNegativeReals,
)
from empire.core.optimization.objective import define_objective
from empire.core.optimization.investment import define_investment_constraints, prep_investment_parameters, define_investment_variables, load_investment_parameters, define_investment_parameters
from empire.core.optimization.shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from empire.core.optimization.results import write_results, run_operational_model, write_operational_results, write_pre_solve
from empire.core.optimization.solver import set_solver
from empire.core.optimization.helpers import pickle_instance, log_problem_statistics, prepare_results_dir, prepare_temp_dir
from empire.core.config import EmpireRunConfiguration, EmpireConfiguration

logger = logging.getLogger(__name__)


def create_master_problem_instance(
        run_config: EmpireRunConfiguration, 
        empire_config: EmpireConfiguration, 
        periods: list[int]
        ) -> None | float:

    prepare_temp_dir(empire_config.use_temporary_directory, temp_dir=empire_config.temporary_directory)
    prepare_results_dir(run_config)
    
    model = AbstractModel()

    
    ########
    ##SETS##
    ########

    define_shared_sets(model, periods, empire_config.north_sea_flag)


    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")
    
    define_shared_parameters(model, empire_config.discountrate, empire_config.leap_years_investment)
    define_investment_parameters(model, empire_config.wacc)

    #Load the data

    data = DataPortal()
    load_shared_sets(model, data, run_config.tab_file_path, empire_config.north_sea_flag)
    load_shared_parameters(model, data, run_config.tab_file_path)
    load_investment_parameters(model, data, run_config.tab_file_path)


    logger.info("Sets and parameters declared and read...")

    #############
    ##VARIABLES##
    #############

    logger.info("Declaring variables...")


    define_investment_variables(model)
    prep_investment_parameters(model)


    ###############
    ##CONSTRAINTS##
    ###############

    # constraint defintions
    define_investment_constraints(model, empire_config.north_sea_flag)


    model.operationalcost = Var(model.periods, model.scenarios, within=NonNegativeReals)


    define_objective(model)


    #################################################################

    #######
    ##RUN##
    #######

    logger.info("Objective and constraints read...")

    logger.info("Building instance...")

    start = time.time()

    instance = model.create_instance(data) #, report_timing=True)
    instance.dual = Suffix(direction=Suffix.IMPORT) #Make sure the dual value is collected into solver results (if solver supplies dual information)

    end = time.time()
    logger.info("Building instance took [sec]: %d", end - start)

    #import pdb; pdb.set_trace()
    #instance.CO2price.pprint()

    # log_problem_statistics(instance, logger)
    # write_pre_solve(
    #     instance,
    #     run_config.results_path,
    #     run_config.run_name,
    #     flags.write_lp_flag,
    #     flags.use_temp_dir_flag,
    #     temp_dir,
    #     logger
    # )
        

    return instance

def solve_master_problem(instance, empire_config, run_config, save_flag=False):
    opt = set_solver(empire_config.optimization_solver, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=run_config.results_path / f"logfile_{run_config.run_name}.log")#, keepfiles=True, symbolic_solver_labels=True)
    if save_flag:
        if empire_config.pickle_instance_flag:
            pickle_instance(instance, run_config.run_name, empire_config.use_temp_dir_flag, logger, empire_config.temporary_directory)

        #instance.display('outputs_gurobi.txt')

        #import pdb; pdb.set_trace()

        write_results(instance, run_config.results_path, run_config.run_name, False, empire_config.emission_cap_flag, empire_config.print_iamc_flag, logger)

    return instance.Obj


def extract_capacity_params(mp_instance):
    """Extract capacity parameters from the master problem instance."""
    capacity_params = {
        'genInvCap': {},
        'storInvCap': {},
        'lineInvCap': {},
        'periods_active': list(mp_instance.periods_active)
    }
    for period in mp_instance.periods_active:
        capacity_params['genInvCap'][period] = {g: mp_instance.genInvCap[g, period].value for g in mp_instance.GeneratorsOfNode}
        capacity_params['storInvCap'][period] = {s: mp_instance.storInvCap[s, period].value for s in mp_instance.StoragesOfNode}
        capacity_params['lineInvCap'][period] = {l: mp_instance.lineInvCap[l, period].value for l in mp_instance.BidirectionalArc}
    return capacity_params