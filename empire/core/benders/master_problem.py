from __future__ import division

import logging
import os
import time
from pathlib import Path



from pyomo.environ import (
    DataPortal,
    AbstractModel,
    Suffix, 
    ConstraintList,
    Var,
    NonNegativeReals,
)
from .objective import define_objective
from .investment import define_investment_constraints, prep_investment_parameters, define_investment_variables, load_investment_parameters, define_investment_parameters
from .shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from .results import write_results, run_operational_model, write_operational_results, write_pre_solve
from .solver import set_solver
from empire.core.optimization.helpers import pickle_instance, log_problem_statistics, prepare_results_dir, prepare_temp_dir
from empire.core.config import EmpireRunConfiguration, OperationalParams, Flags

logger = logging.getLogger(__name__)


def create_master_problem_instance(run_config: EmpireRunConfiguration,
               solver_name: str, 
               temp_dir: Path, 
               periods: list[int], 
               operational_params: OperationalParams,
               discountrate: float, 
               wacc: float,    
               LeapYearsInvestment: float, 
               flags: Flags,
               sample_file_path: Path | None = None,
               ) -> None | float:

    prepare_temp_dir(flags, temp_dir, run_config)
    prepare_results_dir(flags, run_config)
    
    model = AbstractModel()

    
    ########
    ##SETS##
    ########

    define_shared_sets(model, periods, flags.north_sea_flag)


    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")
    
    define_shared_parameters(model, discountrate, LeapYearsInvestment)
    define_investment_parameters(model, wacc)

    #Load the data

    data = DataPortal()
    load_shared_sets(model, data, run_config.tab_file_path, flags.north_sea_flag)
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
    define_investment_constraints(model, flags.north_sea_flag)


    model.operationalcost = Var(model.periods, model.Scenario, within=NonNegativeReals)


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
    if not flags.out_of_sample_flag:	
        log_problem_statistics(instance, logger)
        write_pre_solve(
            instance,
            run_config.results_path,
            run_config.run_name,
            flags.write_lp_flag,
            flags.use_temp_dir_flag,
            temp_dir,
            logger
        )
        

    return instance

def solve_master_problem(instance, solver_name, flags, run_config, temp_dir, save_flag=False):
    opt = set_solver(solver_name, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=run_config.results_path / f"logfile_{run_config.run_name}.log")#, keepfiles=True, symbolic_solver_labels=True)
    if save_flag:
        if flags.pickle_instance_flag:
            pickle_instance(instance, run_config.run_name, flags.use_temp_dir_flag, logger, temp_dir)
                    
        #instance.display('outputs_gurobi.txt')

        #import pdb; pdb.set_trace()

        write_results(instance, run_config.results_path, run_config.run_name, flags.out_of_sample_flag, flags.emission_cap_flag, flags.print_iamc_flag, logger)

        if flags.compute_operational_duals_flag and not flags.out_of_sample_flag:
            run_operational_model(instance, opt, run_config.results_path, run_config.run_name, logger)
            write_operational_results(instance, run_config.results_path, flags.emission_cap_flag, logger)
    return instance.Obj

