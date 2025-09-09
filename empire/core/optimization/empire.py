from __future__ import division

import logging
import os
import time
from pathlib import Path



from pyomo.environ import (
    DataPortal,
    AbstractModel,
    Suffix, 
)
from .objective import define_objective
from .operational import define_operational_sets, define_operational_constraints, prep_operational_parameters, prep_stochastic_parameters, define_operational_variables, define_operational_parameters, load_operational_parameters, define_stochastic_input, load_stochastic_input
from .investment import define_investment_constraints, prep_investment_parameters, define_investment_variables, load_investment_parameters, define_investment_parameters
from .shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from .out_of_sample_functions import set_investments_as_parameters, load_optimized_investments, set_out_of_sample_path
from .lopf_module import LOPFMethod, load_line_parameters
from .results import write_results, run_operational_model, write_operational_results, write_pre_solve
from .solver import set_solver
from .helpers import pickle_instance, log_problem_statistics, prepare_temp_dir, prepare_results_dir
from empire.core.config import EmpireRunConfiguration, OperationalParams, Flags

logger = logging.getLogger(__name__)


def run_empire(run_config: EmpireRunConfiguration,
               solver_name: str, 
               temp_dir: Path, 
               periods: list[int], 
               operational_input_params: OperationalParams,
               discountrate: float, 
               wacc: float,    
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
    define_operational_sets(model, operational_input_params)


    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")
    
    define_shared_parameters(model, discountrate, LeapYearsInvestment)
    define_investment_parameters(model, wacc)
    define_operational_parameters(model, operational_input_params, flags.emission_cap_flag, flags.load_change_module_flag)
    define_stochastic_input(model)

    #Load the data

    data = DataPortal()
    load_shared_sets(model, data, run_config.tab_file_path, flags.north_sea_flag)
    load_shared_parameters(model, data, run_config.tab_file_path)
    load_operational_parameters(model, data, run_config.tab_file_path, flags.emission_cap_flag, flags.load_change_module_flag, flags.out_of_sample_flag, sample_file_path=sample_file_path, scenario_data_path=run_config.scenario_data_path)
    load_stochastic_input(model, data, run_config.tab_file_path, flags.out_of_sample_flag, sample_file_path=sample_file_path)
    load_investment_parameters(model, data, run_config.tab_file_path)

    # Load electrical data for LOPF if requested (need to split investment and operations!)
    if flags.lopf_flag:
        load_line_parameters(model, run_config.tab_file_path, data, lopf_kwargs, logger)


    logger.info("Sets and parameters declared and read...")

    #############
    ##VARIABLES##
    #############

    logger.info("Declaring variables...")

    if flags.out_of_sample_flag:
        set_investments_as_parameters(model)
        load_optimized_investments(model, data, run_config.results_path)
        results_path = set_out_of_sample_path(run_config.results_path, sample_file_path)
        logger.info("Out-of-sample results will be saved to: %s", results_path)

    else:
        define_investment_variables(model)

    define_operational_variables(model)


    # model parameter preparations
    prep_operational_parameters(model)
    prep_stochastic_parameters(model, operational_input_params)


    if not flags.out_of_sample_flag:
        # All constraints exclusively for investment decisions inactive when out_of_sample_flag
        prep_investment_parameters(model)


    ###############
    ##CONSTRAINTS##
    ###############

    # constraint defintions
    define_investment_constraints(model, flags.north_sea_flag)
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
        

    opt = set_solver(solver_name, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=run_config.results_path / f"logfile_{run_config.run_name}.log")#, keepfiles=True, symbolic_solver_labels=True)

    if flags.pickle_instance_flag:
        pickle_instance(instance, run_config.run_name, flags.use_temp_dir_flag, logger, temp_dir)
                
    #instance.display('outputs_gurobi.txt')

    #import pdb; pdb.set_trace()

    write_results(instance, run_config.results_path, run_config.run_name, flags.out_of_sample_flag, flags.emission_cap_flag, flags.print_iamc_flag, logger)

    if flags.compute_operational_duals_flag and not flags.out_of_sample_flag:
        run_operational_model(instance, opt, run_config.results_path, run_config.run_name, logger)
        write_operational_results(instance, run_config.results_path, flags.emission_cap_flag, logger)


