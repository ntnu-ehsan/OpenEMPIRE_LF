from __future__ import division

import logging
import os
import time
from pathlib import Path


from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
    value,
    Set,
    Param,
    Var,
    Constraint,
    NonNegativeReals,
    BuildAction,
    Expression,
    Objective,
    minimize,
    DataPortal,
    AbstractModel,
    Suffix, 
)

from .operational_constraints import define_operational_sets, define_operational_constraints, prep_operational_parameters, define_operational_variables, define_operational_parameters, load_operational_parameters
from .investment_constraints import define_investment_constraints, prep_investment_parameters, define_investment_variables, load_investment_parameters, define_investment_parameters
from .shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from .out_of_sample_functions import set_investments_as_parameters, load_optimized_investments, set_out_of_sample_path
from .lopf_module import LOPFMethod, load_line_parameters
from .results import write_results, run_operational_model, write_operational_results, write_pre_solve
from .solver import set_solver
from .helpers import pickle_instance, log_problem_statistics



logger = logging.getLogger(__name__)


def run_empire(instance_name: str, 
               tab_file_path: Path, 
               result_file_path: Path, 
               scenario_data_path,
               solver_name, 
               temp_dir, 
               FirstHoursOfRegSeason, 
               FirstHoursOfPeakSeason, 
               lengthRegSeason,
               lengthPeakSeason, 
               Period, 
               Operationalhour, 
               Scenario, 
               Season, 
               HoursOfSeason,
               discountrate, 
               wacc, 
               LeapYearsInvestment, 
               print_iamc_flag, 
               write_lp_flag,
               pickle_instance_flag, 
               emission_cap_flag, 
               use_temp_dir_flag, 
               load_change_module_flag, 
               compute_operational_duals_flag, 
               north_sea_flag, 
               out_of_sample_flag: bool = False, 
               sample_file_path: Path | None = None,
               lopf_flag: bool = False, 
               lopf_method: str = LOPFMethod.KIRCHHOFF, 
               lopf_kwargs: dict | None = None
               ) -> None | float:

    if use_temp_dir_flag:
        TempfileManager.tempdir = temp_dir

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    model = AbstractModel()

    ##########
    ##MODULE##
    ##########

    if write_lp_flag:
        logger.info("Will write LP-file...")

    if pickle_instance_flag:
        logger.info("Will pickle instance...")

    if emission_cap_flag:
        logger.info("Absolute emission cap in each scenario...")
    else:
        logger.info("No absolute emission cap...")
    
    ########
    ##SETS##
    ########

    define_shared_sets(model, Period, north_sea_flag)
    define_operational_sets(model, Operationalhour, Season, Scenario, HoursOfSeason, FirstHoursOfRegSeason, FirstHoursOfPeakSeason)


    # 

    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")
    
    define_shared_parameters(model, discountrate, LeapYearsInvestment)
    define_investment_parameters(model, wacc)
    define_operational_parameters(model, lengthRegSeason, lengthPeakSeason, emission_cap_flag, load_change_module_flag)

    #Load the data

    data = DataPortal()
    load_shared_sets(model, data, tab_file_path, north_sea_flag)
    load_shared_parameters(model, data, tab_file_path)
    load_operational_parameters(model, data, tab_file_path, emission_cap_flag, load_change_module_flag, out_of_sample_flag, sample_file_path=sample_file_path, scenario_data_path=scenario_data_path)
    load_investment_parameters(model, data, tab_file_path)
    
    # Load electrical data for LOPF if requested (need to split investment and operations!)
    if lopf_flag:
        load_line_parameters(model, tab_file_path, data, lopf_kwargs, logger) 


    logger.info("Sets and parameters declared and read...")

    #############
    ##VARIABLES##
    #############

    logger.info("Declaring variables...")

    if out_of_sample_flag:
        set_investments_as_parameters(model)
        load_optimized_investments(model, data, result_file_path)
        result_file_path = set_out_of_sample_path(result_file_path, sample_file_path)
        logger.info("Out-of-sample results will be saved to: %s", result_file_path)

    else:
        define_investment_variables(model)

    define_operational_variables(model)


    # model parameter preparations
    prep_operational_parameters(model, load_change_module_flag)

    if not out_of_sample_flag:
        # All constraints exclusively for investment decisions inactive when out_of_sample_flag
        prep_investment_parameters(model)


    ###############
    ##EXPRESSIONS##
    ###############

    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)

    ###############
    ##CONSTRAINTS##
    ###############

    # constraint defintions
    define_investment_constraints(model, north_sea_flag)
    define_operational_constraints(model, logger, emission_cap_flag, include_hydro_node_limit_constraint_flag=True)

    if lopf_flag:
        logger.info("LOPF constraints activated using method: %s", lopf_method)
        from .lopf_module import add_lopf_constraints
        kw = {} if lopf_kwargs is None else dict(lopf_kwargs)
        add_lopf_constraints(model, method=lopf_method, **kw)
    else:
        logger.warning("LOPF constraints not activated: %s", lopf_method)

        
    #############
    ##OBJECTIVE##
    #############

    def Obj_rule(model):
        return sum(model.discount_multiplier[i]*(
            sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
            sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
            sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode )) 
            for i in model.PeriodActive) + \
            sum(model.operationalcost[i, w] for i in model.PeriodActive for w in model.Scenario)

    model.Obj = Objective(rule=Obj_rule, sense=minimize)



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
    if not out_of_sample_flag:	
        log_problem_statistics(instance, logger)
        write_pre_solve(
            instance,
            result_file_path,
            instance_name, 
            write_lp_flag,
            use_temp_dir_flag,
            temp_dir,
            logger
        )
        

    opt = set_solver(solver_name, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=result_file_path / f"logfile_{instance_name}.log")#, keepfiles=True, symbolic_solver_labels=True)

    if pickle_instance_flag:
        pickle_instance(instance, instance_name, use_temp_dir_flag, logger, temp_dir)
                
    #instance.display('outputs_gurobi.txt')

    #import pdb; pdb.set_trace()

    write_results(instance, result_file_path, instance_name, out_of_sample_flag, emission_cap_flag, print_iamc_flag, logger)

    if compute_operational_duals_flag and not out_of_sample_flag:
        run_operational_model(instance, opt, result_file_path, instance_name, logger)
        write_operational_results(instance, result_file_path, emission_cap_flag, logger)


