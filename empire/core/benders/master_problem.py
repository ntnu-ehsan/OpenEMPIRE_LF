from __future__ import division

import logging
import os
import time
from pathlib import Path
from collections import defaultdict


from pyomo.environ import (
    DataPortal,
    AbstractModel,
    Suffix, 
    Var,
    NonNegativeReals,
    Objective,
    minimize,
    Expression,
    value,
    ConcreteModel
)
from empire.core.optimization.objective import investment_obj, multiplier_rule
from empire.core.optimization.investment import define_investment_constraints, prep_investment_parameters, define_investment_variables, load_investment_parameters, define_investment_parameters
from empire.core.optimization.shared_data import define_shared_sets, load_shared_sets, define_shared_parameters, load_shared_parameters
from empire.core.optimization.results import write_results, run_operational_model, write_operational_results, write_pre_solve
from empire.core.optimization.solver import set_solver
from empire.core.optimization.helpers import pickle_instance, log_problem_statistics, prepare_results_dir, prepare_temp_dir
from empire.core.config import EmpireRunConfiguration, EmpireConfiguration

from empire.core.optimization.out_of_sample_functions import set_investments_as_parameters, load_optimized_investments
logger = logging.getLogger(__name__)


def create_master_problem_instance(
        run_config: EmpireRunConfiguration, 
        empire_config: EmpireConfiguration, 
        # capacity_params: dict | None = None,
        # regularization_flag: bool = True,
        # regularization_weight: float = 1e3,
        periods: list[int] | None = None
        ) -> ConcreteModel:

    prepare_temp_dir(empire_config.use_temporary_directory, temp_dir=empire_config.temporary_directory)
    prepare_results_dir(run_config)
    
    model = AbstractModel()
    define_shared_sets(model, empire_config.north_sea_flag)
    define_shared_parameters(model, empire_config.discount_rate, empire_config.leap_years_investment)
    define_investment_parameters(model, empire_config.wacc)
    define_investment_variables(model)

    #Load the data
    data = DataPortal()
    load_shared_sets(model, data, run_config.tab_file_path, empire_config.north_sea_flag, load_period=True, periods_active=periods)
    load_shared_parameters(model, data, run_config.tab_file_path)
    load_investment_parameters(model, data, run_config.tab_file_path)

    prep_investment_parameters(model) 
    define_investment_constraints(model, empire_config.north_sea_flag)

    # Benders specific
    model.theta = Var(model.PeriodActive, within=NonNegativeReals)

    # objective 
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)  # must be specified before defining objective 

    def Obj_rule(model):
        obj = investment_obj(model) + \
            sum(model.discount_multiplier[i] * model.theta[i] for i in model.PeriodActive)
        
        # Regularization: penalize deviation from previous iteration capacities
        # weight can be tuned; smaller = softer stabilization
        # if regularization_flag and capacity_params is not None:

        #     for period in model.PeriodActive:
        #         # Generators

        #         for (n, g) in model.GeneratorsOfNode:
        #             # breakpoint()
        #             prev = capacity_params['genInstalledCap'][period][(n, g, period)]
        #             curr = model.genInstalledCap[n, g, period]
        #             obj += regularization_weight * ((curr - prev)/prev) ** 2

        #         # Storage energy
        #         for (n, b) in model.StoragesOfNode:
        #             prev = capacity_params['storENInstalledCap'][period][(n, b, period)]
        #             curr = model.storENInstalledCap[n, b, period]
        #             obj += regularization_weight * ((curr - prev)/prev) ** 2

        #         # Storage power
        #         for (n, b) in model.StoragesOfNode:
        #             prev = capacity_params['storPWInstalledCap'][period][(n, b, period)]
        #             curr = model.storPWInstalledCap[n, b, period]
        #             obj += regularization_weight * ((curr - prev)/prev) ** 2

        #         # Transmission
        #         for (line_pair) in model.BidirectionalArc:
        #             prev = capacity_params['transmissionInstalledCap'][period][(line_pair, period)]
        #             curr = model.transmissionInstalledCap[line_pair, period]
        #             obj += regularization_weight * ((curr - prev)/prev) ** 2
        return obj

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

def solve_master_problem(
        instance: ConcreteModel, 
        empire_config: EmpireConfiguration, run_config: EmpireRunConfiguration, save_flag=False) -> float:
    opt = set_solver(empire_config.optimization_solver, logger)
    logger.info("Solving...")
    opt.solve(instance, tee=True, logfile=run_config.results_path / f"logfile_{run_config.run_name}.log")#, keepfiles=True, symbolic_solver_labels=True)
    if save_flag:
        if empire_config.pickle_instance_flag:
            pickle_instance(instance, run_config.run_name, empire_config.use_temporary_directory, logger, empire_config.temporary_directory)

        write_results(instance, run_config.results_path, run_config.run_name, False, empire_config.emission_cap_flag, empire_config.print_iamc_flag, logger)

    return value(instance.Obj)


CAPACITY_VARS = [
    'genInstalledCap',  # n,g,i
    'transmissionInstalledCap',  # n1,n2, i
    'storPWInstalledCap',  # n,b,i
    'storENInstalledCap'  # n,b,i
]
def extract_capacity_params(mp_instance) -> dict[str, dict[tuple, float]]:
    """Extract capacity parameters from the master problem instance."""

    capacity_params = {
        var: {} for var in CAPACITY_VARS
    }

    capacity_params['genInstalledCap'] = {(*ng, period): mp_instance.genInstalledCap[ng, period].value for ng in mp_instance.GeneratorsOfNode for period in mp_instance.PeriodActive}
    capacity_params['storENInstalledCap'] = {(*nb, period): mp_instance.storENInstalledCap[nb, period].value for nb in mp_instance.StoragesOfNode for period in mp_instance.PeriodActive}
    capacity_params['storPWInstalledCap'] = {(*nb, period): mp_instance.storPWInstalledCap[nb, period].value for nb in mp_instance.StoragesOfNode for period in mp_instance.PeriodActive}
    capacity_params['transmissionInstalledCap'] = {(*line_pair, period): mp_instance.transmissionInstalledCap[line_pair, period].value for line_pair in mp_instance.BidirectionalArc for period in mp_instance.PeriodActive}

    return capacity_params


def define_initial_capacity_params(mp_instance, base_value=1e4) -> dict[str, dict[tuple, float]]:
    """Initialize capacity params as nested defaultdicts with base_value as default."""
    capacity_params = {
        var: {} for var in CAPACITY_VARS
    }

    capacity_params['genInstalledCap'] = {(*ng, period): base_value for ng in mp_instance.GeneratorsOfNode for period in mp_instance.PeriodActive}
    capacity_params['storENInstalledCap'] = {(*nb, period): base_value for nb in mp_instance.StoragesOfNode for period in mp_instance.PeriodActive}
    capacity_params['storPWInstalledCap'] = {(*nb, period): base_value for nb in mp_instance.StoragesOfNode for period in mp_instance.PeriodActive}
    capacity_params['transmissionInstalledCap'] = {(*line_pair, period): base_value for line_pair in mp_instance.BidirectionalArc for period in mp_instance.PeriodActive}

    # import pickle
    # with open("initial_capacity_params.pkl", "wb") as f:
    #     pickle.dump(capacity_params, f)
    return capacity_params