from __future__ import division

import logging
import os
import time
from pathlib import Path


from empire.utils import get_name_of_last_folder_in_path
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

from .operational_constraints import define_operational_constraints, prep_operational_parameters, define_operational_variables, define_operational_parameters
from .investment_constraints import define_investment_constraints, prep_investment_parameters
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

    #Define the sets

    logger.info("Declaring sets...")

    #Supply technology sets
    model.Generator = Set(ordered=True) #g
    model.Technology = Set(ordered=True) #t
    model.Storage =  Set() #b

    #Temporal sets
    model.Period = Set(ordered=True) #max period
    model.PeriodActive = Set(ordered=True, initialize=Period) #i
    model.Operationalhour = Set(ordered=True, initialize=Operationalhour) #h
    model.Season = Set(ordered=True, initialize=Season) #s

    #Spatial sets
    model.Node = Set(ordered=True) #n
    if north_sea_flag:
        model.OffshoreNode = Set(ordered=True, within=model.Node) #n
    model.DirectionalLink = Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
    model.TransmissionType = Set(ordered=True)

    #Stochastic sets
    model.Scenario = Set(ordered=True, initialize=Scenario) #w

    #Subsets
    model.GeneratorsOfTechnology=Set(dimen=2) #(t,g) for all t in T, g in G_t
    model.GeneratorsOfNode = Set(dimen=2) #(n,g) for all n in N, g in G_n
    model.TransmissionTypeOfDirectionalLink = Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
    model.ThermalGenerators = Set(within=model.Generator) #g_ramp
    model.RegHydroGenerator = Set(within=model.Generator) #g_reghyd
    model.HydroGenerator = Set(within=model.Generator) #g_hyd
    model.StoragesOfNode = Set(dimen=2) #(n,b) for all n in N, b in B_n
    model.DependentStorage = Set() #b_dagger
    model.HoursOfSeason = Set(dimen=2, ordered=True, initialize=HoursOfSeason) #(s,h) for all s in S, h in H_s
    model.FirstHoursOfRegSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfRegSeason)
    model.FirstHoursOfPeakSeason = Set(within=model.Operationalhour, ordered=True, initialize=FirstHoursOfPeakSeason)

    logger.info("Reading sets...")

    #Load the data

    data = DataPortal()
    data.load(filename=str(tab_file_path / 'Sets_Generator.tab'),format="set", set=model.Generator)
    data.load(filename=str(tab_file_path / 'Sets_ThermalGenerators.tab'),format="set", set=model.ThermalGenerators)
    data.load(filename=str(tab_file_path / 'Sets_HydroGenerator.tab'),format="set", set=model.HydroGenerator)
    data.load(filename=str(tab_file_path / 'Sets_HydroGeneratorWithReservoir.tab'),format="set", set=model.RegHydroGenerator)
    data.load(filename=str(tab_file_path / 'Sets_Storage.tab'),format="set", set=model.Storage)
    data.load(filename=str(tab_file_path / 'Sets_DependentStorage.tab'),format="set", set=model.DependentStorage)
    data.load(filename=str(tab_file_path / 'Sets_Technology.tab'),format="set", set=model.Technology)
    data.load(filename=str(tab_file_path / 'Sets_Node.tab'),format="set", set=model.Node)
    if north_sea_flag:
        data.load(filename=str(tab_file_path / 'Sets_OffshoreNode.tab'),format="set", set=model.OffshoreNode)
    data.load(filename=str(tab_file_path / 'Sets_Horizon.tab'),format="set", set=model.Period)
    data.load(filename=str(tab_file_path / 'Sets_DirectionalLines.tab'),format="set", set=model.DirectionalLink)
    data.load(filename=str(tab_file_path / 'Sets_LineType.tab'),format="set", set=model.TransmissionType)
    data.load(filename=str(tab_file_path / 'Sets_LineTypeOfDirectionalLines.tab'),format="set", set=model.TransmissionTypeOfDirectionalLink)
    data.load(filename=str(tab_file_path / 'Sets_GeneratorsOfTechnology.tab'),format="set", set=model.GeneratorsOfTechnology)
    data.load(filename=str(tab_file_path / 'Sets_GeneratorsOfNode.tab'),format="set", set=model.GeneratorsOfNode)
    data.load(filename=str(tab_file_path / 'Sets_StorageOfNodes.tab'),format="set", set=model.StoragesOfNode)

    logger.info("Constructing sub sets...")

    #Build arc subsets

    def NodesLinked_init(model, node):
        retval = []
        for (i,j) in model.DirectionalLink:
            if j == node:
                retval.append(i)
        return retval
    model.NodesLinked = Set(model.Node, initialize=NodesLinked_init)

    def BidirectionalArc_init(model):
        retval = []
        for (i,j) in model.DirectionalLink:
            if i != j and ((j,i) not in retval):
                retval.append((i,j))
        return retval
    model.BidirectionalArc = Set(dimen=2, initialize=BidirectionalArc_init, ordered=True) #l

    ##############
    ##PARAMETERS##
    ##############

    #Define the parameters

    logger.info("Declaring parameters...")

    #Scaling

    model.discountrate = Param(initialize=discountrate) 
    model.WACC = Param(initialize=wacc) 
    model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment)
    model.operationalDiscountrate = Param(mutable=True)
    model.sceProbab = Param(model.Scenario, mutable=True)
    model.seasScale = Param(model.Season, initialize=1.0, mutable=True)
    model.lengthRegSeason = Param(initialize=lengthRegSeason) 
    model.lengthPeakSeason = Param(initialize=lengthPeakSeason) 

    #Cost

    model.genCapitalCost = Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeCapitalCost = Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWCapitalCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENCapitalCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.genFixedOMCost = Param(model.Generator, model.Period, default=0, mutable=True)
    model.transmissionTypeFixedOMCost = Param(model.TransmissionType, model.Period, default=0, mutable=True)
    model.storPWFixedOMCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.storENFixedOMCost = Param(model.Storage, model.Period, default=0, mutable=True)
    model.genInvCost = Param(model.Generator, model.Period, default=9000000, mutable=True)
    model.transmissionInvCost = Param(model.BidirectionalArc, model.Period, default=3000000, mutable=True)
    model.storPWInvCost = Param(model.Storage, model.Period, default=1000000, mutable=True)
    model.storENInvCost = Param(model.Storage, model.Period, default=800000, mutable=True)
    model.transmissionLength = Param(model.BidirectionalArc, default=0, mutable=True)
    model.genVariableOMCost = Param(model.Generator, default=0.0, mutable=True)
    model.genFuelCost = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genMargCost = Param(model.Generator, model.Period, default=600, mutable=True)
    model.genCO2TypeFactor = Param(model.Generator, default=0.0, mutable=True)
    model.nodeLostLoadCost = Param(model.Node, model.Period, default=22000.0)
    model.CO2price = Param(model.Period, default=0.0, mutable=True)
    model.CCSCostTSVariable = Param(model.Period, default=0.0, mutable=True)
    model.CCSRemFrac = Param(initialize=0.9)

    #Node dependent technology limitations

    model.genRefInitCap = Param(model.GeneratorsOfNode, default=0.0, mutable=True)
    model.genScaleInitCap = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.genInitCap = Param(model.GeneratorsOfNode, model.Period, default=0.0, mutable=True)
    model.transmissionInitCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENInitCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.genMaxBuiltCap = Param(model.Node, model.Technology, model.Period, default=500000.0, mutable=True)
    model.transmissionMaxBuiltCap = Param(model.BidirectionalArc, model.Period, default=20000.0, mutable=True)
    model.storPWMaxBuiltCap = Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.storENMaxBuiltCap = Param(model.StoragesOfNode, model.Period, default=500000.0, mutable=True)
    model.genMaxInstalledCapRaw = Param(model.Node, model.Technology, default=0.0, mutable=True)
    model.genMaxInstalledCap = Param(model.Node, model.Technology, model.Period, default=0.0, mutable=True)
    model.transmissionMaxInstalledCapRaw = Param(model.BidirectionalArc, model.Period, default=0.0)
    model.transmissionMaxInstalledCap = Param(model.BidirectionalArc, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storPWMaxInstalledCapRaw = Param(model.StoragesOfNode, default=0.0, mutable=True)
    model.storENMaxInstalledCap = Param(model.StoragesOfNode, model.Period, default=0.0, mutable=True)
    model.storENMaxInstalledCapRaw = Param(model.StoragesOfNode, default=0.0, mutable=True)

    #Type dependent technology limitations

    # investment 
    model.genLifetime = Param(model.Generator, default=0.0, mutable=True)
    model.transmissionLifetime = Param(model.BidirectionalArc, default=40.0, mutable=True)
    model.storageLifetime = Param(model.Storage, default=0.0, mutable=True)


    if emission_cap_flag:
        model.CO2cap = Param(model.Period, default=5000.0, mutable=True)
    
    if load_change_module_flag:
        model.sloadMod = Param(model.Node, model.Operationalhour, model.Scenario, model.Period, default=0.0, mutable=True)

    #Load the parameters

    logger.info("Reading parameters...")
    logger.info("Reading parameters for Generator...")
    data.load(filename=str(tab_file_path / 'Generator_CapitalCosts.tab'), param=model.genCapitalCost, format="table")
    data.load(filename=str(tab_file_path / 'Generator_FixedOMCosts.tab'), param=model.genFixedOMCost, format="table")
    data.load(filename=str(tab_file_path / 'Generator_VariableOMCosts.tab'), param=model.genVariableOMCost, format="table")
    data.load(filename=str(tab_file_path / 'Generator_FuelCosts.tab'), param=model.genFuelCost, format="table")
    data.load(filename=str(tab_file_path / 'Generator_CCSCostTSVariable.tab'), param=model.CCSCostTSVariable, format="table")
    data.load(filename=str(tab_file_path / 'Generator_Efficiency.tab'), param=model.genEfficiency, format="table")
    data.load(filename=str(tab_file_path / 'Generator_RefInitialCap.tab'), param=model.genRefInitCap, format="table")
    data.load(filename=str(tab_file_path / 'Generator_ScaleFactorInitialCap.tab'), param=model.genScaleInitCap, format="table")
    data.load(filename=str(tab_file_path / 'Generator_InitialCapacity.tab'), param=model.genInitCap, format="table") #node_generator_intial_capacity.xlsx
    data.load(filename=str(tab_file_path / 'Generator_MaxBuiltCapacity.tab'), param=model.genMaxBuiltCap, format="table")#?
    data.load(filename=str(tab_file_path / 'Generator_MaxInstalledCapacity.tab'), param=model.genMaxInstalledCapRaw, format="table")#maximum_capacity_constraint_040317_high
    data.load(filename=str(tab_file_path / 'Generator_CO2Content.tab'), param=model.genCO2TypeFactor, format="table")
    data.load(filename=str(tab_file_path / 'Generator_RampRate.tab'), param=model.genRampUpCap, format="table")
    data.load(filename=str(tab_file_path / 'Generator_GeneratorTypeAvailability.tab'), param=model.genCapAvailTypeRaw, format="table")
    data.load(filename=str(tab_file_path / 'Generator_Lifetime.tab'), param=model.genLifetime, format="table") 

    logger.info("Reading parameters for Transmission...")
    data.load(filename=str(tab_file_path / 'Transmission_InitialCapacity.tab'), param=model.transmissionInitCap, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_MaxBuiltCapacity.tab'), param=model.transmissionMaxBuiltCap, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_MaxInstallCapacityRaw.tab'), param=model.transmissionMaxInstalledCapRaw, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_Length.tab'), param=model.transmissionLength, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_TypeCapitalCost.tab'), param=model.transmissionTypeCapitalCost, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_TypeFixedOMCost.tab'), param=model.transmissionTypeFixedOMCost, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_lineEfficiency.tab'), param=model.lineEfficiency, format="table")
    data.load(filename=str(tab_file_path / 'Transmission_Lifetime.tab'), param=model.transmissionLifetime, format="table")
    # Load electrical data for LOPF if requested
    if lopf_flag:
        load_line_parameters(model, tab_file_path, data, lopf_kwargs, logger) 


    logger.info("Reading parameters for Storage...")
    data.load(filename=str(tab_file_path / 'Storage_StorageBleedEfficiency.tab'), param=model.storageBleedEff, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StorageChargeEff.tab'), param=model.storageChargeEff, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StorageDischargeEff.tab'), param=model.storageDischargeEff, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StoragePowToEnergy.tab'), param=model.storagePowToEnergy, format="table")
    data.load(filename=str(tab_file_path / 'Storage_EnergyCapitalCost.tab'), param=model.storENCapitalCost, format="table")
    data.load(filename=str(tab_file_path / 'Storage_EnergyFixedOMCost.tab'), param=model.storENFixedOMCost, format="table")
    data.load(filename=str(tab_file_path / 'Storage_EnergyInitialCapacity.tab'), param=model.storENInitCap, format="table")
    data.load(filename=str(tab_file_path / 'Storage_EnergyMaxBuiltCapacity.tab'), param=model.storENMaxBuiltCap, format="table")
    data.load(filename=str(tab_file_path / 'Storage_EnergyMaxInstalledCapacity.tab'), param=model.storENMaxInstalledCapRaw, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StorageInitialEnergyLevel.tab'), param=model.storOperationalInit, format="table")
    data.load(filename=str(tab_file_path / 'Storage_PowerCapitalCost.tab'), param=model.storPWCapitalCost, format="table")
    data.load(filename=str(tab_file_path / 'Storage_PowerFixedOMCost.tab'), param=model.storPWFixedOMCost, format="table")
    data.load(filename=str(tab_file_path / 'Storage_InitialPowerCapacity.tab'), param=model.storPWInitCap, format="table")
    data.load(filename=str(tab_file_path / 'Storage_PowerMaxBuiltCapacity.tab'), param=model.storPWMaxBuiltCap, format="table")
    data.load(filename=str(tab_file_path / 'Storage_PowerMaxInstalledCapacity.tab'), param=model.storPWMaxInstalledCapRaw, format="table")
    data.load(filename=str(tab_file_path / 'Storage_Lifetime.tab'), param=model.storageLifetime, format="table")

    logger.info("Reading parameters for Node...")
    data.load(filename=str(tab_file_path / 'Node_NodeLostLoadCost.tab'), param=model.nodeLostLoadCost, format="table")
    data.load(filename=str(tab_file_path / 'Node_ElectricAnnualDemand.tab'), param=model.sloadAnnualDemand, format="table") 
    data.load(filename=str(tab_file_path / 'Node_HydroGenMaxAnnualProduction.tab'), param=model.maxHydroNode, format="table") 
    
    logger.info("Reading parameters for Stochastic...")

    if out_of_sample_flag:
        if sample_file_path:
            # Load operational input data EMPIRE has not seen when optimizing (in-sample)
            data.load(filename=str(sample_file_path / 'Stochastic_HydroGenMaxSeasonalProduction.tab'), param=model.maxRegHydroGenRaw, format="table")
            data.load(filename=str(sample_file_path / 'Stochastic_StochasticAvailability.tab'), param=model.genCapAvailStochRaw, format="table") 
            data.load(filename=str(sample_file_path / 'Stochastic_ElectricLoadRaw.tab'), param=model.sloadRaw, format="table")
        else:
            raise ValueError("'out_of_sample_flag = True' needs to be run with existing 'sample_file_path'")
    else:
        data.load(filename=str(tab_file_path / 'Stochastic_HydroGenMaxSeasonalProduction.tab'), param=model.maxRegHydroGenRaw, format="table")
        data.load(filename=str(tab_file_path / 'Stochastic_StochasticAvailability.tab'), param=model.genCapAvailStochRaw, format="table") 
        data.load(filename=str(tab_file_path / 'Stochastic_ElectricLoadRaw.tab'), param=model.sloadRaw, format="table") 
        
    logger.info("Reading parameters for General...")
    data.load(filename=str(tab_file_path / 'General_seasonScale.tab'), param=model.seasScale, format="table") 

    if emission_cap_flag:
        data.load(filename=str(tab_file_path / 'General_CO2Cap.tab'), param=model.CO2cap, format="table")
    else:
        data.load(filename=str(tab_file_path / 'General_CO2Price.tab'), param=model.CO2price, format="table")

    logger.info("Constructing parameter values...")
    if load_change_module_flag:
        data.load(filename=scenario_data_path / 'LoadchangeModule/Stochastic_ElectricLoadMod.tab', param=model.sloadMod, format="table")



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
        model.genInvCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
        model.transmisionInvCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
        model.storPWInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
        model.storENInvCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
        model.genInstalledCap = Var(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
        model.transmissionInstalledCap = Var(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
        model.storPWInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
        model.storENInstalledCap = Var(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)


    ###############
    ##EXPRESSIONS##
    ###############

    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)
    prep_operational_parameters(model, load_change_module_flag)
    define_operational_variables(model)
    define_operational_constraints(model, logger, emission_cap_flag, include_hydro_node_limit_constraint_flag=True)

    #############
    ##OBJECTIVE##
    #############

    def Obj_rule(model):
        return sum(model.discount_multiplier[i]*(
            sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
            sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
            sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode ) + \
            model.operationalcost[i]
        ) for i in model.PeriodActive)
    model.Obj = Objective(rule=Obj_rule, sense=minimize)

    ###############
    ##CONSTRAINTS##
    ###############



    if not out_of_sample_flag:
        # All constraints exclusively for investment decisions inactive when out_of_sample_flag
        define_investment_constraints(model, north_sea_flag)


    if lopf_flag:
        logger.info("LOPF constraints activated using method: %s", lopf_method)
        from .lopf_module import add_lopf_constraints
        kw = {} if lopf_kwargs is None else dict(lopf_kwargs)
        add_lopf_constraints(model, method=lopf_method, **kw)
    else:
        logger.warning("LOPF constraints not activated: %s", lopf_method)

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
        logger.info("----------------------Problem Statistics---------------------")
        logger.info("Nodes: %s", len(instance.Node))
        logger.info("Lines: %s", len(instance.BidirectionalArc))
        logger.info("")
        logger.info("GeneratorTypes: %s", len(instance.Generator))
        logger.info("TotalGenerators: %s", len(instance.GeneratorsOfNode))
        logger.info("StorageTypes: %s", len(instance.Storage))
        logger.info("TotalStorages: %s", len(instance.StoragesOfNode))
        logger.info("")
        logger.info("InvestmentUntil: %s", value(2020+int(len(instance.PeriodActive)*LeapYearsInvestment)))
        logger.info("Scenarios: %s", len(instance.Scenario))
        logger.info("TotalOperationalHoursPerScenario: %s", len(instance.Operationalhour))
        logger.info("TotalOperationalHoursPerInvYear: %s", len(instance.Operationalhour)*len(instance.Scenario))
        logger.info("Seasons: %s", len(instance.Season))
        logger.info("RegularSeasons: %s", len(instance.FirstHoursOfRegSeason))
        logger.info("LengthRegSeason: %s", value(instance.lengthRegSeason))
        logger.info("PeakSeasons: %s", len(instance.FirstHoursOfPeakSeason))
        logger.info("LengthPeakSeason: %s", value(instance.lengthPeakSeason))
        logger.info("")
        logger.info("Discount rate: %s", value(instance.discountrate))
        logger.info("Operational discount scale: %s", value(instance.operationalDiscountrate))
        logger.info("--------------------------------------------------------------")
        
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
        pickle_instance()
                
    #instance.display('outputs_gurobi.txt')

    #import pdb; pdb.set_trace()

    write_results(instance, result_file_path, instance_name, out_of_sample_flag, emission_cap_flag, print_iamc_flag, logger)

    if compute_operational_duals_flag and not out_of_sample_flag:
        run_operational_model(instance, opt, result_file_path, instance_name, logger)
        write_operational_results(instance, result_file_path, emission_cap_flag, logger)


def set_investments_as_parameters(model, data):
    # Redefine investment vars as input parameters
    model.genInvCap = Param(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmisionInvCap = Param(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInvCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInvCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.genInstalledCap = Param(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmissionInstalledCap = Param(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    return 


def load_optimized_investments(model, data, result_file_path):
    """Optimized investment decisions read from result file from in-sample runs"""
    data.load(filename=str(result_file_path / 'genInvCap.tab'), param=model.genInvCap, format="table")
    data.load(filename=str(result_file_path / 'transmisionInvCap.tab'), param=model.transmisionInvCap, format="table")
    data.load(filename=str(result_file_path / 'storPWInvCap.tab'), param=model.storPWInvCap, format="table")
    data.load(filename=str(result_file_path / 'storENInvCap.tab'), param=model.storENInvCap, format="table")
    data.load(filename=str(result_file_path / 'genInstalledCap.tab'), param=model.genInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'transmissionInstalledCap.tab'), param=model.transmissionInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'storPWInstalledCap.tab'), param=model.storPWInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'storENInstalledCap.tab'), param=model.storENInstalledCap, format="table")
    return 


def set_out_of_sample_path(result_file_path, sample_file_path):
    """Update result_file_path to output for given out_of_sample tree"""
    sample_tree = get_name_of_last_folder_in_path(sample_file_path)
    result_file_path = result_file_path / f"OutOfSample/{sample_tree}"
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    return result_file_path

def pickle_instance(
        instance,
        instance_name: str, 
        use_temp_dir_flag: bool,
        temp_dir: None | Path = None
        ):
    """Pickle the Pyomo model instance to a hardcoded location"""
    start = time.time()
    picklestring = f"instance{instance_name}.pkl"
    if use_temp_dir_flag:
        picklestring = temp_dir / picklestring
    with open(picklestring, mode='wb') as file:
        cloudpickle.dump(instance, file)
    end = time.time()
    logger.info("Pickling instance took [sec]: %d", end - start)
    return 
