from pyomo.environ import Constraint, Set, Var, value, BuildAction, Expression, AbstractModel, NonNegativeReals, Param, PercentFraction
import logging
from empire.core.config import OperationalInputParams


logger = logging.getLogger(__name__)


def define_operational_sets(model: AbstractModel, operational_input_params: OperationalInputParams):
    # operational sets
    if scenarios is None:
        scenarios = operational_input_params.scenarios
    model.Operationalhour = Set(ordered=True, initialize=operational_input_params.Operationalhour) #h
    model.Season = Set(ordered=True, initialize=operational_input_params.Season) #s
    model.Scenario = Set(ordered=True) #w
    model.HoursOfSeason = Set(dimen=2, ordered=True, initialize=operational_input_params.HoursOfSeason) #(s,h) for all s in S, h in H_s
    model.FirstHoursOfRegSeason = Set(within=model.Operationalhour, ordered=True, initialize=operational_input_params.FirstHoursOfRegSeason)
    model.FirstHoursOfPeakSeason = Set(within=model.Operationalhour, ordered=True, initialize=operational_input_params.FirstHoursOfPeakSeason)
    return 


def define_operational_variables(
        model: AbstractModel
) -> AbstractModel:
    # Define operational variables for the model
    model.genOperational = Var(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.storOperational = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.transmisionOperational = Var(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals) #flow
    model.storCharge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.storDischarge = Var(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    model.loadShed = Var(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, domain=NonNegativeReals)
    return 


def define_operational_parameters(
        model: AbstractModel,
        operational_input_params: OperationalInputParams,
    ):
    # operational deterministic parameters 
    model.operationalDiscountrate = Param(mutable=True)
    model.seasScale = Param(model.Season, initialize=1.0, mutable=True)
    model.lengthRegSeason = Param(initialize=operational_input_params.lengthRegSeason, mutable=False)
    model.lengthPeakSeason = Param(initialize=operational_input_params.lengthPeakSeason, mutable=False)

    model.lineEfficiency = Param(model.DirectionalLink, default=0.97, mutable=True)
    model.lineReactance   = Param(model.BidirectionalArc, default=0.0, mutable=True)
    model.lineSusceptance = Param(model.BidirectionalArc, default=0.0, mutable=True)
    model.storageChargeEff = Param(model.Storage, default=1.0, mutable=True)
    model.storageDischargeEff = Param(model.Storage, default=1.0, mutable=True)
    model.storageBleedEff = Param(model.Storage, default=1.0, mutable=True)
    model.genRampUpCap = Param(model.ThermalGenerators, default=0.0, mutable=True)
    model.storageDiscToCharRatio = Param(model.Storage, default=1.0, mutable=True) #NB! Hard-coded
    model.genCO2TypeFactor = Param(model.Generator, default=0.0, mutable=True)
    model.genVariableOMCost = Param(model.Generator, default=0.0, mutable=True)
    model.CCSRemFrac = Param(initialize=0.9)
    model.maxHydroNode = Param(model.Node, default=0.0, mutable=True)
    model.genCapAvailTypeRaw = Param(model.Generator, default=1.0, mutable=True)
    model.storOperationalInit = Param(model.Storage, default=0.0, mutable=True) #Percentage of installed energy capacity initially
    return 


def define_period_and_scenario_dependent_parameters(model: AbstractModel, emission_cap_flag: bool):
    # scenario and/or period-dependent parameters
    model.sceProbab = Param(model.Scenario, mutable=True)
    model.genEfficiency = Param(model.Generator, model.Period, default=1.0, mutable=True)
    model.sloadAnnualDemand = Param(model.Node, model.Period, default=0.0, mutable=True)
    model.genMargCost = Param(model.Generator, model.Period, default=600, mutable=True)
    model.nodeLostLoadCost = Param(model.Node, model.Period, default=22000.0)
    model.genFuelCost = Param(model.Generator, model.Period, default=0.0, mutable=True)
    model.CO2price = Param(model.Period, default=0.0, mutable=True)

    model.CCSCostTSVariable = Param(model.Period, default=0.0, mutable=True)
    if emission_cap_flag:
        model.CO2cap = Param(model.Period, default=5000.0, mutable=True)
    return 


def define_stochastic_input(model):

    model.sloadRaw = Param(model.Period, model.Scenario, model.Node, model.Operationalhour, default=0.0, mutable=True)
    model.sload = Param(model.Period, model.Scenario, model.Node, model.Operationalhour, default=0.0, mutable=True)
    model.genCapAvailStochRaw = Param(model.GeneratorsOfNode, model.Operationalhour, model.Period, model.Scenario, default=1., mutable=True, within=PercentFraction)
    model.genCapAvail = Param(model.GeneratorsOfNode, model.Operationalhour, model.Period, model.Scenario, default=0.0, mutable=True, within=PercentFraction)
    model.maxRegHydroGenRaw = Param(model.Period, model.Scenario, model.Node, model.HoursOfSeason, default=0.0, mutable=True)
    model.maxRegHydroGen = Param(model.Period, model.Scenario, model.Node, model.Season, default=0.0, mutable=True)
    return 


def load_operational_sets(model, data, scenarios):
    load_set(data, model.Scenario, scenarios)
    return 

def load_operational_parameters(model, data, tab_file_path, emission_cap_flag, out_of_sample_flag, sample_file_path=None, scenario_data_path=None):
    # Load operational generator parameters
    data.load(filename=str(tab_file_path / 'Generator_VariableOMCosts.tab'), param=model.genVariableOMCost, format="table")
    data.load(filename=str(tab_file_path / 'Generator_CCSCostTSVariable.tab'), param=model.CCSCostTSVariable, format="table")
    data.load(filename=str(tab_file_path / 'Generator_CO2Content.tab'), param=model.genCO2TypeFactor, format="table")
    data.load(filename=str(tab_file_path / 'Generator_GeneratorTypeAvailability.tab'), param=model.genCapAvailTypeRaw, format="table")
    data.load(filename=str(tab_file_path / 'Generator_RampRate.tab'), param=model.genRampUpCap, format="table")

    # Load operational transmission line parameters
    data.load(filename=str(tab_file_path / 'Transmission_lineEfficiency.tab'), param=model.lineEfficiency, format="table")

    # Storage parameters
    data.load(filename=str(tab_file_path / 'Storage_StorageBleedEfficiency.tab'), param=model.storageBleedEff, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StorageChargeEff.tab'), param=model.storageChargeEff, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StorageDischargeEff.tab'), param=model.storageDischargeEff, format="table")
    data.load(filename=str(tab_file_path / 'Storage_StorageInitialEnergyLevel.tab'), param=model.storOperationalInit, format="table")

    data.load(filename=str(tab_file_path / 'Node_HydroGenMaxAnnualProduction.tab'), param=model.maxHydroNode, format="table")
        
    logger.info("Reading parameters for General...")
    data.load(filename=str(tab_file_path / 'General_seasonScale.tab'), param=model.seasScale, format="table")

    data.load(filename=str(tab_file_path / 'Generator_Efficiency.tab'), param=model.genEfficiency, format="table")
    data.load(filename=str(tab_file_path / 'Node_ElectricAnnualDemand.tab'), param=model.sloadAnnualDemand, format="table")
    data.load(filename=str(tab_file_path / 'Node_NodeLostLoadCost.tab'), param=model.nodeLostLoadCost, format="table")
    data.load(filename=str(tab_file_path / 'Generator_FuelCosts.tab'), param=model.genFuelCost, format="table")
    if emission_cap_flag:
        data.load(filename=str(tab_file_path / 'General_CO2Cap.tab'), param=model.CO2cap, format="table")
    else:
        data.load(filename=str(tab_file_path / 'General_CO2Price.tab'), param=model.CO2price, format="table")

    return 


def load_stochastic_input(model, data, tab_file_path, out_of_sample_flag=False, sample_file_path=None):
    if not out_of_sample_flag:
        data.load(filename=str(tab_file_path / 'Stochastic_HydroGenMaxSeasonalProduction.tab'), param=model.maxRegHydroGenRaw, format="table")
        data.load(filename=str(tab_file_path / 'Stochastic_StochasticAvailability.tab'), param=model.genCapAvailStochRaw, format="table")
        data.load(filename=str(tab_file_path / 'Stochastic_ElectricLoadRaw.tab'), param=model.sloadRaw, format="table")
    else:
        if sample_file_path:
                # Load operational input data EMPIRE has not seen when optimizing (in-sample)
                data.load(filename=str(sample_file_path / 'Stochastic_HydroGenMaxSeasonalProduction.tab'), param=model.maxRegHydroGenRaw, format="table")
                data.load(filename=str(sample_file_path / 'Stochastic_StochasticAvailability.tab'), param=model.genCapAvailStochRaw, format="table") 
                data.load(filename=str(sample_file_path / 'Stochastic_ElectricLoadRaw.tab'), param=model.sloadRaw, format="table")
        else:
            raise ValueError("'out_of_sample_flag = True' needs to be run with existing 'sample_file_path'")
    return 

# def update_stochastic_input(instance, tab_file_path):
#     scenario_data = DataPortal()
#     scenario_data.load(filename=str(tab_file_path / 'Stochastic_HydroGenMaxSeasonalProduction.tab'), param=instance.maxRegHydroGenRaw, format="table")
#     scenario_data.load(filename=str(tab_file_path / 'Stochastic_StochasticAvailability.tab'), param=model.genCapAvailStochRaw, format="table") 
#     scenario_data.load(filename=str(tab_file_path / 'Stochastic_ElectricLoadRaw.tab'), param=model.sloadRaw, format="table")

#     for n in instance.Node:
#         for s in instance.Season:
#             for i in instance.PeriodActive:
#                 for w in instance.Scenario:
#                     instance.maxRegHydroGenRaw[i, w, n, s] = scenario_data[w].maxRegHydroGen[i, n, s]
#                     instance.genCapAvailStochRaw[i, w, n, s] = scenario_data[w].genCapAvailStoch[i, n, s]
#                     instance.sloadRaw[i, w, n, s] = scenario_data[w].sload[i, n, s]

def prep_operational_parameters(model) -> None:
    """Prepare operational parameters for the model. 

    """

    def prepSceProbab_rule(model):
        #Build an equiprobable probability distribution for Scenario

        for sce in model.Scenario:
            model.sceProbab[sce] = value(1/len(model.Scenario))

    model.build_SceProbab = BuildAction(rule=prepSceProbab_rule)


    def prepOperationalCostGen_rule(model):
        #Build generator short term marginal costs

        for g in model.Generator:
            for i in model.PeriodActive:
                if ('CCS',g) in model.GeneratorsOfTechnology:
                    costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+(1-model.CCSRemFrac)*model.genCO2TypeFactor[g]*model.CO2price[i])+ \
                    (3.6/model.genEfficiency[g,i])*(model.CCSRemFrac*model.genCO2TypeFactor[g]*model.CCSCostTSVariable[i])+ \
                    model.genVariableOMCost[g]
                else:
                    costperenergyunit=(3.6/model.genEfficiency[g,i])*(model.genFuelCost[g,i]+model.genCO2TypeFactor[g]*model.CO2price[i])+ \
                    model.genVariableOMCost[g]
                model.genMargCost[g,i]=costperenergyunit

    model.build_OperationalCostGen = BuildAction(rule=prepOperationalCostGen_rule)
    
    def prepOperationalDiscountrate_rule(model):
        #Build operational discount rate

        model.operationalDiscountrate = sum((1+model.discountrate)**(-j) for j in list(range(0,value(model.LeapYearsInvestment))))

    model.build_operationalDiscountrate = BuildAction(rule=prepOperationalDiscountrate_rule)     

    return 


def derive_stochastic_parameters(instance):
    """Set values for stochastic parameters."""
    def _set_maxRegHydroGen(instance):
        """Assign values to maxRegHydroGen from maxRegHydroGenRaw."""
        for n in instance.Node:
            for s in instance.Season:
                for i in instance.PeriodActive:
                    for w in instance.Scenario:
                        total = sum(
                            instance.maxRegHydroGenRaw[i, w, n, s, h]
                            for h in instance.Operationalhour
                            if (s, h) in instance.HoursOfSeason
                        )
                        instance.maxRegHydroGen[i, w, n, s] = total
    _set_maxRegHydroGen(instance)

    def _set_sload(instance):
        # Precompute cutoff
        cutoff = list(instance.FirstHoursOfRegSeason)[-1] + instance.lengthRegSeason

        for n in instance.Node:
            for i in instance.PeriodActive:
                # Compute probability-weighted raw demand
                noderawdemand = sum(
                    instance.sceProbab[w] * instance.seasScale[s] * instance.sloadRaw[i, w, n, h]
                    for (s, h) in instance.HoursOfSeason
                    if h < cutoff  # adjust if you want peak hours included
                    for w in instance.Scenario
                )

                # Scaling factor, safe for zero demand
                
                hourlyscale = instance.sloadAnnualDemand[n, i] / noderawdemand if value(noderawdemand) != 0 else 0

                for w in instance.Scenario:
                    for h in instance.Operationalhour:
                        instance.sload[i, w, n, h] = instance.sloadRaw[i, w, n, h] * hourlyscale 

    _set_sload(instance)

    def _set_genCapAvail(instance):
        """Assign generator availability based on type and stochastic raw availability."""
        for (n, g) in instance.GeneratorsOfNode:
            for h in instance.Operationalhour:
                for w in instance.Scenario:
                    for i in instance.PeriodActive:
                        instance.genCapAvail[n, g, h, i, w] = (
                            instance.genCapAvailTypeRaw[g] * instance.genCapAvailStochRaw[n, g, h, i, w]
                        )
    _set_genCapAvail(instance)
    return 


def define_operational_constraints(
        model: AbstractModel, 
        logger: logging.Logger,
        emission_cap_flag: bool, 
        include_hydro_node_limit_constraint_flag: bool
        ) -> None:
    # Define operational constraints for the model

    def shed_component_rule(model,i, w):
        """Defines load shedding cost"""
        return sum(model.operationalDiscountrate*model.seasScale[s]*model.sceProbab[w]*model.nodeLostLoadCost[n,i]*model.loadShed[n,h,i,w] for n in model.Node for (s,h) in model.HoursOfSeason)
    model.shedcomponent=Expression(model.PeriodActive,model.Scenario,rule=shed_component_rule)

    def operational_cost_rule(model,i, w):
        """Defines operational cost"""
        return model.shedcomponent[i,w] + sum(model.operationalDiscountrate*model.seasScale[s]*model.sceProbab[w]*model.genMargCost[g,i]*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason)
    model.operationalcost=Expression(model.PeriodActive,model.Scenario,rule=operational_cost_rule)

    # note: this cannot be included in the Benders
    if include_hydro_node_limit_constraint_flag:
        def hydro_node_limit_rule(model, n, i):
            return sum(model.genOperational[n,g,h,i,w]*model.seasScale[s]*model.sceProbab[w] for g in model.HydroGenerator if (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason for w in model.Scenario) - model.maxHydroNode[n] <= 0   #
        model.hydro_node_limit = Constraint(model.Node, model.PeriodActive, rule=hydro_node_limit_rule)

    # scenario-dependent constraints:

    def FlowBalance_rule(model, n, h, i, w):
        return sum(model.genOperational[n,g,h,i,w] for g in model.Generator if (n,g) in model.GeneratorsOfNode) \
            + sum((model.storageDischargeEff[b]*model.storDischarge[n,b,h,i,w]-model.storCharge[n,b,h,i,w]) for b in model.Storage if (n,b) in model.StoragesOfNode) \
            + sum((model.lineEfficiency[link,n]*model.transmisionOperational[link,n,h,i,w] - model.transmisionOperational[n,link,h,i,w]) for link in model.NodesLinked[n]) \
            - model.sload[i,w,n,h] + model.loadShed[n,h,i,w] \
            == 0
    model.FlowBalance = Constraint(model.Node, model.Operationalhour, model.PeriodActive, model.Scenario, rule=FlowBalance_rule)

    #################################################################

    def genMaxProd_rule(model, n, g, h, i, w):
            return model.genOperational[n,g,h,i,w] - model.genCapAvail[n, g, h, i, w]*model.genInstalledCap[n,g,i] <= 0
    model.maxGenProduction = Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=genMaxProd_rule)

    #################################################################

    def ramping_rule(model, n, g, h, i, w):
        if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
            return Constraint.Skip
        else:
            if g in model.ThermalGenerators:
                return model.genOperational[n,g,h,i,w]-model.genOperational[n,g,(h-1),i,w] - model.genRampUpCap[g]*model.genInstalledCap[n,g,i] <= 0   #
            else:
                return Constraint.Skip
    model.ramping = Constraint(model.GeneratorsOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=ramping_rule)

    #################################################################

    def storage_energy_balance_rule(model, n, b, h, i, w):
        if h in model.FirstHoursOfRegSeason or h in model.FirstHoursOfPeakSeason:
            return model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
        else:
            return model.storageBleedEff[b]*model.storOperational[n,b,(h-1),i,w] + model.storageChargeEff[b]*model.storCharge[n,b,h,i,w]-model.storDischarge[n,b,h,i,w]-model.storOperational[n,b,h,i,w] == 0   #
    model.storage_energy_balance = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_energy_balance_rule)

    #################################################################

    def storage_seasonal_net_zero_balance_rule(model, n, b, h, i, w):
        if h in model.FirstHoursOfRegSeason:
            return model.storOperational[n,b,h+value(model.lengthRegSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
        elif h in model.FirstHoursOfPeakSeason:
            return model.storOperational[n,b,h+value(model.lengthPeakSeason)-1,i,w] - model.storOperationalInit[b]*model.storENInstalledCap[n,b,i] == 0  #
        else:
            return Constraint.Skip
    model.storage_seasonal_net_zero_balance = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_seasonal_net_zero_balance_rule)

    #################################################################

    def storage_operational_cap_rule(model, n, b, h, i, w):
        return model.storOperational[n,b,h,i,w] - model.storENInstalledCap[n,b,i]  <= 0   #
    model.storage_operational_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_operational_cap_rule)

    #################################################################

    def storage_power_discharg_cap_rule(model, n, b, h, i, w):
        return model.storDischarge[n,b,h,i,w] - model.storageDiscToCharRatio[b]*model.storPWInstalledCap[n,b,i] <= 0   #
    model.storage_power_discharg_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_power_discharg_cap_rule)

    #################################################################

    def storage_power_charg_cap_rule(model, n, b, h, i, w):
        return model.storCharge[n,b,h,i,w] - model.storPWInstalledCap[n,b,i] <= 0   #
    model.storage_power_charg_cap = Constraint(model.StoragesOfNode, model.Operationalhour, model.PeriodActive, model.Scenario, rule=storage_power_charg_cap_rule)

    #################################################################

    def hydro_gen_limit_rule(model, n, g, s, i, w):
        if g in model.RegHydroGenerator:
            return sum(model.genOperational[n,g,h,i,w] for h in model.Operationalhour if (s,h) in model.HoursOfSeason) - model.maxRegHydroGen[i,w,n,s] <= 0
        else:
            return Constraint.Skip  #
    model.hydro_gen_limit = Constraint(model.GeneratorsOfNode, model.Season, model.PeriodActive, model.Scenario, rule=hydro_gen_limit_rule)

    #################################################################

    def transmission_cap_rule(model, n1, n2, h, i, w):
        if (n1,n2) in model.BidirectionalArc:
            return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n1,n2),i] <= 0
        elif (n2,n1) in model.BidirectionalArc:
            return model.transmisionOperational[(n1,n2),h,i,w]  - model.transmissionInstalledCap[(n2,n1),i] <= 0
    model.transmission_cap = Constraint(model.DirectionalLink, model.Operationalhour, model.PeriodActive, model.Scenario, rule=transmission_cap_rule)

    #################################################################

    if emission_cap_flag:
        def emission_cap_rule(model, i, w):
            return sum(model.seasScale[s]*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])*model.genOperational[n,g,h,i,w] for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason)/1000000 \
                - model.CO2cap[i] <= 0   #
        model.emission_cap = Constraint(model.PeriodActive, model.Scenario, rule=emission_cap_rule)

    #################################################################
    
    return 
