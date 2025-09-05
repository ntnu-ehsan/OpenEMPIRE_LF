"""Functions that initialize sets, parameters and variables that are used both in investment and operational models"""
from pyomo.environ import (Set, Param, Var, Constraint, NonNegativeReals)


def define_shared_sets(model, Period, north_sea_flag):
    model.Period = Set(ordered=True) #max period
    model.PeriodActive = Set(ordered=True, initialize=Period) #i
    model.Technology = Set(ordered=True) #t
    model.Generator = Set(ordered=True) #g
    model.Storage =  Set() #b
    model.Node = Set(ordered=True) #n
    if north_sea_flag:
        model.OffshoreNode = Set(ordered=True, within=model.Node) #n
    model.DirectionalLink = Set(dimen=2, within=model.Node*model.Node, ordered=True) #a
    model.TransmissionType = Set(ordered=True)

    model.GeneratorsOfTechnology=Set(dimen=2) #(t,g) for all t in T, g in G_t
    model.GeneratorsOfNode = Set(dimen=2) #(n,g) for all n in N, g in G_n
    model.TransmissionTypeOfDirectionalLink = Set(dimen=3) #(n1,n2,t) for all (n1,n2) in L, t in T
    model.ThermalGenerators = Set(within=model.Generator) #g_ramp
    model.RegHydroGenerator = Set(within=model.Generator) #g_reghyd
    model.HydroGenerator = Set(within=model.Generator) #g_hyd
    model.StoragesOfNode = Set(dimen=2) #(n,b) for all n in N, b in B_n
    model.DependentStorage = Set() #b_dagger

    
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
    return 


def load_shared_sets(model, data, tab_file_path, north_sea_flag):
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
    return 

def define_shared_parameters(model, discountrate, LeapYearsInvestment):
    model.storagePowToEnergy = Param(model.DependentStorage, default=1.0, mutable=True)

    # investment and operations
    model.discountrate = Param(initialize=discountrate) 
    model.LeapYearsInvestment = Param(initialize=LeapYearsInvestment)
    
    return 

def load_shared_parameters(model, data, tab_file_path):
    data.load(filename=str(tab_file_path / 'Storage_StoragePowToEnergy.tab'), param=model.storagePowToEnergy, format="table")
    return 