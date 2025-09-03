from pyomo.environ import AbstractModel, Constraint, value


def define_investment_constraints(
    model: AbstractModel,
    north_sea_flag: bool
    ):
    def lifetime_rule_gen(model, n, g, i):
        startPeriod=1
        if value(1+i-(model.genLifetime[g]/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.genLifetime[g]/model.LeapYearsInvestment)
        return sum(model.genInvCap[n,g,j]  for j in model.PeriodActive if j>=startPeriod and j<=i ) - model.genInstalledCap[n,g,i] + model.genInitCap[n,g,i]== 0   #
    model.installedCapDefinitionGen = Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=lifetime_rule_gen)

    ############################################################

    def lifetime_rule_storEN(model, n, b, i):
        startPeriod=1
        if value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storENInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storENInstalledCap[n,b,i] + model.storENInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorEN = Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storEN)

    ############################################################

    def lifetime_rule_storPOW(model, n, b, i):
        startPeriod=1
        if value(1+i-model.storageLifetime[b]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.storageLifetime[b]/model.LeapYearsInvestment)
        return sum(model.storPWInvCap[n,b,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.storPWInstalledCap[n,b,i] + model.storPWInitCap[n,b,i]== 0   #
    model.installedCapDefinitionStorPOW = Constraint(model.StoragesOfNode, model.PeriodActive, rule=lifetime_rule_storPOW)

    ############################################################

    def lifetime_rule_trans(model, n1, n2, i):
        startPeriod=1
        if value(1+i-model.transmissionLifetime[n1,n2]*(1/model.LeapYearsInvestment))>startPeriod:
            startPeriod=value(1+i-model.transmissionLifetime[n1,n2]/model.LeapYearsInvestment)
        return sum(model.transmisionInvCap[n1,n2,j]  for j in model.PeriodActive if j>=startPeriod and j<=i )- model.transmissionInstalledCap[n1,n2,i] + model.transmissionInitCap[n1,n2,i] == 0   #
    model.installedCapDefinitionTrans = Constraint(model.BidirectionalArc, model.PeriodActive, rule=lifetime_rule_trans)

    ############################################################

    def investment_gen_cap_rule(model, t, n, i):
        return sum(model.genInvCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxBuiltCap[n,t,i] <= 0
    model.investment_gen_cap = Constraint(model.Technology, model.Node, model.PeriodActive, rule=investment_gen_cap_rule)

    ############################################################

    def investment_trans_cap_rule(model, n1, n2, i):
        return model.transmisionInvCap[n1,n2,i] - model.transmissionMaxBuiltCap[n1,n2,i] <= 0
    model.investment_trans_cap = Constraint(model.BidirectionalArc, model.PeriodActive, rule=investment_trans_cap_rule)

    ############################################################

    def investment_storage_power_cap_rule(model, n, b, i):
        return model.storPWInvCap[n,b,i] - model.storPWMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_power_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_power_cap_rule)

    ############################################################

    def investment_storage_energy_cap_rule(model, n, b, i):
        return model.storENInvCap[n,b,i] - model.storENMaxBuiltCap[n,b,i] <= 0
    model.investment_storage_energy_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=investment_storage_energy_cap_rule)

    ############################################################

    def installed_gen_cap_rule(model, t, n, i):
        return sum(model.genInstalledCap[n,g,i] for g in model.Generator if (n,g) in model.GeneratorsOfNode and (t,g) in model.GeneratorsOfTechnology) - model.genMaxInstalledCap[n,t,i] <= 0
    model.installed_gen_cap = Constraint(model.Technology, model.Node, model.PeriodActive, rule=installed_gen_cap_rule)

    ############################################################

    def installed_trans_cap_rule(model, n1, n2, i):
        return model.transmissionInstalledCap[n1,n2,i] - model.transmissionMaxInstalledCap[n1,n2,i] <= 0
    model.installed_trans_cap = Constraint(model.BidirectionalArc, model.PeriodActive, rule=installed_trans_cap_rule)

    ############################################################

    def installed_storage_power_cap_rule(model, n, b, i):
        return model.storPWInstalledCap[n,b,i] - model.storPWMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_power_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_power_cap_rule)

    ############################################################

    def installed_storage_energy_cap_rule(model, n, b, i):
        return model.storENInstalledCap[n,b,i] - model.storENMaxInstalledCap[n,b,i] <= 0
    model.installed_storage_energy_cap = Constraint(model.StoragesOfNode, model.PeriodActive, rule=installed_storage_energy_cap_rule)

    ############################################################

    def power_energy_relate_rule(model, n, b, i):
        if b in model.DependentStorage:
            return model.storPWInstalledCap[n,b,i] - model.storagePowToEnergy[b]*model.storENInstalledCap[n,b,i] == 0   #
        else:
            return Constraint.Skip
    model.power_energy_relate = Constraint(model.StoragesOfNode, model.PeriodActive, rule=power_energy_relate_rule)


    if north_sea_flag:
        def wind_farm_tranmission_cap_rule(model, n1, n2, i):
            if n1 in model.OffshoreNode or n2 in model.OffshoreNode:
                if (n1,n2) in model.BidirectionalArc:
                    if n1 in model.OffshoreNode:
                        return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
                    else:
                        return model.transmissionInstalledCap[(n1,n2),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
                elif (n2,n1) in model.BidirectionalArc:
                    if n1 in model.OffshoreNode:
                        return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n1,g,i] for g in model.Generator if (n1,g) in model.GeneratorsOfNode)
                    else:
                        return model.transmissionInstalledCap[(n2,n1),i] <= sum(model.genInstalledCap[n2,g,i] for g in model.Generator if (n2,g) in model.GeneratorsOfNode)
                else:
                    return Constraint.Skip
            else:
                return Constraint.Skip
        model.wind_farm_transmission_cap = Constraint(model.Node, model.Node, model.PeriodActive, rule=wind_farm_tranmission_cap_rule)
    return 