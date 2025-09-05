from pyomo.environ import (Expression, Objective, minimize)

def define_objective(model):
    
    def multiplier_rule(model,period):
        coeff=1
        if period>1:
            coeff=pow(1.0+model.discountrate,(-model.LeapYearsInvestment*(int(period)-1)))
        return coeff
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)

    
    def Obj_rule(model):
        return sum(model.discount_multiplier[i]*(
            sum(model.genInvCost[g,i]* model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode ) + \
            sum(model.transmissionInvCost[n1,n2,i]*model.transmisionInvCap[n1,n2,i] for (n1,n2) in model.BidirectionalArc ) + \
            sum((model.storPWInvCost[b,i]*model.storPWInvCap[n,b,i]+model.storENInvCost[b,i]*model.storENInvCap[n,b,i]) for (n,b) in model.StoragesOfNode )) 
            for i in model.PeriodActive) + \
            sum(model.operationalcost[i, w] for i in model.PeriodActive for w in model.Scenario)

    model.Obj = Objective(rule=Obj_rule, sense=minimize)

    return 