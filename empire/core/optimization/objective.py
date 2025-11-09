from pyomo.environ import (Expression, Objective, minimize, value)

SCALING_FACTOR = 1e-10


def investment_obj(model):
    def _get_block_cap(m, n1, n2):
        """Helper to get block capacity: per-line if available and positive, else global."""
        if (n1, n2) in m.transmissionLineBlockCap:
            blk = value(m.transmissionLineBlockCap[n1, n2])
            if blk > 0:
                return m.transmissionLineBlockCap[n1, n2]
        return m.transmissionLineBlockCapGlobal
    
    return SCALING_FACTOR * sum(
        model.discount_multiplier[i] * (
            # Generator investment costs
            sum(model.genInvCost[g,i] * model.genInvCap[n,g,i] for (n,g) in model.GeneratorsOfNode) +

            # Transmission: candidate lines (binary expansion with fixed block capacity)
            # Cost is per MW; multiply by block size to get total cost per build
            sum(
                model.transmissionInvCost[n1, n2, i]
                * _get_block_cap(model, n1, n2)
                * model.transmissionBuild[n1, n2, i]
                for (n1, n2) in model.CandidateTransmission
            ) +

            # Storage investment costs (power + energy parts)
            sum((model.storPWInvCost[b,i] * model.storPWInvCap[n,b,i] +
                 model.storENInvCost[b,i] * model.storENInvCap[n,b,i])
                for (n,b) in model.StoragesOfNode)
        )
        for i in model.PeriodActive
    )

def multiplier_rule(model,period):
    coeff=1
    if period>1:
        coeff=pow(1.0+model.discountrate,(-model.LeapYearsInvestment*(int(period)-1)))
    return coeff

def define_objective(model, include_investment=True, include_operational=True) -> None:
    model.discount_multiplier=Expression(model.PeriodActive, rule=multiplier_rule)

    def Obj_rule(model):
        obj = 0
        if include_investment:
            obj += investment_obj(model)
        if include_operational:
            obj += SCALING_FACTOR * sum(model.discount_multiplier[i] * model.operationalcost[i, w] for i in model.PeriodActive for w in model.Scenario)
        return obj 
    model.Obj = Objective(rule=Obj_rule, sense=minimize)

    return 