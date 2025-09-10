from pyomo.environ import ConstraintList, value

# def extract_duals(subproblem_instance):
#     dual_maxgen = { (n,g,h,i,w): subproblem_instance.dual[subproblem_instance.maxGenProduction[n,g,h,i,w]]
#                     for n,g in subproblem_instance.GeneratorsOfNode
#                     for h in subproblem_instance.Operationalhour
#                     for i in subproblem_instance.periods_active
#                     for w in subproblem_instance.scenarios }

#     dual_ramp = { (n,g,h,i,w): subproblem_instance.dual[subproblem_instance.ramping[n,g,h,i,w]]
#                 for n,g in subproblem_instance.GeneratorsOfNode
#                 for h in subproblem_instance.Operationalhour
#                 for i in subproblem_instance.periods_active
#                 for w in subproblem_instance.scenarios
#                 if (h not in subproblem_instance.FirstHoursOfRegSeason and 
#                     h not in subproblem_instance.FirstHoursOfPeakSeason and 
#                     g in subproblem_instance.ThermalGenerators) }

#     dual_storage_cap = { (n,b,h,i,w): subproblem_instance.dual[subproblem_instance.storage_operational_cap[n,b,h,i,w]]
#                         for n,b in subproblem_instance.StoragesOfNode
#                         for h in subproblem_instance.Operationalhour
#                         for i in subproblem_instance.periods_active
#                         for w in subproblem_instance.scenarios }

#     dual_storage_charge = { (n,b,h,i,w): subproblem_instance.dual[subproblem_instance.storage_power_charg_cap[n,b,h,i,w]]
#                         for n,b in subproblem_instance.StoragesOfNode
#                         for h in subproblem_instance.Operationalhour
#                         for i in subproblem_instance.periods_active
#                         for w in subproblem_instance.scenarios }

#     dual_storage_discharge = { (n,b,h,i,w): subproblem_instance.dual[subproblem_instance.storage_power_discharg_cap[n,b,h,i,w]]
#                             for n,b in subproblem_instance.StoragesOfNode
#                             for h in subproblem_instance.Operationalhour
#                             for i in subproblem_instance.periods_active
#                             for w in subproblem_instance.scenarios }

#     dual_hydro = { (n,g,s,i,w): subproblem_instance.dual[subproblem_instance.hydro_gen_limit[n,g,s,i,w]]
#                 for n,g in subproblem_instance.GeneratorsOfNode
#                 for s in subproblem_instance.Season
#                 for i in subproblem_instance.periods_active
#                 for w in subproblem_instance.scenarios
#                 if g in subproblem_instance.RegHydroGenerator }

#     dual_transm = { (n1,n2,h,i,w): subproblem_instance.dual[subproblem_instance.transmission_cap[n1,n2,h,i,w]]
#                     for n1,n2 in subproblem_instance.DirectionalLink
#                     for h in subproblem_instance.Operationalhour
#                     for i in subproblem_instance.periods_active
#                     for w in subproblem_instance.scenarios }
#     duals = {
#         "max_gen": dual_maxgen,
#         "ramping": dual_ramp,
#         "storage_cap": dual_storage_cap,
#         "storage_charge": dual_storage_charge,
#         "storage_discharge": dual_storage_discharge,
#         "hydro": dual_hydro,
#         "transmission": dual_transm
#     }
#     return duals


# def extract_duals(subproblem_instance, i, w): # Generation upper bound
#     dual_maxgen = {}

#     for n, g in subproblem_instance.GeneratorsOfNode:
#         for h in subproblem_instance.Operationalhour:
#             dual_maxgen[(n, g, h, i, w)] = subproblem_instance.dual[subproblem_instance.maxGenProduction[n, g, h, i, w]]

#     # Ramping constraints
#     dual_ramp = {}
#     for n, g in subproblem_instance.GeneratorsOfNode:
#         if g in subproblem_instance.ThermalGenerators:
#             for h in subproblem_instance.Operationalhour:
#                 if h not in subproblem_instance.FirstHoursOfRegSeason and h not in subproblem_instance.FirstHoursOfPeakSeason:
#                     dual_ramp[(n, g, h, i, w)] = subproblem_instance.dual[subproblem_instance.ramping[n, g, h, i, w]]

#     # Storage operational capacity
#     dual_storage_cap = {}
#     for n, b in subproblem_instance.StoragesOfNode:
#         for h in subproblem_instance.Operationalhour:
#             dual_storage_cap[(n,b,h,i,w)] = subproblem_instance.dual[subproblem_instance.storage_operational_cap[n,b,h,i,w]]

#     # Storage power charge
#     dual_storage_charge = {}
#     for n, b in subproblem_instance.StoragesOfNode:
#         for h in subproblem_instance.Operationalhour:
#             dual_storage_charge[(n,b,h,i,w)] = subproblem_instance.dual[subproblem_instance.storage_power_charg_cap[n,b,h,i,w]]

#     # Storage power discharge
#     dual_storage_discharge = {}
#     for n, b in subproblem_instance.StoragesOfNode:
#         for h in subproblem_instance.Operationalhour:
#             dual_storage_discharge[(n,b,h,i,w)] = subproblem_instance.dual[subproblem_instance.storage_power_discharg_cap[n,b,h,i,w]]

#     # Hydro generation limit
#     dual_hydro = {}
#     for n, g in subproblem_instance.GeneratorsOfNode:
#         if g in subproblem_instance.RegHydroGenerator:
#             for s in subproblem_instance.Season:
#                 dual_hydro[(n,g,s,i,w)] = subproblem_instance.dual[subproblem_instance.hydro_gen_limit[n,g,s,i,w]]

#     # Transmission capacity
#     dual_transmission = {}
#     for n1, n2 in subproblem_instance.DirectionalLink:
#         for h in subproblem_instance.Operationalhour:
#             dual_transmission[(n1,n2,h,i,w)] = subproblem_instance.dual[subproblem_instance.transmission_cap[n1,n2,h,i,w]]

#     duals = {
#         "max_gen": dual_maxgen,
#         "ramping": dual_ramp,
#         "storage_cap": dual_storage_cap,
#         "storage_charge": dual_storage_charge,
#         "storage_discharge": dual_storage_discharge,
#         "hydro": dual_hydro,
#         "transmission": dual_transmission
#     }
#     return duals


# # Define all terms in a dictionary
# def extract_coefficients(
#         subproblem_instance, i, w
#         ) -> dict[str, dict[tuple, float]]:
#     """
#     !! Note change in indexing order for genCapAvail due to difference in indexing in constraints vs parameter."""
#     coeffs = {
#         'max_gen': {(n, g, h, i, w): value(subproblem_instance.genCapAvail[i,w,n,g,h]) 
#                     for n,g in subproblem_instance.GeneratorsOfNode
#                     for h in subproblem_instance.Operationalhour},
#         'ramping': {(g): value(subproblem_instance.genRampUpCap[g]) 
#                              for g in subproblem_instance.ThermalGenerators},
#         'storage_disc_to_char_ratio': {(n,b,h,i,w): value(subproblem_instance.storageDiscToCharRatio[b])
#                                 for n,b in subproblem_instance.StoragesOfNode
#                                 for h in subproblem_instance.Operationalhour}
#     }
#     return coeffs

# def _cut_definition(master, i, duals, coeffs, q_hat_i, old_capacities):

#     expr = q_hat_i[i]

#     # Start from subproblem objective at previous iteration
    

#     # Gen max production
#     expr += sum(duals["max_gen"][n, g, h, i, w] * coeffs["max_gen"](n, g, h, i, w) * (master.genInstalledCap[n,g,i] - old_capacities.genInstalledCap[n,g,i])
#                 for n,g,h,w in duals["max_gen"])

#     # Ramping
#     expr += sum(duals["ramping"][n, g, h, i, w] * coeffs["ramping"](n, g, h, i, w) * (master.genInstalledCap[n,g,i] - old_capacities.genInstalledCap[n,g,i])
#                 for n,g,h,w in duals["ramping"])

#     # Storage operational capacity
#     expr += sum(duals["storage_cap"][n,b,h,i,w] * (master.storENInstalledCap[n,b,i] - old_capacities.storENInstalledCap[n,b,i])
#                 for n,b,h,w in duals["storage_cap"])

#     # Storage charge
#     expr += sum(duals["storage_charge"][n,b,h,i,w] * (master.storPWInstalledCap[n,b,i] - old_capacities.storPWInstalledCap[n,b,i])
#                 for n,b,h,w in duals["storage_charge"])

#     # Storage discharge
#     expr += sum(duals["storage_discharge"][n,b,h,i,w] * coeffs["storage_disc_to_char_ratio"](n,b,h,i,w) * (master.storPWInstalledCap[n,b,i] - old_capacities.storPWInstalledCap[n,b,i])
#                 for n,b,h,w in duals["storage_discharge"])

#     # Hydro generation limit
#     expr += sum(duals["hydro"][n,g,s,i,w] * (master.maxRegHydroGen[n,g,s,i] - old_capacities.maxRegHydroGen[n,g,s,i])
#                 for n,g,s,w in duals["hydro"])

#     # Transmission
#     expr += sum(duals["transmission"][n1,n2,h,i,w] * (master.transmissionInstalledCap[n1,n2,i] - old_capacities.transmissionInstalledCap[n1,n2,i])
#                 for n1,n2,h,w in duals["transmission"])
    

#     return master.theta[i] >= expr


# def add_cuts(master, period_costs, duals, investment_params, i):
#     """Create a Benders cut based on the dual values from the subproblem and add it to the master problem.

#     Args:
#         master (AbstractModel): The master problem model.
#         sp_objective (float): The objective value of the subproblem.
#         duals (dict): A dictionary containing dual values from the subproblem.
#         investment_params (dict): A dictionary containing investment decision variables.

#     Returns:
#         Constraint: The Benders cut constraint added to the master problem.
#     """
#     cuts = []
#     # define cut constraints
    
#     cut = _cut_definition(master, i, duals, period_costs[i])
#     cuts.append(cut)
#     master.cut_constraints.add(expr=cut)
#     return cuts


        

