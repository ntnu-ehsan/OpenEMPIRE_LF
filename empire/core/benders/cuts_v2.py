
from typing import Callable
import pickle 
from pathlib import Path

from pyomo.environ import value

class CapacityVariableHandler:
    def __init__(
            self, 
            constraint_name: str, 
            constraint_indices: tuple, 
            capacity_var_name: str, 
            capacity_var_index_selection_func: Callable,
            has_coefficient: bool = False,
            coefficients_subproblem_name: None | str = None, 
            coefficient_index_selection_func: None | Callable = None
            ):
        self.constraint_name: str = constraint_name
        self.constraint_indices: list[tuple] = constraint_indices   # all constraint indices
        self.capacity_var_name: str = capacity_var_name
        self.capacity_var_index_selection_func: Callable = capacity_var_index_selection_func  # indices of the capacity variable in the constraint indices
        self.has_coefficient: bool = has_coefficient
        self.coefficient_param_subproblem_name: str = coefficients_subproblem_name
        self.coefficient_index_selection_func: None | Callable = coefficient_index_selection_func

        self.duals: dict | None = None
        self.coefficients: dict | None = None
        self.objective: float | None = None

        self.variable_inds: dict = {
            constraint_index_tuple:
            self.capacity_var_index_selection_func(constraint_index_tuple)
            for constraint_index_tuple in self.constraint_indices
        }
        

    def extract_data(self, subproblem_instance):
        self.extract_duals(subproblem_instance)
        self.extract_coefficients(subproblem_instance)
        return self.duals, self.coefficients, self.variable_inds
    

    def extract_duals(self, subproblem_instance):
        sp_constraint = getattr(subproblem_instance, self.constraint_name)
        self.duals = {
            subproblem_index_tuple:
            subproblem_instance.dual[sp_constraint[subproblem_index_tuple]]
            for subproblem_index_tuple in self.constraint_indices
        }

    def extract_coefficients(self, subproblem_instance):
        if self.has_coefficient:
            subproblem_coefficient_param = getattr(subproblem_instance, self.coefficient_param_subproblem_name)
            self.coefficients = {
                constraint_index_tuple:
                value(subproblem_coefficient_param[self.coefficient_index_selection_func(constraint_index_tuple)])
                    for constraint_index_tuple in self.constraint_indices
                }
        else:
            self.coefficients = {constraint_index_tuple: 1.0 for constraint_index_tuple in self.constraint_indices}


def define_cut_structure(subproblem_instance, i, w):
    cut_structure = [
    CapacityVariableHandler(
        constraint_name="maxGenProduction",
        constraint_indices=[(n, g, h, i, w)  
            for n, g in subproblem_instance.GeneratorsOfNode
            for h in subproblem_instance.Operationalhour],
        capacity_var_name="genInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, g, i
        has_coefficient=True,
        coefficients_subproblem_name="genCapAvail",
        coefficient_index_selection_func=lambda idx: (idx[0], idx[1], idx[2], idx[3], idx[4])  # n, g, h, i, w
    ),
    CapacityVariableHandler(
        constraint_name="ramping",
        constraint_indices=[(n, g, h, i, w)  
            for n, g in subproblem_instance.GeneratorsOfNode
            if g in subproblem_instance.ThermalGenerators
            for h in subproblem_instance.Operationalhour
            if h not in subproblem_instance.FirstHoursOfRegSeason and h not in subproblem_instance.FirstHoursOfPeakSeason],
        capacity_var_name="genInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, g, i
        has_coefficient=True,
        coefficients_subproblem_name="genCapAvail",
        coefficient_index_selection_func=lambda idx: (idx[0], idx[1], idx[2], idx[3], idx[4])  # n, g, h, i, w
    ),
    CapacityVariableHandler(
        constraint_name="storage_operational_cap",
        constraint_indices=[(n,b,h,i,w)
            for n, b in subproblem_instance.StoragesOfNode
            for h in subproblem_instance.Operationalhour],
        capacity_var_name="storENInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, b, i
    ),
    CapacityVariableHandler(
        constraint_name="storage_power_charg_cap",
        constraint_indices=[(n,b,h,i,w)
            for n, b in subproblem_instance.StoragesOfNode
            for h in subproblem_instance.Operationalhour],
        capacity_var_name="storPWInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, b, i
    ),
    CapacityVariableHandler(
        constraint_name="storage_power_discharg_cap",
        constraint_indices=[(n,b,h,i,w)
            for n, b in subproblem_instance.StoragesOfNode
            for h in subproblem_instance.Operationalhour],
        capacity_var_name="storPWInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, b, i
        has_coefficient=True,
        coefficients_subproblem_name="storageDiscToCharRatio",
        coefficient_index_selection_func=lambda idx: (idx[1])
    ),
    CapacityVariableHandler(  
        constraint_name="hydro_gen_limit",
        constraint_indices=[(n,g,s,i,w)
            for n, g in subproblem_instance.GeneratorsOfNode
            if g in subproblem_instance.RegHydroGenerator
            for s in subproblem_instance.Season],
        capacity_var_name="genInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[2], idx[3]),  # n, g, s, i
    ),
    CapacityVariableHandler(
        constraint_name="transmission_cap",
        constraint_indices=[(n1,n2,h,i,w)
            for n1, n2 in subproblem_instance.DirectionalLink
            for h in subproblem_instance.Operationalhour],
        capacity_var_name="transmissionInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n1, n2, i
    )
    ]
    return cut_structure


