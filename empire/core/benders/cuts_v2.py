
from typing import Callable
import pickle 
from pathlib import Path
import pandas as pd
from pyomo.environ import value, ConcreteModel

class CapacityVariableHandler:
    """
    Class to handle extraction of duals and coefficients for a specific capacity variable.
    """
    def __init__(
            self, 
            constraint_name: str, 
            constraint_indices: tuple, 
            constraint_index_names: list[str],
            capacity_var_name: str, 
            capacity_var_index_selection_func: Callable,
            has_coefficient: bool = False,
            coefficients_subproblem_name: None | str = None, 
            coefficient_index_selection_func: None | Callable = None,
            index_names_to_keep: list[str] | None = None,
            coeff_sign: int = 1
            ):
        self.constraint_name: str = constraint_name
        self.constraint_indices: list[tuple] = constraint_indices   # all constraint indices
        self.constraint_index_names: list[str] = constraint_index_names
        self.capacity_var_name: str = capacity_var_name
        self.capacity_var_index_selection_func: Callable = capacity_var_index_selection_func  # indices of the capacity variable in the constraint indices
        self.has_coefficient: bool = has_coefficient
        self.coefficient_param_subproblem_name: str = coefficients_subproblem_name
        self.coefficient_index_selection_func: None | Callable = coefficient_index_selection_func
        self.index_names_to_keep: list[str] | None = index_names_to_keep
        self.coeff_sign: int = coeff_sign

        self.duals: pd.Series | None = None
        self.coefficients: pd.Series | None = None
        self.objective: float | None = None
        self.dual_and_coeff_total: pd.Series | None = None

        self.variable_inds: dict = {
            constraint_index_tuple:
            self.capacity_var_index_selection_func(constraint_index_tuple)
            for constraint_index_tuple in self.constraint_indices
        }


    def extract_data(self, subproblem_instance: ConcreteModel) -> pd.Series:
        self.extract_duals(subproblem_instance)
        self.extract_coefficients(subproblem_instance)
        dual_and_coeff = self.coeff_sign * self.duals * self.coefficients
        self.dual_and_coeff_total = dual_and_coeff.groupby(self.index_names_to_keep).sum()
        return self.dual_and_coeff_total

    def extract_duals(self, subproblem_instance: ConcreteModel):
        sp_constraint = getattr(subproblem_instance, self.constraint_name)
        self.duals = pd.Series({
            subproblem_index_tuple:
            subproblem_instance.dual[sp_constraint[subproblem_index_tuple]]
            for subproblem_index_tuple in self.constraint_indices
        })
        self.duals.index.names = self.constraint_index_names

    def extract_coefficients(self, subproblem_instance: ConcreteModel):
        if self.has_coefficient:
            subproblem_coefficient_param = getattr(subproblem_instance, self.coefficient_param_subproblem_name)
            self.coefficients = pd.Series({
                constraint_index_tuple:
                value(subproblem_coefficient_param[self.coefficient_index_selection_func(constraint_index_tuple)])
                    for constraint_index_tuple in self.constraint_indices
            })
            self.coefficients.index.names = self.constraint_index_names
        else:
            self.coefficients = pd.Series({constraint_index_tuple: 1.0 for constraint_index_tuple in self.constraint_indices})


def define_cut_structure(subproblem_instance: ConcreteModel, i: int, w: str) -> list[CapacityVariableHandler]:
    cut_structure = [
    CapacityVariableHandler(
        constraint_name="maxGenProduction",
        constraint_indices=[(n, g, h, i, w)  
            for n, g in subproblem_instance.GeneratorsOfNode
            for h in subproblem_instance.Operationalhour],
        constraint_index_names=["Node", "Generator", "Operationalhour", "Period", "Scenario"],
        capacity_var_name="genInstalledCap",
        capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, g, i
        has_coefficient=True,
        coefficients_subproblem_name="genCapAvail",
        coefficient_index_selection_func=lambda idx: (idx[0], idx[1], idx[2], idx[3], idx[4]),  # n, g, h, i, w
        index_names_to_keep=["Node", "Generator", "Period"]  # n, g, i
    ),
    # CapacityVariableHandler(
    #     constraint_name="ramping",
    #     constraint_indices=[(n, g, h, i, w)  
    #         for n, g in subproblem_instance.GeneratorsOfNode
    #         if g in subproblem_instance.ThermalGenerators
    #         for h in subproblem_instance.Operationalhour
    #         if h not in subproblem_instance.FirstHoursOfRegSeason and h not in subproblem_instance.FirstHoursOfPeakSeason],
    #     constraint_index_names=["Node", "Generator", "Operationalhour", "Period", "Scenario"],
    #     capacity_var_name="genInstalledCap",
    #     capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, g, i
    #     has_coefficient=True,
    #     coefficients_subproblem_name="genRampUpCap",
    #     coefficient_index_selection_func=lambda idx: (idx[1],),  # n, g, h, i, w
    #     index_names_to_keep=["Node", "Generator", "Period"]  # n, g, i
    # ),
    # CapacityVariableHandler(
    #     constraint_name="storage_operational_cap",
    #     constraint_indices=[(n,b,h,i,w)
    #         for n, b in subproblem_instance.StoragesOfNode
    #         for h in subproblem_instance.Operationalhour],
    #     constraint_index_names=["Node", "Storage", "Operationalhour", "Period", "Scenario"],
    #     capacity_var_name="storENInstalledCap",
    #     capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, b, i
    #     index_names_to_keep=["Node", "Storage", "Period"]  # n, b, i
    # ),
    # CapacityVariableHandler(
    #     constraint_name="storage_power_charg_cap",
    #     constraint_indices=[(n,b,h,i,w)
    #         for n, b in subproblem_instance.StoragesOfNode
    #         for h in subproblem_instance.Operationalhour],
    #     constraint_index_names=["Node", "Storage", "Operationalhour", "Period", "Scenario"],
    #     capacity_var_name="storPWInstalledCap",
    #     capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, b, i
    #     index_names_to_keep=["Node", "Storage", "Period"]  # n, b, i
    # ),
    # CapacityVariableHandler(
    #     constraint_name="storage_power_discharg_cap",
    #     constraint_indices=[(n,b,h,i,w)
    #         for n, b in subproblem_instance.StoragesOfNode
    #         for h in subproblem_instance.Operationalhour],
    #     constraint_index_names=["Node", "Storage", "Operationalhour", "Period", "Scenario"],
    #     capacity_var_name="storPWInstalledCap",
    #     capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, b, i
    #     has_coefficient=True,
    #     coefficients_subproblem_name="storageDiscToCharRatio",
    #     coefficient_index_selection_func=lambda idx: (idx[1]), # b
    #     index_names_to_keep=["Node", "Storage", "Period"]  # n, b, i
    # ),
    # CapacityVariableHandler(   # removed!
    #     constraint_name="hydro_gen_limit",
    #     constraint_indices=[(n,g,s,i,w)
    #         for n, g in subproblem_instance.GeneratorsOfNode
    #         if g in subproblem_instance.RegHydroGenerator
    #         for s in subproblem_instance.Season],
    #     constraint_index_names=["Node", "Generator", "Season", "Period", "Scenario"],
    #     capacity_var_name="genInstalledCap",
    #     capacity_var_index_selection_func=lambda idx: (idx[0], idx[1], idx[3]),  # n, g, i
    #     index_names_to_keep=["Node", "Generator", "Period"]  # n, g, i
    # ),
    # CapacityVariableHandler(
    #     constraint_name="transmission_cap",
    #     constraint_indices=[(n1,n2,h,i,w)
    #         for n1, n2 in subproblem_instance.DirectionalLink
    #         for h in subproblem_instance.Operationalhour],
    #     capacity_var_name="transmissionInstalledCap",
    #     constraint_index_names=["Node1", "Node2", "Operationalhour", "Period", "Scenario"],
    #     capacity_var_index_selection_func=lambda idx: ((idx[0], idx[1]), idx[3]),  # n1, n2, i
    #     index_names_to_keep=["Node1", "Node2", "Period"]  # n1, n2, i
    # )
    ]
    return cut_structure


