

from pyomo.environ import DataPortal, Param
from pathlib import Path
import pandas as pd
import tempfile
import os


def read_tab_file(file_path: Path) -> dict:
    """
    Reads a tab-separated file with the first columns as indices
    and the last column as the value.
    Returns a dict with index tuples as keys.
    """
    df = pd.read_csv(file_path, sep="\t")
    # Assume last column is value
    value_col = df.columns[-1]
    index_cols = df.columns[:-1]
    
    data = {}
    for _, row in df.iterrows():
        idx = tuple(row[col] for col in index_cols)
        data[idx] = row[value_col]
    return data

def filter_param_by_dims(raw_data: dict, dim_indices: dict) -> dict:
    """
    Filters raw_data dict of indexed Param values by allowed values on specified dimensions.
    dim_indices: dict {dim_position: allowed_values}
    """
    return {
        idx: val
        for idx, val in raw_data.items()
        if all(idx[pos] in allowed for pos, allowed in dim_indices.items())
    }




def load_dict_into_dataportal(data_portal, param, data_dict):
    def _return_list(idx):
        b = []
        for i in idx:
            if isinstance(i, tuple):
                b.extend(list(i))
            else:
                b.append(i)
        return b
    
    rows = []
    for idx, val in data_dict.items():
        if isinstance(idx, tuple):
            idx_list = _return_list(idx)
            rows.append((*idx_list, val))
        else:
            rows.append((idx, val))

    df = pd.DataFrame(rows)

    # Name columns: index1, index2, ..., value
    n_index = df.shape[1] - 1
    df.columns = [f"index{i+1}" for i in range(n_index)] + ["value"]

    # Write to a temporary .tab
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tab", delete=False) as tmpfile:
        df.to_csv(tmpfile.name, sep="\t", index=False)
        tmpname = tmpfile.name

    try:
        # Load into DataPortal
        data_portal.load(filename=tmpname, param=param, format="table")
    finally:
        # Clean up
        os.remove(tmpname)


def load_operational_parameter(
    data: DataPortal,
    tab_file_path: Path,
    param_component: Param,
    periods_to_load: list[int] | None = None,
    period_indnr: int | None = None,
    scenarios_to_load: list[str] | None = None,
    scenario_indnr: int | None = None,
):
    """
    Loads operational parameter for an abstract model.
    Only loads entries for the specified periods and scenarios.
    """
    raw_data = read_tab_file(tab_file_path)
    
    dim_indices = {}
    if periods_to_load is not None and period_indnr is not None:
        dim_indices[period_indnr] = periods_to_load
    if scenarios_to_load is not None and scenario_indnr is not None:
        dim_indices[scenario_indnr] = scenarios_to_load
    filtered_data = filter_param_by_dims(
        raw_data,
        dim_indices=dim_indices
    )
    # for idx in filtered_data.keys():
    #     print(idx, type(idx[0]), type(idx[1]))

    load_dict_into_dataportal(data, param_component, filtered_data)


def load_selected_operational_parameters(model, data, tab_file_path, emission_cap_flag, out_of_sample_flag, period, scenario, sample_file_path=None, scenario_data_path=None):
    # Load operational generator parameters
    data.load(filename=str(tab_file_path / 'Generator_VariableOMCosts.tab'), param=model.genVariableOMCost, format="table")

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
        
    # logger.info("Reading parameters for General...")
    data.load(filename=str(tab_file_path / 'General_seasonScale.tab'), param=model.seasScale, format="table")

    # data.load(filename=str(tab_file_path / 'Generator_Efficiency.tab'), param=model.genEfficiency, format="table")
    # # the problem must be related to 
    # data.load(filename=str(tab_file_path / 'Node_ElectricAnnualDemand.tab'), param=model.sloadAnnualDemand, format="table")
    # data.load(filename=str(tab_file_path / 'Node_NodeLostLoadCost.tab'), param=model.nodeLostLoadCost, format="table")
    # data.load(filename=str(tab_file_path / 'Generator_FuelCosts.tab'), param=model.genFuelCost, format="table")
    # if emission_cap_flag:
    #     data.load(filename=str(tab_file_path / 'General_CO2Cap.tab'), param=model.CO2cap, format="table")
    # else:
    #     data.load(filename=str(tab_file_path / 'General_CO2Price.tab'), param=model.CO2price, format="table")
    load_operational_parameter(data, tab_file_path / 'Generator_Efficiency.tab', model.genEfficiency, periods_to_load=[period], period_indnr=1)
    load_operational_parameter(data, tab_file_path / 'Node_ElectricAnnualDemand.tab', model.sloadAnnualDemand, periods_to_load=[period], period_indnr=1)
    load_operational_parameter(data, tab_file_path / 'Node_NodeLostLoadCost.tab', model.nodeLostLoadCost, periods_to_load=[period], period_indnr=1)
    load_operational_parameter(data, tab_file_path / 'Generator_FuelCosts.tab', model.genFuelCost, periods_to_load=[period], period_indnr=1)
    load_operational_parameter(data, tab_file_path / 'Generator_CCSCostTSVariable.tab', model.CCSCostTSVariable, periods_to_load=[period], period_indnr=0)
    if emission_cap_flag:
        load_operational_parameter(data, tab_file_path / 'General_CO2Cap.tab', model.CO2cap, periods_to_load=[period], period_indnr=0)
    else:
        load_operational_parameter(data, tab_file_path / 'General_CO2Price.tab', model.CO2price, periods_to_load=[period], period_indnr=0)

    load_operational_parameter(data, tab_file_path / 'Stochastic_HydroGenMaxSeasonalProduction.tab', model.maxRegHydroGenRaw, periods_to_load=[period], period_indnr=0, scenarios_to_load=[scenario], scenario_indnr=1)
    load_operational_parameter(data, tab_file_path / 'Stochastic_StochasticAvailability.tab', model.genCapAvailStochRaw, periods_to_load=[period], period_indnr=3, scenarios_to_load=[scenario], scenario_indnr=4)
    load_operational_parameter(data, tab_file_path / 'Stochastic_ElectricLoadRaw.tab', model.sloadRaw, periods_to_load=[period], period_indnr=0, scenarios_to_load=[scenario], scenario_indnr=1)

    return 


def load_period(data, period_set, period):
    """Create a temporary .tab file with the specified period and load it into the DataPortal."""
    df = pd.DataFrame({ 'Period': [period] })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tab", delete=False) as tmpfile:
        df.to_csv(tmpfile.name, sep="\t", index=False)
        tmpname = tmpfile.name
    data.load(filename=tmpname, format="set", set=period_set)
    os.remove(tmpname)
    return 