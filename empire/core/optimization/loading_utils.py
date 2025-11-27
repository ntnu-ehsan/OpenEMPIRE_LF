

from pyomo.environ import DataPortal, Param, Set
from pathlib import Path
import pandas as pd
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


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

def _filter_param_by_dims(raw_data: dict, dim_indices: dict) -> dict[tuple, float]:
    """
    Filters raw_data dict of indexed Param values by allowed values on specified dimensions.
    dim_indices: dict {dim_position: allowed_values}
    """
    return {
        idx: val
        for idx, val in raw_data.items()
        if all(idx[pos] in allowed for pos, allowed in dim_indices.items())
    }



def filter_data(
    raw_data: dict[tuple, float],
    periods_to_load: list[int] | None = None,
    period_indnr: int | None = None,
    scenarios_to_load: list[str] | None = None,
    scenario_indnr: int | None = None,
) -> dict[tuple | str | int | float, float]:
    """
    Filters raw_data dict of indexed Param values by allowed values on specified periods and scenarios.
    """
    dim_indices: dict[int, list] = {}
    if periods_to_load is not None and period_indnr is not None:
        dim_indices[period_indnr] = periods_to_load
    if scenarios_to_load is not None and scenario_indnr is not None:
        dim_indices[scenario_indnr] = scenarios_to_load
    return _filter_param_by_dims(
        raw_data,
        dim_indices=dim_indices
    )



def load_dict_into_dataportal(data: DataPortal, param: Param, data_dict: dict[tuple | str | int | float, float]):
    """Load a dictionary of parameter data into a Pyomo DataPortal.
    
    This function converts a dictionary of indexed parameter values into a temporary
    .tab file and loads it into the DataPortal. This is useful when parameter data
    is generated programmatically rather than read from files.

    Args:
        data: Pyomo DataPortal instance to load data into
        param: Pyomo parameter component to load data for
        data_dict: Dictionary with parameter indices as keys and values as data.
                   Keys can be tuples (for multi-indexed params), strings, ints, or floats.
    
    Raises:
        ValueError: If data_dict is empty (no data to load)
        Exception: Re-raises any exception from DataPortal.load() after logging details
    """
    def _return_list(idx):
        b = []
        for i in idx:
            if isinstance(i, tuple):
                b.extend(list(i))
            else:
                b.append(i)
        return b
    

    if not data_dict:
        raise ValueError(f"No data to load for parameter {param.name}")
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
            data.load(filename=tmpname, param=param, format="table")
        except Exception as e:
            logger.error("Error loading parameter %s from temporary file.", param.name)
            logger.error("Dataframe head (max 10 rows):\n%s", df.head(10))
            logger.error("Exception: %s", str(e))
            raise

    os.remove(tmpname)


def load_parameter(
    data: DataPortal,
    tab_file_path: Path,
    param_component: Param,
    periods_to_load: list[int] | None = None,
    period_indnr: int | None = None,
    scenarios_to_load: list[str] | None = None,
    scenario_indnr: int | None = None,
):
    """Load and filter a parameter from a .tab file into a DataPortal.
    
    This function reads parameter data from a tab-separated file and optionally
    filters it by period and/or scenario before loading into the DataPortal.
    
    Args:
        data: Pyomo DataPortal instance
        tab_file_path: Path to the .tab file containing parameter data
        param_component: Pyomo parameter component to load
        periods_to_load: List of period indices to include (None = all periods)
        period_indnr: Position of period index in the parameter tuple (0-based)
        scenarios_to_load: List of scenario names to include (None = all scenarios)
        scenario_indnr: Position of scenario index in the parameter tuple (0-based)
    
    Raises:
        ValueError: If the .tab file contains no data
    
    Note:
        If both periods_to_load and scenarios_to_load are None, all data is loaded.
    """
    raw_data = read_tab_file(tab_file_path)
    if not raw_data:
        raise ValueError(f"No data found in file {tab_file_path} for parameter {param_component.name}")
    if periods_to_load is None and scenarios_to_load is None:
        filtered_data = raw_data
    else:
        filtered_data = filter_data(
            raw_data,
            periods_to_load=periods_to_load,
            period_indnr=period_indnr,
            scenarios_to_load=scenarios_to_load,
            scenario_indnr=scenario_indnr,
        )

    load_dict_into_dataportal(data, param_component, filtered_data)
    return 


def load_set(data: DataPortal, model_set: Set, value: list | int | float | str):
    """Create a temporary .tab file with the specified period and load it into the DataPortal."""
    if isinstance(value, (int, float, str)):
        val = [value]
    elif isinstance(value, list):
        val = value
    else:
        raise ValueError(f"Unsupported type for value: {type(value)}")
    df = pd.Series(val, name="value").to_frame()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tab", delete=False) as tmpfile:
        df.to_csv(tmpfile.name, sep="\t", index=False, header=True)
        tmpname = tmpfile.name
    data.load(filename=tmpname, format="set", set=model_set)
    os.remove(tmpname)
    return 