

from pyomo.environ import DataPortal, Param, Set
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
    """
    Loads a dictionary of data into a DataPortal for a specific parameter.

    Args:
        data (DataPortal): The DataPortal instance to load data into.
        param (Param): The parameter to load data for.
        data_dict (dict[tuple | str | int | float, float]): The data to load, with keys as indices and values as the data.
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
            print(f"Error loading parameter {param.name} from temporary file.")
            print(f"Dataframe: {df}")
            print(e)

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
    """
    Loads a parameter for an abstract model.
    Only loads entries for the specified periods and scenarios.
    If no periods or scenarios are specified (periods_to_load is None and scenarios_to_load is None), loads all data.
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