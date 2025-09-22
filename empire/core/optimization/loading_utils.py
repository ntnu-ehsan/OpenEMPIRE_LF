

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




def load_dict_into_dataportal(data: DataPortal, param, data_dict):
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
        data.load(filename=tmpname, param=param, format="table")
        try:
            data.load(filename=tmpname, param=param, format="table")


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
    try:
        load_dict_into_dataportal(data, param_component, filtered_data)
    except:
        breakpoint()



def load_set(data, model_set, value):
    """Create a temporary .tab file with the specified period and load it into the DataPortal."""
    if isinstance(value, (int, float, str)):
        val = [value]
    else:
        val = value
    df = pd.Series(val, name="value").to_frame()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tab", delete=False) as tmpfile:
        df.to_csv(tmpfile.name, sep="\t", index=False, header=True)
        tmpname = tmpfile.name
    data.load(filename=tmpname, format="set", set=model_set)
    os.remove(tmpname)
    return 