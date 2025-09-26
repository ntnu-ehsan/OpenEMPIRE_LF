import os 
from pathlib import Path
from pyomo.environ import Param, NonNegativeReals
from empire.utils import get_name_of_last_folder_in_path


def set_investments_as_parameters(model, set_only_capacities: bool = False):
    # Redefine investment vars as input parameters
    model.genInstalledCap = Param(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmissionInstalledCap = Param(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInstalledCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    if set_only_capacities: 
        return 
    model.genInvCap = Param(model.GeneratorsOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.transmissionInvCap = Param(model.BidirectionalArc, model.PeriodActive, domain=NonNegativeReals)
    model.storPWInvCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    model.storENInvCap = Param(model.StoragesOfNode, model.PeriodActive, domain=NonNegativeReals)
    return 


def load_optimized_investments(model, data, result_file_path, set_only_capacities: bool = False):
    """Optimized investment decisions read from result file from in-sample runs"""
    data.load(filename=str(result_file_path / 'genInstalledCap.tab'), param=model.genInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'transmissionInstalledCap.tab'), param=model.transmissionInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'storPWInstalledCap.tab'), param=model.storPWInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'storENInstalledCap.tab'), param=model.storENInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'genInvCap.tab'), param=model.genInvCap, format="table")
    data.load(filename=str(result_file_path / 'transmissionInvCap.tab'), param=model.transmissionInvCap, format="table")
    data.load(filename=str(result_file_path / 'storPWInvCap.tab'), param=model.storPWInvCap, format="table")
    data.load(filename=str(result_file_path / 'storENInvCap.tab'), param=model.storENInvCap, format="table")
    return 


def set_out_of_sample_path(result_file_path, sample_file_path) -> Path:
    """Update result_file_path to output for given out_of_sample tree"""
    sample_tree = get_name_of_last_folder_in_path(sample_file_path)
    result_file_path = result_file_path / f"OutOfSample/{sample_tree}"
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    return result_file_path
