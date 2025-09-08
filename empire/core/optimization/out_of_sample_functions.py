import os 
from pyomo.environ import Param, NonNegativeReals
from empire.utils import get_name_of_last_folder_in_path


def set_investments_as_parameters(model, data):
    # Redefine investment vars as input parameters
    model.genInvCap = Param(model.GeneratorsOfNode, model.periods_active, domain=NonNegativeReals)
    model.transmisionInvCap = Param(model.BidirectionalArc, model.periods_active, domain=NonNegativeReals)
    model.storPWInvCap = Param(model.StoragesOfNode, model.periods_active, domain=NonNegativeReals)
    model.storENInvCap = Param(model.StoragesOfNode, model.periods_active, domain=NonNegativeReals)
    model.genInstalledCap = Param(model.GeneratorsOfNode, model.periods_active, domain=NonNegativeReals)
    model.transmissionInstalledCap = Param(model.BidirectionalArc, model.periods_active, domain=NonNegativeReals)
    model.storPWInstalledCap = Param(model.StoragesOfNode, model.periods_active, domain=NonNegativeReals)
    model.storENInstalledCap = Param(model.StoragesOfNode, model.periods_active, domain=NonNegativeReals)
    return 


def load_optimized_investments(model, data, result_file_path):
    """Optimized investment decisions read from result file from in-sample runs"""
    data.load(filename=str(result_file_path / 'genInvCap.tab'), param=model.genInvCap, format="table")
    data.load(filename=str(result_file_path / 'transmisionInvCap.tab'), param=model.transmisionInvCap, format="table")
    data.load(filename=str(result_file_path / 'storPWInvCap.tab'), param=model.storPWInvCap, format="table")
    data.load(filename=str(result_file_path / 'storENInvCap.tab'), param=model.storENInvCap, format="table")
    data.load(filename=str(result_file_path / 'genInstalledCap.tab'), param=model.genInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'transmissionInstalledCap.tab'), param=model.transmissionInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'storPWInstalledCap.tab'), param=model.storPWInstalledCap, format="table")
    data.load(filename=str(result_file_path / 'storENInstalledCap.tab'), param=model.storENInstalledCap, format="table")
    return 


def set_out_of_sample_path(result_file_path, sample_file_path):
    """Update result_file_path to output for given out_of_sample tree"""
    sample_tree = get_name_of_last_folder_in_path(sample_file_path)
    result_file_path = result_file_path / f"OutOfSample/{sample_tree}"
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    return result_file_path
