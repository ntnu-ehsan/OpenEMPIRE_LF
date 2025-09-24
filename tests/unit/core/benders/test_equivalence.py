


from pathlib import Path
import unittest
from pyomo.environ import value
import json 
import pickle 
import pandas as pd
from copy import deepcopy

from empire.core.benders.master_problem import define_initial_capacity_params
from empire.core.benders.subproblem import exe_subproblem_routine
from empire.core.model_runner import define_operational_input_params
from empire.core.config import EmpireConfiguration
from empire.core.model_runner import setup_run_paths
from empire.core.config import read_config_file
from empire.core.reader import generate_tab_files
from empire.core.scenario_random import generate_random_scenario
from empire.core.optimization.empire import run_empire
from empire.core.benders.algorithm import run_benders
from empire.core.optimization.loading_utils import read_tab_file, filter_data

class TestSubProblem(unittest.TestCase):
    def test_benders_equivalence(self, set_initial_capacities=True):
        """
        Test Benders decomposition equivalence to full test model run. 
        """
        # Define fixed capacities for testing
        config = read_config_file(Path("config/testrun.yaml"))
        empire_config = EmpireConfiguration.from_dict(config=config)

        run_path = Path.cwd() / "Results/basic_run/dataset_test"
        run_config = setup_run_paths(version='test', empire_config=empire_config, run_path=run_path)

        # Define operational input parameters
        operational_input_params = define_operational_input_params(empire_config)

        # Define active periods for testing
        periods_active = [1, 2]  

        generate_tab_files(file_path=run_config.dataset_path, tab_file_path=run_config.tab_file_path)

        with open(run_config.empire_path / "config/countries.json", "r", encoding="utf-8") as file:
            dict_countries = json.load(file)

        if empire_config.use_scenario_generation:
            if empire_config.use_fixed_sample and not (run_config.scenario_data_path / "sampling_key.csv").exists():
                raise ValueError("Missing 'sampling_key.csv' in ScenarioData folder.")
            else:
                generate_random_scenario(
                    empire_config=empire_config,
                    dict_countries=dict_countries,
                    scenario_data_path=run_config.scenario_data_path,
                    tab_file_path=run_config.tab_file_path,
                )

        empire_config.benders_flag = False
        obj_regular, instance = run_empire(
            run_config=deepcopy(run_config),
            empire_config=empire_config,
            periods_active=periods_active,
            operational_input_params=operational_input_params,
        )

        if set_initial_capacities:
            capacity_params = ['genInstalledCap']#, 'transmissionInstalledCap']

            capacity_param_values: dict[str, dict[tuple, float]] = {
                param: {key: value(val) for key, val in getattr(instance, param).items()}
                for param in capacity_params
            }
        else:
            capacity_param_values = None

        empire_config.benders_flag = True
        obj_benders, mp_instance = run_benders(
            run_config=deepcopy(run_config),
            empire_config=empire_config,
            periods_active=periods_active,
            operational_input_params=operational_input_params,
            capacity_params_init=capacity_param_values
        )

        params_to_check = [
            'WACC',
            'genCapitalCost',
            'transmissionTypeCapitalCost',
            'storPWCapitalCost',
            'storENCapitalCost',
            'genFixedOMCost',
            'transmissionTypeFixedOMCost',
            'storPWFixedOMCost',
            'storENFixedOMCost',
            'genInvCost',
            'transmissionInvCost',
            'storPWInvCost',
            'storENInvCost',
            'transmissionLength',
            'genRefInitCap',
            'genScaleInitCap',
            'genInitCap',
            'transmissionInitCap',
            'storPWInitCap',
            'storENInitCap',
            'genMaxBuiltCap',
            'transmissionMaxBuiltCap',
            'storPWMaxBuiltCap',
            'storENMaxBuiltCap',
            'genMaxInstalledCapRaw',
            'genMaxInstalledCap',
            'transmissionMaxInstalledCapRaw',
            'transmissionMaxInstalledCap',
            'storPWMaxInstalledCap',
            'storENMaxInstalledCap',
            'storPWMaxInstalledCapRaw',
            'storENMaxInstalledCapRaw',
            'genLifetime',
            'transmissionLifetime',
            'storageLifetime',
        ]

        for param_name in params_to_check:
            for idx in getattr(instance, param_name).keys():
                val_regular = value(getattr(instance, param_name)[idx])
                val_benders = value(getattr(mp_instance, param_name)[idx])
                self.assertAlmostEqual(val_regular, val_benders, places=2, msg=f"Mismatch in parameter {param_name} at index {idx}")    



        for param_name in ['genInvCap', 'transmisionInvCap', 'storPWInvCap', 'storENInvCap',
                            'genInstalledCap', 'transmissionInstalledCap', 'storPWInstalledCap', 'storENInstalledCap']:
            for idx in getattr(instance, param_name).keys():
                val_regular = value(getattr(instance, param_name)[idx])
                val_benders = value(getattr(mp_instance, param_name)[idx])
                self.assertAlmostEqual(val_regular, val_benders, places=2, msg=f"Mismatch in parameter {param_name} at index {idx}")

        self.assertAlmostEqual(obj_regular, obj_benders, places=1)  
        return 

if __name__ == "__main__":
    unittest.main()