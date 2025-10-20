


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
from empire.logger import get_empire_logger


class TestSubProblem(unittest.TestCase):
    def assertAlmostEqualSignificant(self, a: float, b: float, sig_digits: int, msg: str | None = None, abs_tol: float = 1e-4):
        # Normalize by order of magnitude
        if a <= abs_tol and b <= abs_tol:
            return  # both zero --> pass
        scale = max(abs(a), abs(b))
        self.assertAlmostEqual(a / scale, b / scale, places=sig_digits - 1, msg=msg + "absolute values: " + str(a) + " vs " + str(b))

    def test_benders_equivalence(self, set_initial_capacities=True, test_system_flag=True, force_flag=True) -> None:
        """
        Test Benders decomposition equivalence to full test model run. 
        """
        # Define fixed capacities for testing
        if test_system_flag:
            config_file_str = "config/testrun.yaml"
            dataset = "test"  
        else:
            config_file_str = "config/run.yaml"
            dataset = "europe_v51"
            
        config = read_config_file(Path(config_file_str))
        empire_config = EmpireConfiguration.from_dict(config=config)



        run_path = Path.cwd() / f"Results/basic_run/dataset_{dataset}"

        if (run_path / "Output/results_objective.csv").exists() and not force_flag:
            raise ValueError("There already exists results for this analysis run.")

        run_config = setup_run_paths(version=dataset, empire_config=empire_config, run_path=run_path)
        logger = get_empire_logger(run_config=run_config)


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

        obj_regular, instance = run_empire(
            run_config=deepcopy(run_config),
            empire_config=empire_config,
            periods_active=periods_active,
            operational_input_params=operational_input_params,
        )


        if set_initial_capacities:
            capacity_params = [
                'genInstalledCap', 
                'transmissionInstalledCap',
                'storPWInstalledCap',
                'storENInstalledCap',
                ]

            capacity_param_values: dict[str, dict[tuple, float]] = {
                param: {key: value(val) for key, val in getattr(instance, param).items()}
                for param in capacity_params
            }
        else:
            capacity_param_values = None


        obj_benders, mp_instance = run_benders(
            run_config=deepcopy(run_config),
            empire_config=empire_config,
            periods_active=periods_active,
            operational_input_params=operational_input_params,
            capacity_params_init=capacity_param_values
        )
        if obj_benders is None:
            self.fail("Benders decomposition did not converge.")

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
        print(f"Objective regular: {obj_regular}, objective Benders: {obj_benders}")
        self.assertAlmostEqualSignificant(obj_regular, obj_benders, sig_digits=5, msg="Mismatch in objective value between regular and Benders decomposition.")  

        if False: # for degenerate optima this check does not make sense
            for param_name in ['genInvCap', 'transmissionInvCap', 'storPWInvCap', 'storENInvCap',
                                'genInstalledCap', 'transmissionInstalledCap', 'storPWInstalledCap', 'storENInstalledCap']:
                for idx in getattr(instance, param_name).keys():
                    val_regular = value(getattr(instance, param_name)[idx])
                    val_benders = value(getattr(mp_instance, param_name)[idx])
                    self.assertAlmostEqualSignificant(val_regular, val_benders, sig_digits=3, msg=f"Mismatch in parameter {param_name} at index {idx}")


        return 

if __name__ == "__main__":
    unittest.main()