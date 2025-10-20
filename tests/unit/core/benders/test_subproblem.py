from pathlib import Path
import unittest
from pyomo.environ import value
import json 
import pandas as pd
from copy import deepcopy

from empire.core.benders.subproblem import exe_subproblem_routine
from empire.core.model_runner import define_operational_input_params
from empire.core.config import EmpireConfiguration
from empire.core.model_runner import setup_run_paths
from empire.core.config import read_config_file
from empire.core.reader import generate_tab_files
from empire.core.scenario_random import generate_random_scenario
from empire.core.optimization.empire import run_empire
from empire.core.optimization.loading_utils import filter_data
from empire.core.optimization.objective import SCALING_FACTOR

class TestSubProblem(unittest.TestCase):
    def test_subproblem_equivalence(self):
        """
        Test solving a simple subproblem with fixed capacities.
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

        # Define a single scenario for simplicity
        scenario = operational_input_params.scenarios[1]
        period_active = periods_active[1]

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


        obj, instance = run_empire(
            run_config=deepcopy(run_config),
            empire_config=empire_config,
            periods_active=periods_active,
            operational_input_params=operational_input_params,
            out_of_sample_flag=True,
            sample_file_path=Path("Results/basic_run/dataset_test/Input/Tab")
        )
        # breakpoint()
        capacity_params = [
            'genInstalledCap',
            'transmissionInstalledCap',
            'storENInstalledCap',
            'storPWInstalledCap',
            ]

        capacity_param_values = {
            param_name: {key: value for key, value in getattr(instance, param_name).items()}
            for param_name in capacity_params
        }

        
        # run_config = setup_run_paths(version='test', empire_config=empire_config, run_path=run_path)
        sp_instance, opt = exe_subproblem_routine(
            capacity_param_values,
            period_active,
            scenario,
            empire_config,
            run_config,
            operational_input_params,
            )
        
        period_scenario_indices = {
            'genEfficiency': {'period': 1},
            'sload': {'period': 0, 'scenario': 1}
        }

        for param_name, dim_indices in period_scenario_indices.items():
            filtered_values = {}
            for inst_idx, inst in enumerate([instance, sp_instance]):
                values = {idx: value(param) for idx, param in getattr(inst, param_name).items()}
                filtered_values[inst_idx] = filter_data(
                    values,
                    periods_to_load=[period_active],
                    period_indnr=dim_indices.get('period', None),
                    scenarios_to_load=[scenario],
                    scenario_indnr=dim_indices.get('scenario', None)
                )

            if not filtered_values[0]:
                self.fail(f"Filtered data for parameter {param_name} in main instance is empty.")
            for idx in filtered_values[0]:
                self.assertIn(idx, filtered_values[1], msg=f"Index {idx} missing in subproblem for parameter {param_name}")
                self.assertAlmostEqual(filtered_values[0][idx], filtered_values[1][idx], places=2, msg=f"Mismatch in parameter {param_name} at index {idx}")


        operational_costs = pd.Series({
            (i, w): SCALING_FACTOR * instance.discount_multiplier[i] * instance.operationalcost[i, w] for i in periods_active for w in operational_input_params.scenarios
        })

        specific_operational_cost = operational_costs.loc[(period_active, scenario)]
        if not capacity_param_values:
            self.fail("Filtered data is empty.")
        # for capacity_var, capacity_dict in filtered_params.items():
        #     for idx, cap in capacity_dict.items():
        #         self.assertAlmostEqual(installed_caps[idx], cap, places=2)


        self.assertAlmostEqual(value(sp_instance.Obj), value(specific_operational_cost), places=1)  
        

if __name__ == "__main__":
    unittest.main()