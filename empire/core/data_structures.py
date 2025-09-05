from dataclasses import dataclass
from pathlib import Path


@dataclass
class OperationalParams: 
    Operationalhour: int
    Scenario: int
    Season: int
    HoursOfSeason: int
    FirstHoursOfRegSeason: int
    FirstHoursOfPeakSeason: int
    lengthRegSeason: int
    lengthPeakSeason: int


@dataclass 
class Flags:
    print_iamc_flag: bool
    write_lp_flag: bool
    pickle_instance_flag: bool
    emission_cap_flag: bool
    use_temp_dir_flag: bool
    load_change_module_flag: bool
    compute_operational_duals_flag: bool
    north_sea_flag: bool
    out_of_sample_flag: bool
    lopf_flag: bool