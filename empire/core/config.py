from inspect import Parameter, signature
from pathlib import Path
from typing import Dict

import yaml


def read_config_file(path: Path) -> Dict:
    with open(path) as file:
        config = yaml.safe_load(file)

    return config


class EmpireConfiguration:
    def __init__(
        self,
        use_temporary_directory: bool,
        temporary_directory: str | Path,
        forecast_horizon_year: int,
        number_of_scenarios: int,
        length_of_regular_season: int,
        discount_rate: float,
        wacc: float,
        optimization_solver: str,
        use_scenario_generation: bool,
        use_fixed_sample: bool,
        load_change_module: bool,
        filter_make: bool,
        filter_use: bool,
        n_cluster: int,
        moment_matching: bool,
        copula_clusters_make: bool,
        copula_clusters_use: bool,
        copulas_to_use: list[str],
        n_tree_compare: int,
        use_emission_cap: bool,
        compute_operational_duals: bool,
        print_in_iamc_format: bool,
        write_in_lp_format: bool,
        serialize_instance: bool,
        north_sea: bool,
        aggregate_offshore_nodes_in_IAMC: bool = False,
        voronoi_sgr_make: bool = False,
        voronoi_sgr_use: bool = False,
        voronoi_mu_percentile: int = 80,
        regular_seasons: list[str] = ["winter", "spring", "summer", "fall"],
        n_peak_seasons: int = 2,
        len_peak_season: int = 24,
        leap_years_investment: int = 5,
        time_format: str = "%d/%m/%Y %H:%M",
        use_ramping: bool = True,
        generation_growth_limit_flag: bool = False,
        generation_growth_limit_rate: float = 0.04,
        biomass_limit_flag: bool = True,
        biomass_limit_factor: float = 1.2,
        biomass_system_limit_factor: float = 1.04,
        biomass_limit_scope: str = "country",
        bioccs_capacity_limit_factor: float = 1.0,
        transmission_availability: float = 1.0,
        lopf_flag: bool = False,
        lopf_method: str = "kirchhoff",
        lopf_kwargs: dict | None = None,
        use_boundary_conditions: bool = False,
        boundary_bound_type: str = "fixed",
        boundary_include_spain: bool = True,
        solver_method: int = 2,
        solver_crossover: int | None = None,
        solver_presolve: int | None = None,
        solver_threads: int | None = None,
        solver_scaleflag: int | None = None,
        solver_numericfocus: int | None = None,
        solver_barhomogeneous: int | None = None,
        **kwargs,
    ):
        """
        Class containing configurations for running Empire simulations.

        :param use_temporary_directory: Specifies whether to use a temporary directory for operations.
        :param temporary_directory: Path to the temporary directory used for certain operations.
        :param forecast_horizon_year: The last strategic (investment) period used in the optimization run. NB! Must correspond with data for version.
        :param number_of_scenarios: The number of scenarios in every investment period.
        :param length_of_regular_season: The number of chronological time steps in a regular season. NB! Must correspond with data for version.
        :param discount_rate: Rate used to discount future cash flows to present value.
        :param wacc: The Weighted Average Cost of Capital (WACC).
        :param optimization_solver: Mathematical solver used for optimization tasks. Options: “Xpress”, “Gurobi”, “CPLEX”.
        :param use_scenario_generation: If true, new operational scenarios will be generated. NB! If false, .tab-files or sampling key must be manually added to the ‘ScenarioData’-folder in the version.
        :param use_fixed_sample: If true, operational scenarios will be generated according to a fixed sampling key located in the ‘Scenario Data’ folder to ensure the same operational scenarios are generated.
        :param load_change_module:
        :param filter_make:
        :param filter_use:
        :param n_cluster:
        :param moment_matching:
        :param n_tree_compare:
        :param use_emission_cap: If true, emissions in every scenario are capped according to the specified cap in ‘General.xlsx’. If false, the CO2-price specified in ‘General.xlsx’ applies.
        :param compute_operational_duals: If true, investment decisions are fixed and resolved to compute operational duals
        :param print_in_iamc_format: OIf true, selected results are printed on the standard IAMC-format in addition to the normal EMPIRE print.
        :param write_in_lp_format: Problem should be written in Linear Programming format.
        :param serialize_instance: Serialize the data structure or model for later use.
        :param use_north_sea: Whether north-sea is modelled or not.
        :param regular_seasons: Regular seasons.
        :param n_peak_seasons:  Peak seasons.
        :param leap_years_investment: Years between investment decisions
        :param time_format: Time format
        :param use_ramping: If true (default), thermal generator ramp-rate constraints are included. Setting it to
            false removes the inter-hour ramping constraints (fewer rows, less temporal coupling for thermal units);
            only do this if ramping is non-binding at your time resolution, as it is a physical modelling assumption.
        :param generation_growth_limit_flag: If true, add a node-level generation growth cap: total generation
            (summed over all technologies) at each node in a period may not exceed
            (1 + leap_years_investment * rate) times the previous period's expected total generation.
            Default false leaves generation growth unconstrained.
        :param generation_growth_limit_rate: Maximum allowed growth per YEAR when
            generation_growth_limit_flag is true, scaled linearly over the years in a period: 0.06 with
            five-year periods allows at most 30% above the previous period. Used only as a fallback -
            the optional 'GenerationGrowthRate' sheet of General.xlsx overrides it per period whenever
            present. Default 0.04, the reference EMPIRE core's default.
        :param biomass_limit_flag: If true (default), cap system-wide annual biomass generation in each period at
            biomass_limit_factor times the biomass availability given in the optional 'BiomassMaxAnnualActivity'
            sheet of Node.xlsx. The constraint is data-driven: datasets without that sheet are unaffected, so the
            default only takes effect where the data exists. Set false to disable it even when the sheet is present.
        :param biomass_limit_factor: Slack multiplier on the supplied biomass availability when biomass_limit_flag
            is true (e.g. 1.2 = allow at most 20% above the reference value). Under scope "both" this is the
            national multiplier only. Default 1.2.
        :param biomass_system_limit_factor: Slack multiplier for the system-wide row under scope "both". Ignored
            by the other scopes. Default 1.04, matching the reference EMPIRE core.
        :param biomass_limit_scope: Spatial scope of the biomass limit. "country" (default) enforces it nationally,
            summing production and availability over the nodes of each country given by Countries/NodesOfCountry,
            so a country disaggregated into NUTS regions is still limited as one country; a node no country covers
            forms its own group. "system" pools every node into one constraint per period. "both" enforces the two
            together - a loose national ceiling (biomass_limit_factor) plus a tight system-wide one
            (biomass_system_limit_factor) - which is how the reference EMPIRE core is run; use it for
            like-for-like comparison runs against that model.
        :param bioccs_capacity_limit_factor: Multiplier on BioCCS capacity within node- and country-level maximum
            built and maximum installed CCS capacity limits. For example, 1.2 lets one MW of BioCCS consume only
            1/1.2 MW of the applicable CCS capacity budget, so an all-BioCCS build may reach 120% of the supplied
            limit. Other CCS generators continue to consume the capacity budget one-for-one. Default 1.0 preserves
            the supplied limits exactly.
        :param transmission_availability: Fraction (0-1) of each line's installed capacity that may be used in any
            operational hour. Values below 1.0 reserve a reliability/operational margin on every line (e.g. 0.8 = 80%
            usable). Default 1.0 leaves the full installed capacity available.
        :param lopf_flag: If true, add linear (DC) optimal power flow constraints to the transmission network.
            Default false keeps the standard transport (net-transfer) model.
        :param lopf_method: LOPF formulation to use when lopf_flag is true. Supported: "kirchhoff"
            (cycle-based DC-OPF; requires line reactance/susceptance data; continuous transmission
            expansion unchanged) and "angle" (bus-angle DC-OPF; switches transmission investment to
            binary block expansion — candidate corridors listed in a 'CandidateTransmission' sheet of
            Transmission.xlsx build one block at most once, all other corridors stay at initial
            capacity; makes the problem a MIP).
        :param lopf_kwargs: Optional dict of LOPF options. Reader option: "reactance_per_km" (Ohm/km) to compute
            line reactance from line length when no lineReactance sheet/.tab is provided. Constraint options are
            forwarded to the formulation (e.g. "reactance_param_name", "reactance_from_susceptance",
            "dc_line_types" = list of transmission types to treat as HVDC/controllable, excluded from KVL).
        :param use_boundary_conditions: If true, constrain the run to the investment results of an
            original aggregated EMPIRE run, read from the dataset's 'BoundaryConditions' folder
            (produced by scripts/extract_boundary_conditions.py). Default false.
        :param boundary_bound_type: "fixed" (default) applies the generation/storage boundaries as
            equalities, "upper" as upper bounds. Transmission corridors are always equalities.
        :param boundary_include_spain: If true (default), Spain's national totals are pinned to the
            original run as well, so only the distribution of capacity *within* Spain stays free -
            this isolates the effect of the NUTS3 spatial split. Set false to let Spain invest
            freely and be governed by the national limits from the MaxInstalledCapacityCountry /
            MaxBuiltCapacityCountry sheets instead; those limits are inequalities and can never
            bind while the Spanish equalities are active. France/Portugal/EU and the border
            corridors stay fixed either way.
        :param solver_method: Gurobi 'Method' parameter (algorithm). 2 = barrier, best for large LPs.
        :param solver_crossover: Gurobi 'Crossover' parameter. 0 skips the crossover tail for faster
            barrier solves (interior-point solution only; duals/prices become approximate). None leaves the solver default.
        :param solver_presolve: Gurobi 'Presolve' parameter. 2 = aggressive. None leaves the solver default.
        :param solver_threads: Gurobi 'Threads' parameter (max threads). Cap at physical core count to avoid
            hyperthread/NUMA contention on multi-socket machines. None leaves the solver default (all logical cores).
        :param solver_scaleflag: Gurobi 'ScaleFlag' parameter (matrix scaling). 0=off, 1/2/3 increasingly aggressive,
            -1=auto. Scaling is internal; results are returned in original units. None leaves the solver default.
        :param solver_numericfocus: Gurobi 'NumericFocus' parameter. 0=auto, 1-3 spend more effort on numerical
            accuracy (recommended when reading interior-point duals with crossover off). None leaves the solver default.
        :param solver_barhomogeneous: Gurobi 'BarHomogeneous' parameter. 1 enables the homogeneous barrier variant,
            more robust on ill-conditioned models. None leaves the solver default (-1, auto).
        """
        # Model parameters
        self.use_temporary_directory = use_temporary_directory
        self.temporary_directory = Path(temporary_directory).absolute()
        self.forecast_horizon_year = forecast_horizon_year
        self.number_of_scenarios = number_of_scenarios
        self.length_of_regular_season = length_of_regular_season
        self.discount_rate = discount_rate
        self.wacc = wacc
        self.optimization_solver = optimization_solver
        self.use_scenario_generation = use_scenario_generation
        self.use_fixed_sample = use_fixed_sample
        self.load_change_module = load_change_module
        self.filter_make = filter_make
        self.filter_use = filter_use
        self.copulas_to_use = copulas_to_use
        self.copula_clusters_make = copula_clusters_make
        self.copula_clusters_use = copula_clusters_use
        self.n_cluster = n_cluster
        self.moment_matching = moment_matching
        self.n_tree_compare = n_tree_compare
        self.use_emission_cap = use_emission_cap
        self.compute_operational_duals = compute_operational_duals
        self.print_in_iamc_format = print_in_iamc_format
        self.write_in_lp_format = write_in_lp_format
        self.serialize_instance = serialize_instance
        self.north_sea = north_sea
        self.aggregate_offshore_nodes_in_IAMC = aggregate_offshore_nodes_in_IAMC
        self.voronoi_sgr_make = voronoi_sgr_make
        self.voronoi_sgr_use = voronoi_sgr_use
        self.voronoi_mu_percentile = voronoi_mu_percentile

        # Optional parameters
        self.regular_seasons = regular_seasons
        self.n_peak_seasons = n_peak_seasons
        self.len_peak_season = len_peak_season
        self.leap_years_investment = leap_years_investment
        self.time_format = time_format
        self.use_ramping = use_ramping
        self.generation_growth_limit_flag = generation_growth_limit_flag
        self.generation_growth_limit_rate = generation_growth_limit_rate
        self.biomass_limit_flag = biomass_limit_flag
        self.biomass_limit_factor = biomass_limit_factor
        self.biomass_system_limit_factor = biomass_system_limit_factor
        self.biomass_limit_scope = biomass_limit_scope
        self.bioccs_capacity_limit_factor = bioccs_capacity_limit_factor
        self.transmission_availability = transmission_availability

        # Linear Optimal Power Flow (DC-OPF) options
        self.lopf_flag = lopf_flag
        self.lopf_method = lopf_method
        self.lopf_kwargs = {} if lopf_kwargs is None else dict(lopf_kwargs)

        # Spanish-case boundary conditions (capacities constrained to an original EMPIRE run;
        # data in the dataset's 'BoundaryConditions' folder, see scripts/extract_boundary_conditions.py)
        self.use_boundary_conditions = use_boundary_conditions
        self.boundary_bound_type = boundary_bound_type
        self.boundary_include_spain = boundary_include_spain

        # Solver (Gurobi) performance options
        self.solver_method = solver_method
        self.solver_crossover = solver_crossover
        self.solver_presolve = solver_presolve
        self.solver_threads = solver_threads
        self.solver_scaleflag = solver_scaleflag
        self.solver_numericfocus = solver_numericfocus
        self.solver_barhomogeneous = solver_barhomogeneous

        # Computed attributes
        self.n_reg_season = len(regular_seasons)
        self.periods = [i + 1 for i in range(int((self.forecast_horizon_year - 2020) / self.leap_years_investment))]
        self.n_periods = len(self.periods)

        # Validate the configuration
        self.validate()

    def validate(self):
        """
        Validates the configuration. Raises an error if the configuration is invalid.
        """
        if not 0 < self.transmission_availability <= 1:
            raise ValueError(
                f"transmission_availability must be in (0, 1], got {self.transmission_availability}."
            )
        if self.generation_growth_limit_rate < 0:
            raise ValueError(
                f"generation_growth_limit_rate must be >= 0, got {self.generation_growth_limit_rate}."
            )
        if self.biomass_limit_factor < 0:
            raise ValueError(
                f"biomass_limit_factor must be >= 0, got {self.biomass_limit_factor}."
            )
        if self.biomass_system_limit_factor < 0:
            raise ValueError(
                f"biomass_system_limit_factor must be >= 0, got {self.biomass_system_limit_factor}."
            )
        if self.biomass_limit_scope not in ("country", "system", "both"):
            raise ValueError(
                'biomass_limit_scope must be "country", "system" or "both", '
                f"got {self.biomass_limit_scope!r}."
            )
        if self.bioccs_capacity_limit_factor <= 0:
            raise ValueError(
                "bioccs_capacity_limit_factor must be > 0, "
                f"got {self.bioccs_capacity_limit_factor}."
            )

    @classmethod
    def from_dict(cls, config: Dict) -> "EmpireConfiguration":
        """
        Constructs EmpireConfiguration object from a dictionary.

        If constructor arguments are missing and they don't have default values,
        they are added with None value to handle earlier versions of the configuration.

        :param config: Dictionary with configurations.
        :returns: An instance of EmpireConfiguration.
        """
        # Get the signature of the __init__ method
        init_signature = signature(cls.__init__)

        # Prepare a dictionary of arguments
        # Set to None if there is no default value
        init_args = {}
        for param_name, param in init_signature.parameters.items():
            if param_name != "self":
                # Check if the parameter has a default value
                if param.default is Parameter.empty:
                    init_args[param_name] = None
                else:
                    init_args[param_name] = param.default

        # Update the dictionary with values from the config
        init_args.update({k: v for k, v in config.items() if k in init_args})

        # Create an instance of the class with the arguments
        return cls(**init_args)
    
    def to_dict(self) -> dict:
        """
        Used for serialization.

        :return: dictionary
        """
        my_dict = self.__dict__
        for k in my_dict:
            if isinstance(my_dict[k], Path):
                my_dict[k] = str(my_dict[k])
                
        return my_dict


class EmpireRunConfiguration:
    def __init__(
        self,
        run_name: str,
        dataset_path: Path | str,
        tab_path: Path | str,
        scenario_data_path: Path | str,
        results_path: Path | str,
        empire_path: Path | str = Path.cwd(),
    ):
        """
        Class containing configurations for running Empire simulations.

        :param run_name: Name of the run
        :param dataset_path: Folder containing the dataset.
        :param tab_path: Folder containing the .tab files.
        :param scenario_data_path: Folder containing the scenario data.
        :param results_path: Folder where the results should reside.
        :param empire_path: Path to empire project, default is current working directory.
        """

        self.run_name = run_name
        self.dataset_path = Path(dataset_path)
        self.tab_file_path = Path(tab_path)
        self.scenario_data_path = Path(scenario_data_path)
        self.results_path = Path(results_path)
        self.empire_path = Path(empire_path)

        # Validate the configuration
        self.validate()

    def validate(self):
        """
        Validates the configuration. Raises an error if the configuration is invalid.
        """
        if not self.empire_path.exists():
            raise ValueError(f"{self.empire_path} does not exists.")

    @classmethod
    def from_dict(cls, config: dict) -> "EmpireRunConfiguration":
        """
        Constructs EmpireRunConfiguration object from a dictionary.

        :param config: Dictionary with configurations.
        :returns: An instance of EmpireRunConfiguration.
        """
        return cls(**config)
