from pathlib import Path

import pytest

from empire.core.config import EmpireConfiguration
from empire.core.empire import _capacity_limit_contribution


def test_complete_configuration():
    config = {
        "use_temporary_directory": True,
        "temporary_directory": "/tmp",
        "forecast_horizon_year": 2025,
        "number_of_scenarios": 10,
    }
    empire_config = EmpireConfiguration.from_dict(config)
    assert empire_config.use_temporary_directory
    assert empire_config.temporary_directory == Path("/tmp")


def test_incomplete_configuration():
    config = {
        # Only include some parameters
        "use_temporary_directory": True,
        "temporary_directory": "/tmp",
        "forecast_horizon_year": 2025,
        "number_of_scenarios": 10,
    }
    empire_config = EmpireConfiguration.from_dict(config)
    assert empire_config.use_temporary_directory
    assert empire_config.temporary_directory == Path("/tmp")
    # Assert that missing parameters are None or their default values
    assert empire_config.wacc is None
    assert empire_config.regular_seasons == ["winter", "spring", "summer", "fall"]
    assert empire_config.bioccs_capacity_limit_factor == 1.0


def test_bioccs_capacity_limit_factor_can_be_configured():
    config = {
        "use_temporary_directory": True,
        "temporary_directory": "/tmp",
        "forecast_horizon_year": 2025,
        "number_of_scenarios": 10,
        "bioccs_capacity_limit_factor": 1.2,
    }

    empire_config = EmpireConfiguration.from_dict(config)

    assert empire_config.bioccs_capacity_limit_factor == 1.2


def test_bioccs_capacity_limit_factor_must_be_positive():
    config = {
        "use_temporary_directory": True,
        "temporary_directory": "/tmp",
        "forecast_horizon_year": 2025,
        "number_of_scenarios": 10,
        "bioccs_capacity_limit_factor": 0,
    }

    with pytest.raises(ValueError, match="bioccs_capacity_limit_factor must be > 0"):
        EmpireConfiguration.from_dict(config)


def test_bioccs_capacity_consumes_discounted_capacity_budget():
    assert _capacity_limit_contribution(120, "BioCCS", 1.2) == pytest.approx(100)
    assert _capacity_limit_contribution(120, "CoalCCS", 1.2) == 120
