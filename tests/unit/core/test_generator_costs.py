import pytest

from empire.core.generator_costs import (
    ccs_fixed_cost_eur_per_mw,
    generator_marginal_cost_eur_per_mwh,
    resolve_captured_co2_factor,
)


def test_legacy_bioccs_factor_resolves_to_positive_captured_co2():
    captured_factor = resolve_captured_co2_factor(
        net_co2_factor=-0.1,
        capture_rate=0.9,
    )

    assert captured_factor == pytest.approx(0.1)


def test_legacy_fossil_ccs_factor_is_treated_as_residual_emissions():
    captured_factor = resolve_captured_co2_factor(
        net_co2_factor=0.01298059776,
        capture_rate=0.9,
    )

    assert captured_factor == pytest.approx(0.11682537984)


def test_explicit_captured_co2_factor_is_used_without_capture_rate_again():
    captured_factor = resolve_captured_co2_factor(
        net_co2_factor=-0.1,
        capture_rate=0.9,
        explicit_captured_co2_factor=0.1,
    )

    assert captured_factor == pytest.approx(0.1)


def test_negative_explicit_captured_co2_factor_is_rejected():
    with pytest.raises(ValueError, match="must be non-negative"):
        resolve_captured_co2_factor(
            net_co2_factor=-0.1,
            capture_rate=0.9,
            explicit_captured_co2_factor=-0.1,
        )


def test_bioccs_transport_storage_costs_are_positive():
    efficiency = 0.2742054895939017
    captured_factor = resolve_captured_co2_factor(-0.1, 0.9)

    fixed_cost = ccs_fixed_cost_eur_per_mw(
        fixed_cost_coefficient=1_149_873.72,
        captured_co2_factor=captured_factor,
        efficiency=efficiency,
    )
    marginal_cost = generator_marginal_cost_eur_per_mwh(
        fuel_cost_eur_per_gj=8.81123090912966,
        variable_om_eur_per_mwh=7.01568,
        efficiency=efficiency,
        net_co2_factor=-0.1,
        co2_price_eur_per_ton=0.0,
        captured_co2_factor=captured_factor,
        ccs_variable_cost_eur_per_ton=13.62876233745015,
    )

    assert fixed_cost == pytest.approx(1_509_650.8090)
    assert marginal_cost == pytest.approx(140.58990482)


def test_carbon_price_uses_complete_signed_net_emission_factor():
    marginal_cost_without_carbon_price = generator_marginal_cost_eur_per_mwh(
        fuel_cost_eur_per_gj=8.0,
        variable_om_eur_per_mwh=5.0,
        efficiency=0.25,
        net_co2_factor=-0.1,
        co2_price_eur_per_ton=0.0,
    )
    marginal_cost_with_carbon_price = generator_marginal_cost_eur_per_mwh(
        fuel_cost_eur_per_gj=8.0,
        variable_om_eur_per_mwh=5.0,
        efficiency=0.25,
        net_co2_factor=-0.1,
        co2_price_eur_per_ton=100.0,
    )

    # -0.1 t/GJ * 3.6 GJ/MWh / 0.25 = -1.44 t/MWh.
    assert marginal_cost_with_carbon_price - marginal_cost_without_carbon_price == pytest.approx(-144.0)
