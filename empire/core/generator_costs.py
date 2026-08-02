"""Generator cost calculations that distinguish emissions from captured CO2."""


def resolve_captured_co2_factor(
    net_co2_factor: float,
    capture_rate: float,
    explicit_captured_co2_factor: float | None = None,
) -> float:
    """Return a non-negative captured-CO2 factor in tCO2/GJ.

    New datasets should provide ``explicit_captured_co2_factor`` directly.
    For backwards compatibility, legacy positive factors are treated as
    residual post-capture emissions, while a negative factor is treated as a
    net removal whose magnitude equals the captured amount.
    """
    if explicit_captured_co2_factor is not None:
        if explicit_captured_co2_factor < 0:
            raise ValueError("Captured CO2 factors must be non-negative.")
        return explicit_captured_co2_factor

    if not 0 <= capture_rate <= 1:
        raise ValueError("The CCS capture rate must be between zero and one.")
    if net_co2_factor < 0:
        return abs(net_co2_factor)
    if net_co2_factor == 0:
        return 0.0
    if capture_rate == 1:
        raise ValueError(
            "A positive residual CO2 factor is inconsistent with a 100% capture rate."
        )

    gross_co2_factor = net_co2_factor / (1 - capture_rate)
    return gross_co2_factor * capture_rate


def co2_intensity_ton_per_mwh(co2_factor: float, efficiency: float) -> float:
    """Convert a tCO2/GJ fuel factor to tCO2/MWh electricity."""
    if efficiency <= 0:
        raise ValueError("Generator efficiency must be positive.")
    return co2_factor * 3.6 / efficiency


def ccs_fixed_cost_eur_per_mw(
    fixed_cost_coefficient: float,
    captured_co2_factor: float,
    efficiency: float,
) -> float:
    """Calculate the capacity-related CCS transport and storage cost."""
    return fixed_cost_coefficient * co2_intensity_ton_per_mwh(
        captured_co2_factor, efficiency
    )


def generator_marginal_cost_eur_per_mwh(
    fuel_cost_eur_per_gj: float,
    variable_om_eur_per_mwh: float,
    efficiency: float,
    net_co2_factor: float,
    co2_price_eur_per_ton: float,
    captured_co2_factor: float = 0.0,
    ccs_variable_cost_eur_per_ton: float = 0.0,
) -> float:
    """Calculate marginal cost using signed net emissions and positive capture."""
    fuel_cost = fuel_cost_eur_per_gj * 3.6 / efficiency
    carbon_cost = (
        co2_intensity_ton_per_mwh(net_co2_factor, efficiency)
        * co2_price_eur_per_ton
    )
    ccs_transport_storage_cost = (
        co2_intensity_ton_per_mwh(captured_co2_factor, efficiency)
        * ccs_variable_cost_eur_per_ton
    )
    return (
        fuel_cost
        + carbon_cost
        + ccs_transport_storage_cost
        + variable_om_eur_per_mwh
    )
