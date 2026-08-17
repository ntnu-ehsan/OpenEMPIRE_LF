"""Generator cost calculations that distinguish emissions from captured CO2."""


def resolve_captured_co2_factor(
    net_co2_factor: float,
    capture_rate: float,
    explicit_captured_co2_factor: float | None = None,
    gross_co2_factor: float | None = None,
) -> float:
    """Return a non-negative captured-CO2 factor in tCO2/GJ.

    The captured amount is the CO2 that enters the plant with the fuel but never
    reaches the atmosphere, i.e. ``gross - net``. Three sources are used, in
    order of decreasing accuracy:

    1. ``explicit_captured_co2_factor`` -- supplied directly by the dataset.
    2. ``gross_co2_factor`` -- the fuel's CO2 content taken from the generator's
       non-CCS counterpart. This needs no assumption about the capture rate, so
       it stays exact when a plant's real rate differs from ``capture_rate``.
    3. ``capture_rate`` -- last resort, assuming the net factor is exactly the
       residual left after capturing that fraction of the gross.

    A negative net factor is a net removal (biogenic CO2 stored underground);
    with no gross figure available its magnitude is the captured amount.
    """
    if explicit_captured_co2_factor is not None:
        if explicit_captured_co2_factor < 0:
            raise ValueError("Captured CO2 factors must be non-negative.")
        return explicit_captured_co2_factor

    if gross_co2_factor is not None:
        captured = gross_co2_factor - net_co2_factor
        if captured < 0:
            raise ValueError(
                "The non-CCS counterpart's CO2 factor is below the CCS generator's, "
                "which would imply negative capture."
            )
        return captured

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

    return (net_co2_factor / (1 - capture_rate)) * capture_rate


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
