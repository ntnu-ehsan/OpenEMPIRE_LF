"""Fix IAMC output files to comply with IIASA nomenclature rules.

Usage:
    # Split a combined CSV into yearly + subannual files (most common):
    python _fix_iamc_excel.py split path/to/empire_iamc.csv

    # Fix an already-split yearly Excel file:
    python _fix_iamc_excel.py yearly path/to/empire_iamc_yearly.xlsx

    # Fix an already-split subannual Excel file:
    python _fix_iamc_excel.py subannual path/to/empire_subannual.xlsx
"""

import argparse
from pathlib import Path
import re
import shutil

import pandas as pd


EJ_PER_MWH = 3.6e-9
AGGREGATE_CO2_VAR = "Emissions|CO2|Energy|Supply|Electricity"
MODEL_NAME = "EMPIRE v0.1.0/v52"


def _normalize_region_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


NORTH_SEA_REGION_MAP = {
    _normalize_region_key("Borssele"): "EMPIRE v1.0.0/v51|North Sea|Borssele",
    _normalize_region_key("DoggerBank"): "EMPIRE v1.0.0/v51|North Sea|Dogger Bank",
    _normalize_region_key("Dogger Bank"): "EMPIRE v1.0.0/v51|North Sea|Dogger Bank",
    _normalize_region_key("EastAnglia"): "EMPIRE v1.0.0/v51|North Sea|East Anglia",
    _normalize_region_key("East Anglia"): "EMPIRE v1.0.0/v51|North Sea|East Anglia",
    _normalize_region_key("FirthofForth"): "EMPIRE v1.0.0/v51|North Sea|Firth of Forth",
    _normalize_region_key("Firth of Forth"): "EMPIRE v1.0.0/v51|North Sea|Firth of Forth",
    _normalize_region_key("HelgolanderBucht"): "EMPIRE v1.0.0/v51|North Sea|Helgolander Bucht",
    _normalize_region_key("Helgolander Bucht"): "EMPIRE v1.0.0/v51|North Sea|Helgolander Bucht",
    _normalize_region_key("HollandseeKust"): "EMPIRE v1.0.0/v51|North Sea|Hollandsee Kust",
    _normalize_region_key("Hollandsee Kust"): "EMPIRE v1.0.0/v51|North Sea|Hollandsee Kust",
    _normalize_region_key("Hornsea"): "EMPIRE v1.0.0/v51|North Sea|Hornsea",
    _normalize_region_key("MorayFirth"): "EMPIRE v1.0.0/v51|North Sea|Moray Firth",
    _normalize_region_key("Moray Firth"): "EMPIRE v1.0.0/v51|North Sea|Moray Firth",
    _normalize_region_key("Nordsoen"): "EMPIRE v1.0.0/v51|North Sea|Nordsoen",
    _normalize_region_key("Norfolk"): "EMPIRE v1.0.0/v51|North Sea|Norfolk",
    _normalize_region_key("OuterDowsing"): "EMPIRE v1.0.0/v51|North Sea|Outer Dowsing",
    _normalize_region_key("Outer Dowsing"): "EMPIRE v1.0.0/v51|North Sea|Outer Dowsing",
    _normalize_region_key("SorligeNordsjoI"): "EMPIRE v1.0.0/v51|North Sea|Sorlige Nordsjo I",
    _normalize_region_key("Sorlige Nordsjo I"): "EMPIRE v1.0.0/v51|North Sea|Sorlige Nordsjo I",
    _normalize_region_key("SorligeNordsjoII"): "EMPIRE v1.0.0/v51|North Sea|Sorlige Nordsjo II",
    _normalize_region_key("Sorlige Nordsjo II"): "EMPIRE v1.0.0/v51|North Sea|Sorlige Nordsjo II",
    _normalize_region_key("UtsiraNord"): "EMPIRE v1.0.0/v51|North Sea|Utsira Nord",
    _normalize_region_key("Utsira Nord"): "EMPIRE v1.0.0/v51|North Sea|Utsira Nord",
}

NORWAY_REGION_PREFIX_MAP = {
    "Norway|Ostland": "EMPIRE v1.0.0/v51|Norway|Ostland",
    "Norway|Sorland": "EMPIRE v1.0.0/v51|Norway|Sorland",
    "Norway|Norgemidt": "EMPIRE v1.0.0/v51|Norway|Norgemidt",
    "Norway|Troms": "EMPIRE v1.0.0/v51|Norway|Troms",
    "Norway|Vestmidt": "EMPIRE v1.0.0/v51|Norway|Vestmidt",
}

UNIT_MAP = {
    "US$2010/kW": "USD_2010/kW",
    "US$2010/GJ": "USD_2010/GJ",
    "billion US$2010/yr": "billion USD_2010/yr",
}


def _year_columns(df: pd.DataFrame) -> list:
    return [
        c
        for c in df.columns
        if isinstance(c, (int, float)) or (isinstance(c, str) and re.match(r"^\d{4}$", c))
    ]


def _fix_df(df: pd.DataFrame, drop_subannual: bool = False, year_shift: int = 0) -> pd.DataFrame:
    df = df.copy()
    if "variable" not in df.columns:
        return df

    if "model" in df.columns:
        df["model"] = MODEL_NAME

    df["variable"] = df["variable"].astype(str)
    year_cols = _year_columns(df)

    # Active Power -> Secondary Energy with MWh to EJ/yr conversion.
    ap_mask = df["variable"].str.startswith("Active Power|Electricity|", na=False)
    if ap_mask.any():
        df.loc[ap_mask, "variable"] = df.loc[ap_mask, "variable"].str.replace(
            "Active Power|Electricity|", "Secondary Energy|Electricity|", regex=False
        )
        if "unit" in df.columns:
            df.loc[ap_mask, "unit"] = "EJ/yr"
        for yc in year_cols:
            df.loc[ap_mask, yc] = pd.to_numeric(df.loc[ap_mask, yc], errors="coerce").fillna(0.0) * EJ_PER_MWH

    # CO2 typo fix.
    df["variable"] = df["variable"].str.replace(
        "CO2 Emmissions|Electricity|", "Emissions|CO2|Energy|Supply|Electricity|", regex=False
    )

    # Variable naming fixes.
    df["variable"] = df["variable"].str.replace("Run-of-River", "Run of River", regex=False)
    df["variable"] = df["variable"].str.replace(
        "Investment|Energy Supply|Electricity|Electricity storage",
        "Investment|Energy Supply|Electricity|Electricity Storage",
        regex=False,
    )
    df["variable"] = df["variable"].str.replace("Discount rate|Electricity", "Discount Rate|Electricity", regex=False)
    df["variable"] = df["variable"].str.replace("|Electricity|Lignite|", "|Electricity|Coal|Lignite|", regex=False)

    # Remove technology-specific emissions rows, keep aggregate only.
    tech_emissions_mask = df["variable"].str.startswith(AGGREGATE_CO2_VAR + "|")
    if tech_emissions_mask.any():
        df = df.loc[~tech_emissions_mask].copy()

    if "unit" in df.columns:
        df["unit"] = df["unit"].replace(UNIT_MAP)

    # Shift year columns forward by year_shift years.
    if year_shift and year_cols:
        df = df.rename(columns={yc: int(yc) + year_shift for yc in year_cols})

    # Drop subannual column for annual files — ixmp4 only accepts standard IAMC columns.
    if drop_subannual and "subannual" in df.columns:
        df = df.drop(columns=["subannual"])

    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.strip()
        df.loc[df["region"] == "Europe", "region"] = "EU27"
        df.loc[df["region"] == "The Netherlands", "region"] = "Netherlands"
        df.loc[df["region"] == "Czech Republic", "region"] = "Czechia"
        df["region"] = df["region"].replace(NORWAY_REGION_PREFIX_MAP)

        region_keys = df["region"].map(_normalize_region_key)
        for key, prefixed_region in NORTH_SEA_REGION_MAP.items():
            df.loc[region_keys == key, "region"] = prefixed_region

    return df


def _convert_subannual_to_long(df: pd.DataFrame, year_shift: int = 0) -> pd.DataFrame:
    """Convert wide subannual IAMC format to long format with a 'time' datetime column.

    Input:  model, scenario, region, variable, unit, subannual, 2025, 2030, ...
            where subannual contains timestamps like '10-01 00:00+01:00'
    Output: model, scenario, region, variable, unit, time, value
            where time is a full datetime e.g. '2025-10-01 00:00:00+01:00'
    """
    # Fix nomenclature but keep original year columns for melting (shift applied after).
    df = _fix_df(df, drop_subannual=False, year_shift=0)

    meta_cols = ["model", "scenario", "region", "variable", "unit", "subannual"]
    year_cols = _year_columns(df)
    present_meta = [c for c in meta_cols if c in df.columns]

    long = df.melt(id_vars=present_meta, value_vars=year_cols, var_name="year", value_name="value")
    long["year"] = long["year"].astype(int) + year_shift

    # Drop rows that are annual placeholders (subannual == "Year").
    long = long[long["subannual"].astype(str).str.strip() != "Year"].copy()

    # Build full datetime: combine integer year with subannual timestamp string.
    long["time"] = pd.to_datetime(
        long["year"].astype(str) + "-" + long["subannual"].astype(str),
        utc=False,
    )

    long = long.drop(columns=["subannual", "year"])
    long = long[["model", "scenario", "region", "variable", "unit", "time", "value"]]
    long = long.sort_values(["model", "scenario", "region", "variable", "time"]).reset_index(drop=True)
    return long



def _extract_scenario_name(path: Path) -> str | None:
    """Return dataset name from a path containing a 'dataset_<name>' folder, or None.

    Strips the 'dataset_' prefix, removes a trailing '_Europe' suffix, and
    replaces remaining underscores with spaces.
    E.g. 'dataset_Trinity_v1.3.0_Europe' -> 'Trinity v1.3.0'
    """
    for part in path.resolve().parts:
        if part.startswith("dataset_"):
            name = part[len("dataset_"):]
            if name.endswith("_Europe"):
                name = name[: -len("_Europe")]
            return name.replace("_", " ")
    return None


def _rename_scenario_base(df: pd.DataFrame, new_base: str) -> pd.DataFrame:
    """Replace the base scenario name (before the first '|') with new_base."""
    if "scenario" not in df.columns:
        return df
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).apply(
        lambda val: new_base + "|" + val.split("|", 1)[1] if "|" in val else new_base
    )
    return df


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _write_yearly(df: pd.DataFrame, out_path: Path, scenario_name: str | None = None, year_shift: int = 0) -> None:
    fixed = _fix_df(df, drop_subannual=True, year_shift=year_shift)
    if scenario_name:
        fixed = _rename_scenario_base(fixed, scenario_name)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        fixed.to_excel(writer, sheet_name="data", index=False)
    print(f"  Yearly  -> {out_path.name} ({len(fixed)} rows)")


def _write_subannual(df: pd.DataFrame, out_path: Path, scenario_name: str | None = None, year_shift: int = 0) -> None:
    long = _convert_subannual_to_long(df, year_shift=year_shift)
    if scenario_name:
        long = _rename_scenario_base(long, scenario_name)
    long_out = long.copy()
    long_out["time"] = long_out["time"].astype(str)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        long_out.to_excel(writer, sheet_name="data", index=False)
    print(f"  Subannual -> {out_path.name} ({len(long_out)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("type", choices=["split", "yearly", "subannual"],
                        help="'split' reads a combined CSV/xlsx and writes both output files; "
                             "'yearly'/'subannual' fix an already-split file")
    parser.add_argument("input", type=Path, help="Path to the input file (.csv or .xlsx)")
    parser.add_argument("--scenario", default=None,
                        help="Override the scenario base name (default: auto-extracted from the "
                             "dataset_<name> folder in the input path)")
    parser.add_argument("--year-shift", type=int, default=0,
                        help="Shift all years forward by this many years (default: 0)")
    args = parser.parse_args()

    src: Path = args.input
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    scenario_name: str | None = args.scenario or _extract_scenario_name(src)
    year_shift: int = args.year_shift

    if scenario_name:
        print(f"Scenario name: {scenario_name}")
    else:
        print("Warning: could not extract scenario name from path — scenario column unchanged.")
    if year_shift:
        print(f"Year shift: +{year_shift}")

    out_dir = src.parent

    if args.type == "split":
        print(f"Reading {src.name} ...")
        df = _read_input(src)
        print(f"  Total rows: {len(df)}")

        if "subannual" not in df.columns:
            raise ValueError("File has no 'subannual' column — cannot split.")

        yearly_df = df[df["subannual"].astype(str).str.strip() == "Year"].copy()
        subannual_df = df[df["subannual"].astype(str).str.strip() != "Year"].copy()
        print(f"  Yearly rows: {len(yearly_df)}, Subannual rows: {len(subannual_df)}")

        if len(yearly_df):
            _write_yearly(yearly_df, out_dir / "empire_iamc_yearly.xlsx", scenario_name, year_shift)
        if len(subannual_df):
            _write_subannual(subannual_df, out_dir / "empire_subannual_long.xlsx", scenario_name, year_shift)

        print("Done.")

    elif args.type == "yearly":
        df = _read_input(src)
        out_path = src.with_name(src.stem + "_fixed.xlsx")
        _write_yearly(df, out_path, scenario_name, year_shift)
        shutil.copy2(out_path, src.with_suffix(".xlsx"))
        print("Done. Updated file in place.")

    else:  # subannual
        df = _read_input(src)
        out_path = src.with_name(src.stem + "_long.xlsx")
        _write_subannual(df, out_path, scenario_name, year_shift)
        print("Done.")
