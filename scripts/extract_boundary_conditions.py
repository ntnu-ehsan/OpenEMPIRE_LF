"""Extract boundary conditions for the Spanish case from an original (aggregated) EMPIRE run.

Reads generation, storage and transmission capacity results from a ``Results/basic_run/dataset_Agg_*``
output folder and writes per-period boundary capacities for the LF_ES_* datasets:

- France, Portugal: taken directly from the original results.
- EU (rest-of-Europe node): sum of the original results over all nodes except Spain, France, Portugal.
- Spain: national values, used to constrain the SUM over the Spanish NUTS3 nodes.
- Transmission corridors: Spain-France, Spain-Portugal (fixed as corridor sums over the ES* border
  lines) and EU-France (sum of France's original corridors excluding France-Spain).

Output: three CSV files in ``<dataset>/BoundaryConditions/`` with integer model periods (1-based).

The script also rescales the border lines' ``InitialCapacity`` in the dataset's Transmission.xlsx
so that each cross-border corridor sum (ES*-France, ES*-Portugal, EU-France) matches the original
corridor trajectory per period. This is required for feasibility: installed capacity can never
drop below initial capacity, and the disaggregated border lines otherwise start far above the
original corridor totals. The rescaling is proportional per line and idempotent.

Usage:
    python scripts/extract_boundary_conditions.py \
        --results "Results/basic_run/dataset_Agg_NECPEssentials" \
        --dataset "Data handler/LF_ES_NECPEssentials"
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

DIRECT_NODES = ["France", "Portugal"]
SPAIN_NODE = "Spain"
EU_NODE = "EU"


def _period_map(periods: pd.Series) -> dict:
    """Map period labels like '2025-2030' to 1-based model period integers (sorted by start year)."""
    labels = sorted(periods.unique(), key=lambda p: int(str(p).split("-")[0]))
    return {label: i + 1 for i, label in enumerate(labels)}


def extract_generation(output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(output_path / "results_output_gen.csv")
    df["Period"] = df["Period"].map(_period_map(df["Period"]))

    parts = []
    for node in DIRECT_NODES + [SPAIN_NODE]:
        sub = df[df["Node"] == node].groupby(["GeneratorType", "Period"], as_index=False)[
            "genInstalledCap_MW"
        ].sum()
        sub.insert(0, "Node", node)
        parts.append(sub)

    eu = df[~df["Node"].isin(DIRECT_NODES + [SPAIN_NODE])].groupby(
        ["GeneratorType", "Period"], as_index=False
    )["genInstalledCap_MW"].sum()
    eu.insert(0, "Node", EU_NODE)
    parts.append(eu)

    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"genInstalledCap_MW": "InstalledCap_MW"})


def extract_storage(output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(output_path / "results_output_stor.csv")
    df["Period"] = df["Period"].map(_period_map(df["Period"]))

    parts = []
    for node in DIRECT_NODES + [SPAIN_NODE]:
        sub = df[df["Node"] == node].groupby(["StorageType", "Period"], as_index=False)[
            ["storPWInstalledCap_MW", "storENInstalledCap_MWh"]
        ].sum()
        sub.insert(0, "Node", node)
        parts.append(sub)

    eu = df[~df["Node"].isin(DIRECT_NODES + [SPAIN_NODE])].groupby(
        ["StorageType", "Period"], as_index=False
    )[["storPWInstalledCap_MW", "storENInstalledCap_MWh"]].sum()
    eu.insert(0, "Node", EU_NODE)
    parts.append(eu)

    out = pd.concat(parts, ignore_index=True)
    return out.rename(
        columns={
            "storPWInstalledCap_MW": "PWInstalledCap_MW",
            "storENInstalledCap_MWh": "ENInstalledCap_MWh",
        }
    )


def extract_transmission(output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(output_path / "results_output_transmision.csv")
    df["Period"] = df["Period"].map(_period_map(df["Period"]))

    def corridor_sum(mask: pd.Series, from_node: str, to_node: str) -> pd.DataFrame:
        sub = df[mask].groupby("Period", as_index=False)["transmissionInstalledCap_MW"].sum()
        sub.insert(0, "FromNode", from_node)
        sub.insert(1, "ToNode", to_node)
        return sub

    is_spain_france = ((df["BetweenNode"] == SPAIN_NODE) & (df["AndNode"] == "France")) | (
        (df["BetweenNode"] == "France") & (df["AndNode"] == SPAIN_NODE)
    )
    is_spain_portugal = ((df["BetweenNode"] == SPAIN_NODE) & (df["AndNode"] == "Portugal")) | (
        (df["BetweenNode"] == "Portugal") & (df["AndNode"] == SPAIN_NODE)
    )
    # EU-France: everything connected to France in the original run except the Spanish border
    is_eu_france = ((df["BetweenNode"] == "France") | (df["AndNode"] == "France")) & ~is_spain_france

    out = pd.concat(
        [
            corridor_sum(is_spain_france, SPAIN_NODE, "France"),
            corridor_sum(is_spain_portugal, SPAIN_NODE, "Portugal"),
            corridor_sum(is_eu_france, EU_NODE, "France"),
        ],
        ignore_index=True,
    )
    return out.rename(columns={"transmissionInstalledCap_MW": "InstalledCap_MW"})


def rescale_border_initial_capacity(dataset_path: Path, trans: pd.DataFrame) -> None:
    """Rescale border-line InitialCapacity so each corridor sum matches the original trajectory."""
    from empire.input_client.client import EmpireInputClient

    client = EmpireInputClient(dataset_path=dataset_path)
    ic = client.transmission.get_initial_capacity()
    line_col, to_col, period_col, cap_col = ic.columns[:4]

    def corridor_mask(from_node: str, to_node: str) -> pd.Series:
        def side(col, endpoint):
            s = ic[col].astype(str)
            return s.str.startswith(SPAIN_PREFIX) if endpoint == SPAIN_NODE else s.eq(endpoint)

        return (side(line_col, from_node) & side(to_col, to_node)) | (
            side(line_col, to_node) & side(to_col, from_node)
        )

    for (a, b), corridor in trans.groupby(["FromNode", "ToNode"]):
        mask = corridor_mask(a, b)
        if not mask.any():
            print(f"  WARNING: no InitialCapacity rows for corridor {a}-{b}; skipped rescaling.")
            continue
        for row in corridor.itertuples():
            pmask = mask & (ic[period_col].astype(int) == int(row.Period))
            current = ic.loc[pmask, cap_col].sum()
            if current <= 0:
                if row.InstalledCap_MW > 0:
                    # Distribute equally over the corridor's lines in this period
                    n_rows = int(pmask.sum())
                    if n_rows:
                        ic.loc[pmask, cap_col] = row.InstalledCap_MW / n_rows
                    else:
                        print(f"  WARNING: corridor {a}-{b} has no rows for period {row.Period}.")
                continue
            ic.loc[pmask, cap_col] *= row.InstalledCap_MW / current

    client.transmission.set_initial_capacity(ic)
    print("  Rescaled border-line InitialCapacity in Transmission.xlsx to original corridor sums.")


SPAIN_PREFIX = "ES"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the original run results folder, e.g. 'Results/basic_run/dataset_Agg_NECPEssentials'",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the LF_ES dataset folder, e.g. 'Data handler/LF_ES_NECPEssentials'",
    )
    args = parser.parse_args()

    output_path = Path(args.results) / "Output"
    if not output_path.is_dir():
        raise FileNotFoundError(f"No 'Output' folder in {args.results}")

    boundary_path = Path(args.dataset) / "BoundaryConditions"
    boundary_path.mkdir(parents=True, exist_ok=True)

    gen = extract_generation(output_path)
    stor = extract_storage(output_path)
    trans = extract_transmission(output_path)

    gen.to_csv(boundary_path / "boundary_generation.csv", index=False)
    stor.to_csv(boundary_path / "boundary_storage.csv", index=False)
    trans.to_csv(boundary_path / "boundary_transmission.csv", index=False)

    print(f"Wrote boundary conditions to {boundary_path}:")
    print(f"  boundary_generation.csv   ({len(gen)} rows, nodes: {sorted(gen['Node'].unique())})")
    print(f"  boundary_storage.csv      ({len(stor)} rows)")
    print(f"  boundary_transmission.csv ({len(trans)} rows, corridors: "
          f"{sorted(set(zip(trans['FromNode'], trans['ToNode'])))})")

    rescale_border_initial_capacity(Path(args.dataset), trans)


if __name__ == "__main__":
    main()
