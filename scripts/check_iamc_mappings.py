"""
check_iamc_mappings.py
----------------------
Validates that all generator names and node names in a dataset are covered by
the IAMC lookup dicts in empire.py, WITHOUT running the solver.

Usage:
    python scripts/check_iamc_mappings.py              # defaults to north_sea
    python scripts/check_iamc_mappings.py europe_v50
    python scripts/check_iamc_mappings.py test
"""

import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openpyxl

# ── pull the dicts directly from empire.py ────────────────────────────────────
# We import only the dicts, not the heavy Pyomo model.
# Keep these in sync with empire/core/empire.py manually, or refactor them
# into a shared constants module later.

dict_countries = {
    "Austria": "Austria",
    "Bosnia and Herzegovina": "BosniaH",
    "Belgium": "Belgium", "Bulgaria": "Bulgaria",
    "Switzerland": "Switzerland",
    "Czech Republic": "CzechR", "Germany": "Germany",
    "Denmark": "Denmark", "Estonia": "Estonia",
    "Spain": "Spain", "Finland": "Finland",
    "France": "France", "United Kingdom": "GreatBrit.",
    "Greece": "Greece", "Croatia": "Croatia",
    "Hungary": "Hungary", "Ireland": "Ireland",
    "Italy": "Italy", "Lithuania": "Lithuania",
    "Luxembourg": "Luxemb.", "Latvia": "Latvia",
    "North Macedonia": "Macedonia",
    "The Netherlands": "Netherlands", "Norway": "Norway",
    "Poland": "Poland", "Portugal": "Portugal",
    "Romania": "Romania", "Serbia": "Serbia",
    "Sweden": "Sweden", "Slovenia": "Slovenia",
    "Slovakia": "Slovakia", "Norway|Ostland": "NO1",
    "Norway|Sorland": "NO2", "Norway|Norgemidt": "NO3",
    "Norway|Troms": "NO4", "Norway|Vestmidt": "NO5",
}
dict_countries_reversed = {v: k for k, v in dict_countries.items()}

# Fixed node names (spaces introduced by fix_node_names.py differ from original keys)
dict_countries_reversed.update({
    "Bosnia H": "Bosnia and Herzegovina",
    "Czech R": "Czech Republic",
    "Great Brit.": "United Kingdom",
})

dict_generators = {
    "Bio": "Biomass", "Bioexisting": "Biomass",
    "BioCCS": "Biomass|w/ CCS",
    "Coalexisting": "Coal|w/o CCS",
    "Coal": "Coal|w/o CCS", "CoalCCS": "Coal|w/ CCS",
    "CoalCCSadv": "Coal|w/ CCS",
    "Lignite": "Lignite|w/o CCS",
    "Liginiteexisting": "Lignite|w/o CCS",
    "LigniteCCSadv": "Lignite|w/ CCS",
    "LigniteCCS": "Lignite|w/ CCS",
    "Gasexisting": "Gas|CCGT|w/o CCS",
    "GasOCGT": "Gas|OCGT|w/o CCS",
    "GasCCGT": "Gas|CCGT|w/o CCS",
    "GasCCS": "Gas|CCGT|w/ CCS",
    "GasCCSadv": "Gas|CCGT|w/ CCS",
    "Oilexisting": "Oil", "Oil": "Oil",
    "Nuclear": "Nuclear",
    "Wave": "Ocean", "Geo": "Geothermal",
    "Hydroregulated": "Hydro|Reservoir",
    "Hydrorun-of-the-river": "Hydro|Run-of-River",
    "Windonshore": "Wind|Onshore",
    "Windoffshore": "Wind|Offshore",
    "Windoffshoregrounded": "Wind|Offshore",
    "Windoffshorefloating": "Wind|Offshore",
    "Solar": "Solar|PV", "Waste": "Waste",
    "Bio10cofiring": "Coal|w/o CCS",
    "Bio10cofiringCCS": "Coal|w/ CCS",
    "LigniteCCSsup": "Lignite|w/ CCS",
    # Spaced-name variants (north_sea dataset)
    "Bio CCS": "Biomass|w/ CCS",
    "Coal CCS": "Coal|w/ CCS",
    "Gas CCGT": "Gas|CCGT|w/o CCS",
    "Gas CCS": "Gas|CCGT|w/ CCS",
    "Gas OCGT": "Gas|OCGT|w/o CCS",
    "Hydro regulated": "Hydro|Reservoir",
    "Hydro run-of-the-river": "Hydro|Run-of-River",
    "Lignite CCS": "Lignite|w/ CCS",
    "Wind offshore floating": "Wind|Offshore",
    "Wind offshore grounded": "Wind|Offshore",
    "Wind onshore": "Wind|Onshore",
}


def load_generators(sets_path: Path) -> list[str]:
    wb = openpyxl.load_workbook(sets_path, data_only=True, read_only=True)
    for sheet in ("Generators", "Generator"):
        if sheet in wb.sheetnames:
            ws = wb[sheet]
            return [row[0].strip() for row in ws.iter_rows(min_row=2, values_only=True) if row[0]]
    raise KeyError(f"No 'Generators' or 'Generator' sheet in {sets_path}")


def load_nodes(sets_path: Path) -> list[str]:
    wb = openpyxl.load_workbook(sets_path, data_only=True, read_only=True)
    for sheet in ("Nodes", "Node"):
        if sheet in wb.sheetnames:
            ws = wb[sheet]
            return [row[0] for row in ws.iter_rows(min_row=2, values_only=True) if row[0]]
    raise KeyError(f"No 'Nodes' or 'Node' sheet in {sets_path}")


def check(dataset: str) -> bool:
    sets_path = ROOT / "Data handler" / dataset / "Sets.xlsx"
    if not sets_path.exists():
        print(f"ERROR: {sets_path} not found")
        return False

    generators = load_generators(sets_path)
    nodes = load_nodes(sets_path)

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset}")
    print(f"{'='*60}")
    print(f"Generators found : {len(generators)}")
    print(f"Nodes found      : {len(nodes)}")

    # ── check generators ──────────────────────────────────────────────────────
    missing_gen = [g for g in generators if g not in dict_generators]
    if missing_gen:
        print(f"\n[FAIL] {len(missing_gen)} generator(s) missing from dict_generators:")
        for g in missing_gen:
            print(f"       - '{g}'")
    else:
        print(f"\n[OK]   All {len(generators)} generators are in dict_generators.")

    # ── check nodes ───────────────────────────────────────────────────────────
    missing_node = [n for n in nodes if n not in dict_countries_reversed]
    if missing_node:
        print(f"\n[WARN] {len(missing_node)} node(s) not in dict_countries_reversed")
        print(f"       (these will use the node name itself as IAMC region — OK if intentional):")
        for n in missing_node:
            print(f"       - '{n}'")
    else:
        print(f"[OK]   All {len(nodes)} nodes are in dict_countries_reversed.")

    ok = len(missing_gen) == 0
    print(f"\nResult: {'PASS' if ok else 'FAIL'} (nodes with unknown region = {len(missing_node)}, treated as warnings)")
    return ok


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "north_sea"
    passed = check(dataset)
    sys.exit(0 if passed else 1)
