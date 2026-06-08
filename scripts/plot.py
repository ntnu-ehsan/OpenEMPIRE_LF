from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from shapely.geometry import shape


DEFAULT_RESULTS_PATH = Path("Results/basic_run/dataset_north_sea")
DEFAULT_COORDS_PATH = Path("Data handler/north_sea/Sets.xlsx")
DEFAULT_OUTPUT = Path("Results/basic_run/dataset_north_sea/Plots/gen_installed_cap_last_period_nodes.png")
DEFAULT_BASEMAP = Path("scripts/NUTS_RG_60M_2024_3035.geojson")
GEOJSON_SOURCE_CRS = ccrs.epsg(3035)
# If True, offshore-node capacities are aggregated into the closest onshore node before plotting.
AGGREGATE_OFFSHORE_TO_CLOSEST_ONSHORE = True

# Order controls both pie-slice sequence and legend order.
TECH_ORDER = [
    "Lignite",
    "Hcoal",
    "Coal",
    "Gas",
    "Oil",
    "Bio",
    "Geo",
    "Nuclear",
    "Hydro",
    "Wave",
    "Wind",
    "Solar",
    "Waste",
]

TECH_COLORS = {
    "Lignite": "#7f7f7f",
    "Hcoal": "#4f4f4f",
    "Coal": "#9a9a9a",
    "Gas": "#d62728",
    "Oil": "#8c564b",
    "Bio": "#2ca02c",
    "Geo": "#ff9896",
    "Nuclear": "#e377c2",
    "Hydro": "#17becf",
    "Wave": "#1f9ed4",
    "Wind": "#1f77b4",
    "Solar": "#ffbb00",
    "Waste": "#bcbd22",
}

CHANGE_GROUPS = {
    "renewable_generation": ["Bio", "Geo", "Hydro", "Wave", "Wind", "Solar", "Waste"],
    "thermal_generation_including_nuclear": ["Lignite", "Hcoal", "Coal", "Gas", "Oil", "Nuclear"],
    "coal": ["Lignite", "Hcoal", "Coal"],
    "wind_and_solar": ["Wind", "Solar"],
}


def normalize_name(value: str) -> str:
    """Normalize labels so node names from different sources can be matched."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def load_coords(coords_file: Path) -> pd.DataFrame:
    """Load node coordinates from Sets.xlsx/Coords (after metadata rows)."""
    coords = pd.read_excel(coords_file, sheet_name="Coords", skiprows=2, usecols=[0, 1, 2])
    coords.columns = ["Location", "Latitude", "Longitude"]
    coords = coords.dropna(subset=["Location", "Latitude", "Longitude"]).copy()
    coords["norm_node"] = coords["Location"].map(normalize_name)
    return coords


def load_offshore_nodes(results_path: Path) -> set[str]:
    """Read offshore node names from input tab and return normalized labels."""
    offshore_file = results_path / "Input" / "Tab" / "Sets_OffshoreNode.tab"
    if not offshore_file.exists():
        return set()

    offshore = pd.read_csv(offshore_file, sep="\t", header=None, names=["Node"])
    offshore["Node"] = offshore["Node"].astype(str).str.strip()
    offshore = offshore[offshore["Node"] != ""].copy()
    offshore["norm_node"] = offshore["Node"].map(normalize_name)
    offshore = offshore[offshore["norm_node"] != "offshorenode"]
    return set(offshore["norm_node"].tolist())


def build_offshore_to_onshore_map(coords: pd.DataFrame, offshore_norm_nodes: set[str]) -> dict[str, str]:
    """Map each offshore node to the nearest onshore node using coordinate distance."""
    if not offshore_norm_nodes:
        return {}

    coord_lookup = coords[["norm_node", "Latitude", "Longitude"]].drop_duplicates("norm_node")
    onshore_coords = coord_lookup.loc[~coord_lookup["norm_node"].isin(offshore_norm_nodes)].copy()
    offshore_coords = coord_lookup.loc[coord_lookup["norm_node"].isin(offshore_norm_nodes)].copy()
    if onshore_coords.empty or offshore_coords.empty:
        return {}

    nearest_map: dict[str, str] = {}
    for _, row in offshore_coords.iterrows():
        d2 = (onshore_coords["Latitude"] - row["Latitude"]) ** 2 + (onshore_coords["Longitude"] - row["Longitude"]) ** 2
        nearest_idx = d2.idxmin()
        nearest_map[str(row["norm_node"])] = str(onshore_coords.loc[nearest_idx, "norm_node"])
    return nearest_map


def aggregate_offshore_to_nearest_onshore(
    capacity: pd.DataFrame,
    coords: pd.DataFrame,
    offshore_norm_nodes: set[str],
    offshore_to_onshore: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate offshore-node capacities to the closest onshore node (by lat/lon distance).
    Nodes not identified as offshore are kept unchanged.
    """
    if not offshore_norm_nodes:
        return capacity

    coord_lookup = coords[["norm_node", "Location", "Latitude", "Longitude"]].drop_duplicates("norm_node")
    if offshore_to_onshore is None:
        offshore_to_onshore = build_offshore_to_onshore_map(coords, offshore_norm_nodes)
    if not offshore_to_onshore:
        return capacity

    cap = capacity.copy()
    cap["target_norm_node"] = cap["norm_node"].map(offshore_to_onshore).fillna(cap["norm_node"])
    grouped = cap.groupby("target_norm_node", as_index=False)[TECH_ORDER].sum()
    grouped = grouped.rename(columns={"target_norm_node": "norm_node"})
    grouped["total_cap"] = grouped[TECH_ORDER].sum(axis=1)

    name_map = coord_lookup.set_index("norm_node")["Location"].to_dict()
    grouped["Node"] = grouped["norm_node"].map(name_map).fillna(grouped["norm_node"])
    return grouped[["Node", *TECH_ORDER, "total_cap", "norm_node"]]


def load_transmission_by_period(results_path: Path) -> tuple[dict[int, pd.DataFrame], list[int]]:
    """
    Load installed transmission capacity per period.
    Returns undirected links (deduplicated by node pair) with normalized node ids.
    """
    line_file = results_path / "Output" / "transmissionInstalledCap.tab"
    lines = pd.read_csv(line_file, sep="\t")
    required = {"FromNode", "ToNode", "Period", "transmissionInstalledCap"}
    missing = required - set(lines.columns)
    if missing:
        raise ValueError(f"Missing columns in {line_file}: {sorted(missing)}")

    lines["Period"] = pd.to_numeric(lines["Period"], errors="coerce")
    lines["transmissionInstalledCap"] = pd.to_numeric(lines["transmissionInstalledCap"], errors="coerce").fillna(0.0)
    lines = lines.dropna(subset=["FromNode", "ToNode", "Period"]).copy()
    lines["Period"] = lines["Period"].astype(int)
    lines["from_norm"] = lines["FromNode"].map(normalize_name)
    lines["to_norm"] = lines["ToNode"].map(normalize_name)

    # Canonical undirected key to avoid plotting both A->B and B->A.
    lines["u"] = np.where(lines["from_norm"] <= lines["to_norm"], lines["from_norm"], lines["to_norm"])
    lines["v"] = np.where(lines["from_norm"] <= lines["to_norm"], lines["to_norm"], lines["from_norm"])
    lines = lines.groupby(["Period", "u", "v"], as_index=False)["transmissionInstalledCap"].max()

    periods = sorted(lines["Period"].unique().tolist())
    by_period: dict[int, pd.DataFrame] = {
        int(p): lines.loc[lines["Period"] == p, ["u", "v", "transmissionInstalledCap"]].copy()
        for p in periods
    }
    return by_period, [int(p) for p in periods]


def map_generator_to_group(generator: str) -> str | None:
    """Map detailed model generator names into user-requested plotting groups."""
    g = str(generator).strip().lower()
    if "lignite" in g:
        return "Lignite"
    if "hardcoal" in g or g == "hcoal" or "hcoal" in g:
        return "Hcoal"
    if "coal" in g:
        return "Coal"
    if "gas" in g:
        return "Gas"
    if "oil" in g:
        return "Oil"
    if "bio" in g:
        return "Bio"
    if "geo" in g:
        return "Geo"
    if "nuclear" in g:
        return "Nuclear"
    if "hydro" in g:
        return "Hydro"
    if "wave" in g:
        return "Wave"
    if "wind" in g:
        return "Wind"
    if "solar" in g:
        return "Solar"
    if "waste" in g:
        return "Waste"
    return None


def parse_period_end_year(value: str) -> int:
    years = re.findall(r"\d{4}", str(value))
    if years:
        return int(years[-1])
    return -1


def load_capacity_by_period(results_path: Path) -> tuple[dict[int, pd.DataFrame], list[int]]:
    """
    Load installed capacity by node/technology and aggregate for all periods.
    Returns:
    - dict keyed by period with one row per node and grouped technology columns
    - sorted list of available periods
    """
    gen_file = results_path / "Output" / "genInstalledCap.tab"
    gen = pd.read_csv(gen_file, sep="\t")
    required = {"Node", "Generator", "Period", "genInstalledCap"}
    missing = required - set(gen.columns)
    if missing:
        raise ValueError(f"Missing columns in {gen_file}: {sorted(missing)}")

    gen["Period"] = pd.to_numeric(gen["Period"], errors="coerce")
    gen["genInstalledCap"] = pd.to_numeric(gen["genInstalledCap"], errors="coerce").fillna(0.0)
    gen = gen.dropna(subset=["Node", "Generator", "Period"]).copy()
    gen["Period"] = gen["Period"].astype(int)

    gen["TechGroup"] = gen["Generator"].map(map_generator_to_group)
    # Keep track of technologies that are present but not in the aggregation map.
    unknown = sorted(gen.loc[gen["TechGroup"].isna(), "Generator"].astype(str).unique())
    if unknown:
        print("Warning: Unmapped technologies skipped:", ", ".join(unknown))
    gen = gen.dropna(subset=["TechGroup"])

    cap_all = (
        gen.groupby(["Period", "Node", "TechGroup"], as_index=False)["genInstalledCap"]
        .sum()
    )
    print("Unique technologies in dataset:", ", ".join(sorted(gen["Generator"].astype(str).unique())))

    periods = sorted(cap_all["Period"].unique().tolist())
    by_period: dict[int, pd.DataFrame] = {}
    for period in periods:
        cap = (
            cap_all.loc[cap_all["Period"] == period, ["Node", "TechGroup", "genInstalledCap"]]
            .pivot(index="Node", columns="TechGroup", values="genInstalledCap")
            .fillna(0.0)
            .reset_index()
        )
        for tech in TECH_ORDER:
            if tech not in cap.columns:
                cap[tech] = 0.0
        cap["total_cap"] = cap[TECH_ORDER].sum(axis=1)
        cap["norm_node"] = cap["Node"].map(normalize_name)
        by_period[int(period)] = cap

    return by_period, [int(p) for p in periods]


def load_load_curtailment_last_period(results_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Load node-level load curtailment from Operational output (LoadShed_MW).
    For each node/period, sum hourly MW to MWh per scenario and average across scenarios.
    """
    curtail_file = results_path / "Output" / "results_output_Operational.csv"
    curtail = pd.read_csv(curtail_file)
    required = {"Node", "Period", "Scenario", "LoadShed_MW"}
    missing = required - set(curtail.columns)
    if missing:
        raise ValueError(f"Missing columns in {curtail_file}: {sorted(missing)}")

    curtail["LoadShed_MW"] = pd.to_numeric(
        curtail["LoadShed_MW"], errors="coerce"
    ).fillna(0.0)
    periods = sorted(curtail["Period"].astype(str).unique().tolist(), key=parse_period_end_year)
    if not periods:
        raise RuntimeError("No periods found in operational output.")
    last_period = periods[-1]
    curtail_last_raw = curtail.loc[
        curtail["Period"].astype(str) == last_period, ["Node", "Scenario", "LoadShed_MW"]
    ].copy()
    curtail_last = (
        curtail_last_raw.groupby(["Node", "Scenario"], as_index=False)["LoadShed_MW"]
        .sum()
        .groupby("Node", as_index=False)["LoadShed_MW"]
        .mean()
    )
    curtail_last["ExpectedLoadCurtailment_GWh"] = curtail_last["LoadShed_MW"] / 1000.0
    curtail_last = curtail_last[["Node", "ExpectedLoadCurtailment_GWh"]]
    curtail_last["norm_node"] = curtail_last["Node"].map(normalize_name)
    return curtail_last, last_period


def load_geojson_geometries(geojson_file: Path) -> list:
    """Load all geometries from the NUTS GeoJSON for base map shading."""
    with geojson_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", [])
    return [shape(feature["geometry"]) for feature in features if feature.get("geometry")]


def load_nuts2_country_borders(geojson_file: Path, country_codes: set[str]) -> list:
    """Extract NUTS2 polygons for selected countries (e.g., DE/DK) as border overlays."""
    with geojson_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", [])

    borders = []
    for feature in features:
        geom = feature.get("geometry")
        props = feature.get("properties", {})
        if not geom:
            continue

        cntr = str(props.get("CNTR_CODE", "")).strip().upper()
        level_raw = props.get("LEVL_CODE")
        try:
            level = int(level_raw)
        except (TypeError, ValueError):
            level = None

        if cntr in country_codes and level == 2:
            borders.append(shape(geom))
    return borders


def plot_installed_capacity(
    merged: pd.DataFrame,
    lines: pd.DataFrame,
    period: int,
    output_file: Path,
    basemap_file: Path,
) -> None:
    """Render Cartopy map with per-node technology pies and legends."""
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": projection})

    ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f8f8f8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#666666", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#888888", zorder=1)

    if basemap_file.exists():
        geometries = load_geojson_geometries(basemap_file)
        # GeoJSON uses EPSG:3035 coordinates, so draw with that source CRS.
        ax.add_geometries(
            geometries,
            crs=GEOJSON_SOURCE_CRS,
            facecolor="#ededed",
            edgecolor="#b6b6b6",
            linewidth=0.2,
            zorder=1,
        )

        de_dk_nuts2 = load_nuts2_country_borders(basemap_file, {"DE", "DK"})
        if de_dk_nuts2:
            # Highlight only DE/DK NUTS2 borders above the base polygons.
            ax.add_geometries(
                de_dk_nuts2,
                crs=GEOJSON_SOURCE_CRS,
                facecolor="none",
                edgecolor="#6B6B6B",
                linewidth=0.4,
                zorder=6,
            )
    else:
        print(f"Warning: basemap file not found: {basemap_file}")

    cap = merged["total_cap"].clip(lower=0)
    max_cap = cap.max() if len(cap) else 0.0
    if max_cap > 0:
        # Radius scales with sqrt(capacity) to avoid very large nodes dominating.
        radii = 0.15 + 1.2 * np.sqrt(cap / max_cap)
    else:
        radii = np.full(len(merged), 0.15)

    # Draw transmission links below node pies.
    if not lines.empty:
        max_line_cap = float(lines["transmissionInstalledCap"].max())
        for _, line in lines.iterrows():
            if max_line_cap > 0:
                lw = 0.2 + 2.8 * np.sqrt(float(line["transmissionInstalledCap"]) / max_line_cap)
            else:
                lw = 0.2
            ax.plot(
                [line["from_lon"], line["to_lon"]],
                [line["from_lat"], line["to_lat"]],
                color="#2F5D62",
                linewidth=lw,
                alpha=0.8,
                transform=projection,
                zorder=4,
            )

    for (_, row), radius in zip(merged.iterrows(), radii):
        total = float(row["total_cap"])
        if total <= 0:
            continue
        theta1 = 0.0
        for tech in TECH_ORDER:
            value = float(row[tech])
            if value <= 0:
                continue
            # Each wedge angle is proportional to that technology's share.
            theta2 = theta1 + (value / total) * 360.0
            wedge = Wedge(
                center=(row["Longitude"], row["Latitude"]),
                r=float(radius),
                theta1=float(theta1),
                theta2=float(theta2),
                facecolor=TECH_COLORS[tech],
                edgecolor="black",
                linewidth=0.25,
                transform=projection,
                zorder=5,
            )
            ax.add_patch(wedge)
            theta1 = theta2

    top_nodes = merged.nlargest(12, "total_cap")
    # Label only top nodes to keep the map readable.
    for _, row in top_nodes.iterrows():
        ax.text(
            row["Longitude"] + 0.15,
            row["Latitude"] + 0.15,
            row["Node"],
            fontsize=8,
            color="black",
            transform=projection,
            zorder=6,
        )

    lon_min = merged["Longitude"].min() - 4.0
    lon_max = merged["Longitude"].max() + 4.0
    lat_min = merged["Latitude"].min() - 3.0
    lat_max = merged["Latitude"].max() + 3.0
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"North Sea case: installed generation capacity by node\nFinal investment period step: {period}")

    tech_legend = [Patch(facecolor=TECH_COLORS[t], edgecolor="black", label=t) for t in TECH_ORDER]
    legend_tech = ax.legend(
        handles=tech_legend,
        loc="lower left",
        fontsize=8,
        title="Technology group",
        title_fontsize=9,
        framealpha=0.95,
    )
    ax.add_artist(legend_tech)

    if max_cap > 0:
        # Size legend uses representative capacities from the plotted range.
        size_values = [max_cap * 0.1, max_cap * 0.4, max_cap]
        size_values = [v for v in size_values if v > 0]
        size_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=(0.15 + 1.2 * np.sqrt(v / max_cap)) * 9.5,
                label=f"{v:,.0f} MW",
            )
            for v in size_values
        ]
        # Add one handle for transmission line meaning.
        line_handle = Line2D([0], [0], color="#2F5D62", lw=2.0, label="Transmission (scaled by MW)")
        ax.legend(
            handles=[line_handle, *size_handles],
            loc="lower right",
            title="Capacity scale",
            title_fontsize=9,
            fontsize=8,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


def build_change_dataframe(
    first_cap: pd.DataFrame,
    last_cap: pd.DataFrame,
    group_techs: list[str],
) -> pd.DataFrame:
    """Build node-level change (last-first) for a technology group."""
    first = first_cap[["norm_node", *group_techs]].copy()
    last = last_cap[["norm_node", *group_techs]].copy()
    first["first_value"] = first[group_techs].sum(axis=1)
    last["last_value"] = last[group_techs].sum(axis=1)
    merged = (
        first[["norm_node", "first_value"]]
        .merge(last[["norm_node", "last_value"]], on="norm_node", how="outer")
        .fillna(0.0)
    )
    merged["delta_mw"] = merged["last_value"] - merged["first_value"]
    return merged[["norm_node", "delta_mw"]]


def plot_change_bars(
    delta_df: pd.DataFrame,
    coords: pd.DataFrame,
    group_label: str,
    period_first: int,
    period_last: int,
    output_file: Path,
    basemap_file: Path,
) -> None:
    """Plot one vertical bar per node for installed-capacity changes."""
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": projection})

    ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f8f8f8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#666666", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#888888", zorder=1)

    if basemap_file.exists():
        geometries = load_geojson_geometries(basemap_file)
        ax.add_geometries(
            geometries,
            crs=GEOJSON_SOURCE_CRS,
            facecolor="#ededed",
            edgecolor="#b6b6b6",
            linewidth=0.2,
            zorder=1,
        )
        de_dk_nuts2 = load_nuts2_country_borders(basemap_file, {"DE", "DK"})
        if de_dk_nuts2:
            ax.add_geometries(
                de_dk_nuts2,
                crs=GEOJSON_SOURCE_CRS,
                facecolor="none",
                edgecolor="#6B6B6B",
                linewidth=0.4,
                zorder=3,
            )

    plot_df = (
        delta_df.merge(coords[["norm_node", "Location", "Latitude", "Longitude"]], on="norm_node", how="left")
        .dropna(subset=["Latitude", "Longitude"])
        .copy()
    )
    if plot_df.empty:
        raise RuntimeError(f"No coordinates available for change plot: {group_label}")

    max_abs = float(plot_df["delta_mw"].abs().max())
    if max_abs <= 0:
        max_abs = 1.0

    min_h = 0.08
    max_h = 1.5
    for _, row in plot_df.iterrows():
        delta = float(row["delta_mw"])
        if abs(delta) < 1e-9:
            continue
        bar_h = min_h + (max_h - min_h) * np.sqrt(abs(delta) / max_abs)
        y0 = float(row["Latitude"])
        y1 = y0 + (bar_h if delta > 0 else -bar_h)
        color = "#1a9850" if delta > 0 else "#d73027"
        ax.plot(
            [row["Longitude"], row["Longitude"]],
            [y0, y1],
            color=color,
            linewidth=3.6,
            transform=projection,
            zorder=4,
        )

    top_nodes = plot_df.loc[plot_df["delta_mw"].abs().nlargest(10).index]
    for _, row in top_nodes.iterrows():
        ax.text(
            row["Longitude"] + 0.12,
            row["Latitude"] + 0.12,
            str(row["Location"]),
            fontsize=7.5,
            color="black",
            transform=projection,
            zorder=5,
        )

    lon_min = plot_df["Longitude"].min() - 4.0
    lon_max = plot_df["Longitude"].max() + 4.0
    lat_min = plot_df["Latitude"].min() - 3.0
    lat_max = plot_df["Latitude"].max() + 3.0
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        f"North Sea case: installed capacity change by node\n"
        f"{group_label.replace('_', ' ').title()} ({period_first} -> {period_last})"
    )

    legend_handles = [
        Line2D([0], [0], color="#1a9850", lw=3.6, label="Increase"),
        Line2D([0], [0], color="#d73027", lw=3.6, label="Decrease"),
    ]
    size_values = [0.25 * max_abs, 0.6 * max_abs, max_abs]
    for v in size_values:
        h = min_h + (max_h - min_h) * np.sqrt(max(v, 0.0) / max_abs)
        legend_handles.append(
            Line2D([0], [0], color="#444444", lw=max(2.0, 3.6 * h / max_h), label=f"{v:,.0f} MW")
        )
    ax.legend(handles=legend_handles, loc="lower right", title="Bar meaning", title_fontsize=9, fontsize=8)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_load_curtailment_map(
    curtailment: pd.DataFrame,
    coords: pd.DataFrame,
    period_label: str,
    output_file: Path,
    basemap_file: Path,
) -> None:
    """Plot node-level expected load curtailment as bubbles."""
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": projection})

    ax.add_feature(cfeature.OCEAN, facecolor="#dbeeff", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f8f8f8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#666666", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#888888", zorder=1)

    if basemap_file.exists():
        geometries = load_geojson_geometries(basemap_file)
        ax.add_geometries(
            geometries,
            crs=GEOJSON_SOURCE_CRS,
            facecolor="#ededed",
            edgecolor="#b6b6b6",
            linewidth=0.2,
            zorder=1,
        )
        de_dk_nuts2 = load_nuts2_country_borders(basemap_file, {"DE", "DK"})
        if de_dk_nuts2:
            ax.add_geometries(
                de_dk_nuts2,
                crs=GEOJSON_SOURCE_CRS,
                facecolor="none",
                edgecolor="#6B6B6B",
                linewidth=0.4,
                zorder=3,
            )

    plot_df = (
        curtailment.merge(coords[["norm_node", "Location", "Latitude", "Longitude"]], on="norm_node", how="left")
        .dropna(subset=["Latitude", "Longitude"])
        .copy()
    )
    if plot_df.empty:
        raise RuntimeError("No coordinates available for load curtailment map.")

    values = plot_df["ExpectedLoadCurtailment_GWh"].clip(lower=0.0)
    vmax = float(values.max()) if len(values) else 0.0
    if vmax > 0:
        sizes = 25 + 900 * np.sqrt(values / vmax)
    else:
        sizes = np.full(len(plot_df), 25.0)

    sc = ax.scatter(
        plot_df["Longitude"],
        plot_df["Latitude"],
        s=sizes,
        c=values,
        cmap="magma",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.35,
        transform=projection,
        zorder=5,
    )

    top_nodes = plot_df.nlargest(12, "ExpectedLoadCurtailment_GWh")
    for _, row in top_nodes.iterrows():
        ax.text(
            row["Longitude"] + 0.12,
            row["Latitude"] + 0.12,
            str(row["Location"]),
            fontsize=7.5,
            color="black",
            transform=projection,
            zorder=6,
        )

    lon_min = plot_df["Longitude"].min() - 4.0
    lon_max = plot_df["Longitude"].max() + 4.0
    lat_min = plot_df["Latitude"].min() - 3.0
    lat_max = plot_df["Latitude"].max() + 3.0
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"North Sea case: expected annual curtailment by node\nPeriod: {period_label}")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Expected load curtailment [GWh]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot installed generation capacity for the final investment period of the north_sea dataset."
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--coords-file", type=Path, default=DEFAULT_COORDS_PATH)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--basemap-file", type=Path, default=DEFAULT_BASEMAP)
    args = parser.parse_args()

    coords = load_coords(args.coords_file)
    offshore_norm_nodes = load_offshore_nodes(args.results_path)
    offshore_to_onshore = build_offshore_to_onshore_map(coords, offshore_norm_nodes)
    capacity_by_period, gen_periods = load_capacity_by_period(args.results_path)
    transmission_by_period, line_periods = load_transmission_by_period(args.results_path)
    periods = sorted(set(gen_periods).union(line_periods))
    if not periods:
        raise RuntimeError("No investment periods found in generation/transmission outputs.")
    if not gen_periods:
        raise RuntimeError("No generation periods found in genInstalledCap.tab.")

    output_base = args.output_file
    saved_files: list[Path] = []
    for period in periods:
        if period not in capacity_by_period:
            print(f"Warning (period {period}): no generation data, skipping plot.")
            continue
        capacity = capacity_by_period[period]
        if AGGREGATE_OFFSHORE_TO_CLOSEST_ONSHORE:
            capacity = aggregate_offshore_to_nearest_onshore(
                capacity,
                coords,
                offshore_norm_nodes,
                offshore_to_onshore=offshore_to_onshore,
            )
        merged = capacity.merge(coords[["norm_node", "Latitude", "Longitude"]], on="norm_node", how="left")
        missing = merged["Latitude"].isna().sum()
        if missing:
            missing_nodes = merged.loc[merged["Latitude"].isna(), "Node"].sort_values().tolist()
            print(f"Warning (period {period}): {missing} nodes are missing coordinates and will be skipped.")
            print("Missing nodes:", ", ".join(missing_nodes))
        merged = merged.dropna(subset=["Latitude", "Longitude"]).copy()
        if merged.empty:
            print(f"Warning (period {period}): no nodes with coordinates, skipping plot.")
            continue

        lines = transmission_by_period.get(period, pd.DataFrame(columns=["u", "v", "transmissionInstalledCap"])).copy()
        if not lines.empty and AGGREGATE_OFFSHORE_TO_CLOSEST_ONSHORE:
            lines["u"] = lines["u"].map(offshore_to_onshore).fillna(lines["u"])
            lines["v"] = lines["v"].map(offshore_to_onshore).fillna(lines["v"])
            lines = lines[lines["u"] != lines["v"]].copy()
            lines["a"] = np.where(lines["u"] <= lines["v"], lines["u"], lines["v"])
            lines["b"] = np.where(lines["u"] <= lines["v"], lines["v"], lines["u"])
            lines = lines.groupby(["a", "b"], as_index=False)["transmissionInstalledCap"].sum()
            lines = lines.rename(columns={"a": "u", "b": "v"})

        coord_lookup = coords[["norm_node", "Latitude", "Longitude"]].drop_duplicates("norm_node")
        lines = (
            lines.merge(coord_lookup, left_on="u", right_on="norm_node", how="left")
            .rename(columns={"Latitude": "from_lat", "Longitude": "from_lon"})
            .drop(columns=["norm_node"])
            .merge(coord_lookup, left_on="v", right_on="norm_node", how="left")
            .rename(columns={"Latitude": "to_lat", "Longitude": "to_lon"})
            .drop(columns=["norm_node"])
        )
        lines = lines.dropna(subset=["from_lat", "from_lon", "to_lat", "to_lon"]).copy()

        period_output = output_base.with_name(f"{output_base.stem}_period_{period}{output_base.suffix}")
        plot_installed_capacity(
            merged=merged,
            lines=lines,
            period=period,
            output_file=period_output,
            basemap_file=args.basemap_file,
        )
        saved_files.append(period_output)
        print(f"Saved period {period} plot to: {period_output}")
        print(f"Nodes plotted (period {period}): {len(merged)}")

    if not saved_files:
        raise RuntimeError("No figures were generated.")

    period_first = gen_periods[0]
    period_last = gen_periods[-1]
    if period_first in capacity_by_period and period_last in capacity_by_period:
        cap_first = capacity_by_period[period_first]
        cap_last = capacity_by_period[period_last]
        if AGGREGATE_OFFSHORE_TO_CLOSEST_ONSHORE:
            cap_first = aggregate_offshore_to_nearest_onshore(
                cap_first,
                coords,
                offshore_norm_nodes,
                offshore_to_onshore=offshore_to_onshore,
            )
            cap_last = aggregate_offshore_to_nearest_onshore(
                cap_last,
                coords,
                offshore_norm_nodes,
                offshore_to_onshore=offshore_to_onshore,
            )

        for group_label, techs in CHANGE_GROUPS.items():
            delta_df = build_change_dataframe(cap_first, cap_last, techs)
            change_output = output_base.with_name(
                f"{output_base.stem}_change_{group_label}_{period_first}_to_{period_last}{output_base.suffix}"
            )
            plot_change_bars(
                delta_df=delta_df,
                coords=coords,
                group_label=group_label,
                period_first=period_first,
                period_last=period_last,
                output_file=change_output,
                basemap_file=args.basemap_file,
            )
            print(f"Saved change plot ({group_label}) to: {change_output}")

    curtailment, curtail_period = load_load_curtailment_last_period(args.results_path)
    if AGGREGATE_OFFSHORE_TO_CLOSEST_ONSHORE and not curtailment.empty:
        curtailment["target_norm_node"] = curtailment["norm_node"].map(offshore_to_onshore).fillna(curtailment["norm_node"])
        curtailment = (
            curtailment.groupby("target_norm_node", as_index=False)["ExpectedLoadCurtailment_GWh"]
            .sum()
            .rename(columns={"target_norm_node": "norm_node"})
        )
    curtail_output = output_base.with_name(f"{output_base.stem}_load_curtailment_{curtail_period}{output_base.suffix}")
    plot_load_curtailment_map(
        curtailment=curtailment,
        coords=coords,
        period_label=curtail_period,
        output_file=curtail_output,
        basemap_file=args.basemap_file,
    )
    print(f"Saved load curtailment plot to: {curtail_output}")


if __name__ == "__main__":
    main()
