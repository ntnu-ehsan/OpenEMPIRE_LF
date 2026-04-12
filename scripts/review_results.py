from __future__ import annotations

import importlib.util
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

CLIENT_MODULE_PATH = Path(__file__).resolve().parents[1] / "empire" / "output_client" / "client.py"
CLIENT_SPEC = importlib.util.spec_from_file_location("empire_output_client", CLIENT_MODULE_PATH)
if CLIENT_SPEC is None or CLIENT_SPEC.loader is None:
    raise ImportError(f"Could not load output client module from {CLIENT_MODULE_PATH}")
CLIENT_MODULE = importlib.util.module_from_spec(CLIENT_SPEC)
CLIENT_SPEC.loader.exec_module(CLIENT_MODULE)
EmpireOutputClient = CLIENT_MODULE.EmpireOutputClient


def resolve_run_and_output(path: Path) -> tuple[Path, Path]:
    if (path / "Output").exists():
        return path, path / "Output"
    if path.name.lower() == "output":
        return path.parent, path
    raise FileNotFoundError(
        f"Could not infer output folder from '{path}'. "
        "Pass a run folder containing 'Output/' or pass the 'Output/' folder directly."
    )


def safe_get(loader):
    try:
        return loader(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def period_sum(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or value_col not in df.columns or "Period" not in df.columns:
        return pd.DataFrame(columns=["Period", value_col])
    return (
        df.groupby("Period", as_index=False)[value_col]
        .sum()
        .sort_values("Period")
        .reset_index(drop=True)
    )


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    if df.empty:
        print("  (no data)")
        return
    print(df.to_string(index=False))


def main() -> None:
    parser = ArgumentParser(description="Review EMPIRE run results and print key KPIs.")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to run folder (containing Output/) or directly to Output/.",
    )
    parser.add_argument(
        "--node",
        default=None,
        help="Optional node filter for operational output.",
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save computed KPI tables to Output/review_summary.csv.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save basic KPI plots to <run>/Plots.",
    )
    args = parser.parse_args()

    run_path, output_path = resolve_run_and_output(Path(args.path))
    client = EmpireOutputClient(output_path=output_path)

    objective, objective_err = safe_get(client.get_objective)
    gen_df, gen_err = safe_get(client.get_generators_values)
    stor_df, stor_err = safe_get(client.get_storage_values)
    trans_df, trans_err = safe_get(client.get_transmission_values)
    curt_df, curt_err = safe_get(client.get_curtailed_production)
    e_summary_df, e_summary_err = safe_get(client.get_europe_summary_emission_and_energy)

    node_op_df = None
    node_op_err = None
    if args.node is not None:
        node_op_df, node_op_err = safe_get(lambda: client.get_node_operational_values(args.node))

    print(f"Run path: {run_path}")
    print(f"Output path: {output_path}")
    if objective_err:
        print(f"\nObjective: unavailable ({objective_err})")
    else:
        print(f"\nObjective: {objective:,.3f}")

    if e_summary_err:
        print(f"\nEurope summary: unavailable ({e_summary_err})")
    else:
        print_table("Europe summary (emission and energy):", e_summary_df)

    gen_cap = period_sum(gen_df, "genInstalledCap_MW")
    stor_pw = period_sum(stor_df, "storPWInstalledCap_MW")
    stor_en = period_sum(stor_df, "storENInstalledCap_MWh")
    trans_cap = period_sum(trans_df, "transmissionInstalledCap_MW")
    curt = period_sum(curt_df, "ExpectedAnnualCurtailment_GWh")

    print_table("Installed generation capacity by period [MW]:", gen_cap)
    print_table("Installed storage power by period [MW]:", stor_pw)
    print_table("Installed storage energy by period [MWh]:", stor_en)
    print_table("Installed transmission capacity by period [MW]:", trans_cap)
    print_table("Expected annual curtailment by period [GWh]:", curt)

    if args.node is not None:
        if node_op_err:
            print(f"\nNode operational ({args.node}): unavailable ({node_op_err})")
        else:
            print(f"\nNode operational ({args.node}): {len(node_op_df)} rows")

    load_errors = {
        "generators": gen_err,
        "storage": stor_err,
        "transmission": trans_err,
        "curtailment": curt_err,
    }
    missing = {k: v for k, v in load_errors.items() if v is not None}
    if missing:
        print("\nSome files were unavailable:")
        for name, err in missing.items():
            print(f"  - {name}: {err}")

    summary_frames = []
    for metric, df, col in [
        ("genInstalledCap_MW", gen_cap, "genInstalledCap_MW"),
        ("storPWInstalledCap_MW", stor_pw, "storPWInstalledCap_MW"),
        ("storENInstalledCap_MWh", stor_en, "storENInstalledCap_MWh"),
        ("transmissionInstalledCap_MW", trans_cap, "transmissionInstalledCap_MW"),
        ("ExpectedAnnualCurtailment_GWh", curt, "ExpectedAnnualCurtailment_GWh"),
    ]:
        if not df.empty:
            tmp = df.copy()
            tmp.insert(0, "Metric", metric)
            tmp.rename(columns={col: "Value"}, inplace=True)
            summary_frames.append(tmp[["Metric", "Period", "Value"]])

    if args.save_summary and summary_frames:
        summary_df = pd.concat(summary_frames, ignore_index=True)
        if objective is not None:
            objective_row = pd.DataFrame(
                [{"Metric": "Objective", "Period": "ALL", "Value": float(objective)}]
            )
            summary_df = pd.concat([objective_row, summary_df], ignore_index=True)
        output_file = output_path / "review_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSaved summary: {output_file}")

    if args.save_plots:
        import matplotlib.pyplot as plt

        plots_dir = run_path / "Plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for title, df, y_col, fname in [
            ("Installed generation capacity [MW]", gen_cap, "genInstalledCap_MW", "gen_installed_cap_mw.png"),
            ("Installed storage power [MW]", stor_pw, "storPWInstalledCap_MW", "stor_power_installed_mw.png"),
            (
                "Installed transmission capacity [MW]",
                trans_cap,
                "transmissionInstalledCap_MW",
                "transmission_installed_mw.png",
            ),
            ("Expected annual curtailment [GWh]", curt, "ExpectedAnnualCurtailment_GWh", "curtailment_gwh.png"),
        ]:
            if df.empty:
                continue
            ax = df.plot(x="Period", y=y_col, kind="bar", legend=False, title=title, figsize=(10, 4))
            ax.set_xlabel("Period")
            ax.set_ylabel(y_col)
            plt.tight_layout()
            out_file = plots_dir / fname
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"Saved plot: {out_file}")


if __name__ == "__main__":
    main()
