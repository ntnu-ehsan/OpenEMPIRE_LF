import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

def read_file(excelfile: pd.ExcelFile, sheet: str, columns: list, 
              tab_file_path: Path, filename: str, skipheaders: int = 0) -> None:
    """
    Reads data from an Excel file and saves it as a .tab file.
    
    :param excelfile: The Excel file object.
    :param sheet: The name of the sheet to read from.
    :param columns: List of columns to be read.
    :param tab_file_path: Path to save the .tab file.
    :param filename: Base name for the .tab file.
    :param skipheaders: Number of header rows to skip. Defaults to 0.
    """
    logger.info("Reading %s sheet from %s.xlsx", sheet, filename)

    input_sheet = excelfile[sheet]
    data_table = input_sheet.iloc[skipheaders:, columns]
    data_table.columns = pd.Series(data_table.columns).str.replace(' ', '_')
    data_nonempty = data_table.dropna()

    save_csv_frame = pd.DataFrame(data_nonempty)
    save_csv_frame.replace('\s', '', regex=True, inplace=True)

    tab_file_path.mkdir(parents=True, exist_ok=True)
    save_csv_frame.to_csv(tab_file_path / f"{filename}_{sheet}.tab", header=True, index=None, sep='\t', mode='w')

def write_scalar_tab(sheet: pd.DataFrame, tab_file_path: Path, filename: str, param_name: str) -> None:
    """
    Write a single scalar value from a General.xlsx-style sheet to a one-column .tab file
    that Pyomo can load into an unindexed Param via ``format="table"``.

    The sheet is expected to have the value in its first column (with descriptive header/text
    rows above it, as in the NominalVoltage / Sbase sheets). The last numeric value found is used.
    """
    values = pd.to_numeric(sheet.iloc[:, 0], errors='coerce').dropna()
    if values.empty:
        logger.warning("Sheet for '%s' contains no numeric value; skipping %s.tab", param_name, filename)
        return
    scalar = float(values.iloc[-1])
    tab_file_path.mkdir(parents=True, exist_ok=True)
    with open(tab_file_path / f"{filename}.tab", "w", newline="") as f:
        f.write(f"{param_name}\n{scalar}\n")
    logger.info("Wrote %s.tab (%s = %s).", filename, param_name, scalar)


def read_sets(excelfile: pd.ExcelFile, sheet: str, tab_file_path: Path,
              filename: str) -> None:
    """
    Reads sets data from an Excel file and saves each column as a separate .tab file.
    
    :param excelfile: The Excel file object.
    :param sheet: The name of the sheet to read from.
    :param tab_file_path: Path to save the .tab files.
    :param filename: Base name for the .tab files.
    """
    logger.info("Reading %s sheet from %s.xlsx", sheet, filename)

    input_sheet = excelfile[sheet]

    for ind, column in enumerate(input_sheet.columns):
        data_table = input_sheet.iloc[0:, ind]
        data_nonempty = data_table.dropna()
        data_nonempty.replace(" ", "")
        save_csv_frame = pd.DataFrame(data_nonempty)
        save_csv_frame.replace('\s', '', regex=True, inplace=True)
        tab_file_path.mkdir(parents=True, exist_ok=True)
        save_csv_frame.to_csv(tab_file_path / f"{filename}_{column}.tab", header=True, index=None, sep='\t', mode='w')        


def compute_reactance_from_length(excelfile: pd.ExcelFile, reactance_per_km: float,
                                  skipheaders: int = 2) -> pd.DataFrame:
    """
    Compute line reactance from transmission line length and a per-km reactance.

    Reads the 'Length' sheet of Transmission.xlsx and multiplies by ``reactance_per_km``.

    :param excelfile: The Transmission.xlsx workbook (dict of sheet DataFrames).
    :param reactance_per_km: Reactance value per kilometer (Ohm/km).
    :param skipheaders: Number of header rows to skip. Defaults to 2.
    :return: DataFrame with columns [FromNode, ToNode, lineReactance].
    """
    logger.info("Computing lineReactance from Length using reactance_per_km = %.4f Ohm/km", reactance_per_km)

    length_data = excelfile['Length'].iloc[skipheaders:, [0, 1, 2]]
    length_data.columns = pd.Series(length_data.columns).str.replace(' ', '_')
    length_data = length_data.dropna()

    if length_data.empty:
        logger.warning("Length sheet is empty - cannot compute reactance")
        return pd.DataFrame(columns=['FromNode', 'ToNode', 'lineReactance'])

    from_col, to_col, length_col = list(length_data.columns)[:3]
    reactance_data = length_data[[from_col, to_col]].copy()
    reactance_data['lineReactance'] = length_data[length_col] * reactance_per_km
    reactance_data.columns = ['FromNode', 'ToNode', 'lineReactance']
    logger.info("Computed reactance for %d transmission lines", len(reactance_data))
    return reactance_data


def write_directional_reactance_tab(reactance_data: pd.DataFrame, tab_file_path: Path) -> None:
    """
    Write a directional Transmission_lineReactance.tab from a (FromNode, ToNode, lineReactance)
    DataFrame. Reactance is symmetric, so each corridor is written in both directions to match
    the model's DirectionalLink-indexed lineReactance parameter.
    """
    if reactance_data.empty:
        logger.warning("No reactance data to write; skipping Transmission_lineReactance.tab")
        return

    reversed_data = reactance_data.rename(columns={'FromNode': 'ToNode', 'ToNode': 'FromNode'})
    directional = pd.concat([reactance_data, reversed_data], ignore_index=True)
    directional = directional[['FromNode', 'ToNode', 'lineReactance']].drop_duplicates(
        subset=['FromNode', 'ToNode']
    )
    tab_file_path.mkdir(parents=True, exist_ok=True)
    directional.to_csv(tab_file_path / "Transmission_lineReactance.tab",
                       header=True, index=None, sep='\t', mode='w')
    logger.info("Wrote Transmission_lineReactance.tab (%d directional rows) for DC-OPF.", len(directional))


def generate_tab_files(file_path, tab_file_path, lopf_kwargs=None):
    """
    Read column value from excel sheet and save as .tab file "sheet.tab"

    :param file_path: Path to the dataset.
    :param tab_file_path: Path to save the .tab files.
    :param lopf_kwargs: Optional LOPF options. When provided, line reactance is generated for
        the DC optimal power flow: from a 'lineReactance' sheet if present in Transmission.xlsx,
        otherwise computed from the 'Length' sheet using lopf_kwargs['reactance_per_km'].
    """
    
    logger.info("Generating .tab-files...")

    # Reading Excel workbooks using our function read_file

    if not os.path.exists(tab_file_path):
        os.makedirs(tab_file_path)

    logger.info("Reading Sets.xlsx")
    SetsExcelData = pd.read_excel(file_path / "Sets.xlsx", sheet_name=None)
    read_sets(SetsExcelData, 'Nodes', tab_file_path, "Sets")
    read_sets(SetsExcelData, 'OffshoreNodes', tab_file_path, "Sets")
    read_sets(SetsExcelData, 'Horizon', tab_file_path, "Sets")
    read_sets(SetsExcelData, 'LineType', tab_file_path, "Sets")
    read_sets(SetsExcelData, 'Technology', tab_file_path, "Sets")
    read_sets(SetsExcelData, 'Storage', tab_file_path, "Sets")
    read_sets(SetsExcelData, 'Generators', tab_file_path, "Sets")
    read_file(SetsExcelData, 'StorageOfNodes', [0, 1], tab_file_path, "Sets", skipheaders=2)
    read_file(SetsExcelData, 'GeneratorsOfNode', [0, 1], tab_file_path, "Sets", skipheaders=2)
    read_file(SetsExcelData, 'GeneratorsOfTechnology', [0, 1], tab_file_path, "Sets", skipheaders=2)
    read_file(SetsExcelData, 'DirectionalLines', [0, 1], tab_file_path, "Sets", skipheaders=2)
    read_file(SetsExcelData, 'LineTypeOfDirectionalLines', [0, 1, 2], tab_file_path, "Sets", skipheaders=2)

    # Reading GeneratorPeriod
    logger.info("Reading Generator.xlsx")
    GeneratorExcelData = pd.read_excel(file_path / "Generator.xlsx", sheet_name=None)
    read_file(GeneratorExcelData, 'FixedOMCosts', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'CapitalCosts', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'VariableOMCosts', [0, 1], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'FuelCosts', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'CCSCostTSVariable', [0, 1], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'Efficiency', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'RefInitialCap', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'ScaleFactorInitialCap', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'InitialCapacity', [0, 1, 2, 3], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'MaxBuiltCapacity', [0, 1, 2, 3], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'MaxInstalledCapacity', [0, 1, 2], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'RampRate', [0, 1], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'GeneratorTypeAvailability', [0, 1], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'CO2Content', [0, 1], tab_file_path, "Generator", skipheaders=2)
    read_file(GeneratorExcelData, 'Lifetime', [0, 1], tab_file_path, "Generator", skipheaders=2)

    #Reading InterConnector
    logger.info("Reading Transmission.xlsx")
    TransmissionExcelData = pd.read_excel(file_path / "Transmission.xlsx", sheet_name=None)
    read_file(TransmissionExcelData, 'lineEfficiency', [0, 1, 2], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'MaxInstallCapacityRaw', [0, 1, 2, 3], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'MaxBuiltCapacity', [0, 1, 2, 3], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'Length', [0, 1, 2], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'TypeCapitalCost', [0, 1, 2], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'TypeFixedOMCost', [0, 1, 2], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'InitialCapacity', [0, 1, 2, 3], tab_file_path,  "Transmission", skipheaders=2)
    read_file(TransmissionExcelData, 'Lifetime', [0, 1, 2], tab_file_path,  "Transmission", skipheaders=2)

    # Candidate corridors for binary block expansion (angle-based LOPF). Optional sheets:
    # generated whenever present so the dataset drives availability.
    if 'CandidateTransmission' in TransmissionExcelData:
        read_file(TransmissionExcelData, 'CandidateTransmission', [0, 1], tab_file_path, "Transmission", skipheaders=2)
    if 'LineBlockCapacity' in TransmissionExcelData:
        read_file(TransmissionExcelData, 'LineBlockCapacity', [0, 1, 2], tab_file_path, "Transmission", skipheaders=2)
    if 'LineBlockReactance' in TransmissionExcelData:
        read_file(TransmissionExcelData, 'LineBlockReactance', [0, 1, 2], tab_file_path, "Transmission", skipheaders=2)

    # Line reactance for linear (DC) optimal power flow. Only generated when LOPF is enabled.
    if lopf_kwargs is not None:
        if 'lineReactance' in TransmissionExcelData:
            sheet = TransmissionExcelData['lineReactance'].iloc[2:, [0, 1, 2]]
            sheet.columns = ['FromNode', 'ToNode', 'lineReactance']
            write_directional_reactance_tab(sheet.dropna(), tab_file_path)
        elif lopf_kwargs.get("reactance_per_km") is not None:
            reactance_data = compute_reactance_from_length(
                TransmissionExcelData, lopf_kwargs["reactance_per_km"], skipheaders=2
            )
            write_directional_reactance_tab(reactance_data, tab_file_path)
        else:
            logger.warning(
                "LOPF enabled but no line reactance source found: add a 'lineReactance' sheet to "
                "Transmission.xlsx or set lopf_kwargs['reactance_per_km']."
            )

    #Reading Node
    logger.info("Reading Node.xlsx")
    NodeExcelData = pd.read_excel(file_path / "Node.xlsx", sheet_name=None)
    read_file(NodeExcelData , 'ElectricAnnualDemand', [0, 1, 2],tab_file_path,  "Node", skipheaders=2)
    read_file(NodeExcelData , 'NodeLostLoadCost', [0, 1, 2],tab_file_path,  "Node", skipheaders=2)
    read_file(NodeExcelData , 'HydroGenMaxAnnualProduction', [0, 1],tab_file_path,  "Node", skipheaders=2)

    #Reading Season
    logger.info("Reading General.xlsx")
    GeneralExcelData = pd.read_excel(file_path / "General.xlsx", sheet_name=None)
    read_file(GeneralExcelData, 'seasonScale', [0, 1], tab_file_path, "General", skipheaders=2)
    read_file(GeneralExcelData, 'CO2Cap', [0, 1], tab_file_path, "General", skipheaders=2)
    read_file(GeneralExcelData, 'CO2Price', [0, 1], tab_file_path, "General", skipheaders=2)
    # Per-unit system base (MW) for LOPF. Optional: only present in per-unit datasets.
    if 'Sbase' in GeneralExcelData:
        write_scalar_tab(GeneralExcelData['Sbase'], tab_file_path, 'General_Sbase', 'sBase')
    # Optional scalars for the angle-based LOPF / binary block expansion.
    if 'LineBlockCapacityGlobal' in GeneralExcelData:
        write_scalar_tab(GeneralExcelData['LineBlockCapacityGlobal'], tab_file_path, 'General_LineBlockCapacityGlobal', 'transmissionLineBlockCapGlobal')
    if 'LineBlockReactanceGlobal' in GeneralExcelData:
        write_scalar_tab(GeneralExcelData['LineBlockReactanceGlobal'], tab_file_path, 'General_LineBlockReactanceGlobal', 'LineBlockReactanceGlobal')
    if 'NominalVoltage' in GeneralExcelData:
        write_scalar_tab(GeneralExcelData['NominalVoltage'], tab_file_path, 'General_NominalVoltage', 'NominalVoltage')
    
    #Reading Storage
    logger.info("Reading Storage.xlsx")
    StorageExcelData = pd.read_excel(file_path / "Storage.xlsx", sheet_name=None)
    read_file(StorageExcelData, 'StorageBleedEfficiency', [0, 1], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'StorageChargeEff', [0, 1], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'StorageDischargeEff', [0, 1], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'StoragePowToEnergy', [0, 1], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'StorageInitialEnergyLevel', [0, 1], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'InitialPowerCapacity', [0, 1, 2, 3], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'PowerCapitalCost', [0, 1, 2], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'PowerFixedOMCost', [0, 1, 2], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'PowerMaxBuiltCapacity', [0, 1, 2, 3], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'EnergyCapitalCost', [0, 1, 2], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'EnergyFixedOMCost', [0, 1, 2], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'EnergyInitialCapacity', [0, 1, 2, 3], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'EnergyMaxBuiltCapacity', [0, 1, 2, 3], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'EnergyMaxInstalledCapacity', [0, 1, 2], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'PowerMaxInstalledCapacity', [0, 1, 2], tab_file_path, "Storage", skipheaders=2)
    read_file(StorageExcelData, 'Lifetime', [0, 1], tab_file_path, "Storage", skipheaders=2)
