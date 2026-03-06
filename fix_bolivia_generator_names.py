"""
Fix generator names in Bolivia Generator.xlsx to match Sets.xlsx naming convention.
"""
import pandas as pd
from pathlib import Path

# Define the mapping from abbreviated names to full names
GENERATOR_MAPPING = {
    'OCGT': 'Gas OCGT',
    'CCGT': 'Gas CCGT',
    'onwind': 'Wind onshore',
    'solar': 'Solar',
    'ror': 'Hydro run-of-the-river',
    'biomass': 'Bio',
    'oil': 'Oil existing',
    'geo': 'Geo',
    'regulated': 'Hydro regulated',
}

def fix_generator_sheet(excel_path: Path, sheet_name: str, generator_col_name: str):
    """
    Fix generator names in a specific sheet of the Generator.xlsx file.
    
    Args:
        excel_path: Path to the Generator.xlsx file
        sheet_name: Name of the sheet to fix
        generator_col_name: Name of the column containing generator names
    """
    print(f"Processing sheet: {sheet_name}")
    
    # Read the sheet with header at row 2 (skiprows=[0,1])
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=2)
    
    if generator_col_name not in df.columns:
        print(f"  Warning: Column '{generator_col_name}' not found in sheet '{sheet_name}'")
        print(f"  Available columns: {df.columns.tolist()}")
        return df
    
    # Count replacements
    original_values = df[generator_col_name].copy()
    
    # Replace abbreviated names with full names
    df[generator_col_name] = df[generator_col_name].replace(GENERATOR_MAPPING)
    
    # Show changes
    changes = df[generator_col_name] != original_values
    if changes.any():
        print(f"  Replaced {changes.sum()} generator names:")
        for old, new in zip(original_values[changes].unique(), df[generator_col_name][changes].unique()):
            print(f"    {old} -> {new}")
    else:
        print(f"  No changes needed")
    
    return df

def main():
    base_path = Path(r"m:\Work\codes\EMPIRE\OpenEMPIRE_LF\Results\basic_run\dataset_bolivia_v1\Input\Xlsx")
    generator_file = base_path / "Generator.xlsx"
    
    if not generator_file.exists():
        print(f"Error: Generator.xlsx not found at {generator_file}")
        return
    
    print(f"Reading {generator_file}")
    
    # Read all sheets
    excel_file = pd.ExcelFile(generator_file)
    
    # Sheets that need fixing (with their generator column name)
    sheets_to_fix = {
        'RefInitialCap': 'GeneratorTechnology',
        'ScaleFactorInitialCap': 'GeneratorTechnology',
        # Add more sheets if needed
    }
    
    # Process each sheet
    updated_sheets = {}
    for sheet_name in excel_file.sheet_names:
        if sheet_name in sheets_to_fix:
            generator_col = sheets_to_fix[sheet_name]
            updated_sheets[sheet_name] = fix_generator_sheet(generator_file, sheet_name, generator_col)
        else:
            # Keep original sheet unchanged
            updated_sheets[sheet_name] = pd.read_excel(generator_file, sheet_name=sheet_name)
    
    # Save the updated file
    output_file = generator_file.parent / "Generator_fixed.xlsx"
    print(f"\nSaving updated file to: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in updated_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("Done! Please:")
    print(f"1. Review the changes in {output_file}")
    print(f"2. Backup the original file if needed")
    print(f"3. Replace the original with: mv Generator_fixed.xlsx Generator.xlsx")
    print(f"4. Re-run the data conversion: python scripts/run.py -d bolivia_v1 -c config/testrun.yaml -f")

if __name__ == "__main__":
    main()
