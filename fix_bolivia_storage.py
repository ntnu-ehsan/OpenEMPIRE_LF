"""
Fix Bolivia dataset storage inconsistencies.
Option 1: Add HydroPumpStorage to Sets.xlsx
"""
import pandas as pd
from pathlib import Path

base_path = Path(r"m:\Work\codes\EMPIRE\OpenEMPIRE_LF\Results\basic_run\dataset_bolivia_v1\Input\Xlsx")

# Read Sets.xlsx
sets_file = base_path / "Sets.xlsx"
print(f"Reading {sets_file}")

# Load all sheets
sets_data = pd.read_excel(sets_file, sheet_name=None)

# Check current storage types
storage_sheet = sets_data['Storage']
print(f"\nCurrent storage types in Sets.xlsx:")
print(storage_sheet)

# Add HydroPumpStorage if not present
# Find the header row (row with "Storage")
header_row = None
for idx, row in storage_sheet.iterrows():
    if 'Storage' in str(row.values):
        header_row = idx
        break

if header_row is not None:
    # Append new storage type below existing data
    new_row_idx = len(storage_sheet)
    storage_sheet.loc[new_row_idx] = ['Hydro Pump Storage']
    print(f"\nAdded 'Hydro Pump Storage' to storage types")
else:
    print("\nWarning: Could not find header row in Storage sheet")

# Save updated Sets.xlsx
output_file = base_path / "Sets_fixed.xlsx"
print(f"\nSaving to {output_file}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name, df in sets_data.items():
        if sheet_name == 'Storage':
            storage_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print("\nDone! Next steps:")
print(f"1. Review {output_file}")
print(f"2. Backup original: Copy-Item Sets.xlsx Sets_original_backup.xlsx")
print(f"3. Replace: Move-Item -Force Sets_fixed.xlsx Sets.xlsx")
print(f"4. Re-run: python scripts/run.py -d bolivia_v1 -c config/testrun.yaml -f")
