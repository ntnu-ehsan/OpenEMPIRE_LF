import pandas as pd
from pathlib import Path

# Path to General.xlsx
general_file = Path("Data handler/test/General.xlsx")

# Read the NominalVoltage sheet
try:
    df = pd.read_excel(general_file, sheet_name='NominalVoltage', header=None)
    print("=== NominalVoltage Sheet Contents ===")
    print(f"Shape: {df.shape}")
    print(f"\nFull DataFrame:\n{df}")
    print(f"\nData types:\n{df.dtypes}")
    
    print("\n=== Column 0 Analysis ===")
    col0 = df.iloc[:, 0]
    print(f"Column 0 values: {col0.tolist()}")
    print(f"Column 0 dtypes: {col0.dtype}")
    
    print("\n=== Numeric Conversion Attempt ===")
    numeric_col = pd.to_numeric(col0, errors='coerce')
    print(f"After to_numeric: {numeric_col.tolist()}")
    print(f"Non-NaN values: {numeric_col.dropna().tolist()}")
    
except Exception as e:
    print(f"Error reading file: {e}")
