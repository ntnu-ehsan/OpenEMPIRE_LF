"""
Clean up Bolivia generator data files to only include valid GeneratorsOfNode combinations.
"""
import pandas as pd
from pathlib import Path

def main():
    tab_path = Path(r"m:\Work\codes\EMPIRE\OpenEMPIRE_LF\Results\basic_run\dataset_bolivia_v1\Input\Tab")
    
    # Read GeneratorsOfNode set
    print("Reading GeneratorsOfNode set...")
    gen_of_node = pd.read_csv(tab_path / "Sets_GeneratorsOfNode.tab", sep='\t', skiprows=1, names=['Node', 'Generator'])
    valid_combinations = set(zip(gen_of_node['Node'], gen_of_node['Generator']))
    
    print(f"Valid GeneratorsOfNode combinations: {len(valid_combinations)}\n")
    
    # Files to clean
    files_to_clean = [
        'Generator_MaxBuiltCapacity.tab',
        'Generator_MaxInstalledCapacity.tab',
    ]
    
    for filename in files_to_clean:
        filepath = tab_path / filename
        if not filepath.exists():
            print(f"{filename}: File does not exist, skipping")
            continue
            
        print(f"Processing: {filename}")
        
        # Read the file
        df = pd.read_csv(filepath, sep='\t')
        original_count = len(df)
        
        if original_count == 0:
            print(f"  File is empty, skipping\n")
            continue
        
        # Get column names
        cols = df.columns.tolist()
        node_col = cols[0]
        gen_col = cols[1]
        
        # Filter to only valid combinations
        mask = df.apply(lambda row: (row[node_col], row[gen_col]) in valid_combinations, axis=1)
        df_clean = df[mask]
        
        removed_count = original_count - len(df_clean)
        
        if removed_count > 0:
            print(f"  Original rows: {original_count}")
            print(f"  Removed rows: {removed_count}")
            print(f"  Remaining rows: {len(df_clean)}")
            
            # Save cleaned file
            df_clean.to_csv(filepath, sep='\t', index=False)
            print(f"  ✓ Saved cleaned file\n")
        else:
            print(f"  ✓ No invalid rows found\n")
    
    print("="*60)
    print("Cleanup complete! Re-run the model to test.")
    print("="*60)

if __name__ == "__main__":
    main()
