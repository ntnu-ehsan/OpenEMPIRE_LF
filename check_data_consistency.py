"""
Check consistency between GeneratorsOfNode set and generator data files.
"""
import pandas as pd
from pathlib import Path

def main():
    tab_path = Path(r"m:\Work\codes\EMPIRE\OpenEMPIRE_LF\Results\basic_run\dataset_bolivia_v1\Input\Tab")
    
    # Read GeneratorsOfNode set
    gen_of_node = pd.read_csv(tab_path / "Sets_GeneratorsOfNode.tab", sep='\t', skiprows=1, names=['Node', 'Generator'])
    valid_combinations = set(zip(gen_of_node['Node'], gen_of_node['Generator']))
    
    print(f"Valid GeneratorsOfNode combinations: {len(valid_combinations)}")
    print("\nFirst 10 valid combinations:")
    for combo in list(valid_combinations)[:10]:
        print(f"  {combo}")
    
    # Files to check
    files_to_check = [
        'Generator_MaxBuiltCapacity.tab',
        'Generator_RefInitialCap.tab',
        'Generator_InitialCapacity.tab',
        'Generator_MaxInstalledCapacity.tab',
    ]
    
    all_issues = []
    
    for filename in files_to_check:
        filepath = tab_path / filename
        if not filepath.exists():
            print(f"\n{filename}: File does not exist (OK if optional)")
            continue
            
        print(f"\n{'='*60}")
        print(f"Checking: {filename}")
        print('='*60)
        
        # Read the file
        df = pd.read_csv(filepath, sep='\t')
        
        # Get column names (they vary by file)
        cols = df.columns.tolist()
        print(f"Columns: {cols}")
        
        if len(df) == 0:
            print("  File is empty")
            continue
        
        # Assume first two columns are Node and Generator
        node_col = cols[0]
        gen_col = cols[1]
        
        # Check each row
        issues = []
        for idx, row in df.iterrows():
            node = row[node_col]
            gen = row[gen_col]
            combo = (node, gen)
            
            if combo not in valid_combinations:
                issues.append((idx, node, gen))
        
        if issues:
            print(f"\n  Found {len(issues)} INVALID combinations:")
            # Group by generator type to see patterns
            by_gen = {}
            for idx, node, gen in issues:
                if gen not in by_gen:
                    by_gen[gen] = []
                by_gen[gen].append(node)
            
            for gen, nodes in sorted(by_gen.items()):
                print(f"    {gen}: {len(nodes)} invalid nodes")
                print(f"      Nodes: {sorted(set(nodes))[:10]}")  # Show first 10 unique nodes
            
            all_issues.extend([(filename, node, gen) for _, node, gen in issues])
        else:
            print("  ✓ All combinations are valid")
    
    if all_issues:
        print(f"\n{'='*60}")
        print(f"SUMMARY: Found {len(all_issues)} total inconsistencies")
        print('='*60)
        print("\nTo fix, you need to either:")
        print("1. Add missing generator-node combinations to Sets.xlsx → GeneratorsOfNode sheet")
        print("2. Remove invalid entries from Generator.xlsx data sheets")
    else:
        print(f"\n{'='*60}")
        print("✓ All data files are consistent with GeneratorsOfNode set")
        print('='*60)

if __name__ == "__main__":
    main()
