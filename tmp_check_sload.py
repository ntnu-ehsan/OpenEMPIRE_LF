import pandas as pd
from pathlib import Path

dataset = 'Go_RES_v1.3.0_Europe'
data_path = Path(f'Data handler/{dataset}')

# Read all nodes from Sets
sets = pd.read_excel(data_path / 'Sets.xlsx', sheet_name='Nodes')
all_nodes = sets.iloc[:, 0].dropna().str.strip().tolist()
print(f"=== All nodes ({len(all_nodes)}) ===")
print(all_nodes)
print()

# Read annual demand from Node.xlsx
node = pd.read_excel(data_path / 'Node.xlsx', sheet_name='ElectricAnnualDemand', skiprows=2)
print("=== Annual demand (first 5 rows) ===")
print(node.head().to_string())
print()

demand_nodes = set(node.iloc[:, 0].dropna().str.strip().tolist())

# Find nodes with non-zero demand
nonzero = node[node.iloc[:, 2] > 0]  # col 2 is the demand value
nodes_with_demand = set(nonzero.iloc[:, 0].str.strip().tolist())
print(f"Nodes with non-zero demand: {len(nodes_with_demand)}")
print(sorted(nodes_with_demand))
print()

# Compare with a working dataset's nodes
working_node = pd.read_excel('Data handler/DE_DK_NL/Node.xlsx', sheet_name='ElectricAnnualDemand', skiprows=2)
working_demand_nodes = set(working_node[working_node.iloc[:, 2] > 0].iloc[:, 0].str.strip().tolist())
print(f"=== Nodes in Go_RES with demand but not in DE_DK_NL ===")
new_nodes = nodes_with_demand - working_demand_nodes
print(sorted(new_nodes))
