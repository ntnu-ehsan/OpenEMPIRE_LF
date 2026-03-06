# Binary TEP - Input Data & Excel Requirements

## Overview
Based on the model modifications, this document specifies **all required and optional Excel data** needed for the binary transmission expansion planning (TEP) feature to work correctly.

---

## Excel Files Required

### 1. **Transmission.xlsx** (Modified - Contains Key New Data)

#### **REQUIRED SHEETS**

##### 1.1 `CandidateTransmission` Sheet
**Purpose**: Define which transmission lines can be expanded using binary decisions

**Location**: In `Transmission.xlsx`

**Format**:
| Column 0 | Column 1 |
|----------|----------|
| FromNode | ToNode   |
| NO       | SE       |
| SE       | FI       |
| DK       | DE       |
| ...      | ...      |

**Details**:
- **FromNode**: Name of source node (string, must match node names in Sets.xlsx/Nodes)
- **ToNode**: Name of destination node (string, must match node names in Sets.xlsx/Nodes)
- **Required header rows**: 2 (skip first 2 rows with headers/descriptions)
- **Data starts at**: Row 3 (after 2 header rows)
- **Expected columns**: Exactly 2 columns (FromNode, ToNode)

**Example**:
```
From Node        To Node
Name             Name
NO               SE
SE               FI
DK               DE
```

**Notes**:
- Each line should appear exactly once (undirected pairs)
- Lines in this set are eligible for binary build decisions
- Lines NOT in this set are treated as **existing (fixed capacity)**
- Whitespace is automatically stripped

---

##### 1.2 `LineBlockCapacity` Sheet (Per-Line Blocks) - OPTIONAL BUT PREFERRED
**Purpose**: Specify block size (MW) for each candidate transmission line

**Location**: In `Transmission.xlsx`

**Format**:
| Column 0 | Column 1 | Column 2      |
|----------|----------|---------------|
| FromNode | ToNode   | LineBlockCap  |
| NO       | SE       | 500           |
| SE       | FI       | 300           |
| DK       | DE       | 400           |
| ...      | ...      | ...           |

**Details**:
- **FromNode**: Source node name
- **ToNode**: Destination node name
- **LineBlockCap**: Block capacity in MW (numeric, >0)
- **Required header rows**: 2
- **Data starts at**: Row 3

**Data Type**:
- FromNode: String
- ToNode: String
- LineBlockCap: Float or Integer (must be > 0)

**Example**:
```
From Node        To Node          Block Capacity (MW)
Name             Name             MW
NO               SE               500
SE               FI               300
DK               DE               400
```

**Notes**:
- Each candidate line should have exactly one entry
- Block size is the fixed capacity added when binary decision = 1
- If missing: Falls back to global block capacity (see below)
- If provided but value is 0/empty: Falls back to global block capacity
- Only lines in CandidateTransmission should be listed here

---

#### **EXISTING SHEETS** (Unchanged - Still Required)

- `InitialCapacity` - Existing transmission capacity by period
- `MaxBuiltCapacity` - Maximum capacity addition allowed per period per line
- `MaxInstallCapacityRaw` - Resource limit on total installed capacity
- `Length` - Transmission line lengths (km)
- `TypeCapitalCost` - Investment cost by transmission type and period
- `TypeFixedOMCost` - Fixed O&M cost by transmission type and period
- `Lifetime` - Lifetime of transmission infrastructure (years)
- `lineEfficiency` - Line efficiency losses
- `lineReactance` - (Only if `lopf_flag: true`) Reactance for DC OPF - **Now supports bidirectional input format** (see [LINE_REACTANCE_BIDIRECTIONAL_FORMAT.md](LINE_REACTANCE_BIDIRECTIONAL_FORMAT.md))

---

### 2. **General.xlsx** (Optional New Data)

#### `LineBlockCapacityGlobal` Sheet - OPTIONAL FALLBACK
**Purpose**: Global (universal) block capacity for all candidate lines without per-line specification

**Location**: In `General.xlsx` OR provide as standalone `.tab` file

**Format (if in Excel)**:
| Column 0                      |
|-------------------------------|
| transmissionLineBlockCapGlobal |
| 400                            |

**Details**:
- **Single column** with one value
- **Header row**: 1 (description)
- **Data**: Single numeric value
- **Units**: MW

**Example**:
```
Global Block Capacity (MW)
400
```

**Alternative: Standalone Tab File**

If not in Excel, can provide as:
- **File**: `Transmission_LineBlockCapacityGlobal.tab`
- **Location**: Tab files directory (auto-generated)
- **Format**:
  ```
  transmissionLineBlockCapGlobal
  400
  ```

**Usage**:
- Used as fallback when per-line block capacity is missing
- Applies to ALL candidate lines without a specific value
- Must be > 0 or all candidates can't expand

---

### 3. **Sets.xlsx** (Unchanged)

All existing sheets required:
- `Nodes` - Node/region names
- `OffshoreNodes` - Offshore node subset
- `Horizon` - Planning periods
- `LineType` - Transmission line types
- `Technology` - Generation technology types
- `Storage` - Storage technology types
- `Generators` - Generator set
- `StorageOfNodes` - Storage at nodes mapping
- `GeneratorsOfNode` - Generators at nodes mapping
- `GeneratorsOfTechnology` - Generators by technology
- `DirectionalLines` - Directed transmission links
- `LineTypeOfDirectionalLines` - Line type assignments

---

### 4. **Generator.xlsx** (Unchanged)

All existing sheets required (unchanged)

---

### 5. **Node.xlsx** (Unchanged)

All existing sheets required (unchanged)

---

### 6. **Storage.xlsx** (Unchanged)

All existing sheets required (unchanged)

---

## Summary: Excel Data Checklist

### ✅ **MUST HAVE** (Model will fail without these)

- [ ] **Sets.xlsx** - All existing sheets
- [ ] **Generator.xlsx** - All existing sheets
- [ ] **Transmission.xlsx**:
  - [ ] `CandidateTransmission` sheet (defines candidate lines)
  - [ ] `InitialCapacity` sheet (existing capacity)
  - [ ] `MaxBuiltCapacity` sheet (build limits)
  - [ ] `MaxInstallCapacityRaw` sheet (capacity resource limit)
  - [ ] Other transmission parameter sheets (Length, TypeCapitalCost, etc.)
- [ ] **Node.xlsx** - All existing sheets
- [ ] **General.xlsx** - All existing sheets
- [ ] **Storage.xlsx** - All existing sheets

### ⚠️ **REQUIRED FOR BINARY TEP** (Must have one or both)

- [ ] **LineBlockCapacity** sheet in `Transmission.xlsx` (per-line blocks) **OR**
- [ ] **LineBlockCapacityGlobal** value in `General.xlsx` or as `.tab` file

**Without at least one of these, binary builds cannot add capacity!**

### ✅ **OPTIONAL**

- [ ] `lineReactance` in `Transmission.xlsx` (only if `lopf_flag: true`)
- [ ] Multiple block sizes per period (currently not supported)

---

## Data Format Specifications

### Sheet Structure (All Parameter Sheets)

**Header rows**: 2 (skip these)
```
Row 1: Descriptions/Units (skipped)
Row 2: Column Names (skipped)  
Row 3: First data row
```

**Usage in code**:
```python
read_file(excelfile, 'SheetName', [0, 1, 2, ...], 
          tab_file_path, filename, skipheaders=2)
```

### Column Naming
- Spaces automatically converted to underscores: `"From Node"` → `"From_Node"`
- Whitespace trimmed from all string values
- Case-sensitive for node matching

### Data Validation
- **No empty cells** in key columns (automatically dropped)
- **Node names** must match exactly (case-sensitive)
- **Numeric values** must be valid floats/integers
- **Negative values** for capacity will be rejected (default values used)

---

## Typical Data Layout Example

### Transmission.xlsx - CandidateTransmission Sheet

```
Row  | From Node     | To Node
-----|---------------|----------
1    | (Description) | (Description)
2    | Name          | Name
3    | NO            | SE
4    | SE            | FI
5    | DK            | DE
6    | SE            | DK
7    | (empty)       | (empty)  ← Rows with empty data are ignored
```

### Transmission.xlsx - LineBlockCapacity Sheet

```
Row  | From Node     | To Node   | Block Capacity (MW)
-----|---------------|-----------|---------------------
1    | (Description) | (Description) | (Description)
2    | Name          | Name      | MW
3    | NO            | SE        | 500
4    | SE            | FI        | 300
5    | DK            | DE        | 400
6    | SE            | DK        | 350
7    | (empty)       | (empty)   | (empty)
```

### General.xlsx - LineBlockCapacityGlobal Sheet

```
Row | Global Block Capacity (MW)
----|----------------------------
1   | (Description/Units)
2   | transmissionLineBlockCapGlobal
3   | 400
```

---

## Generated .tab Files

The reader automatically generates these tab files from Excel sheets:

### From Transmission.xlsx

- `Transmission_CandidateTransmission.tab` ← From CandidateTransmission sheet
- `Transmission_LineBlockCapacity.tab` ← From LineBlockCapacity sheet (if exists)
- `Transmission_InitialCapacity.tab`
- `Transmission_MaxBuiltCapacity.tab`
- ... (and all other transmission sheets)

### From General.xlsx

- `General_LineBlockCapacityGlobal.tab` ← From LineBlockCapacityGlobal sheet (if exists)
- ... (other general parameters)

**These .tab files are loaded into model parameters and sets.**

---

## Integration with Model

### Set Loading

```python
# From Transmission_CandidateTransmission.tab
data.load(filename='Transmission_CandidateTransmission.tab',
          set=model.CandidateTransmission)
```

### Parameter Loading

```python
# Per-line block capacity (optional)
candidate_block_file = tab_file_path / 'Transmission_LineBlockCapacity.tab'
if candidate_block_file.exists():
    data.load(filename=str(candidate_block_file), 
              param=model.transmissionLineBlockCap, format="table")

# Global block capacity (fallback)
global_block_file = tab_file_path / 'Transmission_LineBlockCapacityGlobal.tab'
if global_block_file.exists():
    data.load(filename=str(global_block_file), 
              param=model.transmissionLineBlockCapGlobal)
```

---

## Common Issues & Solutions

### Issue: "CandidateTransmission set is empty"
**Cause**: `CandidateTransmission` sheet missing or has no data rows

**Solution**:
1. Verify sheet exists in `Transmission.xlsx`
2. Add rows with FromNode, ToNode data
3. Skip first 2 header rows
4. Ensure node names match those in Sets/Nodes

---

### Issue: "Cannot expand transmission capacity (block = 0)"
**Cause**: No block capacity defined for candidates

**Solution**:
1. Add `LineBlockCapacity` sheet to `Transmission.xlsx` with per-line values, OR
2. Add `LineBlockCapacityGlobal` to `General.xlsx` with fallback value
3. Ensure values are > 0

---

### Issue: "KeyError: (node1, node2) not in CandidateTransmission"
**Cause**: Transmission line in another sheet but not in CandidateTransmission

**Solution**:
1. Add line to CandidateTransmission sheet if it should expand, OR
2. Don't include it if it's existing infrastructure

---

### Issue: "Model has no attribute 'transmissionLineBlockCap'"
**Cause**: Neither per-line nor global block capacity provided

**Solution**:
1. Add `LineBlockCapacity` sheet to `Transmission.xlsx`, OR
2. Provide standalone `Transmission_LineBlockCapacity.tab` or `Transmission_LineBlockCapacityGlobal.tab`

---

## Migration Checklist

If converting from continuous TEP to binary TEP:

- [ ] Add `CandidateTransmission` sheet to `Transmission.xlsx`
- [ ] Define which lines can expand (binary decisions)
- [ ] Add `LineBlockCapacity` sheet with block sizes for each candidate
- [ ] OR add `LineBlockCapacityGlobal` fallback value
- [ ] Remove continuous investment parameters (if any)
- [ ] Update existing capacity to initialize properly
- [ ] Set MaxBuiltCapacity to allow at least one block per period
- [ ] Verify all node names match exactly across sheets

---

## Data Precedence & Fallback Logic

```python
# Getting block capacity for a candidate line (n1, n2):

if (n1, n2) in transmissionLineBlockCap AND value > 0:
    block = transmissionLineBlockCap[n1, n2]
else:
    block = transmissionLineBlockCapGlobal

# If transmissionLineBlockCapGlobal is also missing or 0:
#   → Binary expansion disabled (no capacity growth possible)
```

---

## Future Enhancements (Not Yet Implemented)

- [ ] Period-dependent block sizes (tech learning)
- [ ] Multiple discrete block options per line
- [ ] "At most one build per line" constraint
- [ ] Minimum inter-build spacing (e.g., must wait 2 periods)
- [ ] Path-dependent blocks (backup routing)

---

## Summary Table

| File | Sheet | Required | Purpose |
|------|-------|----------|---------|
| Transmission.xlsx | CandidateTransmission | **YES** | Define expandable lines |
| Transmission.xlsx | LineBlockCapacity | (See below) | Per-line block sizes |
| General.xlsx | LineBlockCapacityGlobal | (See below) | Global fallback block |
| **Total block capacity data** | | **YES (at least 1)** | Either per-line OR global |

**Binary TEP will fail without**:
1. CandidateTransmission set defined, AND
2. At least one block capacity parameter (per-line or global)

---

**Date**: November 7, 2025  
**Status**: Complete reference guide for input data requirements
