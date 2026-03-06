# Binary TEP - Example Excel Data

This document provides complete example Excel sheet structures for implementing binary TEP.

---

## Example 1: Simple 3-Node System

### File: Transmission.xlsx

#### Sheet: CandidateTransmission
```
From Node        To Node          
(Description)    (Description)    
Name             Name             
NO               SE               
SE               FI               
DK               DE               
```

#### Sheet: LineBlockCapacity
```
From Node        To Node          Block Capacity (MW)
(Description)    (Description)    (Discrete expansion size)
Name             Name             MW
NO               SE               500
SE               FI               300
DK               DE               400
```

**Notes**:
- 3 candidate lines defined
- Each has different block size
- Covers all expansion options in system

---

### File: General.xlsx

**No changes needed** - all existing sheets work as before

---

## Example 2: Medium 6-Node European System

### File: Transmission.xlsx

#### Sheet: CandidateTransmission
```
From Node        To Node          
(Description)    (Description)    
Name             Name             
NO               SE               
SE               FI               
DK               DE               
DE               FR               
FR               IT               
IT               GR               
SE               DK               
```

#### Sheet: LineBlockCapacity
```
From Node        To Node          Block Capacity (MW)
(Description)    (Description)    (Discrete expansion size)
Name             Name             MW
NO               SE               600
SE               FI               400
DK               DE               500
DE               FR               800
FR               IT               700
IT               GR               300
SE               DK               450
```

---

## Example 3: Using Global Fallback

### File: Transmission.xlsx

#### Sheet: CandidateTransmission
```
From Node        To Node          
(Description)    (Description)    
Name             Name             
NO               SE               
SE               FI               
DK               DE               
DE               FR               
FR               IT               
IT               GR               
SE               DK               
```

**Note**: No LineBlockCapacity sheet - use global fallback instead

### File: General.xlsx

#### Sheet: LineBlockCapacityGlobal
```
Global Block Capacity (MW)
(Description/Units)
transmissionLineBlockCapGlobal
500
```

**Result**: ALL candidate lines expand in 500 MW blocks

---

## Example 4: Mixed Per-Line and Global Fallback

### File: Transmission.xlsx

#### Sheet: CandidateTransmission
```
From Node        To Node          
Name             Name             
NO               SE               
SE               FI               
DK               DE               
DE               FR               
FR               IT               
```

#### Sheet: LineBlockCapacity
```
From Node        To Node          Block Capacity (MW)
Name             Name             MW
NO               SE               600
SE               FI               400
DK               DE               500
```

**Note**: Only 3 out of 5 candidates have per-line blocks

### File: General.xlsx

#### Sheet: LineBlockCapacityGlobal
```
Global Block Capacity (MW)
transmissionLineBlockCapGlobal
450
```

**Result**:
- NO-SE: 600 MW blocks (per-line)
- SE-FI: 400 MW blocks (per-line)
- DK-DE: 500 MW blocks (per-line)
- DE-FR: 450 MW blocks (global fallback)
- FR-IT: 450 MW blocks (global fallback)

---

## Example 5: Real-World Norway-Sweden-Denmark Case

### File: Transmission.xlsx

#### Sheet: CandidateTransmission
```
From Node        To Node          
(Description)    (Description)    
Name             Name             
NO_1             SE_1             
SE_1             SE_2             
SE_1             DK_1             
DK_1             DK_2             
DK_1             DE_1             
SE_2             FI_1             
```

#### Sheet: LineBlockCapacity
```
From Node        To Node          Block Capacity (MW)  Comment
(Description)    (Description)    (Discrete size)      (Optional notes)
Name             Name             MW                   Notes
NO_1             SE_1             700                  HVDC link
SE_1             SE_2             500                  AC cable
SE_1             DK_1             600                  HVDC link
DK_1             DK_2             400                  AC cable
DK_1             DE_1             800                  HVDC link
SE_2             FI_1             300                  Smaller link
```

**Notes**:
- Different block sizes reflect cable/asset availability
- HVDC links (700, 600, 800 MW) larger than AC (500, 400, 300 MW)
- Comments column (Col 3) is optional, not used by model

---

## Example 6: Stochastic Expansion Scenario

### File: Transmission.xlsx

#### Sheet: CandidateTransmission
```
From Node        To Node          
Name             Name             
NO_W             SE_C             
SE_C             SE_E             
NO_W             DK_W             
```

#### Sheet: LineBlockCapacity
```
From Node        To Node          Block Capacity (MW)
Name             Name             MW
NO_W             SE_C             500
SE_C             SE_E             500
NO_W             DK_W             350
```

**Model behavior**:
- 3 possible expansion corridors
- Binary decisions: expand or don't each period
- Discrete: 500, 500, or 350 MW per decision
- Total expansion: 0 to 1350 MW cumulative

---

## Real Data Considerations

### Typical Block Sizes (Based on Cable/Asset Availability)

| Technology | Typical Block Size | Example |
|------------|-------------------|---------|
| HVDC Submarine | 600-800 MW | Viking Link |
| HVDC Land | 500-1200 MW | Fenno-Skan cables |
| AC Cable | 300-500 MW | Regional links |
| AC Overhead | 1000+ MW | Major corridors |
| Upgrade (half capacity) | Varies | Retrofitting |

### Recommended Approach

1. **Start with uniform blocks** (e.g., 400 MW all lines)
2. **Use global fallback**: `LineBlockCapacityGlobal = 400`
3. **Refine later**: Add per-line blocks if sensitivity needed
4. **Document sources**: Note why each block size chosen

---

## Data Validation Examples

### ✅ Valid CandidateTransmission

```
From Node        To Node          
NO               SE               ✓ Valid pair
SE               FI               ✓ Valid pair
DK               DE               ✓ Valid pair
                                  ✓ Empty rows OK (ignored)
```

### ❌ Invalid CandidateTransmission

```
From Node        To Node          
SE               NO               ✗ Duplicate (same as NO-SE, just reversed)
SE               SE               ✗ Self-loop
                 FI               ✗ Missing FromNode
DE                                ✗ Missing ToNode
```

### ✅ Valid LineBlockCapacity

```
From Node        To Node          Block Capacity (MW)
NO               SE               500                  ✓ Candidate line
SE               FI               300                  ✓ Candidate line
                                                       ✓ All MW > 0
```

### ❌ Invalid LineBlockCapacity

```
From Node        To Node          Block Capacity (MW)
NO               SE               500                  ✓ Candidate line
SE               FI               0                    ✗ Block = 0 (will use fallback)
DK               DE               -100                 ✗ Negative value (rejected)
GR               TR               250                  ✗ Not in CandidateTransmission set
```

---

## Column Mapping to Code

### CandidateTransmission Sheet
```python
# In reader.py:
read_file(TransmissionExcelData, 'CandidateTransmission', 
          [0, 1],  # ← Reads first 2 columns
          tab_file_path, "Transmission", skipheaders=2)

# Maps to:
# Column 0 (index [0]) → FromNode
# Column 1 (index [1]) → ToNode
```

### LineBlockCapacity Sheet
```python
# In reader.py (if added):
read_file(TransmissionExcelData, 'LineBlockCapacity', 
          [0, 1, 2],  # ← Reads first 3 columns
          tab_file_path, "Transmission", skipheaders=2)

# Maps to:
# Column 0 (index [0]) → FromNode
# Column 1 (index [1]) → ToNode
# Column 2 (index [2]) → LineBlockCap
```

---

## Migration Example: Converting Existing Data

### If You Have: "Old Continuous TEP" Data

**Old Transmission.xlsx structure:**
```
From Node  | To Node | MinCapacity | MaxCapacity | ... (continuous parameters)
```

### Convert To: Binary TEP Structure

1. **Extract candidate lines** → Create `CandidateTransmission` sheet
   ```
   Select rows where status = "candidate" or "expandable"
   Keep only FromNode, ToNode columns
   ```

2. **Determine block sizes** → Create `LineBlockCapacity` sheet
   ```
   Option A: Use MaxCapacity as block size
   Option B: Use realistic cable size (500-800 MW)
   Option C: Use global average for all lines
   ```

3. **Adjust other sheets** as needed:
   ```
   InitialCapacity: Keep as-is (existing capacity)
   MaxBuiltCapacity: Set to at least block_size (allows 1 build/period)
   MaxInstallCapacityRaw: Update if expansion limits change
   ```

---

## Testing Your Excel Data

### Quick Validation Script Concept
```python
import pandas as pd

# Load data
candidate_df = pd.read_excel('Transmission.xlsx', sheet_name='CandidateTransmission', skiprows=2)
block_df = pd.read_excel('Transmission.xlsx', sheet_name='LineBlockCapacity', skiprows=2)

# Check 1: CandidateTransmission has 2 columns
assert len(candidate_df.columns) == 2, "CandidateTransmission should have 2 columns"

# Check 2: No null values in FromNode, ToNode
assert not candidate_df.iloc[:, 0].isna().any(), "FromNode has null values"
assert not candidate_df.iloc[:, 1].isna().any(), "ToNode has null values"

# Check 3: Block sizes > 0
assert (block_df.iloc[:, 2] > 0).all(), "Some block capacities <= 0"

# Check 4: All block candidates in candidate set
for _, row in block_df.iterrows():
    pair = (row.iloc[0], row.iloc[1])
    assert pair in candidate_df.values, f"Block capacity for {pair} not in CandidateTransmission"

print("✅ Excel data validation passed!")
```

---

## Summary: Required Columns

| Sheet | Col 0 | Col 1 | Col 2 | Type |
|-------|-------|-------|-------|------|
| CandidateTransmission | FromNode | ToNode | - | Set (2 columns) |
| LineBlockCapacity | FromNode | ToNode | LineBlockCap | Parameter (3 columns) |
| LineBlockCapacityGlobal | Value | - | - | Scalar (1 value) |

---

**Next Steps**:
1. Copy these structures to your Transmission.xlsx
2. Fill in your candidate lines
3. Specify block sizes (per-line or global)
4. Run model and verify binary builds work correctly

---

**Date**: November 7, 2025
