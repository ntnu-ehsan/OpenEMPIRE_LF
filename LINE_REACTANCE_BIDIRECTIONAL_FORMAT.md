# Line Reactance Bidirectional Input Format

## Overview

The `lineReactance` (and `lineSusceptance`) sheet in `Transmission.xlsx` now supports **bidirectional input format** while maintaining directional parameters in the model. This reduces the number of rows you need to enter by half.

## Key Features

- **Input**: Bidirectional - enter each line pair only once (e.g., A→B or B→A, not both)
- **Model**: Directional - the model automatically creates both directions internally
- **Assumption**: Reactance/susceptance is symmetric (same value in both directions)

## Format

### Excel Sheet Structure

**File**: `Transmission.xlsx`  
**Sheet**: `lineReactance` (or `lineSusceptance` if using susceptance)  
**Columns**: 3  
**Header rows to skip**: 2

| FromNode | ToNode | lineReactance |
|----------|--------|---------------|
| NO       | SE     | 0.0045        |
| SE       | FI     | 0.0038        |
| DK       | DE     | 0.0052        |

### What Happens

The reader automatically expands bidirectional data to directional:

**Input (3 rows)**:
```
FromNode  ToNode  lineReactance
NO        SE      0.0045
SE        FI      0.0038
DK        DE      0.0052
```

**Generated .tab file (6 rows)**:
```
FromNode  ToNode  lineReactance
NO        SE      0.0045
SE        NO      0.0045
SE        FI      0.0038
FI        SE      0.0038
DK        DE      0.0052
DE        DK      0.0052
```

## Benefits

1. **Reduced Data Entry**: Enter each line pair once instead of twice
2. **Less Error-Prone**: No risk of entering different values for the same line in opposite directions
3. **Model Compatibility**: Model still receives directional data as expected
4. **Backward Compatible**: If you enter directional data (both directions), it will work but you'll have duplicate entries

## Usage

### When LOPF is Enabled

In your configuration file (e.g., `config/run.yaml`):

```yaml
lopf_flag: true
lopf_kwargs:
  reactance_param_name: "lineReactance"  # or "lineSusceptance"
```

### Creating the Sheet

1. Open `Transmission.xlsx`
2. Create/edit sheet `lineReactance` (or `lineSusceptance`)
3. Add 2 header rows (description + column names)
4. Enter data: one row per line pair
5. Choose either direction (A→B or B→A, your choice)

### Example

```
Line Reactance Values (Ohms or per-unit)
FromNode  ToNode  lineReactance
Norway    Sweden  0.0045
Sweden    Finland 0.0038
Denmark   Germany 0.0052
```

**Note**: You can enter as `Norway→Sweden` OR `Sweden→Norway`, not both. The reader will create the reverse direction automatically.

## Important Notes

1. **Symmetric Values Only**: This approach assumes reactance/susceptance is the same in both directions (which is physically correct for transmission lines)

2. **Duplicate Entries**: If you accidentally enter both directions:
   ```
   NO  SE  0.0045
   SE  NO  0.0045
   ```
   The expansion will create 4 rows (NO→SE twice, SE→NO twice). The model will use one of them, but it's redundant.

3. **Consistency with DirectionalLines**: Make sure your `DirectionalLines` set in `Sets.xlsx` includes both directions for each line pair if needed by other parts of the model.

4. **Applied to Both Parameters**: This bidirectional expansion applies to:
   - `lineReactance` (when `reactance_param_name: "lineReactance"`)
   - `lineSusceptance` (when `reactance_param_name: "lineSusceptance"`)

## Migration from Directional Format

If you have existing data in directional format (both directions entered):

**Option 1 - Keep As-Is**: Your existing data will work, just with redundancy

**Option 2 - Migrate to Bidirectional**:
1. Identify all line pairs in your data
2. For each pair (A,B), keep only one row (either A→B or B→A)
3. Delete the reverse direction row
4. Result: Half the rows with same model behavior

## Technical Details

### Implementation

The function `read_bidirectional_to_directional()` in `empire/core/reader.py`:
1. Reads the Excel sheet with columns [FromNode, ToNode, Value]
2. Creates a copy with FromNode and ToNode swapped
3. Keeps the Value column the same (symmetric)
4. Concatenates original and reversed data
5. Saves as directional .tab file

### Logging

When processing, you'll see log messages:
```
INFO: Reading lineReactance sheet (bidirectional) from Transmission.xlsx
INFO: Expanded 3 bidirectional rows to 6 directional rows for sheet 'lineReactance'
```

## Example Workflow

### Before (Directional Input)

```excel
FromNode  ToNode  lineReactance
NO        SE      0.0045
SE        NO      0.0045
SE        FI      0.0038
FI        SE      0.0038
DK        DE      0.0052
DE        DK      0.0052
```
**6 rows to maintain**

### After (Bidirectional Input)

```excel
FromNode  ToNode  lineReactance
NO        SE      0.0045
SE        FI      0.0038
DK        DE      0.0052
```
**3 rows to maintain** → generates the same 6 directional rows automatically

## Configuration Check

To verify this feature is working:

1. Run your model with LOPF enabled
2. Check the log output for:
   ```
   INFO: Reading lineReactance sheet (bidirectional) from Transmission.xlsx
   INFO: Expanded N bidirectional rows to M directional rows for sheet 'lineReactance'
   ```
3. Check the generated .tab file: `Results/.../Tab_Files/Transmission_lineReactance.tab`
4. Verify it has twice as many data rows as your Excel sheet

## FAQ

**Q: Can I mix bidirectional and directional entries?**  
A: Yes, but it's not recommended. Each bidirectional entry will be expanded, including any that you entered in both directions, leading to duplicates.

**Q: What if reactance is different in each direction?**  
A: This is physically unlikely for transmission lines. If needed, use the old directional format and enter both directions manually with different values. The bidirectional reader will still process it but won't add symmetry.

**Q: Does this affect other transmission sheets?**  
A: No, this only applies to `lineReactance` and `lineSusceptance` when LOPF is enabled. Other sheets like `Length`, `InitialCapacity`, etc., are read as before.

**Q: Do I need to update my `DirectionalLines` set?**  
A: No, `DirectionalLines` in `Sets.xlsx` should still list all directions explicitly as needed by the model.

---

**Last Updated**: November 14, 2025
