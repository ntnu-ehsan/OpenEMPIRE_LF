# DC-OPF Voltage Scaling for Actual Units

## Overview

The angle-based DC-OPF formulation in EMPIRE has been updated to support actual unit values instead of per-unit values. This is achieved by incorporating the voltage magnitude squared (V²) into the Ohm's law constraints.

## Physical Background

### Per-Unit Formulation (Original)
In per-unit systems with V = 1:
```
P_pu = B * (θ_i - θ_j)
```
where B = 1/X (susceptance = 1/reactance)

### Actual Units Formulation (Updated)
In actual units:
```
P_MW = (V²/X) * (θ_i - θ_j) = B * V² * (θ_i - θ_j)
```
where:
- V is the system voltage magnitude in kV
- V² is the voltage magnitude squared in kV²
- P_MW is the power flow in MW

## Changes Made

### 1. New Parameters: `NominalVoltage` and `VoltageSquared`
- **Location**: `empire/core/optimization/lopf_module.py`
- **NominalVoltage**: Loaded from General.xlsx, represents system voltage in kV
- **VoltageSquared**: Automatically computed as NominalVoltage² in kV²
- **Default value**: 400 kV (V² = 160,000 kV², typical for Extra High Voltage transmission networks)
- **Description**: System-wide nominal voltage magnitude and its square

### 2. Updated Ohm's Law Constraints
All three Ohm's law constraint rules have been updated to include V²:
- `ohm_exist`: For existing transmission lines
- `ohm_cand_ub`: Upper bound for candidate lines
- `ohm_cand_lb`: Lower bound for candidate lines

**New equation**:
```python
FlowDC[i,j,h,w,p] == B[i,j] * VoltageSquared * (Theta[i,h,w,p] - Theta[j,h,w,p])
```

### 3. Updated Big-M Calculation
The Big-M values for candidate line activation have been scaled by V²:
```python
BigMFlow[i,j] = |B[i,j]| * VoltageSquared * 2.0 * AngleMax
```

### 4. Parameter Loading
- **Reader**: `empire/core/reader.py` now reads the `NominalVoltage` sheet from `General.xlsx`
- **Loader**: `empire/core/optimization/lopf_module.py` loads `General_NominalVoltage.tab`
- **Processing**: VoltageSquared is automatically computed as NominalVoltage²

## How to Use

### Step 1: Add NominalVoltage to General.xlsx

1. Open your `General.xlsx` file in the appropriate dataset folder (e.g., `Data handler/test/ScenarioData/General.xlsx`)

2. Add a new sheet named **`NominalVoltage`**

3. In this sheet, add the nominal voltage value in **kV**:

   | NominalVoltage |
   |----------------|
   | 150            |

   **Example values**:
   - For a 150 kV transmission system: NominalVoltage = 150
   - For a 220 kV transmission system: NominalVoltage = 220
   - For a 400 kV transmission system: NominalVoltage = 400

   The model will automatically compute V² = NominalVoltage²:
   - 150 kV → V² = 22,500 kV²
   - 220 kV → V² = 48,400 kV²
   - 400 kV → V² = 160,000 kV²

4. Format: Single scalar value in the first data row (skip 2 header rows as per EMPIRE convention)

### Step 2: Run EMPIRE

No code changes are needed. Simply run EMPIRE as usual:

```powershell
python .\scripts\run.py -d test -c config\testrun.yaml -f
```

The model will:
- Automatically load the `VoltageSquared` value from `General.xlsx`
- Apply it to all DC power flow equations
- Use actual MW values for power flows

### Step 3: Interpret Results

The `FlowDC` variables and constraints will now represent actual power flows in MW, scaled by the voltage magnitude squared (V² = NominalVoltage²).

## Backward Compatibility

If you **do not** add the `NominalVoltage` sheet to `General.xlsx`:
- The default value of 400 kV will be used (V² = 160,000 kV²)
- This represents a typical Extra High Voltage (EHV) transmission system
- No error will occur; a log message will indicate the default is being used
- **Note**: If you need a different voltage level or per-unit formulation, explicitly add the `NominalVoltage` sheet

## Example: Voltage Levels for European Transmission Networks

Common transmission voltage levels:
- **Extra High Voltage (EHV)**: 220 kV, 380 kV, 400 kV
- **High Voltage (HV)**: 110 kV, 132 kV, 150 kV

Choose the voltage level that best represents your transmission network. For mixed-voltage systems, use the predominant voltage level or a representative average.

## Verification

To verify the voltage scaling is working:

1. Check the log output when running EMPIRE:
   ```
   INFO: Loaded General_NominalVoltage.tab for DC-OPF actual unit conversion.
   ```

2. Compare flow values before/after adding voltage scaling:
   - With V = 400 kV (default): Flows are scaled by V² = 160,000
   - With V = 150 kV: Flows are scaled by V² = 22,500
   - With V = 1 kV (per-unit): Flows are scaled by V² = 1

## Technical Notes

### Why V² and not V?

In DC power flow analysis, the power flow equation is:
```
P = V_i * V_j * B * sin(θ_i - θ_j) ≈ V² * B * (θ_i - θ_j)
```

For small angle differences (DC approximation), we assume:
- sin(Δθ) ≈ Δθ
- V_i ≈ V_j ≈ V (flat voltage profile)

Therefore, the linearized equation uses V².

### Impact on Other Components

This change **only affects** the angle-based DC-OPF formulation in `lopf_module.py`. It does **not** affect:
- Generator variables and constraints
- Storage variables and constraints
- Load variables
- Other transmission representations (transport model, Kirchhoff cycle-based)

### Consistency with Reactance/Susceptance Data

Ensure your reactance (X) or susceptance (B) values in `Transmission_lineReactance.tab` or `Transmission_lineSusceptance.tab` are in appropriate units:
- If using actual ohms: NominalVoltage should be in kV (default 400 kV is appropriate for most EHV networks)
- If using per-unit values: Set NominalVoltage = 1.0 kV to get V² = 1.0

## Questions or Issues?

If you encounter any issues or have questions about the voltage scaling:
1. Check that the `NominalVoltage` sheet exists in `General.xlsx`
2. Verify the voltage value is in kV and appropriate for your system
3. Review the log output for loading confirmation
4. Ensure reactance/susceptance units are consistent

---
**Last Updated**: November 14, 2025
