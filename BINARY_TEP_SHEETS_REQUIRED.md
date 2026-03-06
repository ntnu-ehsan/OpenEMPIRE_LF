# Binary TEP - Excel Data Summary

**Quick answer to: "Which input data and sheet must exist in Excel files?"**

---

## ✅ Required Excel Sheets for Binary TEP

### 1. **MUST HAVE**

#### Transmission.xlsx → `CandidateTransmission` Sheet

| Column | Name | Type | Example |
|--------|------|------|---------|
| 0 | FromNode | String | NO, SE, DK |
| 1 | ToNode | String | SE, FI, DE |

**Purpose**: Define which transmission lines are eligible for binary expansion decisions

**Format**: 2 columns, skip first 2 header rows, data starts row 3

**Content Example**:
```
From Node        To Node          
(Description)    (Description)    
Name             Name             ← Skip these 2 rows
NO               SE               ← Data starts here
SE               FI               
DK               DE               
```

---

### 2. **MUST HAVE - One or Both**

#### Option A: Transmission.xlsx → `LineBlockCapacity` Sheet (Preferred)

| Column | Name | Type | Example |
|--------|------|------|---------|
| 0 | FromNode | String | NO, SE, DK |
| 1 | ToNode | String | SE, FI, DE |
| 2 | LineBlockCap | Float | 500, 300, 400 |

**Purpose**: Define the block size (MW) for each candidate line

**Format**: 3 columns, skip first 2 header rows, data starts row 3

**Why**: Different transmission technologies have different capacity sizes
- HVDC links: 600-800 MW
- AC cables: 300-500 MW
- Asymmetric corridors: different sizes

---

#### Option B: General.xlsx → `LineBlockCapacityGlobal` Sheet (Fallback)

| Column | Name | Type | Value |
|--------|------|------|-------|
| 0 | Value | Float | 400 |

**Purpose**: Universal block size for all candidate lines (if per-line not specified)

**Format**: Single value, skip header row, data row 2

**Alternative**: Standalone file `Transmission_LineBlockCapacityGlobal.tab` with content:
```
transmissionLineBlockCapGlobal
400
```

**Usage**: If per-line block missing for a candidate, use this fallback (400 MW in this example)

---

## 📋 Data Precedence

```
For each candidate transmission line:

STEP 1: Check for per-line block capacity
        ├─ If LineBlockCapacity[FromNode, ToNode] exists AND > 0
        │  └─ ✅ USE: That per-line block size
        │
        └─ If not found or = 0
           └─ STEP 2: Check for global block capacity
              ├─ If LineBlockCapacityGlobal exists AND > 0
              │  └─ ✅ USE: Global block size
              │
              └─ If not found or = 0
                 └─ ❌ ERROR: Cannot expand (block = 0)
```

---

## 🎯 Minimal Excel Setup

**To enable binary TEP, you need**:

✅ **One of these combinations**:

1. **Option 1** (Recommended):
   - Transmission.xlsx with `CandidateTransmission` sheet
   - Transmission.xlsx with `LineBlockCapacity` sheet

2. **Option 2** (Simpler):
   - Transmission.xlsx with `CandidateTransmission` sheet
   - General.xlsx with `LineBlockCapacityGlobal` sheet

3. **Option 3** (Maximum simplicity):
   - Transmission.xlsx with `CandidateTransmission` sheet
   - Pre-made .tab file: `Transmission_LineBlockCapacityGlobal.tab`

---

## 📁 Excel File Structure

### Transmission.xlsx

```
Sheets:
├─ CandidateTransmission        ← NEW: Define candidate lines
├─ LineBlockCapacity             ← NEW (OPTIONAL): Per-line blocks
├─ InitialCapacity              ← Existing (unchanged)
├─ MaxBuiltCapacity             ← Existing (unchanged)
├─ MaxInstallCapacityRaw        ← Existing (unchanged)
├─ Length                        ← Existing (unchanged)
├─ TypeCapitalCost              ← Existing (unchanged)
├─ TypeFixedOMCost              ← Existing (unchanged)
├─ Lifetime                      ← Existing (unchanged)
├─ lineEfficiency               ← Existing (unchanged)
├─ lineReactance                ← Existing (only if lopf_flag=true)
└─ ... (other sheets)
```

### General.xlsx

```
Sheets:
├─ LineBlockCapacityGlobal       ← NEW (OPTIONAL): Global block fallback
├─ seasonScale                   ← Existing (unchanged)
├─ CO2Cap                        ← Existing (unchanged)
├─ CO2Price                      ← Existing (unchanged)
└─ ... (other sheets)
```

### Sets.xlsx, Generator.xlsx, Node.xlsx, Storage.xlsx

```
✅ NO CHANGES - All existing sheets work as-is
```

---

## 🔍 Column Details

### CandidateTransmission

```
╔═════════╦═════════╗
║ Col 0   ║ Col 1   ║
╠═════════╬═════════╣
║ FromNode║ ToNode  ║  ← Header row 2
╠═════════╬═════════╣
║ NO      ║ SE      ║  ← Data row 1
║ SE      ║ FI      ║  ← Data row 2
║ DK      ║ DE      ║  ← Data row 3
╚═════════╩═════════╝

Format: String, String
Skip rows: 2
Data type: Set members (undirected pairs)
```

### LineBlockCapacity

```
╔═════════╦═════════╦══════════════╗
║ Col 0   ║ Col 1   ║ Col 2        ║
╠═════════╬═════════╬══════════════╣
║ FromNode║ ToNode  ║ LineBlockCap ║  ← Header row 2
╠═════════╬═════════╬══════════════╣
║ NO      ║ SE      ║ 500          ║  ← Data row 1
║ SE      ║ FI      ║ 300          ║  ← Data row 2
║ DK      ║ DE      ║ 400          ║  ← Data row 3
╚═════════╩═════════╩══════════════╝

Format: String, String, Float
Skip rows: 2
Data type: Parameter values (MW)
Constraint: Value > 0
```

### LineBlockCapacityGlobal

```
╔════════════════════════════════════╗
║ transmissionLineBlockCapGlobal     ║  ← Header row
╠════════════════════════════════════╣
║ 400                                ║  ← Data (single value)
╚════════════════════════════════════╝

Format: Float
Skip rows: 1
Data type: Single scalar value (MW)
Constraint: Value > 0
```

---

## ⚠️ Common Mistakes to Avoid

| Issue | ❌ Wrong | ✅ Correct |
|-------|---------|-----------|
| Missing CandidateTransmission | No sheet | Add to Transmission.xlsx |
| No block capacity | Forget both sheets | Add LineBlockCapacity or LineBlockCapacityGlobal |
| Block size = 0 | Set value to 0 | Value must be > 0 |
| Node name mismatch | "Norway" in one sheet, "NO" in another | Use same names everywhere |
| Wrong column order | ToNode, FromNode (reversed) | FromNode (Col 0), ToNode (Col 1) |
| Wrong file | Block data in Transmission.xlsx | Per-line in Transmission, global in General |
| Too many columns | Include extra description columns | Use exactly 2 or 3 columns |
| Missing header rows | Data starts at row 1 | Skip first 2 rows, data at row 3 |

---

## 📊 Data Flow

```
Excel Workbooks
    │
    ├─ Transmission.xlsx
    │   ├─ CandidateTransmission sheet
    │   │   └─ → Transmission_CandidateTransmission.tab
    │   │       └─ model.CandidateTransmission (Set)
    │   │
    │   └─ LineBlockCapacity sheet (optional)
    │       └─ → Transmission_LineBlockCapacity.tab
    │           └─ model.transmissionLineBlockCap (Param)
    │
    └─ General.xlsx
        └─ LineBlockCapacityGlobal sheet (optional)
            └─ → General_LineBlockCapacityGlobal.tab
                └─ model.transmissionLineBlockCapGlobal (Param)

Model Variables
    │
    ├─ transmissionBuild[CandidateTransmission, PeriodActive] ∈ {0,1}
    │   └─ Binary expansion decision per period
    │
    └─ transmissionInstalledCap[BidirectionalArc, Period]
        └─ = initCap + Σ(block × build) for candidates
        └─ = initCap for non-candidates
```

---

## 🚦 Pre-Run Validation

Before running the model:

- [ ] `CandidateTransmission` sheet exists and has data
- [ ] Lines in CandidateTransmission have valid node names
- [ ] At least one block capacity source available:
  - [ ] `LineBlockCapacity` sheet in Transmission.xlsx, OR
  - [ ] `LineBlockCapacityGlobal` sheet in General.xlsx, OR
  - [ ] `.tab` files pre-generated in Tab directory
- [ ] Block capacity values > 0
- [ ] Node names consistent across all sheets
- [ ] All Excel files properly formatted (skip 2 header rows for sheets)

---

## 💡 Implementation Steps

1. **Audit existing data**
   - What transmission lines can expand?
   - What are realistic block sizes?

2. **Add to Transmission.xlsx**
   - New sheet: `CandidateTransmission`
   - List FromNode, ToNode for each expandable line
   - (Optional) Add sheet: `LineBlockCapacity` with block sizes

3. **Or add to General.xlsx**
   - New sheet: `LineBlockCapacityGlobal`
   - Single value for all candidates

4. **Test**
   - Run model
   - Check that binary builds = 1 adds block capacity
   - Verify costs calculated correctly

5. **Refine** (optional)
   - Switch to per-line blocks if needed
   - Adjust block sizes based on results
   - Add more candidate corridors

---

## 📖 Reference Documents

For more information, see:
- `BINARY_TEP_INPUT_DATA_REQUIREMENTS.md` - Detailed specifications
- `BINARY_TEP_EXAMPLE_DATA.md` - Complete example sheets
- `BINARY_TEP_EXCEL_QUICK_CARD.md` - One-page reference
- `TRANSMISSION_BINARY_CONVERSION_SUMMARY.md` - Technical details

---

## Summary Table

| Required | File | Sheet | Columns | Purpose |
|----------|------|-------|---------|---------|
| **YES** | Transmission.xlsx | CandidateTransmission | FromNode, ToNode | Define expandable lines |
| **YES (≥1)** | Transmission.xlsx | LineBlockCapacity | FromNode, ToNode, BlockCap | Per-line block sizes |
| **YES (≥1)** | General.xlsx | LineBlockCapacityGlobal | Value | Global fallback block |
| YES | All other files | All other sheets | (varies) | Existing parameters unchanged |

**Binary TEP requires**:
- CandidateTransmission set, AND
- At least one block capacity (per-line or global)

---

**Date**: November 7, 2025  
**Quick Start**: Add CandidateTransmission sheet + one block capacity = ✅ Ready!
