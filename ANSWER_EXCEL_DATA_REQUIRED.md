# Answer: Excel Input Data Required for Binary TEP

## 🎯 Direct Answer

**Based on the binary TEP modifications, these Excel sheets MUST exist:**

### ✅ Required (Will Fail Without)

#### 1. **Transmission.xlsx** → `CandidateTransmission` Sheet

**Columns**:
```
Col 0      | Col 1
-----------|----------
FromNode   | ToNode
```

**Content Example**:
```
FromNode   | ToNode
-----------|----------
NO         | SE
SE         | FI
DK         | DE
```

**Purpose**: List all transmission lines eligible for binary expansion

**Notes**:
- 2 columns exactly
- Skip first 2 header rows
- Data starts at row 3

---

#### 2. **Block Capacity** (Choose ONE or BOTH)

##### Option A: Per-Line Blocks (Preferred)
**Transmission.xlsx** → `LineBlockCapacity` Sheet

**Columns**:
```
Col 0      | Col 1    | Col 2
-----------|----------|------------------
FromNode   | ToNode   | LineBlockCap
```

**Content Example**:
```
FromNode   | ToNode   | LineBlockCap (MW)
-----------|----------|------------------
NO         | SE       | 500
SE         | FI       | 300
DK         | DE       | 400
```

**Purpose**: Define block size for each candidate line

**Notes**:
- 3 columns exactly
- Skip first 2 header rows
- Data starts at row 3
- Block must be > 0

---

##### Option B: Global Fallback
**General.xlsx** → `LineBlockCapacityGlobal` Sheet

**Columns**:
```
Col 0  | Value (MW)
-------|----------
       | 400
```

**Content Example**:
```
transmissionLineBlockCapGlobal
400
```

**Purpose**: Universal block size for candidates without per-line specification

**Notes**:
- Single value
- Skip first 1 header row
- Data at row 2
- Used as fallback
- Block must be > 0

---

### ✅ ALL OTHER EXCEL FILES (No Changes Needed)

The following files continue to work as before:

- ✓ Sets.xlsx (all sheets)
- ✓ Generator.xlsx (all sheets)
- ✓ Node.xlsx (all sheets)
- ✓ Storage.xlsx (all sheets)
- ✓ General.xlsx (all existing sheets continue)
- ✓ Transmission.xlsx (all existing sheets continue)

**Nothing else changes in existing Excel files!**

---

## 📊 Summary Table

| File | Sheet | Status | Columns | Purpose |
|------|-------|--------|---------|---------|
| **Transmission.xlsx** | **CandidateTransmission** | **NEW** | FromNode, ToNode | Define candidates |
| **Transmission.xlsx** | **LineBlockCapacity** | NEW (opt) | FromNode, ToNode, BlockCap | Per-line blocks |
| **General.xlsx** | **LineBlockCapacityGlobal** | NEW (opt) | Value | Global fallback |
| Transmission.xlsx | (all other) | Unchanged | - | Continue as-is |
| General.xlsx | (all other) | Unchanged | - | Continue as-is |
| Sets.xlsx | (all sheets) | Unchanged | - | Continue as-is |
| Generator.xlsx | (all sheets) | Unchanged | - | Continue as-is |
| Node.xlsx | (all sheets) | Unchanged | - | Continue as-is |
| Storage.xlsx | (all sheets) | Unchanged | - | Continue as-is |

---

## 🚨 Minimum Requirements

**Model REQUIRES**:
1. ✅ CandidateTransmission set (define expandable lines)
2. ✅ Block capacity (at least one source)
   - Per-line blocks, OR
   - Global fallback, OR
   - Both

**Without these, binary TEP will NOT work!**

---

## 🔄 Data Precedence

```
For each candidate line:

Check: Does per-line block exist?
├─ YES → Use per-line block
└─ NO → Check: Does global block exist?
        ├─ YES → Use global block
        └─ NO → ❌ Cannot expand (ERROR)
```

---

## 📝 Format Requirements

### All Parameter Sheets

**Header Rows**: 2 (automatically skipped)
```
Row 1: Description/units (ignored)
Row 2: Column names (ignored)
Row 3: First data row
```

### Column Types

| Sheet | Col 0 | Col 1 | Col 2 |
|-------|-------|-------|-------|
| CandidateTransmission | String | String | - |
| LineBlockCapacity | String | String | Float |
| LineBlockCapacityGlobal | - | Float | - |

### Data Rules

- ✅ String values: whitespace trimmed automatically
- ✅ Numeric values: float or integer
- ✅ Empty rows: ignored
- ✅ Node names: must match other sheets exactly (case-sensitive)
- ✅ Block size: must be > 0

---

## 🎯 Setup Steps (3 Simple Steps)

### Step 1: Identify Candidates
Ask: "Which transmission lines can expand?"
- Make a list: NO-SE, SE-FI, DK-DE, ...

### Step 2: Define Blocks
Ask: "How much capacity per build?"
- Option A: Different size per line (500, 300, 400 MW...)
  - → Create `LineBlockCapacity` sheet
- Option B: Same size for all (400 MW...)
  - → Create `LineBlockCapacityGlobal` value

### Step 3: Add to Excel
- Add `CandidateTransmission` sheet to Transmission.xlsx
- Add block capacity (per-line or global)
- Run model!

---

## ✅ Validation Checklist

Before running model:

**Excel Structure**:
- [ ] CandidateTransmission sheet exists in Transmission.xlsx
- [ ] At least one block capacity defined:
  - [ ] LineBlockCapacity in Transmission.xlsx, OR
  - [ ] LineBlockCapacityGlobal in General.xlsx
- [ ] Column counts correct (2 for candidates, 3 for blocks, 1 for global)
- [ ] First 2 rows are headers (auto-skipped)

**Data Content**:
- [ ] All candidate node names exist in Sets/Nodes
- [ ] All block values > 0
- [ ] No duplicate candidates
- [ ] Candidate lines make sense (realistic corridors)

---

## 📚 Complete Documentation Available

For more details, see:

| Document | Content |
|----------|---------|
| BINARY_TEP_SHEETS_REQUIRED.md | Complete sheet specifications |
| BINARY_TEP_INPUT_DATA_REQUIREMENTS.md | Detailed requirements & troubleshooting |
| BINARY_TEP_EXAMPLE_DATA.md | Real example sheets |
| BINARY_TEP_EXCEL_QUICK_CARD.md | One-page visual reference |
| BINARY_TEP_DOCUMENTATION_INDEX.md | Complete index & reading guide |

---

## 🚀 Quick Example

**To enable binary TEP for 3 lines:**

### Transmission.xlsx - NEW Sheet: `CandidateTransmission`
```
From Node        To Node
(Description)    (Description)
Name             Name
NO               SE
SE               FI
DK               DE
```

### Transmission.xlsx - NEW Sheet: `LineBlockCapacity`
```
From Node        To Node          Block Capacity (MW)
(Description)    (Description)    (Discrete expansion size)
Name             Name             MW
NO               SE               500
SE               FI               300
DK               DE               400
```

### Result
✅ Binary TEP enabled for 3 candidates with different block sizes!

---

## ❌ Common Mistakes

| ❌ Problem | ✅ Solution |
|-----------|-----------|
| No CandidateTransmission sheet | Create it with FromNode, ToNode columns |
| No block capacity | Add LineBlockCapacity or LineBlockCapacityGlobal |
| Block size = 0 | Must be > 0 (e.g., 400 MW) |
| Wrong node names | Use exactly same names as in Sets/Nodes |
| Wrong number of columns | Check: 2 for candidates, 3 for blocks |
| Wrong sheet name | Check exact spelling: "CandidateTransmission" |

---

## 📋 Excel Checklist

### ✅ Must Add
- [ ] CandidateTransmission sheet (Transmission.xlsx)
- [ ] Block capacity sheet (LineBlockCapacity OR LineBlockCapacityGlobal)

### ✅ Already Exists (No changes)
- [ ] Sets.xlsx (all sheets)
- [ ] Generator.xlsx (all sheets)
- [ ] Node.xlsx (all sheets)
- [ ] Storage.xlsx (all sheets)
- [ ] General.xlsx (other sheets)
- [ ] Transmission.xlsx (other sheets)

### ✅ Result
- [ ] All existing functionality works
- [ ] Binary TEP enabled for candidates
- [ ] Non-candidates stay fixed
- [ ] Model runs successfully

---

## 🎓 Summary

**Q: Which input data and sheet must exist in Excel files (based on modifications)?**

**A**: 
1. **NEW**: `Transmission.xlsx` → `CandidateTransmission` sheet (2 columns)
2. **NEW**: Block capacity (choose one):
   - `Transmission.xlsx` → `LineBlockCapacity` sheet (3 columns), OR
   - `General.xlsx` → `LineBlockCapacityGlobal` sheet (1 value)
3. **UNCHANGED**: All other Excel sheets work as before

**Minimum to run**: CandidateTransmission + one block capacity source

---

**Date**: November 7, 2025  
**Status**: ✅ Complete answer provided
