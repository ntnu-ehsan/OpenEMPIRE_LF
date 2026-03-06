# Excel Input Data - Quick Card

## 🚀 Must-Have for Binary TEP

### 1️⃣ **Transmission.xlsx** → `CandidateTransmission` Sheet

Defines which lines can be **expanded with binary decisions**

```
╔═══════════╦═══════════╗
║ FromNode  ║ ToNode    ║
╠═══════════╬═══════════╣
║ NO        ║ SE        ║
║ SE        ║ FI        ║
║ DK        ║ DE        ║
╚═══════════╩═══════════╝
```

**Format**: 2 columns, 2 header rows, skip first 2 rows

---

### 2️⃣ **Block Capacity** - Choose ONE or BOTH:

#### Option A: Per-Line (Preferred)
**Transmission.xlsx** → `LineBlockCapacity` Sheet

```
╔═══════════╦═══════════╦═════════════════╗
║ FromNode  ║ ToNode    ║ LineBlockCap    ║
╠═══════════╬═══════════╬═════════════════╣
║ NO        ║ SE        ║ 500             ║
║ SE        ║ FI        ║ 300             ║
║ DK        ║ DE        ║ 400             ║
╚═══════════╩═══════════╩═════════════════╝
```

**Each build decision adds this block (MW)**

---

#### Option B: Global Fallback
**General.xlsx** → `LineBlockCapacityGlobal` Sheet

```
╔═══════════════════════════════════╗
║ transmissionLineBlockCapGlobal    ║
╠═══════════════════════════════════╣
║ 400                               ║
╚═══════════════════════════════════╝
```

**Or standalone file**: `Transmission_LineBlockCapacityGlobal.tab`

**Used when per-line not available**

---

## ✅ All Other Sheets Stay the Same

- ✓ Sets.xlsx (all sheets)
- ✓ Generator.xlsx (all sheets)
- ✓ Node.xlsx (all sheets)
- ✓ Storage.xlsx (all sheets)
- ✓ General.xlsx (other sheets)
- ✓ Transmission.xlsx (other sheets)

**Existing parameter sheets work unchanged!**

---

## 📋 Column Names & Format

### CandidateTransmission
| Col | Name | Type | Example |
|-----|------|------|---------|
| 0 | FromNode | String | NO, SE, DK |
| 1 | ToNode | String | SE, FI, DE |

### LineBlockCapacity
| Col | Name | Type | Example |
|-----|------|------|---------|
| 0 | FromNode | String | NO, SE, DK |
| 1 | ToNode | String | SE, FI, DE |
| 2 | LineBlockCap | Number | 500, 300, 400 |

### LineBlockCapacityGlobal
| Col | Name | Type | Value |
|-----|------|------|-------|
| 0 | Value | Number | 400 |

---

## ⚠️ Common Mistakes

| ❌ Wrong | ✅ Right |
|---------|---------|
| Spaces in node names not trimmed | Spaces auto-trimmed, but use exact names |
| Different node name in different sheets | Same "NO" in Sets, Transmission, CandidateTransmission |
| LineBlockCapacity = 0 | LineBlockCapacity > 0 |
| No block capacity at all | At least 1 block (per-line or global) |
| Including non-candidate lines in block sheet | Only add candidates to LineBlockCapacity |
| Empty rows in middle of data | OK - empty rows skipped automatically |

---

## 🔍 Data Precedence

```
For each candidate line (n1, n2):

if LineBlockCapacity[n1,n2] exists AND > 0:
    use LineBlockCapacity[n1,n2]
else:
    use LineBlockCapacityGlobal
    
if both missing or = 0:
    ❌ Binary expansion DISABLED
```

---

## 📁 Generated Files

After reading Excel, these .tab files created:
- `Transmission_CandidateTransmission.tab` ← Set
- `Transmission_LineBlockCapacity.tab` ← Param (optional)
- `General_LineBlockCapacityGlobal.tab` ← Param (optional)
- (Plus all other usual transmission/general .tab files)

---

## 🎯 Minimal Example

**Minimum setup to test binary TEP**:

```
Transmission.xlsx:
├─ CandidateTransmission (existing)
├─ LineBlockCapacity (new)
│  └─ 3 candidate lines with 500 MW block each
├─ InitialCapacity
├─ MaxBuiltCapacity
└─ ... (other sheets unchanged)

General.xlsx:
└─ ... (all as before, no change needed)
```

**If LineBlockCapacity missing**: Add LineBlockCapacityGlobal with value 400

---

## 🚦 Pre-Run Checklist

Before running model:

- [ ] `CandidateTransmission` sheet populated with candidate lines
- [ ] At least one block capacity defined:
  - [ ] `LineBlockCapacity` in Transmission.xlsx, OR
  - [ ] `LineBlockCapacityGlobal` in General.xlsx
- [ ] All node names match exactly across sheets
- [ ] Block capacity values > 0
- [ ] No missing/corrupt Excel files
- [ ] Header rows skipped correctly (2 rows)

---

## 🔗 Relationships

```
CandidateTransmission (Set)
    ↓
transmissionLineBlockCap (Param) ← From LineBlockCapacity sheet
    ↓
transmissionLineBlockCapGlobal (Param) ← From LineBlockCapacityGlobal
    ↓
transmissionBuild (Binary Var) ← One per candidate per period
    ↓
transmissionInstalledCap = initCap + Σ(build × block)
    ↓
Objective Cost = invCost × block × build
```

---

## 💡 Tips

- **Flexible block sizes**: Different sizes per line? Use LineBlockCapacity
- **Uniform blocks**: Same size everywhere? Use LineBlockCapacityGlobal
- **Both?** Per-line takes priority, global is fallback
- **Node name issues?** Check Sets/Nodes vs Transmission sheets
- **Capacity not growing?** Check if block > 0 and candidate in set

---

**Quick Start**: Add `CandidateTransmission` sheet + one block capacity = ✅ Binary TEP ready!

