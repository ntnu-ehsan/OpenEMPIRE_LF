# Binary TEP - Quick Reference

## Key Concept

### Before (Continuous)
```
transmissionInvCap[n1,n2,i] ∈ ℝ⁺    (any non-negative MW value)
installedCap[i] = initCap + Σ invCap[j] for j ≤ i
Cost = invCost × invCap
```

### After (Binary)
```
transmissionBuild[n1,n2,i] ∈ {0,1}  (build or don't build)
installedCap[i] = initCap + Σ (block × build[j]) for j ≤ i
Cost = invCost × block × build
```

---

## Decision Variables

| Type | Variable | Domain | Set | Meaning |
|------|----------|--------|-----|---------|
| **Before** | `transmissionInvCap` | ℝ⁺ | BidirectionalArc | MW to add this period |
| **After** | `transmissionBuild` | {0,1} | CandidateTransmission | Binary: build block or not |

---

## Capacity Evolution Example

**Line: NO → SE, Block size: 500 MW, Initial: 1000 MW**

| Period | Build Decision | Capacity Added | Installed Capacity |
|--------|---------------|----------------|-------------------|
| 1 | 0 | 0 MW | 1000 MW |
| 2 | 1 | 500 MW | 1500 MW |
| 3 | 0 | 0 MW | 1500 MW |
| 4 | 1 | 500 MW | 2000 MW |

### Continuous (old):
```
invCap[1] = 0 MW       → installedCap[1] = 1000 + 0 = 1000 MW
invCap[2] = 327 MW     → installedCap[2] = 1000 + 327 = 1327 MW
invCap[3] = 173 MW     → installedCap[3] = 1000 + 327 + 173 = 1500 MW
invCap[4] = 500 MW     → installedCap[4] = 1000 + 327 + 173 + 500 = 2000 MW
```

### Binary (new):
```
build[1] = 0, block = 500 → installedCap[1] = 1000 + 0×500 = 1000 MW
build[2] = 1, block = 500 → installedCap[2] = 1000 + 1×500 = 1500 MW
build[3] = 0, block = 500 → installedCap[3] = 1000 + 1×500 = 1500 MW
build[4] = 1, block = 500 → installedCap[4] = 1000 + 2×500 = 2000 MW
```

**Advantage**: Realistic discrete capacity additions (cables, transformers come in standard sizes)

---

## Constraint Differences

### Non-Candidate Lines (existing infrastructure)
```python
# Fixed capacity - no investment allowed
transmissionInstalledCap[n1,n2,i] == transmissionInitCap[n1,n2,i]
```

### Candidate Lines (expansion options)

**OLD - Continuous**:
```python
installedCap[n1,n2,i] = initCap[n1,n2,i] + Σ invCap[n1,n2,j] for j≤i
invCap[n1,n2,i] ≤ maxBuiltCap[n1,n2,i]
```

**NEW - Binary**:
```python
installedCap[n1,n2,i] = initCap[n1,n2,i] + Σ (block × build[n1,n2,j]) for j≤i
block × build[n1,n2,i] ≤ maxBuiltCap[n1,n2,i]
```

---

## Objective Function

### OLD - Continuous
```python
Σ discountMultiplier[i] × (
    Σ transmissionInvCost[n1,n2,i] × transmissionInvCap[n1,n2,i]
    for all (n1,n2) in BidirectionalArc
)
```

### NEW - Binary
```python
Σ discountMultiplier[i] × (
    Σ transmissionInvCost[n1,n2,i] × block[n1,n2] × transmissionBuild[n1,n2,i]
    for (n1,n2) in CandidateTransmission only
)
```

**Key difference**: Cost multiplied by block size (since invCost is per MW)

---

## Data Files Needed

### Required for Binary TEP

1. **CandidateTransmission** (in Transmission.xlsx or as .tab)
   ```
   FromNode  ToNode
   NO        SE
   SE        FI
   DK        DE
   ```

2. **Block Capacity** (at least one of):
   
   a. **Per-line** (preferred): `Transmission_LineBlockCapacity.tab`
   ```
   FromNode  ToNode  LineBlockCap
   NO        SE      500
   SE        FI      300
   DK        DE      400
   ```
   
   b. **Global fallback**: `Transmission_LineBlockCapacityGlobal.tab`
   ```
   transmissionLineBlockCapGlobal
   400
   ```

### Optional (existing files still used)
- `Transmission_InitialCapacity.tab`
- `Transmission_MaxBuiltCapacity.tab`
- `Transmission_MaxInstallCapacityRaw.tab`
- All other transmission parameter files

---

## Code Changes Summary

| File | Change | Purpose |
|------|--------|---------|
| `investment.py` | Removed `transmissionInvCap` Var | No longer continuous decision |
| `investment.py` | Updated `installedCapDefinitionTrans` | Use binary builds × block |
| `investment.py` | Enabled `investment_trans_cap` | Limit per-period builds |
| `objective.py` | Updated transmission term | Multiply by block size |
| `results.py` | Added `transmissionBuild.tab` output | Export binary decisions |
| `results.py` | Compute `transmissionInvCap` from Δcapacity | Backward compatibility |
| `results.py` | Fix binary builds in operational | Was fixing non-existent var |
| `out_of_sample_functions.py` | Document `transmissionInvCap` as Param | Clarify purpose |
| `test_equivalence.py` | Check `transmissionBuild` | Update test logic |

---

## Advantages of Binary TEP

1. **Realistic**: Transmission lines come in discrete sizes (cables, transformers)
2. **Integer optimization**: Matches real-world project decisions (build entire line or not)
3. **LOPF compatible**: Binary builds properly activate/deactivate DC power flow constraints
4. **Clearer economics**: Fixed cost per project (invCost × blockSize) easier to interpret
5. **Easier validation**: Binary decisions (0/1) simpler to verify than arbitrary MW values

---

## Common Issues & Solutions

### Issue: Model unbounded/infeasible
- **Check**: CandidateTransmission set is populated
- **Check**: Block capacity > 0 (either per-line or global)
- **Check**: MaxBuiltCap allows at least one block

### Issue: No transmission investment
- **Check**: Candidate lines are defined in CandidateTransmission set
- **Check**: Initial capacity < demand growth
- **Check**: Block size isn't too large for the system

### Issue: "KeyError: transmissionLineBlockCap"
- **Solution**: Provide `Transmission_LineBlockCapacityGlobal.tab` as fallback
- **Or**: Add LineBlockCapacity column to Transmission.xlsx

### Issue: Out-of-sample runs fail
- **Check**: Previous run generated `transmissionInvCap.tab`
- **Check**: `transmissionInstalledCap.tab` exists and is complete

---

## Testing Checklist

Before considering the conversion complete:

- [ ] Model builds without errors
- [ ] Solver finds optimal solution
- [ ] Binary variables show expected 0/1 values
- [ ] Installed capacity increases by block_size when build=1
- [ ] No capacity change when build=0
- [ ] Investment costs = invCost × block × build
- [ ] Non-candidate lines stay at initial capacity
- [ ] LOPF constraints activate/deactivate with binary builds
- [ ] Out-of-sample runs complete successfully
- [ ] Results visualization shows discrete capacity additions

---

**Quick Start**: To run with binary TEP, ensure you have:
1. `CandidateTransmission` set defined
2. At least one block capacity parameter (per-line or global)
3. All existing transmission data files

The model will automatically use binary decisions for candidate lines and keep existing lines fixed!
