# Binary TEP Documentation Index

Complete guide to the binary transmission expansion planning modifications made to EMPIRE model.

---

## 📚 Documentation Files

### 1. **BINARY_TEP_SHEETS_REQUIRED.md** ⭐ START HERE
   **Quick Answer**: "Which Excel sheets must exist?"
   
   - Minimal requirements
   - Column specifications
   - Data precedence logic
   - Common mistakes
   - Implementation steps
   
   **Best for**: Quick understanding of what data is needed

---

### 2. **BINARY_TEP_INPUT_DATA_REQUIREMENTS.md**
   **Complete Specification**: All input data details
   
   - Excel file-by-file breakdown
   - Sheet-by-sheet specifications
   - Format requirements (headers, columns, types)
   - Data validation rules
   - Generated .tab files
   - Integration with model
   - Troubleshooting guide
   - Checklist for migration
   
   **Best for**: Detailed reference and validation

---

### 3. **BINARY_TEP_EXAMPLE_DATA.md**
   **Real Examples**: Complete sample sheets
   
   - Simple 3-node example
   - Medium 6-node European system
   - Examples with global fallback
   - Mixed per-line and global
   - Real-world Norway-Sweden-Denmark case
   - Stochastic scenario
   - Data validation concepts
   - Migration from continuous to binary
   
   **Best for**: Seeing actual data layouts and structures

---

### 4. **BINARY_TEP_EXCEL_QUICK_CARD.md**
   **One-Page Reference**: Visual quick card
   
   - Must-have sheets (visual tables)
   - Format specifications
   - Column names and types
   - Data precedence diagram
   - Common mistakes table
   - Relationships and flow
   - Tips and tricks
   - Pre-run checklist
   
   **Best for**: Quick lookup while working

---

### 5. **TRANSMISSION_BINARY_CONVERSION_SUMMARY.md**
   **Technical Details**: Model changes and implementation
   
   - Overview of modifications
   - Key changes per file
   - Data requirements
   - Behavior summary
   - Validation checklist
   - Backward compatibility
   - Next steps
   
   **Best for**: Understanding what changed in the code

---

### 6. **BINARY_TEP_QUICK_REFERENCE.md**
   **Visual Concepts**: Comparison and diagrams
   
   - Before/after concept comparison
   - Decision variables table
   - Capacity evolution examples
   - Constraint differences
   - Objective function differences
   - Data files needed
   - Code changes summary
   - Advantages of binary
   - Common issues & solutions
   - Testing checklist
   
   **Best for**: Understanding the concept changes

---

## 🎯 Reading Guide by Use Case

### "I need to set up data for binary TEP"
1. Read: **BINARY_TEP_SHEETS_REQUIRED.md** (5 min)
2. Reference: **BINARY_TEP_EXAMPLE_DATA.md** (10 min)
3. Prepare Excel files
4. Use: **BINARY_TEP_EXCEL_QUICK_CARD.md** while working

### "I want to understand what changed in the model"
1. Read: **TRANSMISSION_BINARY_CONVERSION_SUMMARY.md** (15 min)
2. Read: **BINARY_TEP_QUICK_REFERENCE.md** (10 min)
3. Review: Modified files (investment.py, objective.py, results.py)

### "I'm troubleshooting a problem"
1. Check: **BINARY_TEP_INPUT_DATA_REQUIREMENTS.md** → "Common Issues & Solutions"
2. Verify: Excel data structure against **BINARY_TEP_EXAMPLE_DATA.md**
3. Run: Pre-run checklist from **BINARY_TEP_EXCEL_QUICK_CARD.md**
4. Reference: **BINARY_TEP_QUICK_REFERENCE.md** → "Common Issues & Solutions"

### "I'm migrating from continuous to binary"
1. Read: **BINARY_TEP_QUICK_REFERENCE.md** → Concept differences
2. Follow: **BINARY_TEP_INPUT_DATA_REQUIREMENTS.md** → Migration checklist
3. Implement: Based on **BINARY_TEP_EXAMPLE_DATA.md** → Migration example

### "I need a quick reference while coding"
→ Use: **BINARY_TEP_EXCEL_QUICK_CARD.md** (bookmark it!)

---

## 📋 Key Concepts at a Glance

### What Changed?

**Before**: 
- Continuous investment variable: `transmissionInvCap[n1,n2,i] ∈ ℝ⁺` (any MW)
- Installed cap = initCap + Σ invCap
- Cost = invCost × invCap

**After**:
- Binary build variable: `transmissionBuild[n1,n2,i] ∈ {0,1}` (yes/no)
- Installed cap = initCap + Σ (block × build)
- Cost = invCost × block × build

### Required Excel Data

**Minimum**:
1. `CandidateTransmission` sheet → define expandable lines
2. Block capacity (choose one):
   - `LineBlockCapacity` sheet (per-line), OR
   - `LineBlockCapacityGlobal` sheet (fallback)

### Files Modified

1. `investment.py` - Variable definitions and constraints
2. `objective.py` - Investment cost calculation
3. `results.py` - Output generation
4. `out_of_sample_functions.py` - Parameter definitions
5. `test_equivalence.py` - Test updates

### Model Behavior

- **Non-candidates**: Fixed capacity (no investment)
- **Candidates**: Binary builds, capacity = initCap + Σ (block × build)
- **Objective**: Minimize discounted costs (generators + storage + transmission)

---

## 🔧 Quick Setup (5 Minutes)

### Step 1: Check Excel (2 min)
- [ ] Identify candidate transmission lines (which can expand?)
- [ ] Determine block sizes (MW per build decision)

### Step 2: Create Sheets (2 min)
- [ ] Add `CandidateTransmission` to Transmission.xlsx
  - Columns: FromNode, ToNode
- [ ] Add block capacity data (choose one):
  - Option A: `LineBlockCapacity` in Transmission.xlsx
  - Option B: `LineBlockCapacityGlobal` in General.xlsx

### Step 3: Run Model (1 min)
- [ ] Model automatically reads sheets
- [ ] Generates .tab files
- [ ] Runs optimization with binary TEP

### Result
✅ Binary transmission expansion enabled!

---

## 📊 Documentation Statistics

| Document | Pages | Read Time | Purpose |
|----------|-------|-----------|---------|
| BINARY_TEP_SHEETS_REQUIRED.md | 4 | 5 min | Quick answer |
| BINARY_TEP_INPUT_DATA_REQUIREMENTS.md | 12 | 20 min | Complete spec |
| BINARY_TEP_EXAMPLE_DATA.md | 10 | 15 min | Real examples |
| BINARY_TEP_EXCEL_QUICK_CARD.md | 3 | 3 min | Quick card |
| TRANSMISSION_BINARY_CONVERSION_SUMMARY.md | 8 | 15 min | Tech details |
| BINARY_TEP_QUICK_REFERENCE.md | 8 | 10 min | Concepts |

**Total**: ~55 pages, ~70 minutes comprehensive reading

**Quick start**: ~13 pages, ~10 minutes essential reading

---

## ✅ Validation Checklist

Before running model:

**Data Structure**:
- [ ] CandidateTransmission sheet has FromNode, ToNode columns
- [ ] At least one block capacity source available
- [ ] All node names match across sheets
- [ ] Block values > 0

**Excel Format**:
- [ ] First 2 rows are headers (auto-skipped)
- [ ] Data starts at row 3
- [ ] No merged cells
- [ ] Consistent spacing

**Content**:
- [ ] Candidate lines make sense (realistic expansion corridors)
- [ ] Block sizes are realistic (300-800 MW typical)
- [ ] InitialCapacity makes sense
- [ ] MaxBuiltCapacity > 0 (allows builds)

---

## 🔗 Related Files in Repository

### Code Files Modified
- `empire/core/optimization/investment.py` - Binary variable definition
- `empire/core/optimization/objective.py` - Cost calculation
- `empire/core/optimization/results.py` - Output generation
- `empire/core/optimization/out_of_sample_functions.py` - Parameters
- `tests/unit/core/benders/test_equivalence.py` - Tests

### Configuration
- `config/testrun.yaml` - Example configuration
- `config/run.yaml` - Main configuration

### Data
- `Data handler/` - Sample datasets
- `Results/` - Output storage

---

## 💬 Common Questions Answered

**Q: Must I use binary TEP or is continuous still available?**  
A: Binary is now the only option. Continuous variable removed.

**Q: What if I don't have per-line block capacity?**  
A: Use global block capacity fallback.

**Q: Can block size vary by period?**  
A: Not in current version (future enhancement possible).

**Q: How do I disable transmission expansion?**  
A: Don't include lines in CandidateTransmission set.

**Q: What if I don't have any candidates?**  
A: CandidateTransmission can be empty; non-candidates stay fixed.

**Q: Can I have both candidate and non-candidate lines?**  
A: Yes - candidates get binary builds, others stay fixed.

**Q: Is backward compatibility maintained?**  
A: Yes for out-of-sample runs (transmissionInvCap computed from capacity).

---

## 📞 Support Resources

For each issue type:

| Issue | Document | Section |
|-------|----------|---------|
| "I don't know what data to provide" | BINARY_TEP_SHEETS_REQUIRED.md | Top section |
| "What columns should sheet have?" | BINARY_TEP_EXAMPLE_DATA.md | Examples |
| "What format for block capacity?" | BINARY_TEP_INPUT_DATA_REQUIREMENTS.md | Data Format |
| "My model failed, what's wrong?" | BINARY_TEP_INPUT_DATA_REQUIREMENTS.md | Common Issues |
| "How do I migrate existing data?" | BINARY_TEP_INPUT_DATA_REQUIREMENTS.md | Migration |
| "Show me a real example" | BINARY_TEP_EXAMPLE_DATA.md | Examples |
| "What's the quick reference?" | BINARY_TEP_EXCEL_QUICK_CARD.md | Quick Card |

---

## 🎓 Learning Path

**Level 1 - Beginner** (10 min)
- BINARY_TEP_SHEETS_REQUIRED.md
- BINARY_TEP_EXCEL_QUICK_CARD.md

**Level 2 - Intermediate** (30 min)
- + BINARY_TEP_EXAMPLE_DATA.md
- + BINARY_TEP_QUICK_REFERENCE.md

**Level 3 - Advanced** (60 min)
- + BINARY_TEP_INPUT_DATA_REQUIREMENTS.md
- + TRANSMISSION_BINARY_CONVERSION_SUMMARY.md
- + Code review (investment.py, objective.py, results.py)

**Level 4 - Expert** (120+ min)
- Detailed code analysis
- Constraint formulation review
- Model optimization
- Extension implementation

---

## 🚀 Quick Start Path

```
START
  ↓
Read: BINARY_TEP_SHEETS_REQUIRED.md (5 min)
  ↓
Understand: 3 required things
  ├─ CandidateTransmission sheet
  ├─ Block capacity (per-line or global)
  └─ Node names consistent
  ↓
Check: BINARY_TEP_EXAMPLE_DATA.md (10 min)
  ├─ See example sheets
  ├─ Copy structure to your Excel
  └─ Fill in your data
  ↓
Use: BINARY_TEP_EXCEL_QUICK_CARD.md (1 min)
  ├─ Quick reference while working
  ├─ Validate data format
  └─ Pre-run checklist
  ↓
RUN MODEL ✅
  ↓
Success? → Done!
Error? → Check BINARY_TEP_INPUT_DATA_REQUIREMENTS.md "Common Issues"
```

---

## 📝 Document Versioning

| Document | Version | Date | Status |
|----------|---------|------|--------|
| BINARY_TEP_SHEETS_REQUIRED.md | 1.0 | Nov 7, 2025 | ✅ Current |
| BINARY_TEP_INPUT_DATA_REQUIREMENTS.md | 1.0 | Nov 7, 2025 | ✅ Current |
| BINARY_TEP_EXAMPLE_DATA.md | 1.0 | Nov 7, 2025 | ✅ Current |
| BINARY_TEP_EXCEL_QUICK_CARD.md | 1.0 | Nov 7, 2025 | ✅ Current |
| TRANSMISSION_BINARY_CONVERSION_SUMMARY.md | 1.0 | Nov 6, 2025 | ✅ Current |
| BINARY_TEP_QUICK_REFERENCE.md | 1.0 | Nov 6, 2025 | ✅ Current |

---

## 🎯 Next Steps

1. **Choose your reading path** based on use case (see above)
2. **Prepare your Excel files** using examples
3. **Run the model** with binary TEP
4. **Bookmark** the quick card for reference
5. **Refer to** relevant documents when needed

---

**Main Question Answered**: 

"Which input data and sheet must exist in Excel files?"

**Answer**:
1. ✅ `Transmission.xlsx` → `CandidateTransmission` sheet (2 columns: FromNode, ToNode)
2. ✅ Block capacity (at least one):
   - `Transmission.xlsx` → `LineBlockCapacity` sheet (3 columns), OR
   - `General.xlsx` → `LineBlockCapacityGlobal` sheet (1 value)
3. ✅ All other existing Excel sheets (unchanged)

**See**: BINARY_TEP_SHEETS_REQUIRED.md for detailed information

---

**Date**: November 7, 2025  
**Status**: ✅ Complete Documentation Set
