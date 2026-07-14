# Data specification: national-level generation limits for NUTS-disaggregated EMPIRE datasets

**Purpose of this document.** Instructions for preparing an OpenEMPIRE input dataset
(the folder of Excel workbooks: `Sets.xlsx`, `Generator.xlsx`, `Node.xlsx`,
`Storage.xlsx`, `Transmission.xlsx`, `General.xlsx`, `ScenarioData/`) that supports
**national-level** limits on generation capacity in datasets where some countries have
been disaggregated into NUTS regions. The model code that consumes these sheets lives
on the `NUTS` branch; the data must follow this spec exactly.

---

## 1. Background and reasoning — why these changes are needed

In EMPIRE, every spatial unit is a **node**. In the original datasets one country was
one node (e.g. `Denmark`), so sheets like `MaxBuiltCapacity` and `MaxInstalledCapacity`
in `Generator.xlsx`, which are indexed by node, effectively expressed *national* limits.

In the NUTS-disaggregated datasets (e.g. `NESP_GoRES`), several countries were split
into NUTS-level nodes (`DK01…DK05`, `FI19/FI1B/FI1C/FI1D`, `NO02/NO06/NO07/NO08/NO09/NO0A`,
`SE11…SE33`), while other countries remain single nodes (`France`, `Germany`, `Austria`, …).
This breaks every limit that is national in nature:

- **Resource/policy ceilings** (max installed capacity): e.g. a national nuclear cap of
  12 GW. Applied per NUTS node, the model could build 12 GW in *each* region. Spread
  as 12/N GW per region, the model loses the freedom to concentrate capacity where it
  is most valuable. Neither is correct — the cap must bind on the **sum over the
  country's regions**.
- **Fuel-supply-type ceilings**: e.g. limited national gas availability capping total
  gas capacity. Same problem: the limit belongs to the country, not to any one region.
- **Build-rate limits** (max built capacity per investment period): supply chains,
  permitting and construction capacity are national. Per-node copies of the national
  rate would multiply the allowed build rate by the number of regions.
- **Policy-mandated build-out** (min built capacity per investment period): national
  plans (e.g. NECP targets: "at least X GW of offshore wind commissioned by period Y")
  mandate deployment at country level, but should not dictate *which* region hosts it —
  the model should choose the optimal regions subject to the national floor.

Therefore three limits become definable at **country level**, on top of the existing
nodal sheets which remain in place:

| National limit | Meaning | Analogous nodal sheet |
|---|---|---|
| `MaxInstalledCapacityCountry` | Total installed MW of a technology across the country, every period | `MaxInstalledCapacity` (Generator.xlsx) |
| `MaxBuiltCapacityCountry` | MW of a technology the country may **invest in within one investment period** | `MaxBuiltCapacity` (Generator.xlsx) |
| `MinBuiltCapacityCountry` | MW of a technology the country **must at least invest in within one investment period** | *(none — new concept, no nodal counterpart exists in EMPIRE)* |

To connect NUTS nodes to countries, two new **set** sheets are added to `Sets.xlsx`.
The mapping is explicit (a table), not inferred from node-name prefixes, because node
names are heterogeneous (`DK01`, but also `Czech R`, `Bosnia H`, `Great Brit.`,
offshore nodes like `Sorlige Nordsjo I`) and an explicit table is self-documenting
and unambiguous.

---

## 2. General formatting rules (apply to all new sheets)

These match the conventions of the existing workbooks; the EMPIRE reader
(`empire/core/reader.py`) depends on them:

1. **Single-column set sheets** (like existing `Nodes`, `Technology`): the header word
   in cell A1, values from A2 down. No Source/Description rows.
2. **Multi-column sheets** (like existing `GeneratorsOfNode`, `MaxBuiltCapacity`):
   - Row 1, cell A1: `Source: <data source>`
   - Row 2, cell A2: `Description: <one-line description>`
   - Row 3: the column headers
   - Row 4 onward: data. No blank rows inside the data block, no merged cells,
     no formulas (paste values), no extra columns to the right of the data.
3. **Exact string matching.** Node names must match the `Nodes` sheet of `Sets.xlsx`
   character-for-character (including spaces and dots: `Czech R`, `Great Brit.`,
   `Luxemb.`, `NO0A`). Technology names must match the `Technology` sheet exactly.
   Country names must be consistent across all new sheets. A typo does not raise an
   error — it silently weakens or drops a constraint. Trailing whitespace counts as
   a mismatch.
4. **Units:** all capacity values in **MW**.
5. **Periods:** integer investment-period indices matching the `Horizon` sheet of
   `Sets.xlsx` (e.g. 1–7 in NESP_GoRES). Not calendar years.

---

## 3. New sheets in `Sets.xlsx`

### 3.1 Sheet `Countries` (single-column set sheet)

|   | A |
|---|---|
| 1 | `Country` |
| 2 | `Denmark` |
| 3 | `Finland` |
| 4 | `Norway` |
| 5 | `Sweden` |

- List **only** countries that will carry at least one national limit. Countries not
  listed are simply untouched by this feature.
- Country names are new identifiers (they need not be node names). `Denmark` is valid
  even though no node is named `Denmark`.
- A country may also be a non-disaggregated one (e.g. `Netherlands`) if a national
  limit is wanted for it; it then maps to its single node in `NodesOfCountry`.

### 3.2 Sheet `NodesOfCountry` (multi-column sheet)

|   | A | B |
|---|---|---|
| 1 | `Source: -` | |
| 2 | `Description: Assignment of nodes to countries for national-level limits` | |
| 3 | `Country` | `Node` |
| 4 | `Denmark` | `DK01` |
| 5 | `Denmark` | `DK02` |
| … | … | … |

Full mapping for the NESP-style datasets (adjust to the actual `Nodes` sheet of the
target dataset):

| Country | Nodes |
|---|---|
| Denmark | DK01, DK02, DK03, DK04, DK05 |
| Finland | FI19, FI1B, FI1C, FI1D |
| Norway  | NO02, NO06, NO07, NO08, NO09, NO0A |
| Sweden  | SE11, SE12, SE21, SE22, SE23, SE31, SE32, SE33 |

Rules:
- **Completeness is critical.** Every onshore node of a listed country must appear.
  National constraints are sums over exactly these nodes; a forgotten region is a
  loophole (for Max limits) or an unfair burden on the other regions (for Min limits).
- Each node belongs to **at most one** country.
- **Offshore nodes** (`Nordsoen`, `Utsira Nord`, `Sorlige Nordsjo I/II`,
  `Dogger Bank`, …): include one under a country **only if** its capacity should count
  toward that country's national limits. Default recommendation: leave offshore nodes
  out unless a national limit on offshore wind is intended, in which case assign each
  offshore node to the country whose budget it consumes.
- A non-disaggregated country maps to itself, e.g. `Netherlands | Netherlands`.

---

## 4. New sheets in `Generator.xlsx`

All three use the multi-column format (Source row, Description row, header row, data).
In all three, **column B is a technology** from the `Technology` set (in NESP_GoRES:
`CCS`, `Lignite`, `Hcoal`, `Coal`, `Gas_OCGT`, `Oil`, `Bio`, `Geo`, `Nuclear`,
`Hydro_reg`, `Hydro_ror`, `Wave`, `Wind_onshr`, `Wind_offshr_grounded`,
`Wind_offshr_floating`, `Solar`, `Waste`, `Gas_CCGT`) — **not** an individual generator
name. A limit on a technology covers the sum over all generators mapped to it in
`GeneratorsOfTechnology` (e.g. a `Gas_CCGT` limit covers plain and CCS-retrofit CCGT
generators alike, if both are mapped to that technology).

### 4.1 Sheet `MaxInstalledCapacityCountry` — national resource/policy ceiling

|   | A | B | C |
|---|---|---|---|
| 1 | `Source: <e.g. national policy, NECP>` | | |
| 2 | `Description: Maximum total installed capacity of a technology across all nodes of a country, enforced in every period (only listed pairs constrained)` | | |
| 3 | `Country` | `GeneratorTechnology` | `generatorMaxInstallCapacity in MW` |
| 4 | `Sweden` | `Nuclear` | `12000` |
| 5 | `Finland` | `Nuclear` | `6000` |
| 6 | `Denmark` | `Nuclear` | `0` |
| 7 | `Norway` | `Gas_OCGT` | `1500` |

*(values are placeholders — insert real national limits)*

Semantics:
- One row per (Country, Technology). **No Period column** — the ceiling applies in
  every period (it is a resource/policy limit, mirroring the nodal
  `MaxInstalledCapacity` sheet which is also period-independent).
- Constraint: Σ installed capacity of that technology over the country's nodes ≤ value,
  in every investment period.
- **A missing (Country, Technology) pair is unconstrained.** Only add rows for limits
  that really exist. (Note: this is deliberately the *opposite* of the nodal
  `MaxInstalledCapacity` sheet, where a missing row freezes the node at initial
  capacity.)
- `0` means "no capacity allowed"; if the country has pre-existing (initial) capacity,
  the model clamps the effective limit up to that initial capacity so existing plants
  can live out their lifetime — but nothing new is built.

### 4.2 Sheet `MaxBuiltCapacityCountry` — national build-rate ceiling per period

|   | A | B | C | D |
|---|---|---|---|---|
| 1 | `Source: <e.g. supply-chain / permitting assessment>` | | | |
| 2 | `Description: Maximum capacity expansion of a technology across all nodes of a country in one investment period (only listed triples constrained)` | | | |
| 3 | `Country` | `GeneratorTechnology` | `Period` | `generatorMaxBuildCapacity in MW` |
| 4 | `Sweden` | `Nuclear` | `1` | `0` |
| 5 | `Sweden` | `Nuclear` | `2` | `1600` |
| 6 | `Sweden` | `Nuclear` | `3` | `3200` |
| 7 | `Denmark` | `Wind_offshr_grounded` | `1` | `3000` |

Semantics:
- One row per (Country, Technology, Period). Period-indexed, mirroring the nodal
  `MaxBuiltCapacity` sheet (`Node, GeneratorTechnology, Period, MW`).
- Constraint: Σ **new investment** in that technology over the country's nodes in that
  period ≤ value. This limits the *rate* of expansion, not the total stock (the stock
  is what `MaxInstalledCapacityCountry` limits).
- **A missing (Country, Technology, Period) triple is unconstrained.** If a build-rate
  limit should apply in every period, a row must be written for every period in the
  `Horizon` sheet.

### 4.3 Sheet `MinBuiltCapacityCountry` — national mandated build-out per period

|   | A | B | C | D |
|---|---|---|---|---|
| 1 | `Source: <e.g. NECP target, national energy plan>` | | | |
| 2 | `Description: Minimum capacity expansion of a technology across all nodes of a country in one investment period (only listed triples constrained)` | | | |
| 3 | `Country` | `GeneratorTechnology` | `Period` | `generatorMinBuildCapacity in MW` |
| 4 | `Denmark` | `Wind_offshr_grounded` | `2` | `2000` |
| 5 | `Sweden` | `Wind_onshr` | `1` | `1000` |

Semantics:
- One row per (Country, Technology, Period). Same layout as 4.2.
- Constraint: Σ **new investment** in that technology over the country's nodes in that
  period ≥ value. The model chooses freely *which* NUTS regions host it.
- **A missing triple means 0 (no mandated build).** Omit rows entirely where nothing
  is mandated — do not fill zeros.
- **This is a hard floor and can make the model infeasible or force uneconomic
  investment.** It has *no nodal counterpart in EMPIRE* — it is a new concept
  introduced for national policy targets. Use it only for genuine commitments.
- If the real-world target is cumulative ("X GW installed *by* period Y" rather than
  "X GW built *in* period Y"), convert it to per-period increments before filling the
  sheet: mandated build in period p = max(0, target(p) − target(p−1) − expected initial
  capacity already covering the target). Document the conversion in the Source row.

---

## 5. Required consistency checks (run after filling the sheets)

The model will apply safety clamps where it can, but the data should be consistent by
construction. Verify:

1. **Name integrity.** Every `Node` in `NodesOfCountry` exists in `Sets.xlsx → Nodes`.
   Every `Country` in the three Generator sheets exists in `Sets.xlsx → Countries` and
   has at least one row in `NodesOfCountry`. Every `GeneratorTechnology` exists in
   `Sets.xlsx → Technology`. Every `Period` exists in `Sets.xlsx → Horizon`.
2. **Nodal ceilings must not undercut national ones.** For every
   (Country, Technology) in `MaxInstalledCapacityCountry`: each node of that country
   should have a row in the **nodal** `MaxInstalledCapacity` sheet with a generous
   value (≥ the national limit is the simple safe choice). Reason: in the nodal sheet
   a *missing* row freezes the node at its initial capacity, so nodal limits would
   silently bind before the national one. (In `NESP_GoRES` this already holds —
   NUTS nodes carry 200 000 MW placeholders for Nuclear/Gas/etc. Keep genuinely
   resource-based nodal values — regional Wind/Solar/Hydro potentials — exactly as
   they are; they are correct at NUTS level and coexist with national caps.)
3. **Min vs Max coherence.** For every (Country, Technology, Period) in
   `MinBuiltCapacityCountry`:
   - value ≤ the corresponding `MaxBuiltCapacityCountry` value, if one exists;
   - value ≤ Σ over the country's nodes of the nodal `MaxBuiltCapacity` (missing nodal
     rows default to 500 000 MW, i.e. effectively unlimited, so only *explicit* small
     nodal values can conflict);
   - the cumulative mandated build must not exceed `MaxInstalledCapacityCountry`
     minus retirements — as a simple safe check: Σ over all periods of the min-build
     values ≤ the national `MaxInstalledCapacityCountry` for the same pair, if one
     exists.
4. **Ceiling vs installed reality.** For every `MaxInstalledCapacityCountry` row,
   compare against the country's summed initial capacity (`InitialCapacity` ×
   `ScaleFactorInitialCap` per node/generator): a limit below existing capacity is
   allowed (phase-out) but should be intentional, not accidental.
5. **No duplicate index rows** in any of the new sheets (a duplicated
   (Country, Tech[, Period]) row is a data error).

---

## 6. What must NOT change

- All existing sheets in all workbooks keep their current structure and content
  (subject only to check 5.2 above).
- The nodal `MaxInstalledCapacity` and `MaxBuiltCapacity` sheets stay — nodal and
  national limits are enforced simultaneously (both must hold).
- Datasets that omit all five new sheets remain valid: the model treats absent sheets
  as "national limits disabled".

## 7. Summary of missing-row semantics (quick reference)

| Sheet | Missing row means |
|---|---|
| `MaxInstalledCapacity` (nodal, existing) | Node frozen at initial capacity (no expansion) |
| `MaxBuiltCapacity` (nodal, existing) | Effectively unlimited (default 500 000 MW/period) |
| `MaxInstalledCapacityCountry` (new) | Unconstrained |
| `MaxBuiltCapacityCountry` (new) | Unconstrained |
| `MinBuiltCapacityCountry` (new) | No mandated build (0) |
