# OpenEMPIRE NUTS2 Performance Investigation — Handoff

> **STATUS (2026-06-12, Fable 5):** Recommended actions 1 and 3 are implemented:
> - `empire/core/empire.py`: emission cap reformulated — per-node `nodeEmission`
>   variables (tonnes) feed the cap row, removing the dense all-generators row and
>   the 5e-8 coefficients from the matrix. CO2-price dual reporting updated to match
>   (the `*1e6` divisor removed in both report blocks).
> - `config/run.yaml`: accelerators (`solver_method: 2`, `solver_crossover: 0`,
>   `solver_presolve: 2`) enabled; `numericfocus`/`barhomogeneous`/`scaleflag`
>   commented out and documented as enable-only-on-numerical-failure.
> - Bonus bug found & fixed: `results_co2_price_resolved.csv` was never populated —
>   the file ended mid-loop (missing `writer.writerow`/`f.close()` at EOF). Now writes
>   rows with a `CO2Cap_Ton` column.
> Verified on the `test` dataset with `use_emission_cap: True`: clean solve,
> Matrix range [1e-3, 3e+1] (no 5e-8), binding-cap shadow prices in plausible
> EUR/tonne magnitudes, zero when non-binding. Flag reverted to False afterwards.
> Still open: cluster-log tail, node RAM vs Gurobi peak memory (action 2),
> and a re-timed full NESP run with the new formulation.
>
> **UPDATE (2026-06-13): the code fix was never actually exercised on the cluster.**
> Cluster runs kept showing the OLD signature (`Matrix [5e-08, ...]`, and a
> `Non-default parameters` block listing only `Method 2` / `QCPDual 1` — no
> `Crossover 0` / `Presolve 2`). Root cause is an **import-path / worktree mismatch**,
> NOT the model code:
> - The conda env `empire_env` has `empire` installed **editable, pointed at the
>   OLD clone** `/mnt/beegfs/users/ehsanno/openempire` (`pip show empire` →
>   `Editable project location: .../openempire`).
> - `python scripts/run.py` imports `empire` from that editable target regardless of
>   which directory the job runs in. So worktrees `openempire_accel` and
>   `openempire_v2` supplied only their **config + data + Results paths** (resolved
>   from cwd); the **Python code always came from `openempire`**.
> - This is why interactive `cd openempire_v2; python -c "import empire..."` looked
>   correct (cwd is on sys.path first) but jobs ran stale code. The reliable test is
>   from a neutral dir: `cd /tmp && python -c "import empire.core.empire as m; print(m.__file__)"`.
> - LOCAL machine (Windows) has it installed editable to the repo, which is why local
>   `test`/`NS_GoRES` runs DID show the new formulation (`Matrix [1e-3, ...]`). The fix
>   itself is verified correct; only the cluster was running old bytes.
>
> **Constraint:** ongoing runs from `openempire` AND `openempire_accel` must not be
> interrupted. Already-running processes hold their modules in memory and are safe,
> but a global `pip install -e` re-point would affect any QUEUED job that starts after.
>
> **Chosen workaround (no global change):** per-job `PYTHONPATH` override in the v2
> submission script (PYTHONPATH wins over site-packages):
> ```bash
> export PYTHONPATH=/mnt/beegfs/users/ehsanno/openempire_v2:$PYTHONPATH
> cd /mnt/beegfs/users/ehsanno/openempire_v2
> python scripts/run.py -d <dataset> -f
> ```
> Verify (no submit needed):
> `cd /tmp && PYTHONPATH=/mnt/.../openempire_v2 python -c "import empire.core.empire as m; print(m.__file__)"`
> Runtime confirmation in the fresh log header: `Crossover 0` + `Presolve 2` in the
> non-default parameters, and NO `5e-08` in the matrix range.
> Also check the sbatch script `cd`s into `openempire_v2` (so config/data/Results match).
>
> **Permanent cleanup once old runs finish:** pick ONE canonical checkout,
> `pip install -e` it, delete/quarantine the others. One shared conda env = exactly
> one import target, no matter how many worktrees exist.

## Goal
Some countries (NO, DK, SE, FI) were upgraded from NUTS0 (1 node each) to NUTS2
multi-node resolution. The resulting datasets (the `NESP_*` family, e.g.
`Data handler/NESP_GoRES`) solve extremely slowly in Gurobi. A full NUTS2 run
(`NESP`, horizon **2045**, **all** solver options in `config/run.yaml` enabled)
ran for **~122,000 s (~34 h)** on a cluster.

Original user hypothesis: distributing country-level values down to NUTS2 created
tiny coefficients that cause a numerical-scaling problem. **This hypothesis was
investigated and largely ruled out** — see below. The real story is problem size +
crossover + a dense constraint row, compounded by counter-productive solver flags.

---

## Dataset facts
- `NESP_GoRES` split: NO→6 nodes, DK→5, SE→8, FI→4 (NO02/06/07/08/09/0A, DK01–05,
  FI19/1B/1C/1D, SE11/12/21/22/23/31/32/33).
- `NESP_GoRES/Sets.xlsx`: **64 nodes, 418 directional lines, 654 GeneratorsOfNode,
  130 StorageOfNodes, 14 offshore nodes.**
- Full Excel inputs exist only for the `NESP_*` datasets. Sibling dirs
  (`NS_GoRES`, `Orig_GoRES`, `GoRES`, `Agg_GoRES`) contain only `ScenarioData/`.

---

## Evidence collected

### 1. Input-data scan — NO tiny-value pathology from the NUTS2 split
Scanned every node-indexed sheet (demand, generator/storage/transmission
capacities) and all 6 ScenarioData profiles, comparing the 23 NUTS2 nodes vs
aggregated countries:
- Smallest non-zero capacity anywhere = **0.25 MW**, and it belongs to **Austria**
  (a non-split country), not a NUTS2 node. Smallest NUTS2 capacity ≈ 0.3 MW.
- Profile minima (~1e-4 wind, ~1e-3 solar) are **identical** between NUTS2 nodes and
  aggregated countries (profiles are normalized per-unit, so splitting doesn't
  change them).
- Conclusion: the NUTS2 distribution did **not** inject pathological tiny
  coefficients into the input data.

### 2. Gurobi coefficient statistics (from real run logs)
Logs live under `Results/basic_run/dataset_*/Output/`.

**Test (reg24, sce2)** — `dataset_test/.../logfile_..._202606102152_resolved.log`:
```
81,190 rows / 50,267 cols / 243,773 nonzeros
Matrix range     [1e-03, 1e+01]
Objective range  [2e-01, 1e+12]
Bounds range     [1e+00, 1e+00]
RHS range        [7e+00, 8e+07]
Warning: Model contains large objective coefficients
         Consider reformulating model or setting NumericFocus parameter
         to avoid numerical issues.
Solved in 0.28 s.
```

**NS_GoRES (reg168, sce1)** — `dataset_NS_GoRES/.../...202606050837.log`:
```
1,914,038 rows / 1,470,971 cols / 6,215,498 nonzeros
Matrix range  [5e-08, 1e+01]   Objective [2e-02, 3e+06]   RHS [2e-01, 1e+08]
Barrier solved in 131 iters / 139.65 s
Crossover: 975,907 iterations  ->  TOTAL 1327.11 s   (crossover ≈ 9x barrier)
```

**NS_NECPEssentials (reg168, sce2)** — `dataset_NS_NECPEssentials/.../...202606091345.log`:
```
8,199,318 rows / 6,224,181 cols / 26,685,233 nonzeros
Matrix range  [5e-08, 6e+00]   Objective [6e-03, 5e+06]   RHS [5e-03, 8e+07]
Barrier solved in 239 iters / 1201.41 s
Crossover: 4,081,068 iterations  ->  TOTAL 12,331.25 s   (crossover ≈ 10x barrier)
```

Key reads:
- **Barrier is fast; crossover dominates** (9–10× the barrier time, millions of
  pivots). This is degeneracy/size, not coefficient range.
- The **objective-range warning only fired on the tiny test** (`1e+12`); on the
  big models the objective range was modest (`~3e6`–`5e6`). So objective scaling
  is a real-but-secondary issue.
- The wide **Matrix range `[5e-08, …]`** appears identically in the aggregated
  `NS_NECPEssentials` run → the 5e-8 is structural, NOT a NUTS2 artifact.

### 3. Source of the 5e-08 matrix coefficient + a dense row (likely the real killer)
`empire/core/empire.py:716-720`, the emission-cap constraint:
```python
def emission_cap_rule(model, i, w):
    return sum(model.seasScale[s]*model.genCO2TypeFactor[g]*(3.6/model.genEfficiency[g,i])
               *model.genOperational[n,g,h,i,w]
               for (n,g) in model.GeneratorsOfNode for (s,h) in model.HoursOfSeason)/1000000 \
           - model.CO2cap[i] <= 0
model.emission_cap = Constraint(model.PeriodActive, model.Scenario, rule=emission_cap_rule)
```
- The `/1000000` (CO2 expressed in Mt) is exactly where the `5e-08` coefficients
  come from.
- This is **one dense constraint row per (period, scenario)** summing over EVERY
  generator × EVERY hour operational variable (hundreds of thousands of nonzeros
  per row). Dense rows cause **catastrophic fill-in in the barrier Cholesky
  factor** — the prime suspect for super-linear blow-up as the model grows.

### 4. Objective construction (source of large objective coefficients)
`empire/core/empire.py:577-596`:
- `shedcomponent`: `lostLoadCost(22000 €/MWh) × seasScale × operationalDiscountrate
  × sceProbab × discount_multiplier`.
- Investment terms: `capitalCost ×1000 × annuity` (capital cost up to 9357 €/kW),
  plus a hard-coded `CCSCostTSFix = 1,149,873.72` (`empire.py:197`).
- `seasScale ≈ 12.96` for regular seasons, 1 for peak. `NodeLostLoadCost = 22000`.

---

## Analysis / conclusions
1. **Tiny-value-from-distribution hypothesis: ruled out.** No pathological small
   inputs; the 5e-8 is from the `/1e6` in the emission constraint, not the split.
2. **`ScaleFlag` will not fix this.** It scales the *matrix* (helps `[5e-8,1e1]`),
   not the *objective* range, and does nothing for size/degeneracy/density.
3. **"All options enabled" is likely hurting.** `solver_numericfocus` and
   `solver_barhomogeneous` are *robustness* options that make Gurobi **slower**.
   Stacking them on a huge model adds brakes. `solver_crossover: 0` is the one true
   accelerator (removes the 9–10× crossover cost).
4. **Dominant costs:** (a) sheer size — NUTS2 (×~5 nodes) × horizon-2045 (more
   5-year periods) × scenarios; (b) the dense emission-cap row → barrier fill-in;
   (c) crossover; and possibly (d) **memory swapping** on the cluster node (a
   NUTS2/2045 barrier factor may need hundreds of GB — if it spills to disk that
   alone is ~100× slowdown, which fits 122,000 s).

---

## Recommended next actions (ranked)
1. **Re-time with the brakes off:** keep `solver_method: 2`, `solver_crossover: 0`;
   set `solver_numericfocus` and `solver_barhomogeneous` back to default (absent).
   Likely the single biggest win.
2. **Check memory / swapping** on the cluster node (RAM vs Gurobi peak; set
   `NodefileStart`, or use a larger-RAM node). If it's swapping, no flag matters.
3. **Reformulate, don't band-aid:** express `CO2cap` in tonnes so the `/1e6` leaves
   the matrix (kills the 5e-8 *and* the dense-row conditioning); rescale all costs
   to k€/M€ to compress the objective range. This is what Gurobi's warning actually
   asks for.
4. **Reduce resolution as a fallback / for iteration speed:** fewer representative
   hours (`length_of_regular_season`) or scenarios — barrier cost falls
   super-linearly.

## Open questions / data still needed
- **Where did the 122,000 s go?** Need the tail of the cluster log from
  `Optimize a model with …` onward: the `rows, columns, nonzeros` line, the four
  `range` lines, and whether it was stuck in *Presolve* / *Barrier iterations* /
  *Crossover*. This decides which fix matters most.
- **How many periods does horizon 2045 produce?** (Need start year;
  `leap_years_investment: 5`.)
- **Cluster node RAM and Gurobi's reported peak memory** — to confirm/rule out swap.
- **Was `crossover` actually 0 in the cluster run?** If crossover ran, that alone
  could explain the 34 h.

## Relevant files
- Model: `empire/core/empire.py` (emission cap ~L716, objective ~L577-596,
  CCS const L197, lost-load L195).
- Config: `config/run.yaml` (solver options block near the bottom; user has it open).
- Solver options plumbing: `empire/core/config.py`, `empire/core/model_runner.py`
  (added in commit 588c516 "Add configurable Gurobi solver options and ramping toggle").
- Logs: `Results/basic_run/dataset_*/Output/logfile_*.log`.
