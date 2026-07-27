# Alignment with the reference EMPIRE core

This note records the differences found between this repository's model core and a
reference copy of another group's EMPIRE core, and what was done about each one. It
exists so that results from the two models can be compared on a like-for-like basis,
and so the remaining deliberate divergences are not rediscovered later.

The comparison was made by diffing parameter declarations and the bodies of all 47
constraint/build rules shared by the two `empire.py` files.

## Summary

| Item | Reference model | This model (after alignment) |
| --- | --- | --- |
| Biomass usage limit | `1.2`, hardcoded, pooled Europe-wide | `biomass_limit_factor`, default `1.2`; scoped per country by default |
| Node generation growth | `1.3`, hardcoded, always on | `generation_growth_limit_rate: 0.3`, flag on |
| Per-node yearly availability | in `genMaxProd_rule` | added, data-gated |
| Emission cap scaling | `1e6 * CO2cap` | identical |
| Parameter defaults (60+) | — | identical |

The reference model contains exactly two bare tuning constants, `1.3` and `1.2`.
Everything else is data-driven, and every shared `Param` default matches.

## 1. Biomass usage limit (new)

Caps expected annual biomass electricity production in each period:

```
sum over nodes in group, biomass generators, hours, scenarios of
    seasScale * sceProbab * genOperational
        <= biomass_limit_factor * sum over nodes in group of maxBiomassNode[n, i]
```

where the group of nodes is set by `biomass_limit_scope` below.

### Spatial scope

The reference model pools every node into a single constraint row per period, i.e. a
Europe-wide biomass pool. This model instead treats the limit as **national** by
default, because biomass availability is a national resource assessment and a
country split into NUTS regions must not receive one full allowance per region.

`biomass_limit_scope` selects between the two:

- `"country"` (default) — one row per country per period. Production and availability
  are both summed over that country's nodes as given by `Countries` / `NodesOfCountry`
  in `Sets.xlsx`. Biomass may therefore be traded between a country's NUTS regions but
  not across borders. A node that no `NodesOfCountry` row covers forms its own group,
  which is the correct reading for datasets where one node is one country — so
  aggregated datasets get a genuine per-country limit without needing the country sheets
  at all.
- `"system"` — the reference model's single pooled row per period. Use this for
  like-for-like comparison runs.

Note that the default therefore diverges from the reference model. On the datasets
that currently supply the sheet (`energy_vis_*`, `Laura_*`, one node per country and no
country sheets) `"country"` binds per node, which is materially tighter than the
reference model's Europe-wide pool.

Input is the optional `BiomassMaxAnnualActivity` sheet of `Node.xlsx`, with columns
`Node`, `Period`, `maxBiomassNode` in MWh per year. The scaling convention matches the
existing `hydro_node_limit`.

Biomass generators are selected by name prefix (`bio`, case-insensitive) so that `Bio`,
`BioCCS` and `Bioexisting` are all covered. The matched generators are written to the
log on model build. Note that a co-firing unit named e.g. `Bio10cofiring` would be
counted at its full output; no dataset that currently supplies the sheet contains one.

Controlled by `biomass_limit_flag` (default true), `biomass_limit_factor` (default 1.2)
and `biomass_limit_scope` (default `"country"`). The sheet is the real gate: datasets
without it are unaffected and log that the limit is disabled. The flag allows switching
the constraint off even where the data exists.

The sheet's own description text mentions a factor of 1.1, but the reference
implementation applies 1.2. The default follows the code, not the description.

## 2. Per-node yearly generator availability (new)

The reference model derates maximum production by a per-node, per-period factor:

```
genOperational <= genYearlyAvailability * genCapAvail * genInstalledCap
```

Input is the optional `YearlyAvailability` sheet of `Generator.xlsx`, with columns
`Node`, `GeneratorTechnology`, `Period`, `Value`. A value of `0` forces the technology
off in that node and period; `1.0` (the default) leaves it unaffected.

In the `energy_vis_*` and `Laura_*` datasets this sheet carries 237 rows for coal and
lignite that are almost all zero, i.e. it encodes a coal phase-out schedule. Without
it this model was free to keep running plant that the reference model forces off, so
this was the single largest source of divergence between the two.

When the sheet is absent the factor is omitted from the constraint expression
entirely, so there is no additional LP cost for datasets that do not use it.

## 3. Generation growth limit (constant changed)

The growth limit already existed here as a config-gated feature, off by default at a
rate of 0.2. The reference model applies it unconditionally with a hardcoded 1.3.

`config/run.yaml` now sets `generation_growth_limit_flag: True` and
`generation_growth_limit_rate: 0.3` to match. This affects all runs using that config,
not only comparison runs.

For the record, the reference model defines this rule twice; the second definition,
inside its `not OUT_OF_SAMPLE` block, silently overwrites the first. Both copies are
identical, so the duplication has no effect on results.

## Deliberate remaining differences

### Storage installed-capacity floor (no practical effect)

This model floors `storENMaxInstalledCap` / `storPWMaxInstalledCap` at the storage's
initial capacity; the reference model assigns the raw value directly. On the datasets
in use, no row has `MaxInstalledCapacityRaw < InitialCapacity` (0 of 203 for both
energy and power), so the two formulations coincide. Left as is.

### CCS cost treatment (not aligned, intentionally)

The reference model treats the `CO2Content` column as a *gross* fuel factor and derives
residual emissions in the marginal cost as `(1 - CCSRemFrac) * CO2Content`. This model
treats the column as an already-*net* factor and uses it directly, deriving the captured
quantity separately (see `empire/core/generator_costs.py`).

The datasets in use supply net values: `Coal` is 0.216 against `Coal CCS` at 0.0288, and
`Bio CCS` is negative at -0.0898. A negative value cannot be a gross fuel factor.

The reference model is also internally inconsistent on this point: its `emission_cap`
uses the raw factor directly as a net quantity, while its marginal cost applies the
`(1 - CCSRemFrac)` multiplier. It therefore prices CO2 for CCS units at one tenth of
the rate at which it counts the same emissions against the cap, and grants BioCCS only
a tenth of its negative-emission credit.

Aligning with the reference here would mean reintroducing that inconsistency, so this
model keeps its own treatment. If a strictly like-for-like run is ever needed, the
legacy behaviour should be added behind an explicit flag rather than by reverting.

## Configuration

```yaml
generation_growth_limit_flag: True   # was False
generation_growth_limit_rate: 0.3    # was 0.2
biomass_limit_flag: True             # new
biomass_limit_factor: 1.2            # new
biomass_limit_scope: "country"       # new; "system" reproduces the reference model
```

`YearlyAvailability` has no config switch; it is enabled purely by the presence of the
sheet.

## Verification

- Biomass limit binds exactly at the configured multiple: a fixture supplying 5 TWh per
  node across 3 nodes produced 18,000 GWh with the limit on (1.2 x 15 TWh) against
  430,366 GWh with it off.
- Biomass scope, on a fixture with country `DEDK` = {Germany, Denmark} and France
  uncovered, supplying 4 / 1 / 3 TWh: `"country"` bound DEDK at 6,000 GWh (Germany
  drawing on Denmark's share, Denmark at 0) and France separately at 3,600 GWh, while
  `"system"` bound the three together at 9,600 GWh. With no country sheets present the
  same data bound each node on its own at 4,800 / 1,200 / 3,600 GWh.
- Yearly availability: forcing a node's coal to zero produced 0 GWh there while the same
  technology ran unconstrained at another node.
- All 237 `(node, generator)` pairs in the real `YearlyAvailability` sheets resolve
  against `GeneratorsOfNode`, and the sheet periods match each dataset's horizon.
- Datasets without either sheet build and solve unchanged, logging both features as
  disabled.
