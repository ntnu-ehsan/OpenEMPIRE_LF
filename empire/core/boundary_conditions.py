"""Spanish-case boundary conditions.

Constrains an LF_ES_* run (Spanish NUTS3 + France + Portugal + aggregated EU node) to the
investment results of an original aggregated EMPIRE run (``Results/basic_run/dataset_Agg_*``):

- ``France``, ``Portugal``, ``EU``: node generation / storage capacities set to the original
  results (EU = sum of all original nodes except Spain, France, Portugal).
- Spanish NUTS3 nodes: the SUM over all ES* nodes per generator/storage type is constrained to
  Spain's national value from the original run. Optional — see ``include_spain``.
- Transmission: corridor sums fixed — sum of ES*-France border lines to the original Spain-France
  capacity, likewise ES*-Portugal, and the single EU-France line to the sum of France's original
  non-Spanish corridors. Internal Spanish lines remain freely expandable.

Boundary data is read from ``BoundaryConditions/*.csv`` in the dataset folder, produced by
``scripts/extract_boundary_conditions.py``. Periods in the CSVs are 1-based model periods.

``bound_type`` selects equality (``"fixed"``, default) or upper bounds (``"upper"``). Note that
equality can render the model infeasible if the LF_ES initial capacities exceed the original
results for some node/technology/period.

``include_spain`` selects which experiment is being run:

- ``True`` (default) — Spain's national totals are pinned to the original run as well, so the
  only remaining freedom is *where inside Spain* capacity is built. This isolates the effect of
  the spatial (NUTS3) split on its own.
- ``False`` — the three Spanish constraints are dropped and Spain invests freely. Use this when
  Spain's build-out should instead be governed by the national limits from the
  ``MaxInstalledCapacityCountry`` / ``MaxBuiltCapacityCountry`` sheets. Note those national
  limits are inequalities (``<=``) while the Spanish boundary constraints are equalities, so
  leaving both active is not an error — but the equality always wins and the national limits
  can never bind. Border corridors stay fixed either way.
"""

import logging
from pathlib import Path

import pandas as pd
from pyomo.environ import Constraint, Set, value

logger = logging.getLogger(__name__)

SPAIN_PREFIX = "ES"
SPAIN_NODE = "Spain"
DIRECT_NODES = ("France", "Portugal", "EU")


def _bound(expr, target, bound_type):
    if bound_type == "upper":
        return expr <= target
    return expr == target


def _reachable(target, floor, label):
    """Installed capacity can never drop below initial capacity; clamp the target to that floor."""
    if floor > target + 1e-6:
        logger.warning(
            "Boundary conditions: %s target %.1f below initial capacity %.1f — clamped to initial "
            "(datasets disagree on existing capacity).",
            label, target, floor,
        )
        return floor
    return target


def load_boundary_data(boundary_path: Path) -> dict:
    """Read the three boundary CSVs and return lookup dicts keyed on model indices."""
    boundary_path = Path(boundary_path)
    for name in ("boundary_generation.csv", "boundary_storage.csv", "boundary_transmission.csv"):
        if not (boundary_path / name).exists():
            raise FileNotFoundError(
                f"Boundary conditions enabled but '{name}' not found in {boundary_path}. "
                "Run scripts/extract_boundary_conditions.py first."
            )

    gen = pd.read_csv(boundary_path / "boundary_generation.csv")
    stor = pd.read_csv(boundary_path / "boundary_storage.csv")
    trans = pd.read_csv(boundary_path / "boundary_transmission.csv")

    data = {
        "gen": {
            (r.Node, r.GeneratorType, int(r.Period)): float(r.InstalledCap_MW)
            for r in gen.itertuples()
        },
        "stor_pw": {
            (r.Node, r.StorageType, int(r.Period)): float(r.PWInstalledCap_MW)
            for r in stor.itertuples()
        },
        "stor_en": {
            (r.Node, r.StorageType, int(r.Period)): float(r.ENInstalledCap_MWh)
            for r in stor.itertuples()
        },
        "trans": {
            (r.FromNode, r.ToNode, int(r.Period)): float(r.InstalledCap_MW)
            for r in trans.itertuples()
        },
        "periods": set(gen["Period"].astype(int)) | set(trans["Period"].astype(int)),
    }
    logger.info(
        "Boundary conditions loaded from %s: %d gen, %d storage, %d transmission entries "
        "covering periods %s",
        boundary_path, len(data["gen"]), len(data["stor_pw"]), len(data["trans"]),
        sorted(data["periods"]),
    )
    return data


def add_boundary_conditions(
    model, boundary_path: Path, bound_type: str = "fixed", include_spain: bool = True
):
    """Attach boundary-condition constraints to the (abstract) EMPIRE model."""
    bound_type = str(bound_type).lower()
    if bound_type not in ("fixed", "upper"):
        raise ValueError(f"boundary_bound_type must be 'fixed' or 'upper', got '{bound_type}'")

    bc = load_boundary_data(boundary_path)
    periods = bc["periods"]

    logger.info(
        "Adding Spanish-case boundary conditions (bound type: %s, Spanish national totals: %s)...",
        bound_type,
        "pinned" if include_spain else "free",
    )

    # --- Generation: France / Portugal / EU fixed per node ---------------------------------
    def bc_gen_direct_rule(model, n, g, i):
        if n not in DIRECT_NODES or i not in periods:
            return Constraint.Skip
        target = _reachable(
            bc["gen"].get((n, g, i), 0.0), value(model.genInitCap[n, g, i]), f"gen {n}/{g}/p{i}"
        )
        return _bound(model.genInstalledCap[n, g, i], target, bound_type)

    model.bc_gen_direct = Constraint(model.GeneratorsOfNode, model.PeriodActive, rule=bc_gen_direct_rule)

    # --- Generation: Spain national bound on the sum over NUTS3 nodes ----------------------
    # Skipped entirely when include_spain is False, leaving Spain's national total to the
    # MaxInstalledCapacityCountry / MaxBuiltCapacityCountry sheets instead.
    def bc_gen_spain_rule(model, g, i):
        if not include_spain or i not in periods:
            return Constraint.Skip
        nodes = [n for n in model.Node if str(n).startswith(SPAIN_PREFIX) and (n, g) in model.GeneratorsOfNode]
        target = bc["gen"].get((SPAIN_NODE, g, i))
        if not nodes:
            if target:
                logger.warning(
                    "Boundary conditions: Spain has %.1f MW of '%s' in period %d in the original "
                    "results, but no Spanish node carries that generator — constraint skipped.",
                    target, g, i,
                )
            return Constraint.Skip
        target = _reachable(
            target if target is not None else 0.0,
            sum(value(model.genInitCap[n, g, i]) for n in nodes),
            f"gen Spain-sum/{g}/p{i}",
        )
        return _bound(
            sum(model.genInstalledCap[n, g, i] for n in nodes),
            target,
            bound_type,
        )

    model.bc_gen_spain = Constraint(model.Generator, model.PeriodActive, rule=bc_gen_spain_rule)

    # --- Storage: France / Portugal / EU fixed per node ------------------------------------
    def bc_stor_pw_direct_rule(model, n, b, i):
        if n not in DIRECT_NODES or i not in periods:
            return Constraint.Skip
        target = _reachable(
            bc["stor_pw"].get((n, b, i), 0.0), value(model.storPWInitCap[n, b, i]), f"storPW {n}/{b}/p{i}"
        )
        return _bound(model.storPWInstalledCap[n, b, i], target, bound_type)

    model.bc_stor_pw_direct = Constraint(model.StoragesOfNode, model.PeriodActive, rule=bc_stor_pw_direct_rule)

    def bc_stor_en_direct_rule(model, n, b, i):
        if n not in DIRECT_NODES or i not in periods:
            return Constraint.Skip
        target = _reachable(
            bc["stor_en"].get((n, b, i), 0.0), value(model.storENInitCap[n, b, i]), f"storEN {n}/{b}/p{i}"
        )
        return _bound(model.storENInstalledCap[n, b, i], target, bound_type)

    model.bc_stor_en_direct = Constraint(model.StoragesOfNode, model.PeriodActive, rule=bc_stor_en_direct_rule)

    # --- Storage: Spain national bound on the sum over NUTS3 nodes -------------------------
    # Also skipped when include_spain is False, see bc_gen_spain_rule above.
    def bc_stor_pw_spain_rule(model, b, i):
        if not include_spain or i not in periods:
            return Constraint.Skip
        nodes = [n for n in model.Node if str(n).startswith(SPAIN_PREFIX) and (n, b) in model.StoragesOfNode]
        if not nodes:
            return Constraint.Skip
        target = _reachable(
            bc["stor_pw"].get((SPAIN_NODE, b, i), 0.0),
            sum(value(model.storPWInitCap[n, b, i]) for n in nodes),
            f"storPW Spain-sum/{b}/p{i}",
        )
        return _bound(
            sum(model.storPWInstalledCap[n, b, i] for n in nodes),
            target,
            bound_type,
        )

    model.bc_stor_pw_spain = Constraint(model.Storage, model.PeriodActive, rule=bc_stor_pw_spain_rule)

    def bc_stor_en_spain_rule(model, b, i):
        if not include_spain or i not in periods:
            return Constraint.Skip
        nodes = [n for n in model.Node if str(n).startswith(SPAIN_PREFIX) and (n, b) in model.StoragesOfNode]
        if not nodes:
            return Constraint.Skip
        target = _reachable(
            bc["stor_en"].get((SPAIN_NODE, b, i), 0.0),
            sum(value(model.storENInitCap[n, b, i]) for n in nodes),
            f"storEN Spain-sum/{b}/p{i}",
        )
        return _bound(
            sum(model.storENInstalledCap[n, b, i] for n in nodes),
            target,
            bound_type,
        )

    model.bc_stor_en_spain = Constraint(model.Storage, model.PeriodActive, rule=bc_stor_en_spain_rule)

    # --- Transmission: corridor sums fixed --------------------------------------------------
    corridors = sorted({(a, b) for (a, b, _) in bc["trans"]})
    model.BoundaryCorridor = Set(dimen=2, initialize=corridors, ordered=True)

    def _arc_matches(node, endpoint):
        if endpoint == SPAIN_NODE:
            return str(node).startswith(SPAIN_PREFIX)
        return node == endpoint

    def bc_trans_rule(model, a, b, i):
        if i not in periods:
            return Constraint.Skip
        arcs = [
            (n1, n2)
            for (n1, n2) in model.BidirectionalArc
            if (_arc_matches(n1, a) and _arc_matches(n2, b))
            or (_arc_matches(n1, b) and _arc_matches(n2, a))
        ]
        target = bc["trans"].get((a, b, i), 0.0)
        if not arcs:
            logger.warning(
                "Boundary conditions: no arcs found for corridor %s-%s (period %d) — skipped.",
                a, b, i,
            )
            return Constraint.Skip
        target = _reachable(
            target,
            sum(value(model.transmissionInitCap[n1, n2, i]) for (n1, n2) in arcs),
            f"transmission {a}-{b}/p{i}",
        )
        # Transmission capacities are always fixed to the original results (equality),
        # independent of bound_type, as requested for the Spanish case.
        return sum(model.transmissionInstalledCap[n1, n2, i] for (n1, n2) in arcs) == target

    model.bc_transmission = Constraint(model.BoundaryCorridor, model.PeriodActive, rule=bc_trans_rule)

    logger.info("Boundary-condition constraints added.")
