from __future__ import annotations
from pyexpat import model
from pyomo.environ import (Set, Var, Constraint, PositiveReals, value, Param, Expression, Reals, ConstraintList)
from collections import defaultdict, deque
from typing import  Dict,List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class LOPFMethod:
    """Enumeration of available LOPF methods."""
    KIRCHHOFF = "kirchhoff"     # Cycle-based DC-OPF without angles
    ANGLE = "angle"             # Classical bus-angle DC-OPF
    PTDF = "ptdf"               # PTDF-based formulation

    @classmethod
    def list_methods(cls):
        """Return a list of all available methods."""
        return [cls.KIRCHHOFF, cls.ANGLE, cls.PTDF]




# ---------------------------
# Public entrypoint (router)
# ---------------------------
def add_lopf_constraints(model, method: str = LOPFMethod.KIRCHHOFF, **kwargs):
    """
    method:
      - "kirchhoff": cycle-based DC power flow (KCL + KVL, no angles)
      - "angle":     classic DC-OPF with bus angles

    Common kwargs (both methods):
      - capacity_expr: callable (m,i,j,p) -> capacity expression for line (i,j) in period p
      - couple_to_existing_flows: bool (default True)
      - existing_flow_candidates: tuple[str,...] (names of directed flow var to bind)
    Kirchhoff-specific:
      - reactance_param_name: str (default "lineReactance") on BidirectionalArc
      - susceptance_param_name: str (default "lineSusceptance")
      - reactance_from_susceptance: bool (default False)
    Angle-specific:
      - susceptance_param_name: str (default "lineSusceptance") on DirectionalLink
      - fix_angle_reference: bool (default True)
      - slack_node_set_name: str (default "SlackNode")  # optional Set(model.Node)
    """

    if method.lower() == LOPFMethod.KIRCHHOFF:
        return _add_kirchhoff_constraints(model, **kwargs)
    elif method.lower() == LOPFMethod.ANGLE:
        return _add_angle_constraints(model, **kwargs)
    elif method.lower() == LOPFMethod.PTDF:
        return _add_ptdf_constraints(model, **kwargs)
    else:
        raise ValueError(f"Unknown LOPF method: {method}")


# ---------------------------
# Shared Helpers
# ---------------------------
def _infer_capacity_expr(model):
    """Return (m,i,j,p) -> capacity expression for undirected (i,j).

    This helper inspects the model for common components that represent
    transmission capacity and returns a single callable with signature
    (m, i, j, p) -> capacity. The callable abstracts over different
    representations (installed-capacity Var, initial-capacity Param,
    incremental build Vars/Params, or legacy component names) so the
    LOPF code can request a capacity value without duplicating lookup
    logic.

    Behaviour highlights:
        - Tries several common component names to support historical
            naming variants.
        - If only init + build components exist, constructs capacity by
            summing initial capacity and cumulative builds up to period p.
        - Performs safe lookups that try the reversed node ordering
            (j,i,p) when (i,j,p) is not present to tolerate orientation
            differences in input data.
        - Raises a RuntimeError if no plausible capacity source is found
            (caller may instead pass an explicit capacity_expr argument).
    """
    # helper: look up component by name on the instance
    def _by_name(name: str):
        # Return a callable that safely looks up component values by (i,j,p).
        # If the direct ordering (i,j,p) is not present, try the reversed ordering (j,i,p).
        # If neither exists, log a warning and return 0.0 as a safe fallback to allow model
        # construction to continue. This prevents a KeyError during Expression construction
        # when input data uses the opposite orientation for corridor keys.
        logger.debug('_by_name: looking up capacity component "%s"', name)
        def _safe_lookup(m, i, j, p, _n=name):
            comp = getattr(m, _n)
            try:
                return comp[i, j, p]
            except Exception:
                logger.debug('Capacity lookup: component "%s" missing index (%s,%s,%s); trying reversed order', _n, i, j, p)
                try:
                    return comp[j, i, p]
                except Exception:
                    logger.warning("Capacity lookup: component '%s' missing index (%s,%s,%s) and (%s,%s,%s); returning 0.0",
                                   _n, i, j, p, j, i, p)
                    return 0.0
        return _safe_lookup

    # Prefer an installed-capacity variable if it exists
    for name in ("transmissionInstalledCap", "installedTransmissionCap", "TransInstalledCap", 'Transmission_InitialCapacity'):
        if hasattr(model, name):
            return _by_name(name)

    # Build from init + built (if both exist)
    init_name = "transmissionInitCap" if hasattr(model, "transmissionInitCap") else None
    built_name = next((nm for nm in ("transmissionBuilt", "TransBuilt", "lineExpansion")
                       if hasattr(model, nm)), None)

    if init_name and built_name:
        def _cap_from_init_plus_built(m, i, j, p, _init=init_name, _built=built_name):
            init = getattr(m, _init)
            built = getattr(m, _built)
            total = init[i, j, p] if (i, j, p) in init else 0.0
            for pp in m.PeriodActive:
                if pp <= p and (i, j, pp) in built:
                    total += built[i, j, pp]
            return total
        return _cap_from_init_plus_built

    # Safe fallbacks — note: these are *params* loaded from .tab
    for name in ("transmissionMaxInstalledCap", "transmissionMaxInstalledCapRaw", "transmissionInitCap", 'Transmission_InitialCapacity'):
        if hasattr(model, name):
            return _by_name(name)

    raise RuntimeError("Cannot infer transmission capacity; pass capacity_expr=... to add_lopf_constraints().")



def _bind_to_existing_flows(model, flow_dc_like, existing_flow_candidates=()):
    """
    Try to bind to an existing directed flow var FlowDir[i,j,h,w,p] used in KCL:
        FlowDir[i,j,h,w,p] == +FlowUndir[(i,j),h,w,p]
        FlowDir[j,i,h,w,p] == -FlowUndir[(i,j),h,w,p]
    Returns the bound variable or None.
    """
    if not existing_flow_candidates:
        existing_flow_candidates = ("transmissionOperational", "transFlow", "lineFlow", "flow")
    FlowDir = None
    for name in existing_flow_candidates:
        if hasattr(model, name):
            FlowDir = getattr(model, name)
            break
    if FlowDir is None:
        return None

    # map undirected to directed
    def _fwd(m, i, j, h, w, p):
        return FlowDir[i, j, h, w, p] == flow_dc_like[i, j, h, w, p]
    def _bwd(m, i, j, h, w, p):
        return FlowDir[j, i, h, w, p] == -flow_dc_like[i, j, h, w, p]

    model.LOPF_BindFwd = Constraint(model.BidirectionalArc, model.Operationalhour, model.Scenario, model.PeriodActive, rule=_fwd)
    model.LOPF_BindBwd = Constraint(model.BidirectionalArc, model.Operationalhour, model.Scenario, model.PeriodActive, rule=_bwd)
    return FlowDir

# ---------------------------
# Method A: Kirchhoff (Cycle-based)
# ---------------------------

def _add_kirchhoff_constraints(
    model,
    *,
    reactance_param_name: str = "lineReactance",
    susceptance_param_name: str = "lineSusceptance",
    reactance_from_susceptance: bool = False,
    store_debug: bool = False,
    capacity_expr: Optional[Callable] = None,
    couple_to_existing_flows: bool = True,
    existing_flow_candidates: tuple[str, ...] = (
        "transmissionOperational",   # EMPIRE’s usual name (fixed missspelling!)
        "transFlow",
        "lineFlow",
        "flow",
    ),
    # (any remaining kwargs)
):
    """
    Cycle-based DC-OPF (Kirchhoff) without voltage angles:
      - Variables: undirected net flow on each corridor (i,j) ∈ BidirectionalArc
      - KCL is satisfied by EMPIRE's existing directed flow variables in FlowBalance;
        we link our net-flow to those directed flows linearly: F_ij^dir - F_ji^dir = F_ij^net
      - KVL is enforced on a fundamental cycle basis: Σ s_c,ij * x_ij * F_ij^net = 0
      - Capacity: |F_ij^net| ≤ cap_ij(p) (cap inferred unless provided)

    All declarations are done with Pyomo rules/initializers so this works with AbstractModel.
    """

    # ---- Guard required sets (Abstract) ---------------------------------------
    for s in ("BidirectionalArc", "DirectionalLink", "Node", "Operationalhour", "Scenario", "PeriodActive"):
        if not hasattr(model, s):
            raise ValueError(f"Model is missing required set: {s}")

    L = model.BidirectionalArc
    H, W, P = model.Operationalhour, model.Scenario, model.PeriodActive

    # ---- Reactance X[i,j] selection/derivation (Abstract-safe) ----------------
    # If a reactance Param exists, use it. Else, if asked, derive X = 1/B from susceptance.
    # Otherwise, error out with a clear message.
    if hasattr(model, reactance_param_name):
        X = getattr(model, reactance_param_name)
    elif reactance_from_susceptance:
        if not hasattr(model, susceptance_param_name):
            raise RuntimeError(
                f"Need '{reactance_param_name}' or '{susceptance_param_name}', "
                f"or set reactance_from_susceptance=True with the latter present."
            )
        B = getattr(model, susceptance_param_name)
        eps = 1e-9  # guard against divide-by-zero
        def _x_init(m, i, j):
            bij = float(B[i, j])
            if abs(bij) < eps:
                return 1.0 / eps
            return 1.0 / bij
        # Create a derived reactance Param on BidirectionalArc
        model._lopf_reactance = Param(L, initialize=_x_init, within=PositiveReals)
        X = model._lopf_reactance
    else:
        raise RuntimeError(f"Provide '{reactance_param_name}' or set reactance_from_susceptance=True.")

    # ---- Capacity expression ---------------------------------------------------
    # If none is supplied, infer from model components (init + built etc.).
    cap_rule = capacity_expr or _infer_capacity_expr(model)  # returns (m,i,j,p) -> cap

    # ---- Lazy cycle-basis construction (Abstract-safe) -------------------------
    # We compute cycles & their edge signs when the instance is constructed.
    model._lopf_edge_signs = {}  # {cycle_idx: {(i,j): -1/0/+1}}

    def _cycles_index_init(m):
        cycles, edge_signs = _fundamental_cycles(m.Node, m.BidirectionalArc)
        m._lopf_edge_signs = edge_signs
        return range(1, len(cycles) + 1)

    model.Cycle = Set(ordered=True, initialize=_cycles_index_init)

    def _sign_init(m, c, i, j):
        return m._lopf_edge_signs.get(c, {}).get((i, j), 0)

    model.CycleEdgeSign = Param(model.Cycle, L, initialize=_sign_init, within=Reals, default=0)

    if store_debug:
        # Optional: expose number of cycles and a quick check for missing reverse arcs
        def _missing_rev_init(m):
            missing = []
            for (i, j) in m.BidirectionalArc:
                if (i, j) not in model.DirectionalLink or (j, i) not in model.DirectionalLink:
                    missing.append((i, j))
            return tuple(missing)
        model.LOPF_MissingReverseArcs = Set(dimen=2, initialize=_missing_rev_init)

    # ---- Decision var: undirected NET flow on each corridor --------------------
    model.FlowK = Var(L, H, W, P, domain=Reals)

    # ---- Bind to existing directed flow variables (linear, net-flow mapping) ---
    # We map EMPIRE's directional, nonnegative flows F_ij^dir and F_ji^dir to our
    # net flow: F_ij^dir - F_ji^dir = FlowK_ij (works with NonNegativeReals).
    if couple_to_existing_flows:
        FlowDir = None
        for name in existing_flow_candidates:
            if hasattr(model, name):
                FlowDir = getattr(model, name)
                break
        if FlowDir is None:
            raise ValueError(
                "Could not find an existing directed flow variable to bind to. "
                f"Tried: {existing_flow_candidates}"
            )

        def _net_bind(m, i, j, h, w, p):
            # Require both directions to exist in DirectionalLink
            if (i, j) not in m.DirectionalLink or (j, i) not in m.DirectionalLink:
                # If a direction is missing, best to fail fast with a clear message
                raise KeyError(f"DirectionalLink missing ({i},{j}) or ({j},{i}) needed for net-flow binding.")
            return FlowDir[i, j, h, w, p] - FlowDir[j, i, h, w, p] == m.FlowK[i, j, h, w, p]


        model.LOPF_NetBinding = Constraint(L, H, W, P, rule=_net_bind)

    # ---- KVL on cycles: sum s_c,ij * X_ij * FlowK_ij = 0 -----------------------
    def _kvl_rule(m, c, h, w, p):
        return sum(m.CycleEdgeSign[c, i, j] * X[i, j] * m.FlowK[i, j, h, w, p]
                   for (i, j) in m.BidirectionalArc) == 0
    model.KVL = Constraint(model.Cycle, H, W, P, rule=_kvl_rule)

    # ---- Thermal capacity on corridors: |FlowK_ij| <= cap_ij(p) ----------------
    def _cap_pos(m, i, j, h, w, p):
        return m.FlowK[i, j, h, w, p] <= cap_rule(m, i, j, p)
    def _cap_neg(m, i, j, h, w, p):
        return -m.FlowK[i, j, h, w, p] <= cap_rule(m, i, j, p)

    model.K_CapacityPos = Constraint(L, H, W, P, rule=_cap_pos)
    model.K_CapacityNeg = Constraint(L, H, W, P, rule=_cap_neg)

    # Done.
    return model


def _fundamental_cycles(NodeSet, LineSet) -> Tuple[List[List[Tuple]], Dict[Tuple[int, Tuple], int]]:
    """Build a fundamental cycle basis via spanning tree + chords; return cycles and edge signs."""
    adj = defaultdict(list)
    edges = set()
    for (i, j) in LineSet:
        adj[i].append(j); adj[j].append(i)
        edges.add((i, j))  # stored orientation

    parent = {}
    tree_edges = set()
    visited = set()
    for root in NodeSet:
        if root in visited: continue
        visited.add(root); parent[root] = None
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v); parent[v] = u; q.append(v)
                    e = (u, v) if (u, v) in edges else (v, u)
                    tree_edges.add(e)

    chords = [e for e in edges if e not in tree_edges]

    def path_edges(u, v):
        pu, pv = [], []
        uu = u
        while uu is not None:
            pu.append(uu); uu = parent.get(uu, None)
        vv = v
        while vv is not None:
            pv.append(vv); vv = parent.get(vv, None)
        pu = pu[::-1]; pv = pv[::-1]
        l = 0
        while l < min(len(pu), len(pv)) and pu[l] == pv[l]:
            l += 1
        l -= 1
        nodes_path = pu[l:] + pv[:l:-1]
        epath = []
        cur = u
        for k in range(len(nodes_path) - 1):
            a, b = nodes_path[k], nodes_path[k+1]
            epath.append((a, b) if (a, b) in edges else (b, a))
            cur = b
        return epath

    cycles = []
    edge_signs: Dict[Tuple[int, Tuple], int] = {}
    for chord in chords:
        u, v = chord
        pth = path_edges(u, v)
        cyc_edges = pth + [chord]
        c_idx = len(cycles)
        cycles.append(cyc_edges)

        # signs: +1 if traversal aligns with stored orientation, else -1
        cur = u
        for e in pth:
            a, b = e
            sgn = +1 if a == cur else -1
            edge_signs[(c_idx, e)] = sgn
            cur = b if a == cur else a
        edge_signs[(c_idx, chord)] = -1  # closing chord traversed v->u (opposite stored (u,v))
    return cycles, edge_signs

# ---------------------------
# Method B: Angle-based DC
# ---------------------------
def _add_angle_constraints(
    model,
    *,
    reactance_param_name: str = "lineReactance",  # Required: reactance X on DirectionalLink
    # Angle-specific args
    capacity_expr: Optional[Callable] = None,   # If None, infer from model components
    couple_to_existing_flows: bool = True,  # If True (recommended), couple FlowDC to the existing flow variables
    existing_flow_candidates = ("transmissionOperational", "TransFlow", "lineFlow", "flow"),    # Existing flow variable candidates. In Existing version of EMPIRE it is "transmissionOperational".
    fix_angle_reference: bool = True,   # If True, fix angle at slack node(s) to zero
    slack_node_set_name: str = "SlackNode", # Name of SlackNode Set(model.Node)
    **kwargs  # Accept any other kwargs without error (e.g., susceptance_param_name, reactance_from_susceptance)
):
    """
    Angle-based DC-OPF formulation.
    
    Requires reactance X[i,j] to be provided on DirectionalLink.
    Susceptance B = 1/X is derived internally for Ohm's law constraints.
    """
    # log what kwargs were passed
    logger.info("=" * 70)
    logger.info("SETTING UP ANGLE-BASED DC-OPF")
    logger.info("=" * 70)
    logger.debug("Angle-based LOPF called with kwargs: %s", kwargs)
    
    # Required sets
    for s in ("DirectionalLink", "Node", "Operationalhour", "Scenario", "PeriodActive"):
        if not hasattr(model, s):
            raise RuntimeError(f"Model is missing required set '{s}'")

    A = model.DirectionalLink; 
    N = model.Node; 
    H = model.Operationalhour; 
    W = model.Scenario; 
    P = model.PeriodActive; 

    # CandidateDirectional set (derived from CandidateTransmission)     
    if not hasattr(model, "CandidateDirectional"):
        def _cand_dir_init(m):
            arcs = []
            for (i, j) in m.DirectionalLink:
                if (i, j) in m.CandidateTransmission or (j, i) in m.CandidateTransmission:
                    arcs.append((i, j))
            return arcs
        model.CandidateDirectional = Set(within=A, initialize=_cand_dir_init)

    CAND_DIR = model.CandidateDirectional

    # --- Reactance input & Susceptance derivation (accept BidirectionalArc or DirectionalLink) ---
    if not hasattr(model, reactance_param_name):
        raise RuntimeError(
            f"Angle-based LOPF requires reactance parameter '{reactance_param_name}' present on either "
            f"DirectionalLink or BidirectionalArc. Load it via load_line_parameters() before calling add_lopf_constraints()."
        )

    logger.debug("Reactance parameter '%s' found.", reactance_param_name)
    eps = 1e-9  # Guard against division by zero

    # Log index set type on the abstract component (for info)
    logger.debug("Reactance parameter '%s' indexed by: %s", reactance_param_name, type(getattr(model, reactance_param_name).index_set()))

    # Derive a directional reactance Param regardless of the original index set.
    # IMPORTANT: Reference the instance component via 'm', not the abstract one via 'model'.
    # This guarantees we can look up X[i,j] for every directed arc during instance construction.
    def _react_dir_init(m, i, j):
        RX = getattr(m, reactance_param_name)
        # Extra diagnostics: check set membership to help trace orientation/index issues
        if logger.isEnabledFor(logging.DEBUG):
            try:
                A_dbg = getattr(m, 'DirectionalLink', None)
                C_dbg = getattr(m, 'CandidateTransmission', None)
                in_A_ij = (i, j) in A_dbg if A_dbg is not None else None
                in_A_ji = (j, i) in A_dbg if A_dbg is not None else None
                in_C_ij = (i, j) in C_dbg if C_dbg is not None else None
                in_C_ji = (j, i) in C_dbg if C_dbg is not None else None
                logger.debug(
                    "RX init: arc (%s,%s): in DirectionalLink=%s (rev=%s); in Candidate=%s (rev=%s)",
                    i, j, in_A_ij, in_A_ji, in_C_ij, in_C_ji
                )
            except Exception:
                pass
        # helper to fetch numeric value safely
        def _get(ii, jj):
            try:
                val = float(value(RX[ii, jj]))
                # Reject zero or negative values
                if val <= 0:
                    return None
                return val
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug("_react_dir_init: Reactance lookup failed for (%s,%s) with %s.", ii, jj, type(e).__name__)
                    except Exception:
                        pass
                return None
        x = _get(i, j)
        if x is None:
            x = _get(j, i)
        if x is None:
            logger.warning("Missing reactance data for arc (%s,%s); using reciprocal of eps.", i, j)
            try:
                logger.debug("%s sample keys: %s", reactance_param_name, list(RX.keys())[:10])
            except Exception:
                pass
            return 1.0 / eps
        return x
    model._reactance_dir = Param(A, initialize=_react_dir_init, within=PositiveReals)
    X = model._reactance_dir
    logger.debug("Directional reactance parameter synthesized for angle-based DC-OPF from '%s'.", reactance_param_name)

    # Helper: compute susceptance B = 1/X on-the-fly in a safe way.
    def _get_B_val(m, i, j):
        """Return susceptance for directional arc (i,j).

        This avoids relying on a derived Param that may not yet be
        constructed. It attempts to read a directional reactance `X[i,j]`.
        If that fails, it tries the reversed ordering `X[j,i]` (to support
        undirected input). If both fail, it falls back to a large reactance
        (small susceptance) and logs a warning.
        """
        # Use the synthesized instance Param instead of the abstract reference
        X_inst = m._reactance_dir
        # Extra diagnostics: membership and availability
        if logger.isEnabledFor(logging.DEBUG):
            try:
                A_dbg = getattr(m, 'DirectionalLink', None)
                C_dbg = getattr(m, 'CandidateTransmission', None)
                in_A_ij = (i, j) in A_dbg if A_dbg is not None else None
                in_A_ji = (j, i) in A_dbg if A_dbg is not None else None
                in_C_ij = (i, j) in C_dbg if C_dbg is not None else None
                in_C_ji = (j, i) in C_dbg if C_dbg is not None else None
                try:
                    has_X_ij = (i, j) in X_inst
                except Exception:
                    has_X_ij = None
                try:
                    has_X_ji = (j, i) in X_inst
                except Exception:
                    has_X_ji = None
                logger.debug(
                    "Arc (%s,%s): in DirectionalLink=%s (rev=%s); in Candidate=%s (rev=%s); X has ij=%s, ji=%s",
                    i, j, in_A_ij, in_A_ji, in_C_ij, in_C_ji, has_X_ij, has_X_ji
                )
            except Exception:
                pass
        try:
            xij = float(X_inst[i, j])
        except Exception as e_ij:
            logger.debug("_get_B_val: Directional reactance lookup failed for (%s,%s); trying reversed ordering.", i, j)
            try:
                xij = float(X_inst[j, i])
            except Exception as e_ji:
                logger.warning("Missing reactance for directed arc (%s,%s); using reciprocal of eps.", i, j)
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        logger.debug("Lookup errors: ij=%s, ji=%s", type(e_ij).__name__, type(e_ji).__name__)
                    except Exception:
                        pass
                xij = 1.0 / eps
        if abs(xij) < eps:
            logger.debug("_get_B_val: Reactance near zero for line (%s,%s); using minimum value for susceptance.", i, j)
            logger.warning("_get_B_val: Reactance near zero for line (%s,%s); using minimum value for susceptance.", i, j)
            xij = eps
        return 1.0 / xij
    

    # --- Existing / Candidate sets ---
    # CandidateTransmission: corridors that CAN BE EXPANDED (may have existing capacity)
    # ExistingTransmission: corridors that are FIXED (cannot be expanded)
    # 
    # IMPORTANT: Big-M formulation is used for candidate lines because:
    #   - Ohm's law should only apply when transmissionBuild = 1
    #   - Without big-M, zero-capacity candidates over-constrain angle differences
    #   - With big-M, Ohm's law is "deactivated" when transmissionBuild = 0
    
    if not hasattr(model, "CandidateTransmission"):
        raise RuntimeError("Model is missing set 'CandidateTransmission'.")
    CAND = model.CandidateTransmission
    
    # Reactance for candidate block (new circuit)
    if hasattr(model, "LineBlockReactance"):
        def _rx_cand_dir_init(m, i, j):

            # First check if (i,j) or (j,i) is in LineBlockReactance
            if (i, j) in m.LineBlockReactance:
                X_ohm = value(m.LineBlockReactance[i, j])
            elif (j, i) in m.LineBlockReactance:
                X_ohm = value(m.LineBlockReactance[j, i])
            elif hasattr(m, "LineBlockReactanceGlobal"):
                # No special reactance provided → check global candidate reactance
                X_ohm = value(m.LineBlockReactanceGlobal)
            else:
                # No special reactance provided → fallback to existing reactance
                return value(m._reactance_dir[i, j])

            # # Convert Ω → pu
            # V_kv = value(m.NominalVoltage)
            # V_base = V_kv * 1e3         # convert to volts
            # S_base = 100e6              # assume 100 MVA system base
            # X_pu = X_ohm * (V_base**2) / S_base

            return X_ohm

        model._reactance_cand_dir = Param(
            model.DirectionalLink,
            initialize=_rx_cand_dir_init,
            mutable=True
        )
    else:
        model._reactance_cand_dir = model._reactance_dir
        
    def _get_B_cand(m, i, j):
        X = value(m._reactance_cand_dir[i, j])
        return 1.0 / X

    # ExistingTransmission = lines that are in DirectionalLink but NOT expandable
    # (i.e., not in CandidateTransmission in either direction)
    if not hasattr(model, "ExistingTransmission"):
        def _existing_init(m):
            arcs = []
            for (i, j) in m.DirectionalLink:

                #  Find the corresponding node in BidirectionalArc
                if (i, j) in m.BidirectionalArc:
                    bi, bj = i, j
                elif (j, i) in m.BidirectionalArc:
                    bi, bj = j, i
                else:
                     # If for any reason this direction is not in BidirectionalArc, skip it
                    continue

                # If in any period the initial capacity > 0, consider this direction as an existing line
                for p in m.PeriodActive:
                    try:
                        if value(m.transmissionInitCap[bi, bj, p]) > 0:
                            arcs.append((i, j))
                            break  # It is enough that one period is positive
                    except Exception:
                        # If no data, do not add anything
                        pass
            return arcs

        model.ExistingTransmission = Set(within=A, initialize=_existing_init)
    EXIST = model.ExistingTransmission

    # --- Angle bounds & Big-M for candidate Ohm's law activation ---
    # Angle bounds for numerical stability (radians). You can adjust via .tab or after instance creation.
    if not hasattr(model, "AngleMax"):
        model.AngleMax = Param(default=0.6, mutable=True)  # ~34 degrees

    # --- Voltage magnitude squared (V²) for converting per-unit to actual values ---
    # This parameter converts the per-unit DC power flow equation to actual MW.
    # In per-unit: P = B * (θ_i - θ_j)
    # In actual units: P_MW = (V²/X) * (θ_i - θ_j) = B * V² * (θ_i - θ_j)
    # where V is the system nominal voltage magnitude in kV
    # NominalVoltage is defined in operational.py and loaded from General.xlsx (NominalVoltage sheet)
    # Compute VoltageSquared from NominalVoltage (V² = V_kV²)
    if not hasattr(model, "VoltageSquared"):
        def _voltage_squared_init(m):
            return value(m.NominalVoltage) ** 2
        model.VoltageSquared = Param(initialize=_voltage_squared_init, mutable=True)

    # Bus angles with bounds ±AngleMax
    def _theta_bounds(m, *idx):
        amax = value(m.AngleMax)
        return (-amax, amax)

    model.Theta = Var(N, H, W, P, domain=Reals, bounds=_theta_bounds)
    
    model.Flow_exist = Var(EXIST, H, W, P, domain=Reals)
    model.Flow_new   = Var(CAND_DIR, H, W, P, domain=Reals)


    def total_flow_expr(m, i, j, h, w, p):
        base = m.Flow_exist[i,j,h,w,p] \
            if (i,j) in m.ExistingTransmission else 0.0

        new = m.Flow_new[i,j,h,w,p] \
            if (i,j) in m.CandidateDirectional else 0.0

        return base + new

    model.FlowDC = Expression(
        model.DirectionalLink,
        model.Operationalhour,
        model.Scenario,
        model.PeriodActive,
        rule=total_flow_expr
    )

    # If caller didn’t pass a capacity expression, infer it (this should end up using transmissionInstalledCap)
    if capacity_expr is None:
        capacity_expr = _infer_capacity_expr(model)

    model.CapacityDir = Expression(A, P, rule=lambda m,i,j,p: capacity_expr(m,i,j,p))

    # --- Big-M per candidate arc: M_flow[i,j] ≈ |B[i,j]| * V² * (2*AngleMax) ---
    # Big-M must be large enough to relax Ohm's law when transmissionBuild = 0
    # Updated to include V² scaling for actual units.
    eps_local = eps
    if hasattr(model, "BigMFlow"):
        logger.debug("BigMFlow already defined; skipping re-definition.")
    else:
        def _bigm_expr(m, i, j):
            # Use the synthesized directional reactance Param directly.
            try:
                xval = float(m._reactance_dir[i, j])
            except Exception:
                # Fallback: rely on _get_B_cand (will log once if truly missing)
                Bij = _get_B_cand(m, i, j)
                return abs(Bij) * value(m.VoltageSquared) * 2.0 * value(m.AngleMax)
            if xval < eps_local:  # guard near-zero reactance
                xval = eps_local
            Bij = 1.0 / xval
            return abs(Bij) * value(m.VoltageSquared) * 2.0 * value(m.AngleMax)
        model.BigMFlow = Expression(CAND, rule=_bigm_expr)

    # -----------------------
    # Ohm's law constraints
    # -----------------------
    # DC power flow equation in actual units:
    # P_MW = (V²/X) * (θ_i - θ_j) = B * V² * (θ_i - θ_j)
    # where V² is the voltage magnitude squared in kV²

    # 1) Existing lines: equality always active (these have fixed capacity)
    def ohm_exist_rule(m, i, j, h, w, p):
        B = _get_B_val(m, i, j)   
        return m.Flow_exist[i,j,h,w,p] == B * m.VoltageSquared * (m.Theta[i,h,w,p] - m.Theta[j,h,w,p])
    model.Ohm_exist = Constraint(
        model.ExistingTransmission,
        model.Operationalhour,
        model.Scenario,
        model.PeriodActive,
        rule=ohm_exist_rule
    )

    # 2) Candidate lines: big-M activation using binary build var
    #    When transmissionOn = 0, Ohm's law is relaxed (deactivated)
    #    When transmissionOn = 1, Ohm's law is enforced (activated)
    if not hasattr(model, "transmissionOn"):
        raise RuntimeError("Angle-based LOPF with variable susceptance requires 'transmissionOn[n1,n2,p]' to be defined in the investment model.")

    
    # Upper & lower linearized envelopes:
    def ohm_cand_ub(m, i, j, h, w, p):
        B = _get_B_cand(m, i, j)
        # Find corridor representative in CandidateTransmission:
        if (i, j) in m.CandidateTransmission:
            ci, cj = i, j
        elif (j, i) in m.CandidateTransmission:
            ci, cj = j, i
        else:
            raise ValueError(f"Directional arc ({i},{j}) not found in CandidateTransmission as ({i},{j}) or ({j},{i})")
        M = m.BigMFlow[ci, cj]     
        t_diff = m.Theta[i,h,w,p] - m.Theta[j,h,w,p]
        return m.Flow_new[i,j,h,w,p] <= B * m.VoltageSquared * t_diff + M * (1 - m.transmissionOn[ci, cj, p])

    def ohm_cand_lb(m, i, j, h, w, p):
        B = _get_B_cand(m, i, j)
        # Find corridor representative in CandidateTransmission:
        if (i, j) in m.CandidateTransmission:
            ci, cj = i, j
        elif (j, i) in m.CandidateTransmission:
            ci, cj = j, i
        else:
            raise ValueError(f"Directional arc ({i},{j}) not found in CandidateTransmission as ({i},{j}) or ({j},{i})")
        M = m.BigMFlow[ci, cj]
        t_diff = m.Theta[i,h,w,p] - m.Theta[j,h,w,p]
        return m.Flow_new[i,j,h,w,p] >= B * m.VoltageSquared * t_diff - M * (1 - m.transmissionOn[ci, cj, p])
    model.OhmLawDC_Cand_UB = Constraint(CAND_DIR, H, W, P, rule=ohm_cand_ub)
    model.OhmLawDC_Cand_LB = Constraint(CAND_DIR, H, W, P, rule=ohm_cand_lb)

    # -----------------------
    # Thermal limits (all arcs)
    # -----------------------
    def flow_cap_up(m, i, j, h, w, p):
        return  m.FlowDC[i,j,h,w,p] <= m.CapacityDir[i,j,p]
    def flow_cap_lo(m, i, j, h, w, p):
        return -m.CapacityDir[i,j,p] <= m.FlowDC[i,j,h,w,p]
    model.FlowCapUp = Constraint(A, H, W, P, rule=flow_cap_up)
    model.FlowCapLo = Constraint(A, H, W, P, rule=flow_cap_lo)
    
    
    # Capacity limits for candidate lines (new flow component)
    def cap_new_up_rule(m, i, j, h, w, p):
        # Find the correct corridor representative in CandidateTransmission:
        if (i, j) in m.CandidateTransmission:
            ci, cj = i, j
        elif (j, i) in m.CandidateTransmission:
            ci, cj = j, i
        else:
            raise ValueError(f"Directional arc ({i},{j}) not found in CandidateTransmission as ({i},{j}) or ({j},{i})")

        # The capacity expression for candidate lines:
        if hasattr(m, "transmissionLineBlockCap") and (ci, cj) in m.transmissionLineBlockCap:
            cap = m.transmissionLineBlockCap[ci, cj]
        else:
            cap = m.transmissionLineBlockCapGlobal  # If there is a global block capacity

        on = m.transmissionOn[ci, cj, p]
        return m.Flow_new[i,j,h,w,p] <= cap * on

    def cap_new_lo_rule(m, i, j, h, w, p):
        # Find the correct corridor representative in CandidateTransmission:
        if (i, j) in m.CandidateTransmission:
            ci, cj = i, j
        elif (j, i) in m.CandidateTransmission:
            ci, cj = j, i
        else:
            raise ValueError(f"Directional arc ({i},{j}) not found in CandidateTransmission as ({i},{j}) or ({j},{i})")

        # The capacity expression for candidate lines:
        if hasattr(m, "transmissionLineBlockCap") and (ci, cj) in m.transmissionLineBlockCap:
            cap = m.transmissionLineBlockCap[ci, cj]
        else:
            cap = m.transmissionLineBlockCapGlobal  # If there is a global block capacity

        on = m.transmissionOn[ci, cj, p]
        return -cap * on <= m.Flow_new[i,j,h,w,p]

    model.CapNewUp = Constraint(CAND_DIR, H, W, P, rule=cap_new_up_rule)
    model.CapNewLo = Constraint(CAND_DIR, H, W, P, rule=cap_new_lo_rule)
    # -----------------------
    # Angle reference (slack)
    # -----------------------
    if fix_angle_reference:
        # Determine slack node at instance-construction time inside the constraint rule
        def _angle_ref_rule(m, h, w, p):
            # Prefer an explicit SlackNode set if provided
            slack_n = None
            if hasattr(m, slack_node_set_name):
                for nn in getattr(m, slack_node_set_name):
                    slack_n = nn
                    break
            if slack_n is None:
                for nn in m.Node:
                    slack_n = nn
                    break
            if slack_n is None:
                raise RuntimeError("No nodes available to fix angle reference.")
            return m.Theta[slack_n, h, w, p] == 0.0

        model.AngleRef = Constraint(H, W, P, rule=_angle_ref_rule)

    # -----------------------
    # Bind to existing directed flow var (keep KCL intact)
    # -----------------------
    if couple_to_existing_flows:
        FlowDir = None
        for nm in existing_flow_candidates:
            if hasattr(model, nm):
                FlowDir = getattr(model, nm)
                logger.info(f"Coupling DC-OPF flows to existing variable '{nm}'")
                break
        if FlowDir is not None:
            model.DC_Bind = Constraint(A, H, W, P, rule=lambda m,i,j,h,w,p: FlowDir[i,j,h,w,p] == m.FlowDC[i,j,h,w,p])
            logger.info("DC-OPF flow binding constraint (DC_Bind) added successfully")
        else:
            logger.warning(f"Could not find existing flow variable to bind. Tried: {existing_flow_candidates}")
    
    # Log completion summary
    logger.info("=" * 70)
    logger.info("ANGLE-BASED DC-OPF SETUP COMPLETE")
    logger.info("=" * 70)
    logger.info("  Bus angle variables (Theta) created for all nodes")
    logger.info("  DC flow variables (FlowDC) created for all directional arcs")
    logger.info("  Constraints added:")
    logger.info("    - Ohm's law for existing lines (OhmLawDC_Exist)")
    logger.info("    - Big-M Ohm's law for candidate lines (OhmLawDC_Cand_UB/LB)")
    logger.info("    - Thermal capacity limits (FlowCapUp/Lo)")
    if fix_angle_reference:
        logger.info(f"    - Angle reference constraint (AngleRef) for slack node")
    logger.info(f"  Angle bounds: defined by AngleMax parameter (default ±0.6 rad or ±34°)")
    logger.info(f"  Flow coupling: {'Active' if couple_to_existing_flows else 'Inactive'}")
    logger.info("=" * 70)
    
    return model


def _add_ptdf_constraints(model, **kwargs):
    """Placeholder for a PTDF-based DC-OPF formulation.

    This is not yet implemented. The router can call this stub,
    so keep a clear exception until the implementation is provided.
    """
    raise NotImplementedError("PTDF formulation is not implemented yet.")


def load_line_parameters(model, tab_file_path, data, lopf_kwargs, logger):
    """Load line parameters for linear OPF
    Read kwargs to see whether we should derive X from B

    Args:
        model (_type_): _description_
        tab_file_path (_type_): _description_
        data (_type_): _description_
        LOPF_KWARGS (_type_): _description_
        logger (_type_): _description_
    """
    rx_from_b = bool(lopf_kwargs and lopf_kwargs.get("reactance_from_susceptance", False))

    # Try to load reactance (preferred for Kirchhoff formulation)
    reactance_tab = tab_file_path / 'Transmission_lineReactance.tab'
    susceptance_tab = tab_file_path / 'Transmission_lineSusceptance.tab'
    block_reactance_tab = tab_file_path / 'Transmission_LineBlockReactance.tab'
    block_reactance_tab_global = tab_file_path / 'General_LineBlockReactanceGlobal.tab'
    
    if block_reactance_tab.exists():
        data.load(filename=str(block_reactance_tab), param=model.LineBlockReactance, format="table", range="LineBlockReactance")
        logger.info("Loaded Transmission_LineBlockReactance.tab for DC-OPF.")
    elif block_reactance_tab_global.exists():
        data.load(filename=str(block_reactance_tab_global), param=model.LineBlockReactanceGlobal, format="value", range="LineBlockReactanceGlobal")
        logger.info("Loaded General_LineBlockReactanceGlobal.tab for DC-OPF.")

    if reactance_tab.exists() and not rx_from_b:
        data.load(filename=str(reactance_tab), param=model.lineReactance, format="table")
        logger.info("Loaded Transmission_lineReactance.tab for DC-OPF.")
        logger.debug("lineReactance dimen: %s", model.lineReactance.index_set().dimen)
        #logger.debug("lineReactance keys: %s", list(model.lineReactance.keys())[:10])
    elif susceptance_tab.exists():
        data.load(filename=str(susceptance_tab), param=model.lineSusceptance, format="table")
        logger.info("Loaded Transmission_lineSusceptance.tab for DC-OPF (will invert to reactance if configured).")
    else:
        logger.warning("No electrical line parameter (.tab) found for DC-OPF (reactance/susceptance). "
                    "If using Kirchhoff, set lopf_kwargs.reactance_from_susceptance=True and provide susceptance, "
                    "or provide Transmission_lineReactance.tab.")
    
    # Load nominal voltage for converting per-unit to actual values
    # The voltage is loaded in kV and squared internally to get V² in kV²
    nominal_voltage_tab = tab_file_path / 'General_NominalVoltage.tab'
    if nominal_voltage_tab.exists():
        data.load(filename=str(nominal_voltage_tab), param=model.NominalVoltage, format="table")
        logger.info("Loaded General_NominalVoltage.tab for DC-OPF actual unit conversion.")
    else:
        logger.info("General_NominalVoltage.tab not found; using default value (400 kV, V²=160,000 kV²).")