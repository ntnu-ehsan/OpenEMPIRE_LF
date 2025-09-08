from __future__ import annotations
from pyomo.environ import (Set, Var, Constraint, NonNegativeReals, value, Param, Expression, Reals, ConstraintList)
from collections import defaultdict, deque
from typing import  Dict,List, Tuple, Optional, Callable

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
    """Return (m,i,j,p) -> capacity expression for undirected (i,j)."""
    # helper: look up component by name on the instance
    def _by_name(name: str):
        return lambda m, i, j, p, _n=name: getattr(m, _n)[i, j, p]

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
            for pp in m.periods_active:
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

    model.LOPF_BindFwd = Constraint(model.BidirectionalArc, model.Operationalhour, model.scenarios, model.periods_active, rule=_fwd)
    model.LOPF_BindBwd = Constraint(model.BidirectionalArc, model.Operationalhour, model.scenarios, model.periods_active, rule=_bwd)
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
        "transmissionOperational",   # EMPIRE’s usual name
        "transmisionOperational",    # common misspelling in some code paths
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
    for s in ("BidirectionalArc", "DirectionalLink", "Node", "Operationalhour", "scenarios", "periods_active"):
        if not hasattr(model, s):
            raise ValueError(f"Model is missing required set: {s}")

    L = model.BidirectionalArc
    H, W, P = model.Operationalhour, model.scenarios, model.periods_active

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
def _add_angle_opf(
    model,
    *,
    susceptance_param_name: str = "lineSusceptance",  # on DirectionalLink
    capacity_expr: Optional[Callable] = None,
    couple_to_existing_flows: bool = True,
    existing_flow_candidates = ("transmissionOperational", "TransFlow", "lineFlow", "flow"),
    fix_angle_reference: bool = True,
    slack_node_set_name: str = "SlackNode",
):
    # Required sets
    for s in ("DirectionalLink", "Node", "Operationalhour", "scenarios", "periods_active"):
        if not hasattr(model, s):
            raise RuntimeError(f"Model is missing required set '{s}'")
    A = model.DirectionalLink
    N = model.Node
    H, W, P = model.Operationalhour, model.scenarios, model.periods_active

    # Susceptance B[(i,j)]
    if not hasattr(model, susceptance_param_name):
        # allow creation if absent; user can populate later
        setattr(model, susceptance_param_name, Param(A, default=0.0, mutable=True))
    B = getattr(model, susceptance_param_name)

    # Variables
    model.Theta = Var(N, H, W, P, domain=Reals)         # bus angles
    model.FlowDC = Var(A, H, W, P, domain=Reals)         # directed DC flow

    # Capacity
    if capacity_expr is None:
        capacity_expr = _infer_capacity_expr(model)
    model.CapacityDir = Expression(A, P, rule=lambda m,i,j,p: capacity_expr(m,i,j,p))

    # Ohm's law
    model.OhmLawDC = Constraint(A, H, W, P,
        rule=lambda m,i,j,h,w,p: m.FlowDC[i,j,h,w,p] == B[i,j] * (m.Theta[i,h,w,p] - m.Theta[j,h,w,p]))

    # Thermal limits
    model.FlowCapUp = Constraint(A, H, W, P, rule=lambda m,i,j,h,w,p:  m.FlowDC[i,j,h,w,p] <= m.CapacityDir[i,j,p])
    model.FlowCapLo = Constraint(A, H, W, P, rule=lambda m,i,j,h,w,p: -m.CapacityDir[i,j,p] <= m.FlowDC[i,j,h,w,p])

    # Angle reference
    if fix_angle_reference:
        # pick a slack node
        slack = None
        if hasattr(model, slack_node_set_name):
            for n in getattr(model, slack_node_set_name):
                slack = n; break
        if slack is None:
            for n in N:
                slack = n; break
        if slack is None:
            raise RuntimeError("No nodes available to fix angle reference.")
        model.AngleRef = Constraint(H, W, P, rule=lambda m,h,w,p, _n=slack: m.Theta[_n,h,w,p] == 0.0)

    # Bind to existing directed flow (so KCL stays unchanged)
    if couple_to_existing_flows:
        # here we *equate* FlowDC (this method’s flow) to existing flow var
        FlowDir = None
        for nm in existing_flow_candidates:
            if hasattr(model, nm):
                FlowDir = getattr(model, nm)
                break
        if FlowDir is not None:
            model.DC_Bind = Constraint(A, H, W, P, rule=lambda m,i,j,h,w,p: FlowDir[i,j,h,w,p] == m.FlowDC[i,j,h,w,p])


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

    if reactance_tab.exists() and not rx_from_b:
        data.load(filename=str(reactance_tab), param=model.lineReactance, format="table")
        logger.info("Loaded Transmission_lineReactance.tab for DC-OPF.")
    elif susceptance_tab.exists():
        data.load(filename=str(susceptance_tab), param=model.lineSusceptance, format="table")
        logger.info("Loaded Transmission_lineSusceptance.tab for DC-OPF (will invert to reactance if configured).")
    else:
        logger.warning("No electrical line parameter (.tab) found for DC-OPF (reactance/susceptance). "
                    "If using Kirchhoff, set lopf_kwargs.reactance_from_susceptance=True and provide susceptance, "
                    "or provide Transmission_lineReactance.tab.")