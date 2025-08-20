from __future__ import annotations
from pyomo.environ import (Var, Constraint, NonNegativeReals, value, Param, Expression, Reals)
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
    # Prefer installed cap variable
    for name in ("transmissionInstalledCap", "installedTransmissionCap", "TransInstalledCap"):
        if hasattr(model, name):
            cap = getattr(model, name)
            return lambda m, i, j, p, _cap=cap: _cap[i, j, p]

    # Build from init + built
    init = getattr(model, "transmissionInitCap", None)
    built = None
    for name in ("transmissionBuilt", "TransBuilt", "lineExpansion"):
        if hasattr(model, name):
            built = getattr(model, name)
            break
    if init is not None and built is not None:
        def _cap_from_init_plus_built(m, i, j, p, _init=init, _built=built):
            total = _init[i, j, p] if (i, j, p) in _init else 0.0
            for pp in m.PeriodActive:
                if pp <= p and (i, j, pp) in _built:
                    total = total + _built[i, j, pp]
            return total
        return _cap_from_init_plus_built

    # Fallback params
    for name in ("transmissionInitCap", "transmissionMaxInstalledCap"):
        if hasattr(model, name):
            cap = getattr(model, name)
            return lambda m, i, j, p, _cap=cap: _cap[i, j, p]

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
    reactance_param_name: str = "lineReactance", # on BidirectionalArc
    susceptance_param_name: str = "lineSusceptance", # alt source if reactance missing
    reactance_from_susceptance: bool = False,
    capacity_expr: Optional[Callable] = None,
    couple_to_existing_flows: bool = True,
    existing_flow_candidates: ("transmisionOperational"),
    store_debug: bool = False,
):
    #Required model sets
    for s in ("BidirectionalArc", "DirectionalLink", "Node", "Operationalhour", "Scenario", "PeriodActive"):
        if not hasattr(model, s):
            raise ValueError(f"Model is missing required set: {s}")

    L = model.BidirectionalArc
    H, W, P = model.Operationalhour, model.Scenario, model.PeriodActive

    # Reactance X[i,j]
    if hasattr(model, reactance_param_name):
        X = getattr(model, reactance_param_name)
    elif reactance_from_susceptance:
        if not hasattr(model, susceptance_param_name):
            raise RuntimeError(f"Need '{reactance_param_name}' or '{susceptance_param_name}'")
        B = getattr(model, susceptance_param_name)
        eps = 1e-7
        def _x_rule(m, i, j):
            bij = B[i, j]
            return 1.0 / (bij if abs(bij) > eps else (eps if bij >= 0 else -eps))
        model.LOPF_Reactance = Expression(L, rule=_x_rule)
        X = model.LOPF_Reactance
    else:
        raise RuntimeError(f"Provide '{reactance_param_name}' or set reactance_from_susceptance=True.")

    # flow on undirected arc (i,j) in period p
    model.FlowK = Var(L, H, W, P, domain=Reals)
    
    # Capacity expression
    if capacity_expr is None:
        capacity_expr = _infer_capacity_expr(model)
    model.LineCapacity = Expression(L, P, rule=lambda m, i, j, p: capacity_expr(m, i, j, p))

    # Thermal limits
    model.KVL_flowCapUp = Constraint(L, H, W, P, rule=lambda m, i, j, h, w, p:
        m.FlowK[i, j, h, w, p] <= m.LineCapacity[i, j, p] 
    )
    model.KVL_FlowCapLo = Constraint(L, H, W, P, rule=lambda m,i,j,h,w,p: 
    -m.LineCapacity[i,j,p] <= m.FlowK[i,j,h,w,p]
    )

    # Bind to existing directed flow (so KCL stays unchanged)
    if couple_to_existing_flows:
        _bind_to_existing_flows(model, model.FlowK, existing_flow_candidates)

    # Build fundamental cycles and add KVL
    cycles, edge_signs = _fundamental_cycles(model.Node, L)
    if store_debug:
        model._KVL_cycles = cycles
        model._KVL_edge_signs = edge_signs
    model.KVL_Index = range(len(cycles))
    def _kvl_rule(m, c_idx, h, w, p):
        cyc = cycles[c_idx]
        return sum(edge_signs[(c_idx, e)] * X[e] * m.FlowK[e[0], e[1], h, w, p] for e in cyc) == 0.0
    model.KVL_Constraints = Constraint(model.KVL_Index, H, W, P, rule=_kvl_rule)


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
    for s in ("DirectionalLink", "Node", "Operationalhour", "Scenario", "PeriodActive"):
        if not hasattr(model, s):
            raise RuntimeError(f"Model is missing required set '{s}'")
    A = model.DirectionalLink
    N = model.Node
    H, W, P = model.Operationalhour, model.Scenario, model.PeriodActive

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