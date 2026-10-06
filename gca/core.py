"""Shared decision core: utilities, beliefs, value of alignment (VOA), gates and fusion rules.

The acting specialist (the arm) must choose one of three skills each decision step:
  pull        - keep pulling the handle
  reposition  - stop and brace the base against the cabinet
  release     - stop and back off the force
Two binary latent variables decide which skill is right:
  s  - the cabinet is sliding instead of the drawer opening   (process violation if pulled)
  m  - the pulling force is near the declared force limit     (force violation if pulled)
Each neighbor specialist holds evidence about one latent. Opening an alignment edge to a
neighbor costs c and returns a binary report with a known accuracy. The gate opens an edge only
when the expected decision gain exceeds the cost: the COO refinement criterion with the
"refined observation" read as "my belief plus the neighbor's report".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ACTIONS = ("pull", "reposition", "release")

# Declared per-step utilities U[a][(s, m)].
UTIL = {
    "pull": {(0, 0): 1.0, (1, 0): -3.0, (0, 1): -3.0, (1, 1): -3.0},
    "reposition": {(0, 0): -0.3, (1, 0): 0.5, (0, 1): -0.3, (1, 1): -0.3},
    "release": {(0, 0): -0.3, (1, 0): -0.3, (0, 1): 0.5, (1, 1): 0.5},
}


def expected_utility(action: str, p: float, q: float) -> float:
    """p = P(s=1 sliding), q = P(m=1 near limit); treated as independent."""
    u = UTIL[action]
    return ((1 - p) * (1 - q) * u[(0, 0)] + p * (1 - q) * u[(1, 0)]
            + (1 - p) * q * u[(0, 1)] + p * q * u[(1, 1)])


def best_action(p: float, q: float) -> tuple[str, float]:
    vals = {a: expected_utility(a, p, q) for a in ACTIONS}
    a = max(ACTIONS, key=lambda k: vals[k])
    return a, vals[a]


def bayes_update(prior: float, report: int, acc: float, tnr: float | None = None) -> float:
    """Posterior P(latent=1) after a binary report. acc = P(report=1 | latent=1) (true-positive
    rate); tnr = P(report=0 | latent=0), defaulting to acc for a symmetric sensor."""
    tnr = acc if tnr is None else tnr
    l1 = acc if report == 1 else 1 - acc
    l0 = (1 - tnr) if report == 1 else tnr
    num = prior * l1
    den = num + (1 - prior) * l0
    return num / den if den > 0 else prior


@dataclass
class Edge:
    name: str          # neighbor specialist
    latent: str        # "s" or "m"
    acc: float         # P(report=1 | latent=1), the neighbor's true-positive rate
    cost: float        # declared alignment cost (utility units)
    tnr: float | None = None   # P(report=0 | latent=0); None = symmetric (= acc)


def fused_belief(prior: float, report: int, acc: float, fuse: str, tnr: float | None = None) -> float:
    """Belief after one neighbor report under a fusion rule.

    exact: Bayes with the neighbor's accuracy (pooled evidence, common prior counted once).
    can:   the consensus of fixed-weight ring diffusion on two nodes, i.e. the mean of the two
           beliefs, where the neighbor's belief is its own report confidence. This is the
           fixed-weight CAN mixing rule used as a fusion module behind the same gate.
    """
    tnr = acc if tnr is None else tnr
    if fuse == "exact":
        return bayes_update(prior, report, acc, tnr)
    if fuse == "can":
        # the neighbor's own belief is its report confidence: precision of a positive / negative report
        # at a flat prior, which is what a fixed-weight pool mixes in
        nb = acc / (acc + (1 - tnr)) if report == 1 else (1 - acc) / ((1 - acc) + tnr)
        return 0.5 * prior + 0.5 * nb
    raise ValueError(fuse)


def voa(edge: Edge, p: float, q: float, fuse: str = "exact") -> float:
    """Value of alignment for one edge given the arm's current belief (preposterior analysis).

    The realised utility of acting on the fused belief is scored under the exact posterior, so a
    fusion rule that mis-weights evidence pays for it inside the VOA itself.
    """
    prior = p if edge.latent == "s" else q
    tnr = edge.acc if edge.tnr is None else edge.tnr
    _, base = best_action(p, q)
    val = 0.0
    for report in (1, 0):
        p_report = prior * (edge.acc if report else 1 - edge.acc) + (1 - prior) * ((1 - tnr) if report else tnr)
        post = fused_belief(prior, report, edge.acc, fuse, tnr)
        pp, qq = (post, q) if edge.latent == "s" else (p, post)
        a, _ = best_action(pp, qq)
        exact_post = bayes_update(prior, report, edge.acc, tnr)
        ep, eq = (exact_post, q) if edge.latent == "s" else (p, exact_post)
        val += p_report * expected_utility(a, ep, eq)
    return val - base - edge.cost


@dataclass
class GateResult:
    opened: list[str] = field(default_factory=list)
    p: float = 0.0
    q: float = 0.0
    action: str = "pull"


def run_gate(policy: str, edges: list[Edge], p: float, q: float, reports: dict[str, int],
             truth: dict[str, int] | None = None, fuse: str = "exact",
             event_flags: dict[str, bool] | None = None, random_rate: dict[str, float] | None = None,
             rng=None) -> GateResult:
    """Apply one alignment policy at one decision step.

    policy in {never, always, event, random, voa, oracle}
    reports: the report each neighbor would give if its edge were opened
    truth:   true latent values (oracle only)
    """
    res = GateResult(p=p, q=q)
    closed = list(edges)

    def open_edge(e: Edge):
        prior = res.p if e.latent == "s" else res.q
        post = fused_belief(prior, reports[e.name], e.acc, fuse, e.tnr)
        if e.latent == "s":
            res.p = post
        else:
            res.q = post
        res.opened.append(e.name)
        closed.remove(e)

    if policy == "always":
        for e in list(closed):
            open_edge(e)
    elif policy == "never":
        pass
    elif policy == "event":
        for e in list(closed):
            if event_flags and event_flags.get(e.name, False):
                open_edge(e)
    elif policy == "random":
        for e in list(closed):
            if rng.random() < random_rate.get(e.name, 0.0):
                open_edge(e)
    elif policy == "voa":
        while closed:
            vals = [(voa(e, res.p, res.q, fuse), e) for e in closed]
            v, e = max(vals, key=lambda t: t[0])
            if v <= 0:
                break
            open_edge(e)
    elif policy == "oracle":
        a0, _ = best_action(res.p, res.q)
        for e in list(closed):
            pp, qq = (float(truth["s"]), res.q) if e.latent == "s" else (res.p, float(truth["m"]))
            a1, _ = best_action(pp, qq)
            if a1 != a0:
                open_edge(e)
                a0, _ = best_action(res.p, res.q)
    else:
        raise ValueError(policy)
    res.action, _ = best_action(res.p, res.q)
    return res


def can_rounds_to_consensus(n: int, tol: float = 1e-4, w_self: float = 0.75) -> int:
    """Rounds for fixed-weight ring diffusion on n nodes to reach consensus within tol (cap 100)."""
    if n <= 1:
        return 0
    lam = max(abs(w_self + (1 - w_self) * math.cos(2 * math.pi * k / n)) for k in range(1, n))
    if lam >= 1:
        return 100
    return min(math.ceil(math.log(tol) / math.log(lam)), 100)


def always_align_cost(n: int, fuse: str) -> float:
    """Messages per decision step when every edge is aligned.

    exact: each neighbor sends one summary to the actor (n-1 messages).
    can:   ring diffusion, every node sends to two neighbors each round until consensus.
    """
    if fuse == "exact":
        return float(n - 1)
    return float(2 * n * can_rounds_to_consensus(n))


def theorem_precondition(edge: Edge, p: float, q: float) -> bool:
    """Finite check of the strict-value precondition: the coarse belief aliases cases whose
    optimal actions differ, and the neighbor's report separates them."""
    prior = p if edge.latent == "s" else q
    acts = set()
    for report in (1, 0):
        post = bayes_update(prior, report, edge.acc, edge.tnr)
        pp, qq = (post, q) if edge.latent == "s" else (p, post)
        acts.add(best_action(pp, qq)[0])
    return len(acts) > 1
