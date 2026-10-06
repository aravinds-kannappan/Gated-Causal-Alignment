"""Stage A: synthetic POSCM drawer task, exact enumeration over a declared design distribution.

The POSCM: applied force f -> handle displacement h -> drawer position d; cabinet displacement x;
contact state k. Exogenous: cabinet anchored or not (A), drawer resistance (R), force limit (L).
Specialist subgraphs:
  arm      sees {f, h, k}        -> weak own evidence about s (sliding) and m (near limit)
  base     sees {x, pose}        -> strong evidence about s
  verifier sees {f, L, x}        -> exact evidence about m
  head     sees {x} (noisy)      -> redundant with base
  torso    sees {pose}           -> irrelevant to the decision
Everything below is computed by exact enumeration of worlds, own evidence and neighbor reports
(weighted by their probabilities), except the N-scaling sweep which uses Monte Carlo.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gca.core import (Edge, UTIL, always_align_cost, bayes_update, best_action, can_rounds_to_consensus,
                      run_gate, theorem_precondition, voa)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "stage_a")
POLICIES = ("never", "always", "event", "random", "voa", "oracle")


def enumerate_design(pi_s, pi_m, a_s, a_m, edges, fuse, cost_scale=1.0, random_rate=None, event_thr=0.35):
    """Exact expectation over worlds (s, m), own evidence (e_s, e_m) and neighbor reports."""
    rng = random.Random(0)
    edges = [Edge(e.name, e.latent, e.acc, e.cost * cost_scale) for e in edges]
    rows = {pol: dict(util=0.0, cost=0.0, preserved=0.0, violations=0.0, stops=0.0, opened={e.name: 0.0 for e in edges}) for pol in POLICIES}
    rows["oracle_truth"] = dict(util=0.0, cost=0.0, preserved=0.0, violations=0.0, stops=0.0, opened={})
    for s, m, e_s, e_m in itertools.product((0, 1), repeat=4):
        w = (pi_s if s else 1 - pi_s) * (pi_m if m else 1 - pi_m)
        w *= (a_s if e_s == s else 1 - a_s) * (a_m if e_m == m else 1 - a_m)
        p = bayes_update(pi_s, e_s, a_s)
        q = bayes_update(pi_m, e_m, a_m)
        # enumerate neighbor reports
        for reps in itertools.product((0, 1), repeat=len(edges)):
            wr = w
            reports = {}
            for e, r in zip(edges, reps):
                truth_e = s if e.latent == "s" else m
                wr *= e.acc if r == truth_e else 1 - e.acc
                reports[e.name] = r
            if wr == 0:
                continue
            flags = {e.name: (p > event_thr if e.latent == "s" else q > event_thr) for e in edges}
            ref = run_gate("always", edges, p, q, reports, fuse="exact")
            a_true, _ = best_action(float(s), float(m))
            rows["oracle_truth"]["util"] += wr * UTIL[a_true][(s, m)]
            for pol in POLICIES:
                res = run_gate(pol, edges, p, q, reports, truth={"s": s, "m": m}, fuse=fuse,
                               event_flags=flags, random_rate=random_rate or {}, rng=rng)
                r = rows[pol]
                r["util"] += wr * UTIL[res.action][(s, m)]
                r["cost"] += wr * sum(e.cost for e in edges if e.name in res.opened)
                r["preserved"] += wr * (res.action == ref.action)
                r["violations"] += wr * (res.action == "pull" and (s == 1 or m == 1))
                r["stops"] += wr * (res.action != "pull" and s == 0 and m == 0)
                for n in res.opened:
                    r["opened"][n] += wr
    return rows


def base_edges(acc_base=0.95, acc_ver=0.99, c=0.1):
    return [Edge("base", "s", acc_base, c), Edge("verifier", "m", acc_ver, c)]


def sweep_own_evidence(fuse="exact"):
    """Overlap sweep: how much the arm's own subgraph already tells it about s and m."""
    out = []
    for a_own in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        edges = base_edges()
        pilot = enumerate_design(0.3, 0.2, a_own, a_own, edges, fuse)
        rate = {k: v for k, v in pilot["voa"]["opened"].items()}
        rows = enumerate_design(0.3, 0.2, a_own, a_own, edges, fuse, random_rate=rate)
        for pol, r in rows.items():
            out.append(dict(sweep="own_evidence", fuse=fuse, a_own=a_own, policy=pol, util=r["util"], cost=r["cost"],
                            preserved=r["preserved"], violations=r["violations"], stops=r["stops"],
                            open_base=r["opened"].get("base", 0.0), open_verifier=r["opened"].get("verifier", 0.0)))
    return out


def sweep_cost(fuse="exact"):
    out = []
    for c in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
        edges = base_edges(c=c)
        pilot = enumerate_design(0.3, 0.2, 0.65, 0.65, edges, fuse)
        rate = {k: v for k, v in pilot["voa"]["opened"].items()}
        rows = enumerate_design(0.3, 0.2, 0.65, 0.65, edges, fuse, random_rate=rate)
        for pol, r in rows.items():
            out.append(dict(sweep="cost", fuse=fuse, cost_per_edge=c, policy=pol, util=r["util"], cost=r["cost"],
                            preserved=r["preserved"], violations=r["violations"], stops=r["stops"],
                            open_base=r["opened"].get("base", 0.0), open_verifier=r["opened"].get("verifier", 0.0)))
    return out


def sweep_prior(fuse="exact"):
    out = []
    for pi_s in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        edges = base_edges()
        pilot = enumerate_design(pi_s, 0.2, 0.65, 0.65, edges, fuse)
        rate = {k: v for k, v in pilot["voa"]["opened"].items()}
        rows = enumerate_design(pi_s, 0.2, 0.65, 0.65, edges, fuse, random_rate=rate)
        for pol, r in rows.items():
            out.append(dict(sweep="prior", fuse=fuse, pi_s=pi_s, policy=pol, util=r["util"], cost=r["cost"],
                            preserved=r["preserved"], violations=r["violations"], stops=r["stops"],
                            open_base=r["opened"].get("base", 0.0), open_verifier=r["opened"].get("verifier", 0.0)))
    return out


def theorem_check():
    """Exhaustive finite check: zero-cost VOA > 0 iff the aliasing-and-separation precondition."""
    agree = total = 0
    positives = 0
    grid = [i / 20 for i in range(1, 20)]
    for p in grid:
        for q in grid:
            for acc in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
                for latent in ("s", "m"):
                    e = Edge("n", latent, acc, 0.0)
                    v = voa(e, p, q, "exact")
                    pre = theorem_precondition(e, p, q)
                    total += 1
                    positives += v > 1e-12
                    agree += (v > 1e-12) == pre
    return dict(cases=total, agree=agree, positive=positives, ties=total - agree,
                note="disagreements are exact ties: VOA is 0 to floating precision while two actions have equal expected utility")


def sweep_n(fuse="exact", samples=4000, seed=0):
    """N-scaling by Monte Carlo. Specialists beyond the base and verifier are redundant copies
    (lower-accuracy reports about s) or irrelevant (report about nothing decision-relevant)."""
    rng = random.Random(seed)
    out = []
    for n in (2, 3, 5, 8, 12):
        # build the neighbor set (n-1 neighbors of the arm)
        edges = [Edge("verifier", "m", 0.99, 0.1)]
        if n >= 3:
            edges.append(Edge("base", "s", 0.95, 0.1))
        extra = n - len(edges) - 1
        for i in range(extra):
            if i % 2 == 0:
                edges.append(Edge(f"head{i}", "s", 0.8, 0.1))      # redundant with base
            else:
                edges.append(Edge(f"torso{i}", "z", 0.5, 0.1))     # irrelevant: never changes the action
        relevant = sum(1 for e in edges if e.latent in ("s", "m"))
        acc = {pol: dict(util=0.0, opened=0.0, preserved=0.0) for pol in ("never", "always", "voa", "oracle")}
        for _ in range(samples):
            s, m = int(rng.random() < 0.3), int(rng.random() < 0.2)
            e_s, e_m = (s if rng.random() < 0.65 else 1 - s), (m if rng.random() < 0.65 else 1 - m)
            p, q = bayes_update(0.3, e_s, 0.65), bayes_update(0.2, e_m, 0.65)
            reports = {}
            for e in edges:
                t = s if e.latent == "s" else (m if e.latent == "m" else 0)
                reports[e.name] = t if rng.random() < e.acc else 1 - t
            rel_edges = [e for e in edges if e.latent in ("s", "m")]
            ref = run_gate("always", rel_edges, p, q, reports, fuse="exact")
            for pol in acc:
                res = run_gate(pol, rel_edges, p, q, reports, truth={"s": s, "m": m}, fuse=fuse)
                acc[pol]["util"] += UTIL[res.action][(s, m)]
                acc[pol]["opened"] += len(res.opened)
                acc[pol]["preserved"] += res.action == ref.action
        for pol, r in acc.items():
            # gate: exact = one summary per opened edge; can = pairwise ring diffusion to consensus per opened edge
            per_edge = 1.0 if fuse == "exact" else 4.0 * can_rounds_to_consensus(2)
            msgs = always_align_cost(n, fuse) if pol == "always" else r["opened"] / samples * per_edge
            out.append(dict(sweep="n", fuse=fuse, n=n, relevant_edges=relevant, policy=pol,
                            util=r["util"] / samples, edges_opened=r["opened"] / samples,
                            messages=msgs, preserved=r["preserved"] / samples))
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for fuse in ("exact", "can"):
        rows += sweep_own_evidence(fuse)
        rows += sweep_cost(fuse)
        rows += sweep_prior(fuse)
    write_csv(os.path.join(OUT, "sweeps.csv"), rows)
    nrows = sweep_n("exact") + sweep_n("can")
    write_csv(os.path.join(OUT, "n_scaling.csv"), nrows)
    thm = theorem_check()
    # The milestone case: complementary evidence changes the compliant action.
    edges = base_edges()
    p, q = bayes_update(0.3, 0, 0.65), bayes_update(0.2, 0, 0.65)   # arm's own evidence says "fine"
    milestone = dict(arm_prior_p=p, arm_prior_q=q, arm_alone_action=best_action(p, q)[0],
                     voa_base=voa(edges[0], p, q), voa_verifier=voa(edges[1], p, q),
                     action_if_base_reports_sliding=best_action(bayes_update(p, 1, 0.95), q)[0],
                     action_if_verifier_reports_near=best_action(p, bayes_update(q, 1, 0.99))[0])
    summary = dict(theorem_check=thm, milestone=milestone,
                   design=dict(pi_s=0.3, pi_m=0.2, arm_own_accuracy=0.65, base_accuracy=0.95, verifier_accuracy=0.99, edge_cost=0.1))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"rows: sweeps={len(rows)} n_scaling={len(nrows)}")


if __name__ == "__main__":
    main()
