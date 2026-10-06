"""Record step-by-step traces of real simulator episodes for the animated site.

Each trace holds, per decision step: commanded force, cabinet displacement, drawer opening, handle
displacement, base position, brace contact, the arm's own beliefs, the neighbors' reports, which edges
the gate opened, the chosen action, and the latent truth. The site replays these; nothing in the
animation is hand-drawn data.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gca.core import UTIL, best_action, run_gate
from gca.drawer_env import PRIOR_LIMIT, DrawerConfig, DrawerEnv
from gca.stage_b import Models, all_features, arm_features, cfg_label, collect_training, edges_for

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "stage_b", "traces.json")


def trace(cfg, models, controller, fuse="exact", seed=0, T=300):
    env = DrawerEnv(cfg, seed=seed)
    edges = [e for e in edges_for(3, models) if e.latent in ("s", "m")]
    o = env.step("pull")
    steps = []
    for t in range(1, T):
        p = models.arm_s.prob(arm_features(o))
        q = models.arm_m.prob(arm_features(o))
        cap_safe = env.F_cap <= (env.known_limit if env.known_limit is not None else PRIOR_LIMIT)
        if cap_safe:
            q = 0.0
        reports = models.reports(o)
        active = [e for e in edges if not (cap_safe and e.name == "verifier")]
        truth = {"s": o["s"], "m": o["m"]}
        if controller == "centralized":
            pc, qc = models.cent_s.prob(all_features(o)), (0.0 if cap_safe else models.cent_m.prob(all_features(o)))
            action, _ = best_action(pc, qc)
            opened, pf, qf = [], pc, qc
            if action == "release":
                env.known_limit = o["limit"]
        else:
            res = run_gate(controller, active, p, q, reports, truth=truth, fuse=fuse)
            action, opened, pf, qf = res.action, res.opened, res.p, res.q
            if "verifier" in opened:
                env.known_limit = o["limit"]
        steps.append(dict(t=t, F=round(o["F_true"], 1), cap=round(env.F_cap, 1), xc=round(o["xc"], 4), d=round(o["d"], 4),
                          h=round(o["h"], 4), base=round(float(env.data.xpos[env.b_base][0]), 4),
                          cab=round(float(env.data.xpos[env.b_cab][0]), 4), braced=int(env.braced),
                          p=round(p, 3), q=round(q, 3), pf=round(pf, 3), qf=round(qf, 3),
                          rep_base=reports["base"], rep_ver=reports["verifier"], opened=opened, action=action,
                          s=o["s"], m=o["m"], util=UTIL[action][(o["s"], o["m"])]))
        o = env.step(action)
        if env.done():
            break
    return dict(config=cfg_label(cfg), controller=controller, fuse=fuse, steps=steps,
                completed=int(env.done()), violations=env.violations, drawer_open=round(env.drawer_open(), 4))


def main():
    rows = collect_training()
    models = Models(rows)
    cfgs = {
        "free_cabinet": DrawerConfig(drawer_fric=60, cab_fric=30.0, sensor_noise=0.5),
        "over_limit": DrawerConfig(drawer_fric=85, cab_fric=1e5, sensor_noise=0.5),
        "easy_anchored": DrawerConfig(drawer_fric=35, cab_fric=1e5, sensor_noise=0.5),
    }
    out = {}
    for key, cfg in cfgs.items():
        out[key] = {ctl: trace(cfg, models, ctl) for ctl in ("voa", "always", "never")}
        out[key]["voa_can"] = trace(cfg, models, "voa", fuse="can")
    with open(OUT, "w") as f:
        json.dump(out, f)
    for key, d in out.items():
        for ctl, tr in d.items():
            n_open = sum(len(s["opened"]) for s in tr["steps"])
            print(f"{key:14s} {ctl:8s} steps={len(tr['steps']):3d} completed={tr['completed']} viol={tr['violations']} edges_opened={n_open}")


if __name__ == "__main__":
    main()
