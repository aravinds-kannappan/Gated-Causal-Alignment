"""Stage B: gated causal alignment in the MuJoCo drawer environment.

Pipeline
  1. Collect training episodes on training configurations with an exploring controller.
  2. Fit each specialist's local model (logistic regression on its own features) and a monolithic
     centralized model on all features. Calibrate each neighbor's report accuracy.
  3. Evaluate controllers on held-out configurations: compliant completion, violations,
     unnecessary stops, decision preservation, regret vs the oracle gate, and alignment cost.
  4. Sweeps: fusion rule, number of specialists N, horizon T, and robustness variants.
All numbers written to results/stage_b/*.csv come from these runs.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gca.core import (ACTIONS, UTIL, Edge, always_align_cost, bayes_update, best_action,
                      can_rounds_to_consensus, run_gate)
from gca.drawer_env import PRIOR_LIMIT, DrawerConfig, DrawerEnv

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "stage_b")
EDGE_COST = 0.1
T_DEFAULT = 300


# ----------------------------------------------------------------------------- features
def arm_features(o):
    """Arm subgraph {f, h, k}: force, handle motion, cap. Cannot see the cabinet."""
    return np.array([1.0, o["F"] / 100, o["hdot"] / 0.1, o["h"] / 0.3, (o["F"] / 100) * (o["hdot"] / 0.1),
                     o["F_cap"] / 150, o["t"] / 300])


def base_features(o):
    """Base subgraph {x, pose}: cabinet displacement and its rate, brace contact."""
    return np.array([1.0, o["xc"] / 0.1, o["xcdot"] / 0.1, o["braced"]])


def head_features(o):
    return np.array([1.0, o["xc_head"] / 0.1])


def verifier_features(o):
    """Verifier subgraph {f, L, x}: calibrated force against the declared limit, handle motion."""
    return np.array([1.0, o["F_true"] / o["limit"], o["hdot"] / 0.1, (o["F_true"] / o["limit"]) * (o["hdot"] < 0.02)])


def all_features(o):
    return np.concatenate([arm_features(o), base_features(o)[1:], verifier_features(o)[1:]])


# ----------------------------------------------------------------------------- logistic model
class Logit:
    def __init__(self, dim):
        self.w = np.zeros(dim)

    def fit(self, X, y, l2=1e-3, iters=4000, lr=0.3):
        """Class-balanced logistic regression (positives up-weighted to the negative mass)."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if y.min() == y.max():
            self.w[:] = 0
            self.w[0] = 6.0 if y.max() == 1 else -6.0
            return self
        sw = np.where(y == 1, (1 - y.mean()) / y.mean(), 1.0)
        sw = sw / sw.mean()
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (sw * (p - y)) / len(y) + l2 * w
            w -= lr * g
        self.w = w
        return self

    def prob(self, x):
        return float(1 / (1 + np.exp(-np.dot(self.w, x))))


# ----------------------------------------------------------------------------- configurations
def training_configs():
    cfgs = []
    for res in (25, 45, 70, 90):
        for cab in (1e5, 20.0, 40.0):
            for stiff in (False, True):
                for limit in (60.0, 80.0, 110.0):
                    cfgs.append(DrawerConfig(drawer_fric=res, cab_fric=cab, stiff_start=stiff, sensor_noise=1.0, force_limit=limit))
    return cfgs


def heldout_configs():
    cfgs = []
    for res in (35, 60, 85):
        for cab in (1e5, 30.0):
            for noise in (0.5, 4.0):
                for stiff in (False, True):
                    cfgs.append(DrawerConfig(drawer_fric=res, cab_fric=cab, sensor_noise=noise, stiff_start=stiff))
    return cfgs


def cfg_label(c: DrawerConfig):
    return f"res{int(c.drawer_fric)}_{'anch' if c.anchored else 'free'}_n{c.sensor_noise}_{'stiff' if c.stiff_start else 'std'}"


# ----------------------------------------------------------------------------- training
def collect_training(seeds=(0, 1)):
    rows = []
    for cfg in training_configs():
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            env = DrawerEnv(cfg, seed=seed)
            mode, hold = "pull", 0
            env.F_cap = rng.uniform(60, 150)
            for t in range(T_DEFAULT):
                # exploring controller: mostly pull with a random cap; occasional brace or release,
                # after which it resumes pulling with a fresh cap so episodes do not stall
                if mode == "pull" and rng.random() < 0.04:
                    mode = "reposition" if rng.random() < 0.6 else "release"
                    hold = int(rng.integers(1, 6)) if mode == "reposition" else 1
                o = env.step(mode)
                rows.append(o)
                if mode != "pull":
                    hold -= 1
                    if hold <= 0:
                        mode = "pull"
                        env.F_cap = rng.uniform(60, 150)
                if env.done():
                    break
    return rows


class Models:
    def __init__(self, rows):
        self.arm_s = Logit(7).fit([arm_features(o) for o in rows], [o["s"] for o in rows])
        self.arm_m = Logit(7).fit([arm_features(o) for o in rows], [o["m"] for o in rows])
        self.base_s = Logit(4).fit([base_features(o) for o in rows], [o["s"] for o in rows])
        self.head_s = Logit(2).fit([head_features(o) for o in rows], [o["s"] for o in rows])
        self.ver_m = Logit(4).fit([verifier_features(o) for o in rows], [o["m_phys"] for o in rows])
        self.cent_s = Logit(13).fit([all_features(o) for o in rows], [o["s"] for o in rows])
        self.cent_m = Logit(13).fit([all_features(o) for o in rows], [o["m"] for o in rows])
        # calibrated report rates: true-positive and true-negative rate of each thresholded belief
        def rates(model, feat, key):
            pred = np.array([model.prob(feat(o)) > 0.5 for o in rows])
            lab = np.array([o[key] for o in rows], bool)
            tpr = float(np.mean(pred[lab])) if lab.any() else 0.5
            tnr = float(np.mean(~pred[~lab])) if (~lab).any() else 0.5
            return dict(tpr=tpr, tnr=tnr, acc=float(np.mean(pred == lab)))
        self.rates = {"base": rates(self.base_s, base_features, "s"), "head": rates(self.head_s, head_features, "s"),
                      "verifier": rates(self.ver_m, verifier_features, "m_phys"), "torso": dict(tpr=0.5, tnr=0.5, acc=0.5)}
        self.acc = {k: v["acc"] for k, v in self.rates.items()}
        self.arm_rates = dict(s=rates(self.arm_s, arm_features, "s"), m=rates(self.arm_m, arm_features, "m"))
        self.cent_rates = dict(s=rates(self.cent_s, all_features, "s"), m=rates(self.cent_m, all_features, "m"))
        self.arm_acc_s, self.arm_acc_m = self.arm_rates["s"]["acc"], self.arm_rates["m"]["acc"]
        self.cent_acc_s, self.cent_acc_m = self.cent_rates["s"]["acc"], self.cent_rates["m"]["acc"]

    def reports(self, o):
        return {"base": int(self.base_s.prob(base_features(o)) > 0.5),
                "head": int(self.head_s.prob(head_features(o)) > 0.5),
                "verifier": int(self.ver_m.prob(verifier_features(o)) > 0.5),
                "torso": 0}


def edges_for(n: int, models: Models):
    r = models.rates
    mk = lambda name, latent: Edge(name, latent, r[name]["tpr"], EDGE_COST, tnr=r[name]["tnr"])
    if n == 2:
        return [mk("verifier", "m")]
    e = [mk("base", "s"), mk("verifier", "m")]
    if n >= 5:
        e.append(mk("head", "s"))
        e.append(Edge("torso", "z", 0.5, EDGE_COST))
    return e


# ----------------------------------------------------------------------------- one episode
def run_episode(cfg, seed, models: Models, controller: str, fuse: str = "exact", n: int = 3,
                T: int = T_DEFAULT, random_rate=None, robustness: str | None = None):
    """controller in {centralized, never, always, event, random, voa, oracle}."""
    env = DrawerEnv(cfg, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    edges = [e for e in edges_for(n, models) if e.latent in ("s", "m")]
    if robustness == "corrupt_map":
        # the base's alignment map is wrong: reports are flipped 30% of the time, unknown to the gate
        pass
    o = env.step("pull")  # first step: nothing is known yet
    util = regret = 0.0
    msgs = 0.0
    preserved = steps = stops = 0
    opened_counts = {e.name: 0 for e in edges}
    delay_buf = []
    for t in range(1, T):
        p = models.arm_s.prob(arm_features(o))
        q = models.arm_m.prob(arm_features(o))
        # the cap is the arm's own state: once it sits below the known (or prior) limit, no pull can
        # violate the force limit, so the force latent is settled and the verifier edge has no value
        cap_safe = env.F_cap <= (env.known_limit if env.known_limit is not None else PRIOR_LIMIT)
        if cap_safe:
            q = 0.0
        reports = models.reports(o)
        if robustness == "corrupt_map" and rng.random() < 0.3:
            reports["base"] = 1 - reports["base"]
        if robustness and robustness.startswith("delay"):
            d = int(robustness[5:])
            delay_buf.append(dict(reports))
            if len(delay_buf) > d:
                reports = delay_buf[-1 - d]
        active = list(edges)
        if robustness == "drop_base" and t >= 60:
            active = [e for e in edges if e.name != "base"]
        if cap_safe:
            active = [e for e in active if e.name != "verifier"]
        truth = {"s": o["s"], "m": o["m"]}
        flags = {"base": o["hdot"] > 0.03, "verifier": o["F"] > 55.0, "head": o["hdot"] > 0.03}
        if controller == "centralized":
            pc, qc = models.cent_s.prob(all_features(o)), (0.0 if cap_safe else models.cent_m.prob(all_features(o)))
            action, _ = best_action(pc, qc)
            opened = []
            if action == "release":
                env.known_limit = o["limit"]
        else:
            res = run_gate(controller, active, p, q, reports, truth=truth, fuse=fuse,
                           event_flags=flags, random_rate=random_rate or {}, rng=rng)
            action, opened = res.action, res.opened
            if "verifier" in opened:
                env.known_limit = o["limit"]
        # reference action: always-align with exact fusion on the same evidence
        ref = run_gate("always", active, p, q, reports, fuse="exact").action
        orc = run_gate("oracle", active, p, q, reports, truth=truth, fuse="exact").action
        u = UTIL[action][(o["s"], o["m"])]
        util += u
        regret += UTIL[orc][(o["s"], o["m"])] - u
        preserved += action == ref
        stops += action != "pull" and o["s"] == 0 and o["m"] == 0
        for nme in opened:
            opened_counts[nme] += 1
        if controller == "always":
            msgs += always_align_cost(len(active) + 1, fuse)
        elif controller == "centralized":
            msgs += len(active)  # full sensing: every channel every step
        else:
            msgs += len(opened) * (1.0 if fuse == "exact" else 4.0 * can_rounds_to_consensus(2))
        steps += 1
        o = env.step(action)
        if env.done():
            break
    done = env.done()
    viol = env.violations["process"] + env.violations["force"]
    return dict(config=cfg_label(cfg), seed=seed, controller=controller, fuse=fuse, n=n, T=T,
                robustness=robustness or "none", completed=int(done), violations=viol,
                compliant=int(done and viol == 0), process_violations=env.violations["process"],
                force_violations=env.violations["force"], steps=steps, util=util / steps,
                regret=regret / steps, preserved=preserved / steps, stops=stops / steps,
                messages=msgs / steps, opened_base=opened_counts.get("base", 0) / steps,
                opened_verifier=opened_counts.get("verifier", 0) / steps,
                opened_head=opened_counts.get("head", 0) / steps, drawer_open=env.drawer_open())


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = collect_training()
    models = Models(rows)
    info = dict(training_rows=len(rows), training_configs=len(training_configs()),
                report_rates=models.rates, arm_own_rates=models.arm_rates, centralized_train_rates=models.cent_rates,
                label_rates=dict(s=float(np.mean([o["s"] for o in rows])), m=float(np.mean([o["m"] for o in rows]))),
                edge_cost=EDGE_COST, utilities={a: {f'({k[0]},{k[1]})': v for k, v in d.items()} for a, d in UTIL.items()},
                held_out_configs=len(heldout_configs()))
    print(json.dumps(info, indent=1, default=str))
    seeds = (0, 1, 2)
    held = heldout_configs()

    # pilot for the budget-matched random control: the gate's open rate on training configs
    pilot = [run_episode(c, 0, models, "voa", "exact") for c in training_configs()]
    rate = {"base": float(np.mean([r["opened_base"] for r in pilot])),
            "verifier": float(np.mean([r["opened_verifier"] for r in pilot]))}
    info["random_rate"] = rate

    # main comparison on held-out configs
    main_rows = []
    for cfg in held:
        for seed in seeds:
            main_rows.append(run_episode(cfg, seed, models, "centralized"))
            main_rows.append(run_episode(cfg, seed, models, "never"))
            for fuse in ("exact", "can"):
                for ctl in ("always", "event", "random", "voa", "oracle"):
                    main_rows.append(run_episode(cfg, seed, models, ctl, fuse, random_rate=rate))
    write_csv(os.path.join(OUT, "main.csv"), main_rows)
    print(f"main: {len(main_rows)} episodes, {time.time() - t0:.0f}s")

    # N scaling
    n_rows = []
    for n in (2, 3, 5):
        for cfg in held:
            for seed in seeds:
                for fuse in ("exact", "can"):
                    for ctl in ("always", "voa"):
                        n_rows.append(run_episode(cfg, seed, models, ctl, fuse, n=n))
    write_csv(os.path.join(OUT, "n_scaling.csv"), n_rows)
    print(f"n scaling: {len(n_rows)} episodes, {time.time() - t0:.0f}s")

    # horizon x family grid
    t_rows = []
    for T in (120, 300, 600):
        for cfg in held:
            for seed in seeds:
                for ctl in ("always", "voa", "never"):
                    t_rows.append(run_episode(cfg, seed, models, ctl, "exact", T=T))
    write_csv(os.path.join(OUT, "horizon.csv"), t_rows)
    print(f"horizon: {len(t_rows)} episodes, {time.time() - t0:.0f}s")

    # robustness
    r_rows = []
    for rob in ("drop_base", "delay1", "delay3", "corrupt_map"):
        for cfg in held:
            for seed in seeds:
                for ctl in ("always", "voa", "event"):
                    r_rows.append(run_episode(cfg, seed, models, ctl, "exact", robustness=rob))
    write_csv(os.path.join(OUT, "robustness.csv"), r_rows)
    print(f"robustness: {len(r_rows)} episodes, {time.time() - t0:.0f}s")

    info["episodes"] = dict(main=len(main_rows), n_scaling=len(n_rows), horizon=len(t_rows), robustness=len(r_rows))
    info["wall_seconds"] = time.time() - t0
    with open(os.path.join(OUT, "info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


if __name__ == "__main__":
    main()
