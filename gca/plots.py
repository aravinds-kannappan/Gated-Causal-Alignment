"""Figures for the report and the site, drawn from results/*.csv only."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# Validated categorical palette (reference instance of the dataviz method): fixed slot order.
C = {"voa": "#2a78d6", "always": "#eb6834", "never": "#1baf7a", "oracle": "#4a3aa7", "event": "#eda100",
     "random": "#e87ba4", "centralized": "#008300", "can": "#e34948"}
LABEL = {"voa": "VOA gate", "always": "Always-align", "never": "Isolated (never)", "oracle": "Oracle gate",
         "event": "Event-triggered", "random": "Budget-matched random", "centralized": "Centralized network"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a9a", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555",
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 160})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".svg"))
    fig.savefig(os.path.join(FIG, name + ".png"))


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fl(r, k):
    return float(r[k])


def mean_sd(vals):
    v = np.asarray(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


# --------------------------------------------------------------------------- Stage A
def stage_a():
    rows = read(os.path.join(RES, "stage_a", "sweeps.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for fuse, ls in (("exact", "-"), ("can", "--")):
        for pol in ("always", "voa", "oracle", "never", "event", "random"):
            if fuse == "can" and pol not in ("always", "voa"):
                continue
            rr = sorted([r for r in rows if r["sweep"] == "own_evidence" and r["fuse"] == fuse and r["policy"] == pol], key=lambda r: fl(r, "a_own"))
            x = [fl(r, "a_own") for r in rr]
            lab = LABEL[pol] + (" (CAN pool)" if fuse == "can" else "")
            axes[0].plot(x, [fl(r, "util") for r in rr], ls, color=C[pol], lw=2, label=lab)
            axes[1].plot(x, [fl(r, "preserved") for r in rr], ls, color=C[pol], lw=2)
            axes[2].plot(x, [fl(r, "cost") for r in rr], ls, color=C[pol], lw=2)
    axes[0].set_title("Utility per decision", fontsize=10)
    axes[1].set_title("Decision preservation", fontsize=10)
    axes[2].set_title("Cost paid per decision", fontsize=10)
    for a in axes:
        a.set_xlabel("arm's own evidence accuracy", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "a_overlap")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for pol in ("always", "voa", "oracle", "random"):
        rr = sorted([r for r in rows if r["sweep"] == "cost" and r["fuse"] == "exact" and r["policy"] == pol], key=lambda r: fl(r, "cost_per_edge"))
        x = [fl(r, "cost_per_edge") for r in rr]
        axes[0].plot(x, [fl(r, "util") for r in rr], "-", color=C[pol], lw=2, label=LABEL[pol])
        axes[1].plot(x, [fl(r, "open_base") + fl(r, "open_verifier") for r in rr], "-", color=C[pol], lw=2)
    axes[0].set_title("Utility vs declared edge cost")
    axes[1].set_title("Edges opened per decision")
    for a in axes:
        a.set_xlabel("cost per edge (utility units)")
    axes[0].legend(fontsize=8)
    save(fig, "a_cost")
    plt.close(fig)

    nrows = read(os.path.join(RES, "stage_a", "n_scaling.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for fuse, ls in (("exact", "-"), ("can", "--")):
        for pol in ("always", "voa"):
            rr = sorted([r for r in nrows if r["fuse"] == fuse and r["policy"] == pol], key=lambda r: int(r["n"]))
            x = [int(r["n"]) for r in rr]
            axes[0].plot(x, [fl(r, "messages") for r in rr], ls, marker="o", color=C[pol], lw=2, label=f"{LABEL[pol]}, {fuse}")
            axes[1].plot(x, [fl(r, "edges_opened") for r in rr], ls, marker="o", color=C[pol], lw=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Messages per decision vs specialists N")
    axes[1].set_title("Edges opened per decision vs N")
    for a in axes:
        a.set_xlabel("number of specialists N")
    axes[0].legend(fontsize=8)
    save(fig, "a_nscaling")
    plt.close(fig)


# --------------------------------------------------------------------------- Stage B
def agg(rows, keys, metrics):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {m: mean_sd([fl(r, m) for r in rs]) for m in metrics}
    return out


def stage_b():
    rows = read(os.path.join(RES, "stage_b", "main.csv"))
    metrics = ["compliant", "violations", "stops", "preserved", "regret", "messages", "util", "steps", "completed"]
    A = agg(rows, ["controller", "fuse"], metrics)
    order = [("centralized", "exact"), ("never", "exact"), ("always", "exact"), ("event", "exact"), ("random", "exact"),
             ("voa", "exact"), ("oracle", "exact"), ("always", "can"), ("voa", "can")]
    order = [o for o in order if o in A]
    names = [LABEL[c] + (" / CAN pool" if f == "can" else "") for c, f in order]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (m, title) in zip(axes, [("compliant", "Compliant completion rate"), ("stops", "Unnecessary stops per decision"),
                                     ("preserved", "Decision preservation"), ("messages", "Messages per decision")]):
        vals = [A[o][m][0] for o in order]
        errs = [A[o][m][1] / np.sqrt(A[o][m][2]) for o in order]
        cols = [C[o[0]] if o[1] == "exact" else C["can"] for o in order]
        ax.barh(range(len(order)), vals, xerr=errs, color=cols, height=0.62, error_kw=dict(ecolor="#555", lw=1))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(order), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        if m == "messages":
            ax.set_xscale("symlog", linthresh=1)
    save(fig, "b_main")
    plt.close(fig)

    # cost vs quality scatter
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for o in order:
        ax.scatter(A[o]["messages"][0], A[o]["util"][0], s=70, color=C[o[0]] if o[1] == "exact" else C["can"],
                   edgecolor="white", zorder=3)
        off = {("oracle", "exact"): (-8, 8), ("always", "exact"): (6, -12), ("centralized", "exact"): (6, 6), ("voa", "exact"): (6, -4),
               ("never", "exact"): (6, 6), ("event", "exact"): (6, -10), ("random", "exact"): (6, 4), ("always", "can"): (-6, 8), ("voa", "can"): (-6, -12)}
        ha = "right" if off.get(o, (6, 4))[0] < 0 else "left"
        ax.annotate(LABEL[o[0]] + (" (CAN)" if o[1] == "can" else ""), (A[o]["messages"][0], A[o]["util"][0]),
                    textcoords="offset points", xytext=off.get(o, (6, 4)), fontsize=7.5, ha=ha)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("messages per decision (symlog)")
    ax.set_ylabel("utility per decision")
    ax.set_title("Quality against alignment cost, held-out configs")
    save(fig, "b_pareto")
    plt.close(fig)

    # per-family breakdown for the gate vs always vs never
    fam = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["fuse"] != "exact":
            continue
        cab = "free cabinet" if "_free_" in r["config"] else "anchored"
        res = r["config"].split("_")[0].replace("res", "") + " N"
        fam[(res, cab)][r["controller"]].append(r)
    keys = sorted(fam.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    w = 0.2
    for i, ctl in enumerate(("never", "always", "voa", "centralized")):
        for ax, m in zip(axes, ("compliant", "stops")):
            vals = [np.mean([fl(x, m) for x in fam[k][ctl]]) for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 1.5) * w, vals, w, color=C[ctl], label=LABEL[ctl])
    for ax, t in zip(axes, ("Compliant completion by family", "Unnecessary stops per decision by family")):
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{a}\n{b}" for a, b in keys], fontsize=8)
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    save(fig, "b_families")
    plt.close(fig)

    # N scaling
    nrows = read(os.path.join(RES, "stage_b", "n_scaling.csv"))
    N = agg(nrows, ["controller", "fuse", "n"], ["messages", "compliant", "preserved", "util"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for fuse, ls in (("exact", "-"), ("can", "--")):
        for ctl in ("always", "voa"):
            ns = sorted({int(k[2]) for k in N if k[0] == ctl and k[1] == fuse})
            axes[0].plot(ns, [N[(ctl, fuse, str(n))]["messages"][0] for n in ns], ls, marker="o", color=C[ctl], lw=2, label=f"{LABEL[ctl]}, {fuse}")
            axes[1].plot(ns, [N[(ctl, fuse, str(n))]["util"][0] for n in ns], ls, marker="o", color=C[ctl], lw=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Messages per decision vs N")
    axes[1].set_title("Utility per decision vs N")
    for a in axes:
        a.set_xlabel("number of specialists N")
        a.set_xticks([2, 3, 5])
    axes[0].legend(fontsize=8)
    save(fig, "b_nscaling")
    plt.close(fig)

    # horizon x family grid
    trows = read(os.path.join(RES, "stage_b", "horizon.csv"))
    fams = ["anchored", "free cabinet"]
    Ts = sorted({int(r["T"]) for r in trows})
    grid_dc = np.zeros((len(fams), len(Ts)))
    grid_save = np.zeros((len(fams), len(Ts)))
    for i, f in enumerate(fams):
        for j, T in enumerate(Ts):
            sel = lambda ctl: [r for r in trows if int(r["T"]) == T and r["controller"] == ctl and (("_free_" in r["config"]) == (f == "free cabinet"))]
            a, v = sel("always"), sel("voa")
            grid_dc[i, j] = np.mean([fl(r, "compliant") for r in v]) - np.mean([fl(r, "compliant") for r in a])
            grid_save[i, j] = 1 - np.mean([fl(r, "messages") for r in v]) / np.mean([fl(r, "messages") for r in a])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, g, t, cmap, vmin, vmax in ((axes[0], grid_dc, "Δ compliant completion (gate − always)", "RdBu", -0.3, 0.3),
                                       (axes[1], grid_save, "Message saving (gate vs always)", "Blues", 0, 1)):
        im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([f"T={T}" for T in Ts])
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(fams)
        ax.set_title(t, fontsize=10)
        ax.grid(False)
        for i in range(len(fams)):
            for j in range(len(Ts)):
                ax.text(j, i, f"{g[i, j]:+.2f}" if g is grid_dc else f"{g[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="black" if abs(g[i, j]) < 0.6 * max(abs(vmin), abs(vmax)) else "white")
    save(fig, "b_grid")
    plt.close(fig)

    # robustness
    rrows = read(os.path.join(RES, "stage_b", "robustness.csv"))
    R = agg(rrows, ["robustness", "controller"], ["compliant", "violations", "preserved", "messages"])
    base = agg([r for r in rows if r["fuse"] == "exact"], ["controller"], ["compliant", "violations", "preserved", "messages"])
    robs = ["none", "delay1", "delay3", "drop_base", "corrupt_map"]
    rob_label = {"none": "nominal", "delay1": "reports delayed 1", "delay3": "reports delayed 3", "drop_base": "base dropped at t=60", "corrupt_map": "base map corrupted 30%"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    w = 0.25
    for i, ctl in enumerate(("always", "voa", "event")):
        for ax, m in zip(axes, ("compliant", "messages")):
            vals = []
            for rb in robs:
                vals.append(base[(ctl,)][m][0] if rb == "none" else R[(rb, ctl)][m][0])
            ax.bar(np.arange(len(robs)) + (i - 1) * w, vals, w, color=C[ctl], label=LABEL[ctl])
    for ax, t in zip(axes, ("Compliant completion under perturbation", "Messages per decision under perturbation")):
        ax.set_xticks(range(len(robs)))
        ax.set_xticklabels([rob_label[r] for r in robs], fontsize=7.5, rotation=15)
        ax.set_title(t, fontsize=10)
    axes[0].legend(fontsize=8)
    save(fig, "b_robust")
    plt.close(fig)

    # summary numbers for the report / site
    summary = {"main": {f"{c}/{f}": {m: dict(mean=A[(c, f)][m][0], sd=A[(c, f)][m][1], n=A[(c, f)][m][2]) for m in metrics} for c, f in order},
               "n_scaling": {f"{k[0]}/{k[1]}/N{k[2]}": {m: v[m][0] for m in ("messages", "compliant", "preserved", "util")} for k, v in N.items()},
               "grid": {"families": fams, "T": Ts, "delta_compliant": grid_dc.tolist(), "message_saving": grid_save.tolist()},
               "robustness": {f"{k[0]}/{k[1]}": {m: v[m][0] for m in ("compliant", "violations", "preserved", "messages")} for k, v in R.items()}}
    with open(os.path.join(RES, "stage_b", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    stage_a()
    s = stage_b()
    for k, v in s["main"].items():
        print(f"{k:18s} compliant={v['compliant']['mean']:.3f} viol={v['violations']['mean']:.2f} pres={v['preserved']['mean']:.3f} "
              f"msgs={v['messages']['mean']:.2f} util={v['util']['mean']:.3f} regret={v['regret']['mean']:.3f} stops={v['stops']['mean']:.3f}")
