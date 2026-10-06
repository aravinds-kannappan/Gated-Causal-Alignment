# Gated Causal Alignment

A value-of-alignment gate for multi-specialist robot control. Each specialist (arm, base, verifier, and optional head and torso) holds evidence about a different part of a task's causal graph. Before acting, the arm opens an alignment edge to a neighbor only when the expected decision gain from that neighbor's evidence exceeds the declared cost. The verifier checks process constraints (a force limit and prohibited cabinet motion) against the task specification rather than a proxy reward.

The gate is the middle layer between two existing lines of work: cost-aware observation refinement for a single agent (what to observe) and belief fusion across agents (how to combine). This repository asks the question in between: which neighbors to align with, and when.

## What is here

| Path | Contents |
|---|---|
| `gca/core.py` | Utilities, beliefs, value of alignment (VOA), gate policies, exact and CAN-pool fusion rules, the finite theorem check |
| `gca/stage_a.py` | Stage A: exact enumeration over a synthetic POSCM drawer task (overlap, cost, prior and N sweeps) |
| `gca/drawer_env.py` | Stage B environment: a MuJoCo drawer with a sliding cabinet, a bracing base, a force-limited pull and a stiff-start handle variant |
| `gca/stage_b.py` | Stage B: training, local models, seven controllers, fusion comparison, N and horizon sweeps, robustness variants |
| `gca/plots.py` | Figures and the summary JSON, drawn only from `results/*.csv` |
| `gca/build_site.py` | Builds `site/index.html` from the results (animation, results, interpretation, next steps) |
| `gca/build_report.py` | Builds `report/report.html` and `report/report.pdf` from the results |
| `results/` | Raw per-episode CSVs, summaries, figures |
| `site/` | The animated website and the earlier gate-episode animation |
| `docs/proposal.md` | The original proposal that defined the design space and success criteria |
| `lit/related_work.json` | Verified related work with one-line relevance notes |

## Reproduce

```bash
pip install -r requirements.txt
python gca/stage_a.py        # seconds
python gca/stage_b.py        # minutes on a laptop CPU
python gca/plots.py
python gca/build_site.py
python gca/build_report.py   # needs Google Chrome for the PDF step
```

Every number in the site and the report is read from the CSV and JSON files that these scripts write. Rerunning them regenerates everything.

## Controllers compared

| Controller | What it does |
|---|---|
| Centralized network | One classifier over every specialist's features, no messages |
| Isolated (never) | The arm acts on its own evidence only |
| Always-align | Every edge opened at every decision (exact fusion or fixed-weight CAN pool) |
| Event-triggered | Open an edge on a hand-set trigger (handle motion, force threshold) |
| Budget-matched random | Open each edge with the gate's empirical open rate |
| VOA gate | Open an edge when expected decision gain exceeds its cost (this work) |
| Oracle gate | Open an edge only if the true latent would change the action (ceiling) |

## Metrics

Compliant completion (drawer open with zero violations), violations per episode, unnecessary stops, decision preservation against always-align, regret against the oracle gate, and messages per decision.
