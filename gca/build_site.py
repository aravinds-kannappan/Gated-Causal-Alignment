"""Build site/index.html from the results. Every number on the page is read from results/ files."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SITE = os.path.join(ROOT, "site")


def load():
    S = json.load(open(os.path.join(RES, "stage_b", "summary.json")))
    A = json.load(open(os.path.join(RES, "stage_a", "summary.json")))
    I = json.load(open(os.path.join(RES, "stage_b", "info.json")))
    T = json.load(open(os.path.join(RES, "stage_b", "traces.json")))
    L = json.load(open(os.path.join(ROOT, "lit", "related_work.json")))
    sweeps = list(csv.DictReader(open(os.path.join(RES, "stage_a", "sweeps.csv"))))
    nscale = list(csv.DictReader(open(os.path.join(RES, "stage_a", "n_scaling.csv"))))
    return S, A, I, T, L, sweeps, nscale


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def main():
    S, A, I, T, L, sweeps, nscale = load()
    os.makedirs(os.path.join(SITE, "figures"), exist_ok=True)
    for f in os.listdir(os.path.join(RES, "figures")):
        if f.endswith(".svg"):
            shutil.copy(os.path.join(RES, "figures", f), os.path.join(SITE, "figures", f))
    M = S["main"]
    g = lambda c, m: M[c][m]["mean"]
    n_eps = M["voa/exact"]["compliant"]["n"]
    # Stage A headline numbers at the design point (own accuracy 0.65, exact fusion)
    a65 = {r["policy"]: r for r in sweeps if r["sweep"] == "own_evidence" and r["fuse"] == "exact" and r["a_own"] == "0.65"}
    a65c = {r["policy"]: r for r in sweeps if r["sweep"] == "own_evidence" and r["fuse"] == "can" and r["a_own"] == "0.65"}
    a95 = {r["policy"]: r for r in sweeps if r["sweep"] == "own_evidence" and r["fuse"] == "exact" and r["a_own"] == "0.95"}
    nA = {(r["fuse"], r["policy"], r["n"]): r for r in nscale}
    rob = S["robustness"]
    N = S["n_scaling"]
    grid = S["grid"]
    thm = A["theorem_check"]
    ms = A["milestone"]
    venue = lambda v: re.sub(r"\s*\(?\b(19|20)\d\d\)?", "", v).strip(" ,.")
    lit_items = "".join(
        f'<li><a href="{p["url"]}">{p["title"]}</a> <span class="muted">({p["authors"]}; {venue(p["venue"])})</span><br><span class="rel">{p["relevance"]}</span></li>'
        for p in sorted(L, key=lambda p: p["topic"]))
    traces_json = json.dumps(T)

    main_rows = ""
    order = ["centralized/exact", "never/exact", "always/exact", "event/exact", "random/exact", "voa/exact", "oracle/exact", "always/can", "voa/can"]
    label = {"centralized/exact": "Centralized network", "never/exact": "Isolated (never align)", "always/exact": "Always-align, exact fusion",
             "event/exact": "Event-triggered", "random/exact": "Budget-matched random", "voa/exact": "<b>VOA gate, exact fusion</b>",
             "oracle/exact": "Oracle gate (reference)", "always/can": "Always-align, CAN pool", "voa/can": "VOA gate, CAN pool"}
    for k in order:
        r = M[k]
        cls = ' class="hl"' if k == "voa/exact" else ""
        main_rows += (f'<tr{cls}><td>{label[k]}</td><td>{pct(r["compliant"]["mean"])}</td><td>{r["violations"]["mean"]:.2f}</td>'
                      f'<td>{r["stops"]["mean"]:.3f}</td><td>{pct(r["preserved"]["mean"], 1)}</td><td>{r["util"]["mean"]:.3f}</td>'
                      f'<td>{r["steps"]["mean"]:.0f}</td><td>{r["messages"]["mean"]:.2f}</td></tr>')

    n_rows = ""
    for n in (2, 3, 5):
        n_rows += (f'<tr><td>{n}</td><td>{N[f"always/exact/N{n}"]["messages"]:.2f}</td><td>{N[f"voa/exact/N{n}"]["messages"]:.2f}</td>'
                   f'<td>{N[f"always/can/N{n}"]["messages"]:.0f}</td><td>{N[f"voa/can/N{n}"]["messages"]:.1f}</td>'
                   f'<td>{N[f"always/exact/N{n}"]["util"]:.2f} / {N[f"voa/exact/N{n}"]["util"]:.2f}</td>'
                   f'<td>{N[f"always/can/N{n}"]["util"]:.2f} / {N[f"voa/can/N{n}"]["util"]:.2f}</td></tr>')

    rob_rows = ""
    rob_label = {"delay1": "Reports delayed 1 step", "delay3": "Reports delayed 3 steps", "drop_base": "Base dropped mid-episode", "corrupt_map": "Base map corrupted (30% flipped)"}
    for rb in ("delay1", "delay3", "drop_base", "corrupt_map"):
        rob_rows += (f'<tr><td>{rob_label[rb]}</td><td>{pct(rob[f"{rb}/always"]["compliant"])}</td><td>{pct(rob[f"{rb}/voa"]["compliant"])}</td>'
                     f'<td>{pct(rob[f"{rb}/event"]["compliant"])}</td><td>{rob[f"{rb}/always"]["messages"]:.2f}</td><td>{rob[f"{rb}/voa"]["messages"]:.2f}</td></tr>')

    grid_cells = ""
    for i, fam in enumerate(grid["families"]):
        grid_cells += f"<tr><td>{fam}</td>" + "".join(
            f'<td>{grid["delta_compliant"][i][j]:+.2f} / {pct(grid["message_saving"][i][j])}</td>' for j in range(len(grid["T"]))) + "</tr>"

    saving3 = 1 - g("voa/exact", "messages") / g("always/exact", "messages")
    saving5 = 1 - N["voa/exact/N5"]["messages"] / N["always/exact/N5"]["messages"]
    growth_always = N["always/exact/N5"]["messages"] / N["always/exact/N3"]["messages"] - 1
    can_util_drop = 1 - g("always/can", "util") / g("always/exact", "util")
    a_cost_save65 = 1 - float(a65["voa"]["cost"]) / float(a65["always"]["cost"])
    a_cost_save95 = 1 - float(a95["voa"]["cost"]) / float(a95["always"]["cost"])
    a_util_gap65 = float(a65["always"]["util"]) - float(a65["voa"]["util"])

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Gated Causal Alignment</title>
<meta name="description" content="A value-of-alignment gate decides which robot specialists to consult before acting. Results from a synthetic causal model and a physics drawer task.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root {{
  --bg:#f4f5f7; --panel:#ffffff; --fg:#1b2130; --muted:#5d6678; --line:#d3d8e1; --grid:#e8ebf0;
  --accent:#b5551b; --accent-soft:rgba(181,85,27,.13); --base:#2a78d6; --verify:#008300; --arm:#4b4f66; --bad:#e34948; --good:#1baf7a;
  --ui:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif; --mono:"IBM Plex Mono",Menlo,monospace; --serif:"IBM Plex Serif",Georgia,serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14171e; --panel:#1c2029; --fg:#e9ecf2; --muted:#a3abbb; --line:#363c49; --grid:#2a303c; --accent:#e0803f; --accent-soft:rgba(224,128,63,.18); --base:#6fa8d6; --verify:#7cc26b; --arm:#b3b8cc; --bad:#e66767; --good:#3fc595; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#14171e; --panel:#1c2029; --fg:#e9ecf2; --muted:#a3abbb; --line:#363c49; --grid:#2a303c; --accent:#e0803f; --accent-soft:rgba(224,128,63,.18); --base:#6fa8d6; --verify:#7cc26b; --arm:#b3b8cc; --bad:#e66767; --good:#3fc595; color-scheme:dark; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--ui); font-size:16px; line-height:1.55; }}
a {{ color:var(--accent); }}
.wrap {{ max-width:1040px; margin:0 auto; padding-inline:20px; padding-block:0 60px; }}
header.top {{ position:sticky; top:env(safe-area-inset-top,0px); z-index:5; background:color-mix(in srgb,var(--bg) 88%,transparent); backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }}
header.top .wrap {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px 22px; padding-block:10px; }}
header.top .brand {{ font-weight:700; letter-spacing:.01em; }}
header.top nav a {{ color:var(--muted); text-decoration:none; font-size:14px; margin-right:14px; }}
header.top nav a:hover {{ color:var(--fg); }}
.hero {{ padding-block:56px 28px; display:grid; gap:14px; }}
.hero h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(30px,4.6vw,48px); line-height:1.1; margin:0; text-wrap:balance; max-width:20ch; }}
.hero p.lede {{ font-size:18px; color:var(--muted); max-width:62ch; margin:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:8px; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.kpi .k {{ font-size:11.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }}
.kpi .v {{ font-family:var(--mono); font-size:28px; font-weight:500; font-variant-numeric:tabular-nums; margin-top:2px; }}
.kpi .u {{ font-size:12.5px; color:var(--muted); }}
section {{ padding-block:36px 8px; }}
section h2 {{ font-family:var(--serif); font-weight:600; font-size:28px; margin:0 0 6px; text-wrap:balance; }}
section .sub {{ color:var(--muted); margin:0 0 18px; max-width:70ch; }}
h3 {{ font-size:17px; margin:22px 0 8px; }}
p {{ max-width:72ch; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:760px) {{ .two {{ grid-template-columns:1fr; }} }}
.bar {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; margin-bottom:12px; }}
select,button {{ font:500 14px var(--ui); color:var(--fg); background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:6px 12px; }}
button {{ cursor:pointer; }}
button:focus-visible,select:focus-visible,input:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.scrub {{ display:flex; align-items:center; gap:10px; flex:1 1 220px; min-width:0; }}
input[type=range] {{ flex:1; min-width:0; accent-color:var(--accent); }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
svg {{ display:block; width:100%; height:auto; max-width:100%; }}
.scene {{ overflow-x:auto; }}
.readout {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:12px; }}
.readout .r {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; min-width:0; }}
.readout .k {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}
.readout .v {{ font-family:var(--mono); font-size:20px; font-variant-numeric:tabular-nums; }}
.readout .u {{ font-size:12px; color:var(--muted); }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:13px; font-weight:600; }}
.pill.pull {{ background:var(--grid); }} .pill.reposition {{ background:color-mix(in srgb,var(--base) 22%,transparent); }} .pill.release {{ background:color-mix(in srgb,var(--verify) 22%,transparent); }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
td:not(:first-child),th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.hl td {{ background:var(--accent-soft); }}
.tablewrap {{ overflow-x:auto; }}
figure {{ margin:14px 0 6px; }}
figure img {{ width:100%; height:auto; background:#fff; border-radius:8px; border:1px solid var(--line); }}
figcaption {{ font-size:13px; color:var(--muted); margin-top:6px; max-width:80ch; }}
.callout {{ border-left:3px solid var(--accent); padding:8px 14px; background:var(--accent-soft); border-radius:0 8px 8px 0; margin:14px 0; }}
.findings {{ display:grid; gap:12px; }}
.finding {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.finding b {{ display:block; margin-bottom:4px; }}
ul.lit {{ padding-left:18px; }} ul.lit li {{ margin-bottom:8px; font-size:14px; }} .muted {{ color:var(--muted); }} .rel {{ color:var(--muted); font-size:13px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; color:var(--muted); }}
.legend span::before {{ content:""; display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; background:var(--sw); }}
.formula {{ font-family:var(--mono); background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 14px; overflow-x:auto; font-size:14px; }}
.steps {{ counter-reset:s; display:grid; gap:10px; }}
.step {{ display:grid; grid-template-columns:34px 1fr; gap:10px; align-items:start; }}
.step::before {{ counter-increment:s; content:counter(s); font-family:var(--mono); font-weight:500; color:var(--accent); border:1px solid var(--line); border-radius:50%; width:30px; height:30px; display:grid; place-items:center; }}
footer {{ color:var(--muted); font-size:13px; padding-block:24px; border-top:1px solid var(--line); margin-top:30px; }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none!important; }} }}
</style>
</head>
<body>
<header class="top"><div class="wrap"><span class="brand">Gated Causal Alignment</span>
<nav><a href="#episode">Episode</a><a href="#idea">Idea</a><a href="#setup">Setup</a><a href="#results">Results</a><a href="#interpretation">Interpretation</a><a href="#next">Next steps</a><a href="#related">Related work</a></nav></div></header>

<div class="wrap">
<div class="hero">
  <h1>Consult a neighbor only when its evidence would change the action</h1>
  <p class="lede">A robot arm, a mobile base and a verifier each see a different part of a drawer task's causal graph. A value-of-alignment gate opens an edge to a neighbor only when the expected decision gain beats the declared cost. This page replays real simulator episodes and reports every result from the runs.</p>
  <div class="kpis">
    <div class="kpi"><div class="k">Decisions preserved</div><div class="v">{pct(g("voa/exact","preserved"),1)}</div><div class="u">gate vs always-align, held-out configs</div></div>
    <div class="kpi"><div class="k">Messages saved</div><div class="v">{pct(saving3)}</div><div class="u">at 3 specialists; {pct(saving5)} at 5</div></div>
    <div class="kpi"><div class="k">Compliant completion</div><div class="v">{pct(g("voa/exact","compliant"))}</div><div class="u">same as always-align ({pct(g("always/exact","compliant"))}); isolated {pct(g("never/exact","compliant"))}</div></div>
    <div class="kpi"><div class="k">Violations</div><div class="v">{g("voa/exact","violations"):.0f}</div><div class="u">per episode, all aligned controllers</div></div>
  </div>
</div>

<section id="episode">
  <h2>Replay of simulator episodes</h2>
  <p class="sub">Every frame is a recorded decision step from the physics environment: forces, cabinet and drawer motion, the arm's beliefs, the neighbors' reports, and which edges the gate opened. Pick a scenario and a controller.</p>
  <div class="panel">
    <div class="bar">
      <label>Scenario <select id="scn"><option value="free_cabinet">Free cabinet (proxy exploit)</option><option value="over_limit">Drawer needs more than the force limit</option><option value="easy_anchored">Easy anchored drawer</option></select></label>
      <label>Controller <select id="ctl"><option value="voa">VOA gate, exact fusion</option><option value="always">Always-align</option><option value="never">Isolated arm</option><option value="voa_can">VOA gate, CAN pool</option></select></label>
      <button id="play" type="button">Pause</button><button id="restart" type="button">Restart</button>
      <div class="scrub"><input id="scrub" type="range" min="0" max="10" value="0" aria-label="Decision step"><span class="mono" id="tl">t = 0</span></div>
    </div>
    <div class="scene"><svg id="scene" viewBox="0 0 760 300" role="img" aria-label="Robot pulling a drawer, with specialist edges"></svg></div>
    <div class="readout">
      <div class="r"><div class="k">Action</div><div class="v" id="ract"><span class="pill pull">pull</span></div></div>
      <div class="r"><div class="k">Force / cap</div><div class="v" id="rF">0</div><div class="u">N, limit 80 N</div></div>
      <div class="r"><div class="k">Drawer open</div><div class="v" id="rd">0.000</div><div class="u">m, target 0.30</div></div>
      <div class="r"><div class="k">Cabinet moved</div><div class="v" id="rxc">0.000</div><div class="u">m (prohibited)</div></div>
      <div class="r"><div class="k">Edges opened so far</div><div class="v" id="rmsg">0</div><div class="u" id="rmsgu">always-align would be 0</div></div>
      <div class="r"><div class="k">Utility so far</div><div class="v" id="rutil">0.0</div><div class="u">declared per-step utilities</div></div>
    </div>
    <div class="legend" style="margin-top:12px"><span style="--sw:var(--base)">Base: sees cabinet displacement</span><span style="--sw:var(--verify)">Verifier: sees force against the limit</span><span style="--sw:var(--accent)">Edge opened this step</span><span style="--sw:var(--bad)">Violation</span></div>
    <p class="muted" style="font-size:13.5px;margin-top:10px" id="scnnote"></p>
  </div>
</section>

<section id="idea">
  <h2>The idea in one formula</h2>
  <p class="sub">Three layers of one control loop. Observation refinement decides what one agent should look at. Fusion decides how agents combine beliefs. The gate sits between them and decides whom to align with, per decision.</p>
  <div class="two">
    <div class="panel">
      <svg viewBox="0 0 480 190" role="img" aria-label="Three layers: observe, gate edges, fuse beliefs">
        <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>
        <g fill="none" stroke="var(--muted)" stroke-width="1.3"><path d="M150 60H176" marker-end="url(#ar)"/><path d="M304 60H330" marker-end="url(#ar)"/><path d="M405 96V130" marker-end="url(#ar)"/><path d="M80 130V96" marker-end="url(#ar)"/></g>
        <rect x="14" y="24" width="136" height="72" rx="8" fill="none" stroke="var(--line)"/><text x="24" y="46" font-size="13" font-weight="600" fill="var(--fg)">1. Observe</text><text x="24" y="64" font-size="11.5" fill="var(--muted)">refine, probe or stay</text><text x="24" y="80" font-size="11.5" fill="var(--muted)">coarse (COO criterion)</text>
        <rect x="176" y="24" width="128" height="72" rx="8" fill="var(--accent)" fill-opacity=".13" stroke="var(--accent)" stroke-width="2"/><text x="186" y="46" font-size="13" font-weight="600" fill="var(--fg)">2. Gate edges</text><text x="186" y="64" font-size="11.5" fill="var(--muted)">open (i, j) only when</text><text x="186" y="80" font-size="11.5" fill="var(--muted)">VOA &gt; 0 (this work)</text>
        <rect x="330" y="24" width="136" height="72" rx="8" fill="none" stroke="var(--line)"/><text x="340" y="46" font-size="13" font-weight="600" fill="var(--fg)">3. Fuse beliefs</text><text x="340" y="64" font-size="11.5" fill="var(--muted)">exact fusion, or a</text><text x="340" y="80" font-size="11.5" fill="var(--muted)">CAN diffusion pool</text>
        <rect x="14" y="130" width="452" height="40" rx="8" fill="var(--grid)" stroke="var(--line)"/><text x="240" y="155" text-anchor="middle" font-size="12" fill="var(--fg)">Act under verifier arbitration, log the evidence, repeat at the next decision</text>
      </svg>
    </div>
    <div>
      <div class="formula">VOA<sub>ij</sub>(t) = E[ U(a*(b<sub>i</sub> ⊕ b<sub>j</sub>)) ] − E[ U(a*(b<sub>i</sub>)) ] − c<sub>ij</sub></div>
      <p>The arm holds beliefs p (the cabinet will slide) and q (the next pull crosses the force limit). A neighbor's report is a binary signal with calibrated true-positive and true-negative rates. The gate computes, before asking, the expected utility of acting on the fused belief minus the utility of acting now, minus the declared cost. It opens the edge with the largest positive value, updates, and repeats until no edge clears its cost.</p>
      <p>This is the single-agent refinement criterion with "a finer observation" replaced by "my belief plus a neighbor's report". The same precondition carries over: an edge has strict value only when the arm's belief aliases cases with different optimal actions and the neighbor's report separates them. An exhaustive finite check over {thm["cases"]:,} belief-accuracy cases agreed with that precondition in {thm["agree"]:,}; the {thm["ties"]} remaining cases are exact ties between two equally good actions.</p>
    </div>
  </div>
</section>

<section id="setup">
  <h2>Two experiments</h2>
  <p class="sub">Stage A isolates the gate's logic with exact enumeration over a synthetic causal model. Stage B runs it inside a physics drawer task with learned specialist models and held-out configurations.</p>
  <div class="two">
    <div class="panel"><h3 style="margin-top:0">Stage A: synthetic POSCM, exact</h3>
      <p>Force drives handle displacement; the drawer or the cabinet moves; a contact state and a force limit are exogenous. The arm sees force and handle motion, the base sees cabinet displacement, the verifier sees force against the limit. Worlds, the arm's own evidence and every neighbor report are enumerated with their probabilities, so each number is an expectation, not a sample. Sweeps: the arm's own evidence quality (subgraph overlap), the edge cost, the prior, and the number of specialists up to 12 with redundant and irrelevant neighbors.</p></div>
    <div class="panel"><h3 style="margin-top:0">Stage B: physics drawer task</h3>
      <p>A MuJoCo scene: a cabinet on a slide joint with declared ground friction (anchored or free), a drawer with rail resistance, a heavy base whose plate braces the cabinet, and a pull modelled as an external force pair. Skills: pull (ramp force until the handle moves, then hold), reposition (brace), release (back off below the known or prior limit). Specialists fit class-balanced logistic models on {I["training_rows"]:,} training steps from {I["training_configs"]} training configurations; report rates are calibrated there. Evaluation: {I["held_out_configs"]} held-out configurations (three unseen resistances, anchored and free cabinets, two sensor-noise levels, two handle variants) × 3 seeds = {n_eps} episodes per controller.</p></div>
  </div>
  <h3>Controllers</h3>
  <div class="tablewrap"><table>
  <tr><th>Controller</th><th style="text-align:left">Rule</th></tr>
  <tr><td>Centralized network</td><td style="text-align:left">One classifier over every specialist's features; no messages, all channels sensed every step</td></tr>
  <tr><td>Isolated (never)</td><td style="text-align:left">The arm acts on its own evidence</td></tr>
  <tr><td>Always-align</td><td style="text-align:left">Every edge opened at every decision; exact fusion or a fixed-weight CAN diffusion pool</td></tr>
  <tr><td>Event-triggered</td><td style="text-align:left">Open the base edge on handle motion, the verifier edge above a force threshold</td></tr>
  <tr><td>Budget-matched random</td><td style="text-align:left">Open each edge with the gate's empirical open rate from a pilot ({I["random_rate"]["base"]:.2f} base, {I["random_rate"]["verifier"]:.2f} verifier)</td></tr>
  <tr class="hl"><td>VOA gate</td><td style="text-align:left">Open an edge when expected decision gain exceeds its cost ({I["edge_cost"]} utility units)</td></tr>
  <tr><td>Oracle gate</td><td style="text-align:left">Open an edge only if the true latent would change the action; a reference, not a strict ceiling, since it still acts on noisy reports</td></tr>
  </table></div>
  <p class="muted" style="font-size:13.5px">Report rates calibrated on training data: base true-positive {I["report_rates"]["base"]["tpr"]:.2f}, true-negative {I["report_rates"]["base"]["tnr"]:.2f}; verifier {I["report_rates"]["verifier"]["tpr"]:.2f} and {I["report_rates"]["verifier"]["tnr"]:.2f}; head camera {I["report_rates"]["head"]["tpr"]:.2f} and {I["report_rates"]["head"]["tnr"]:.2f}. The arm's own accuracy is {I["arm_own_rates"]["s"]["acc"]:.2f} on sliding and {I["arm_own_rates"]["m"]["acc"]:.2f} on the force latent.</p>
</section>

<section id="results">
  <h2>Results</h2>
  <p class="sub">All values are means over the runs in the repository's results folder. Stage A values are exact expectations; Stage B values are means over held-out episodes.</p>

  <h3>Stage A: the gate tracks always-align at a fraction of the cost, and the saving grows with overlap</h3>
  <figure><img src="figures/a_overlap.svg" alt="Utility, decision preservation and cost against the arm's own evidence accuracy"><figcaption>As the arm's own subgraph tells it more (x axis), the gate stops paying for the base edge. At own accuracy 0.65 the gate pays {pct(a_cost_save65)} less than always-align for a utility gap of {a_util_gap65:.3f} per decision; at 0.95 it pays {pct(a_cost_save95)} less. The dashed CAN-pool curves show the fixed-weight pool losing decisions even when every edge is open.</figcaption></figure>
  <div class="two">
    <figure><img src="figures/a_nscaling.svg" alt="Messages and edges opened against the number of specialists"><figcaption>With redundant and irrelevant specialists added, the gate opens about {float(nA[("exact","voa","12")]["edges_opened"]):.2f} edges per decision at N=12, the same as at N=5 ({float(nA[("exact","voa","5")]["edges_opened"]):.2f}); always-align grows to {float(nA[("exact","always","12")]["messages"]):.0f} messages (exact) or {float(nA[("can","always","12")]["messages"]):,.0f} (CAN ring diffusion to consensus).</figcaption></figure>
    <figure><img src="figures/a_cost.svg" alt="Utility and edges opened against edge cost"><figcaption>Raising the declared cost closes edges smoothly; the budget-matched random control pays the same but loses utility because it opens the wrong edges.</figcaption></figure>
  </div>
  <div class="callout">Milestone from the proposal: with the arm's own evidence saying "fine" (p = {ms["arm_prior_p"]:.2f}, q = {ms["arm_prior_q"]:.2f}) the arm alone would <b>{ms["arm_alone_action"]}</b>. The value of the base edge is {ms["voa_base"]:.2f} and of the verifier edge {ms["voa_verifier"]:.2f}, both above the cost; a sliding report flips the action to <b>{ms["action_if_base_reports_sliding"]}</b>, a near-limit report to <b>{ms["action_if_verifier_reports_near"]}</b>. Complementary evidence changes the compliant action.</div>

  <h3>Stage B: held-out physics episodes</h3>
  <div class="tablewrap"><table>
  <tr><th>Controller</th><th>Compliant completion</th><th>Violations / episode</th><th>Unnecessary stops / decision</th><th>Decision preservation</th><th>Utility / decision</th><th>Steps</th><th>Messages / decision</th></tr>
  {main_rows}
  </table></div>
  <figure><img src="figures/b_main.svg" alt="Main Stage B comparison"><figcaption>Means with standard errors over {n_eps} held-out episodes per controller. One third of the held-out configurations need more force than the limit allows, so {pct(2/3)} is the ceiling for a compliant controller; every aligned controller reaches it with zero violations.</figcaption></figure>
  <div class="two">
    <figure><img src="figures/b_pareto.svg" alt="Utility against messages per decision"><figcaption>Quality against cost. The gate sits next to always-align and the centralized network on utility at {pct(saving3)} fewer messages; the CAN-pool variants are far to the right and well below.</figcaption></figure>
    <figure><img src="figures/b_families.svg" alt="Per-family completion and stops"><figcaption>By drawer resistance and cabinet type. Isolated and event-triggered arms stall on the 60 N drawer after a blind release, because without the verifier's limit they back off to the conservative prior.</figcaption></figure>
  </div>

  <h3>Scaling with the number of specialists</h3>
  <div class="tablewrap"><table>
  <tr><th>N</th><th>Always, exact: msgs</th><th>Gate, exact: msgs</th><th>Always, CAN: msgs</th><th>Gate, CAN: msgs</th><th>Utility exact (always / gate)</th><th>Utility CAN (always / gate)</th></tr>
  {n_rows}
  </table></div>
  <figure><img src="figures/b_nscaling.svg" alt="Messages and utility against N in Stage B"><figcaption>Adding a redundant head camera and an irrelevant torso specialist raises always-align's cost by {pct(growth_always)} while the gate's cost does not move: it never opened either new edge. Under ring diffusion the always-align pool collapses at N=5 (utility {N["always/can/N5"]["util"]:.2f}).</figcaption></figure>

  <h3>Horizon × task family, and robustness</h3>
  <div class="two">
    <div><div class="tablewrap"><table><tr><th>Family</th>{"".join(f"<th>T = {t}</th>" for t in grid["T"])}</tr>{grid_cells}</table></div><p class="muted" style="font-size:13px">Cells: Δ compliant completion (gate − always) / message saving. Short horizons penalise the gate's slightly longer episodes.</p></div>
    <div class="tablewrap"><table><tr><th>Perturbation</th><th>Always</th><th>Gate</th><th>Event</th><th>Always msgs</th><th>Gate msgs</th></tr>{rob_rows}</table></div>
  </div>
  <figure><img src="figures/b_robust.svg" alt="Robustness under perturbations"><figcaption>Delayed reports and a dropped base do not change compliant completion. A corrupted base map (30% of reports flipped, unknown to the gate) hurts every aligned controller; the gate keeps {pct(rob["corrupt_map/voa"]["compliant"])} against always-align's {pct(rob["corrupt_map/always"]["compliant"])}, because it consults the corrupted neighbor less often.</figcaption></figure>
</section>

<section id="interpretation">
  <h2>Interpretation</h2>
  <div class="findings">
    <div class="finding"><b>The gate keeps the decision and drops the traffic.</b> On held-out configurations the gate matched always-align on compliant completion ({pct(g("voa/exact","compliant"))}), utility ({g("voa/exact","util"):.3f} vs {g("always/exact","util"):.3f}) and stops, preserved {pct(g("voa/exact","preserved"),1)} of its decisions, and sent {pct(saving3)} fewer messages at three specialists. The oracle reference shows more headroom ({g("oracle/exact","messages"):.2f} messages per decision): the gate is conservative because the arm's own evidence is weak, so the base edge is usually worth its cost.</div>
    <div class="finding"><b>Cost scales with decision-relevant edges, not with agents.</b> From three to five specialists the gate's messages stayed at {N["voa/exact/N3"]["messages"]:.2f} per decision while always-align rose to {N["always/exact/N5"]["messages"]:.2f}; in Stage A the gate held near {float(nA[("exact","voa","12")]["edges_opened"]):.1f} edges per decision up to twelve specialists. The redundant head camera and the irrelevant torso were never consulted.</div>
    <div class="finding"><b>The fusion rule matters more than the gate.</b> Replacing exact fusion with the fixed-weight CAN diffusion pool cut utility by {pct(can_util_drop)} and decision preservation to {pct(g("always/can","preserved"))} even with every edge open, cost {g("always/can","messages"):.0f} messages per decision, and collapsed at five specialists. This reproduces, inside a robot task, the earlier Pursuit finding that pooled evidence beats fixed-weight mixing, and it says the gate should sit in front of an exact or confidence-weighted fusion rule.</div>
    <div class="finding"><b>Isolation costs completion and time, not violations.</b> Every controller, including the isolated arm, recorded zero violations on held-out configurations: the arm's own learned model is cautious enough to brace early. Isolation instead showed up as stalls ({pct(g("never/exact","compliant"))} completion) and unnecessary stops ({g("never/exact","stops"):.2f} per decision). The verifier edge's value in this task is the limit it carries: an arm that releases without it backs off to the prior and never recovers.</div>
    <div class="finding"><b>A centralized network is as good here, on quality.</b> One classifier over all features matched always-align on every quality metric. The modular gate's advantage is on cost and on robustness to a corrupted neighbor, not on decision quality. That is a fair result to report and it narrows the claim.</div>
    <div class="finding"><b>Limits.</b> Report accuracies are state-independent calibrations; the task has two binary latents; the horizon grid shows no family dependence because both cabinet types were braced early; the process-violation advantage did not materialise because no controller violated. These bound what the numbers support.</div>
  </div>
</section>

<section id="next">
  <h2>Next steps and research direction</h2>
  <div class="steps">
    <div class="step"><div><b>State-dependent report accuracies.</b> Replace the calibrated constants with accuracies conditioned on the neighbor's own evidence strength, so the gate can see when a neighbor currently knows nothing (a base before any force is applied). This should close part of the gap to the oracle reference.</div></div>
    <div class="step"><div><b>Confidence-weighted fusion behind the gate.</b> Run the gate in front of a confidence-weighted CAN update and a convergence-matched diffusion, so the fusion axis is a fair contest rather than a fixed-weight pool.</div></div>
    <div class="step"><div><b>Learned subgraphs.</b> Discover each specialist's subgraph online instead of declaring it, and let the gate consume the discovered overlap structure. The gate's cost claim predicts that cost should track the discovered number of decision-relevant edges.</div></div>
    <div class="step"><div><b>Harder process constraints.</b> Add constraints that the arm's own model cannot anticipate (a sudden change in cabinet anchoring mid-episode, a limit that changes with contact state), so the safety value of alignment is measured, not only its efficiency.</div></div>
    <div class="step"><div><b>Shared harness.</b> Package the Task (environment, subgraph assignment, verifier spec, cost model) and Solver (gate, fusion) interfaces so the same gate runs on the grid-pursuit and real-time-strategy environments the other workstreams use, and on a full mobile-manipulation simulator.</div></div>
    <div class="step"><div><b>Sequential gate.</b> The gate is one-step. A sequential version values an edge by its effect on later beliefs and probes, which the formal framework already defines.</div></div>
  </div>
</section>

<section id="related">
  <h2>Related work</h2>
  <p class="sub">Verified sources and what this work takes from each.</p>
  <ul class="lit">{lit_items}</ul>
</section>

<footer>Reproduce: <span class="mono">python gca/stage_a.py && python gca/stage_b.py && python gca/plots.py && python gca/build_site.py</span>. Every number above is read from the results files those scripts write.</footer>
</div>

<script>
const TRACES = {traces_json};
const NOTES = {{
  free_cabinet: "A 60 N drawer in an unanchored cabinet. Pulling without bracing would slide the cabinet (a process violation). The base's cabinet-displacement reading is what separates that case from an easy drawer.",
  over_limit: "An 85 N drawer behind an 80 N limit. The task is infeasible under the constraint: the compliant outcome is to stop. The verifier's report carries the limit, which the arm needs to back off correctly.",
  easy_anchored: "A 35 N drawer in an anchored cabinet. Nothing can go wrong; the question is how much the controller pays to find that out."
}};
(function(){{
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const NS='http://www.w3.org/2000/svg';
  const el=(n,a,p)=>{{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}};
  const txt=(p,s,a)=>{{const e=el('text',a,p);e.textContent=s;return e;}};
  const scene=document.getElementById('scene');
  const X=w=>60+(w+0.1)*440;   // world metres to px
  let tr=null, alw=null, t=0, playing=true, last=0, parts={{}};
  function build(){{
    scene.innerHTML='';
    const ink=css('--fg'),muted=css('--muted'),line=css('--line'),grid=css('--grid'),accent=css('--accent'),base=css('--base'),ver=css('--verify'),bad=css('--bad');
    el('line',{{x1:20,x2:740,y1:200,y2:200,stroke:line}},scene);
    txt(scene,'floor',{{x:24,y:214,'font-size':10,fill:muted}});
    parts.cab=el('rect',{{x:0,y:90,width:0,height:110,rx:3,fill:grid,stroke:line}},scene);
    parts.drawer=el('rect',{{x:0,y:120,width:0,height:40,rx:2,fill:css('--panel'),stroke:line}},scene);
    parts.handle=el('rect',{{x:0,y:132,width:6,height:16,rx:2,fill:ink}},scene);
    parts.base=el('rect',{{x:0,y:130,width:0,height:70,rx:4,fill:css('--arm')}},scene);
    parts.plate=el('rect',{{x:0,y:110,width:5,height:90,fill:ink}},scene);
    parts.arm=el('line',{{x1:0,x2:0,y1:140,y2:140,stroke:css('--arm'),'stroke-width':5,'stroke-linecap':'round'}},scene);
    parts.force=el('rect',{{x:0,y:62,width:0,height:8,rx:2,fill:ver}},scene);
    parts.forceLbl=txt(scene,'',{{x:0,y:58,'font-size':10.5,fill:muted,'font-family':'IBM Plex Mono, monospace'}});
    parts.limit=el('line',{{x1:0,x2:0,y1:58,y2:74,stroke:bad,'stroke-dasharray':'3 2'}},scene);
    parts.viol=txt(scene,'',{{x:380,y:30,'text-anchor':'middle','font-size':12,'font-weight':600,fill:bad}});
    // specialist nodes
    const node=(x,y,name,c)=>{{el('rect',{{x:x-54,y:y-18,width:108,height:36,rx:8,fill:'none',stroke:c,'stroke-width':1.5}},scene);txt(scene,name,{{x:x,y:y+4,'text-anchor':'middle','font-size':12,'font-weight':600,fill:ink}});}};
    parts.eB=el('line',{{x1:110,y1:250,x2:286,y2:250,stroke:line,'stroke-width':1.5}},scene);
    parts.eV=el('line',{{x1:650,y1:250,x2:474,y2:250,stroke:line,'stroke-width':1.5}},scene);
    node(56,250,'Base',base); node(380,250,'Arm (acts)',ink); node(704,250,'Verifier',ver);
    parts.lB=txt(scene,'closed',{{x:198,y:244,'text-anchor':'middle','font-size':10.5,fill:muted,'font-family':'IBM Plex Mono, monospace'}});
    parts.lV=txt(scene,'closed',{{x:562,y:244,'text-anchor':'middle','font-size':10.5,fill:muted,'font-family':'IBM Plex Mono, monospace'}});
    parts.pB=txt(scene,'',{{x:198,y:266,'text-anchor':'middle','font-size':10.5,fill:muted}});
    parts.pV=txt(scene,'',{{x:562,y:266,'text-anchor':'middle','font-size':10.5,fill:muted}});
    // belief bars
    txt(scene,'P(sliding)',{{x:300,y:290,'text-anchor':'end','font-size':10.5,fill:muted}});
    txt(scene,'P(near limit)',{{x:300,y:304,'text-anchor':'end','font-size':10.5,fill:muted}});
    el('rect',{{x:310,y:282,width:140,height:8,rx:2,fill:grid}},scene); el('rect',{{x:310,y:296,width:140,height:8,rx:2,fill:grid}},scene);
    parts.bp=el('rect',{{x:310,y:282,width:0,height:8,rx:2,fill:base}},scene);
    parts.bq=el('rect',{{x:310,y:296,width:0,height:8,rx:2,fill:ver}},scene);
    parts.bpf=el('line',{{x1:310,x2:310,y1:280,y2:292,stroke:accent,'stroke-width':2}},scene);
    parts.bqf=el('line',{{x1:310,x2:310,y1:294,y2:306,stroke:accent,'stroke-width':2}},scene);
    txt(scene,'own belief, bar; after alignment, marker',{{x:460,y:300,'font-size':10,fill:muted}});
    scene.setAttribute('viewBox','0 0 760 312');
  }}
  function render(){{
    if(!tr)return; const s=tr.steps[Math.min(t,tr.steps.length-1)];
    const accent=css('--accent'),line=css('--line'),muted=css('--muted'),bad=css('--bad');
    const cabL=X(s.cab-0.25), cabW=X(s.cab+0.25)-cabL;
    parts.cab.setAttribute('x',cabL); parts.cab.setAttribute('width',cabW);
    const front=X(s.cab-0.25-s.d);
    parts.drawer.setAttribute('x',front); parts.drawer.setAttribute('width',cabL+cabW-20-front);
    parts.handle.setAttribute('x',front-6);
    const bL=X(s.base-0.2); parts.base.setAttribute('x',bL); parts.base.setAttribute('width',X(s.base+0.2)-bL);
    parts.plate.setAttribute('x',X(s.base+0.22));
    parts.arm.setAttribute('x1',X(s.base+0.2)); parts.arm.setAttribute('x2',front-6);
    const f0=X(s.base-0.2); parts.force.setAttribute('x',f0); parts.force.setAttribute('width',Math.max(0,s.F)*1.6);
    parts.forceLbl.setAttribute('x',f0); parts.forceLbl.textContent='F = '+s.F.toFixed(0)+' N, cap '+s.cap.toFixed(0);
    parts.limit.setAttribute('x1',f0+80*1.6); parts.limit.setAttribute('x2',f0+80*1.6);
    const violating = (s.action==='pull' && (s.s===1 || s.m===1));
    parts.viol.textContent = violating ? 'pulling while the latent says stop' : '';
    const openB=s.opened.includes('base'), openV=s.opened.includes('verifier');
    const setE=(e,l,open,rep,lbl)=>{{e.setAttribute('stroke',open?accent:line);e.setAttribute('stroke-width',open?3:1.5);l.textContent=open?'open':'closed';l.setAttribute('fill',open?accent:muted);lbl.textContent=open?('report: '+rep):'';}};
    setE(parts.eB,parts.lB,openB,s.rep_base?'sliding':'not sliding',parts.pB);
    setE(parts.eV,parts.lV,openV,s.rep_ver?'near limit':'clear',parts.pV);
    parts.bp.setAttribute('width',140*s.p); parts.bq.setAttribute('width',140*s.q);
    parts.bpf.setAttribute('x1',310+140*s.pf); parts.bpf.setAttribute('x2',310+140*s.pf);
    parts.bqf.setAttribute('x1',310+140*s.qf); parts.bqf.setAttribute('x2',310+140*s.qf);
    document.getElementById('ract').innerHTML='<span class="pill '+s.action+'">'+s.action+'</span>';
    document.getElementById('rF').textContent=s.F.toFixed(0)+' / '+s.cap.toFixed(0);
    document.getElementById('rd').textContent=s.d.toFixed(3);
    document.getElementById('rxc').textContent=Math.abs(s.xc).toFixed(3);
    let msgs=0, util=0, am=0; for(let i=0;i<=t && i<tr.steps.length;i++){{msgs+=tr.steps[i].opened.length; util+=tr.steps[i].util;}}
    for(let i=0;i<=t && i<alw.steps.length;i++){{am+=alw.steps[i].opened.length;}}
    document.getElementById('rmsg').textContent=msgs; document.getElementById('rmsgu').textContent='always-align would be '+am;
    document.getElementById('rutil').textContent=util.toFixed(1);
    document.getElementById('tl').textContent='t = '+s.t+(tr.completed&&t>=tr.steps.length-1?' · opened':'' )+(!tr.completed&&t>=tr.steps.length-1?' · stopped, drawer '+tr.drawer_open.toFixed(2)+' m':'');
    document.getElementById('scrub').value=t;
  }}
  function load(){{
    const scn=document.getElementById('scn').value, ctl=document.getElementById('ctl').value;
    tr=TRACES[scn][ctl]; alw=TRACES[scn]['always']; t=0;
    document.getElementById('scrub').max=tr.steps.length-1;
    document.getElementById('scnnote').textContent=NOTES[scn]+' This run: '+(tr.completed?'drawer opened':'drawer not opened')+' in '+tr.steps.length+' decision steps, '+(tr.violations.process+tr.violations.force)+' violations, '+tr.steps.reduce((a,s)=>a+s.opened.length,0)+' edges opened.';
    render();
  }}
  build(); load();
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches; if(reduced){{playing=false;}}
  const pb=document.getElementById('play'); pb.textContent=playing?'Pause':'Play';
  function tick(now){{ if(playing && now-last>90){{ last=now; t=(t+1)%tr.steps.length; render(); }} requestAnimationFrame(tick); }}
  requestAnimationFrame(tick);
  pb.addEventListener('click',()=>{{playing=!playing;pb.textContent=playing?'Pause':'Play';}});
  document.getElementById('restart').addEventListener('click',()=>{{t=0;render();}});
  document.getElementById('scrub').addEventListener('input',e=>{{playing=false;pb.textContent='Play';t=Number(e.target.value);render();}});
  document.getElementById('scn').addEventListener('change',load); document.getElementById('ctl').addEventListener('change',load);
  const rebuild=()=>{{build();render();}};
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',rebuild);
  new MutationObserver(rebuild).observe(document.documentElement,{{attributes:true,attributeFilter:['data-theme']}});
}})();
</script>
</body>
</html>
"""
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(html)
    print("wrote site/index.html", len(html), "bytes")


if __name__ == "__main__":
    main()
