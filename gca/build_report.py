"""Build report/report.html and report/report.pdf from the results. Numbers come from results/ only."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
REP = os.path.join(ROOT, "report")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def main():
    os.makedirs(os.path.join(REP, "figures"), exist_ok=True)
    for f in os.listdir(os.path.join(RES, "figures")):
        if f.endswith(".svg"):
            shutil.copy(os.path.join(RES, "figures", f), os.path.join(REP, "figures", f))
    S = json.load(open(os.path.join(RES, "stage_b", "summary.json")))
    A = json.load(open(os.path.join(RES, "stage_a", "summary.json")))
    I = json.load(open(os.path.join(RES, "stage_b", "info.json")))
    L = json.load(open(os.path.join(ROOT, "lit", "related_work.json")))
    sweeps = list(csv.DictReader(open(os.path.join(RES, "stage_a", "sweeps.csv"))))
    nscale = list(csv.DictReader(open(os.path.join(RES, "stage_a", "n_scaling.csv"))))
    M, N, rob, grid, thm, ms = S["main"], S["n_scaling"], S["robustness"], S["grid"], A["theorem_check"], A["milestone"]
    g = lambda c, m: M[c][m]["mean"]
    n_eps = M["voa/exact"]["compliant"]["n"]
    a65 = {r["policy"]: r for r in sweeps if r["sweep"] == "own_evidence" and r["fuse"] == "exact" and r["a_own"] == "0.65"}
    a95 = {r["policy"]: r for r in sweeps if r["sweep"] == "own_evidence" and r["fuse"] == "exact" and r["a_own"] == "0.95"}
    nA = {(r["fuse"], r["policy"], r["n"]): r for r in nscale}
    saving3 = 1 - g("voa/exact", "messages") / g("always/exact", "messages")
    growth_always = N["always/exact/N5"]["messages"] / N["always/exact/N3"]["messages"] - 1
    can_util_drop = 1 - g("always/can", "util") / g("always/exact", "util")
    a_cost_save65 = 1 - float(a65["voa"]["cost"]) / float(a65["always"]["cost"])
    a_cost_save95 = 1 - float(a95["voa"]["cost"]) / float(a95["always"]["cost"])
    a_util_gap65 = float(a65["always"]["util"]) - float(a65["voa"]["util"])

    order = ["centralized/exact", "never/exact", "always/exact", "event/exact", "random/exact", "voa/exact", "oracle/exact", "always/can", "voa/can"]
    label = {"centralized/exact": "Centralized network", "never/exact": "Isolated (never)", "always/exact": "Always-align, exact",
             "event/exact": "Event-triggered", "random/exact": "Budget-matched random", "voa/exact": "<b>VOA gate, exact</b>",
             "oracle/exact": "Oracle gate (ref.)", "always/can": "Always-align, CAN pool", "voa/can": "VOA gate, CAN pool"}
    rows = "".join(
        f'<tr{" class=hl" if k == "voa/exact" else ""}><td>{label[k]}</td><td>{pct(M[k]["compliant"]["mean"])}</td><td>{M[k]["violations"]["mean"]:.2f}</td>'
        f'<td>{M[k]["stops"]["mean"]:.3f}</td><td>{pct(M[k]["preserved"]["mean"], 1)}</td><td>{M[k]["util"]["mean"]:.3f}</td><td>{M[k]["steps"]["mean"]:.0f}</td><td>{M[k]["messages"]["mean"]:.2f}</td></tr>'
        for k in order)
    nrows = "".join(
        f'<tr><td>{n}</td><td>{N[f"always/exact/N{n}"]["messages"]:.2f}</td><td>{N[f"voa/exact/N{n}"]["messages"]:.2f}</td><td>{N[f"always/can/N{n}"]["messages"]:.0f}</td>'
        f'<td>{N[f"voa/can/N{n}"]["messages"]:.1f}</td><td>{N[f"always/exact/N{n}"]["util"]:.2f} / {N[f"voa/exact/N{n}"]["util"]:.2f}</td><td>{N[f"always/can/N{n}"]["util"]:.2f} / {N[f"voa/can/N{n}"]["util"]:.2f}</td></tr>'
        for n in (2, 3, 5))
    rob_label = {"delay1": "Reports delayed 1 step", "delay3": "Reports delayed 3 steps", "drop_base": "Base dropped mid-episode", "corrupt_map": "Base map corrupted (30%)"}
    rrows = "".join(
        f'<tr><td>{rob_label[rb]}</td><td>{pct(rob[f"{rb}/always"]["compliant"])}</td><td>{pct(rob[f"{rb}/voa"]["compliant"])}</td><td>{pct(rob[f"{rb}/event"]["compliant"])}</td>'
        f'<td>{rob[f"{rb}/always"]["messages"]:.2f}</td><td>{rob[f"{rb}/voa"]["messages"]:.2f}</td></tr>' for rb in ("delay1", "delay3", "drop_base", "corrupt_map"))
    venue = lambda v: re.sub(r"\s*\(?\b(19|20)\d\d\)?", "", v).strip(" ,.")
    refs = "".join(f'<li>{p["authors"]}. <i>{p["title"]}</i>. {venue(p["venue"])}. <span class="u">{p["url"]}</span></li>' for p in sorted(L, key=lambda p: p["topic"]))
    grid_cells = "".join(f"<tr><td>{fam}</td>" + "".join(f'<td>{grid["delta_compliant"][i][j]:+.2f} / {pct(grid["message_saving"][i][j])}</td>' for j in range(len(grid["T"]))) + "</tr>"
                         for i, fam in enumerate(grid["families"]))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Gated Causal Alignment</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono&display=swap">
<style>
@page {{ size: Letter; margin: 16mm 16mm 16mm 16mm; }}
:root {{ --fg:#1b2130; --muted:#5d6678; --line:#d3d8e1; --accent:#b5551b; --soft:#f6ece4; }}
body {{ font-family:"IBM Plex Sans",Arial,sans-serif; color:var(--fg); font-size:10.2pt; line-height:1.38; margin:0; }}
h1 {{ font-family:"IBM Plex Serif",Georgia,serif; font-size:22pt; margin:0 0 4px; line-height:1.15; }}
.sub {{ color:var(--muted); font-size:10.5pt; margin:0 0 10px; }}
h2 {{ font-family:"IBM Plex Serif",Georgia,serif; font-size:13.5pt; margin:14px 0 5px; border-bottom:1px solid var(--line); padding-bottom:2px; break-after:avoid; }}
h3 {{ font-size:10.8pt; margin:9px 0 3px; }}
p {{ margin:0 0 6px; }}
.abstract {{ background:var(--soft); border-left:3px solid var(--accent); padding:8px 12px; margin:8px 0 10px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:8.8pt; margin:4px 0 6px; }}
th,td {{ padding:3px 5px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ font-size:7.8pt; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
td:not(:first-child),th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.hl td {{ background:var(--soft); }}
figure {{ margin:6px 0 8px; break-inside:avoid; }}
figure img {{ width:100%; display:block; }}
figcaption {{ font-size:8.6pt; color:var(--muted); margin-top:2px; }}
.formula {{ font-family:"IBM Plex Mono",monospace; font-size:9.6pt; background:#f3f4f6; padding:5px 9px; border-radius:4px; margin:4px 0 8px; }}
ul {{ margin:2px 0 6px; padding-left:18px; }} li {{ margin-bottom:2px; }}
ol.refs {{ font-size:8.3pt; padding-left:16px; column-count:2; column-gap:14px; }} ol.refs li {{ margin-bottom:3px; break-inside:avoid; }} .u {{ color:var(--muted); font-size:7.6pt; word-break:break-all; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:6px 0 8px; }}
.kpi {{ border:1px solid var(--line); border-radius:6px; padding:6px 8px; }} .kpi .k {{ font-size:7.6pt; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }} .kpi .v {{ font-family:"IBM Plex Mono",monospace; font-size:15pt; }} .kpi .d {{ font-size:8pt; color:var(--muted); }}
.pb {{ break-before:page; }}
</style></head><body>
<h1>Gated Causal Alignment: consulting a neighbor only when its evidence would change the action</h1>
<p class="sub">A value-of-alignment gate for multi-specialist robot control, evaluated in a synthetic causal model and a physics drawer task</p>
<div class="abstract"><b>Summary.</b> A robot arm, a mobile base and a verifier each hold evidence about a different part of a drawer task's causal graph. Before each decision the arm may align with a neighbor, at a declared cost. We propose a gate that opens an alignment edge only when the expected decision gain exceeds that cost, the single-agent observation-refinement criterion applied to agent-to-agent edges. In exact enumeration over a synthetic causal model the gate stays within {a_util_gap65:.3f} utility per decision of always-align while paying {pct(a_cost_save65)} less, and its cost stays flat as redundant specialists are added. In {n_eps} held-out physics episodes it matches always-align on compliant completion ({pct(g("voa/exact","compliant"))}) and utility, preserves {pct(g("voa/exact","preserved"),1)} of its decisions, and sends {pct(saving3)} fewer messages at three specialists and {pct(1 - N["voa/exact/N5"]["messages"]/N["always/exact/N5"]["messages"])} fewer at five. The fusion rule matters more than the gate: a fixed-weight diffusion pool loses {pct(can_util_drop)} of utility even with every edge open. No controller violated a constraint on held-out configurations; isolation cost completion and time instead.</div>
<div class="kpis">
<div class="kpi"><div class="k">Decisions preserved</div><div class="v">{pct(g("voa/exact","preserved"),1)}</div><div class="d">gate vs always-align</div></div>
<div class="kpi"><div class="k">Messages saved</div><div class="v">{pct(saving3)}</div><div class="d">N=3; {pct(1 - N["voa/exact/N5"]["messages"]/N["always/exact/N5"]["messages"])} at N=5</div></div>
<div class="kpi"><div class="k">Compliant completion</div><div class="v">{pct(g("voa/exact","compliant"))}</div><div class="d">equal to always-align; ceiling given infeasible cases</div></div>
<div class="kpi"><div class="k">CAN-pool utility loss</div><div class="v">{pct(can_util_drop)}</div><div class="d">fixed-weight diffusion vs exact fusion</div></div>
</div>

<h2>1. Problem and idea</h2>
<div class="two"><div>
<p>Process-compliant manipulation can fail while a proxy succeeds: a robot that pulls a drawer handle makes progress on handle displacement whether the drawer opens or the whole cabinet slides. Specialists hold complementary causal evidence (the base sees the cabinet move; the verifier knows the force limit), but continuous alignment is costly and fixed-weight consensus can wash out a confident neighbor. The question is which neighbors to consult, and when.</p>
<p>Existing lines of work answer two neighbouring questions: when one agent should pay to refine its own observation, and how agents should combine beliefs once they share. This work sits between them.</p>
</div><div>
<div class="formula">VOA<sub>ij</sub>(t) = E[U(a*(b<sub>i</sub> ⊕ b<sub>j</sub>))] − E[U(a*(b<sub>i</sub>))] − c<sub>ij</sub></div>
<p>The arm holds beliefs p (the cabinet will slide) and q (the next pull crosses the force limit) and three skills: pull, reposition (brace), release (back off). A neighbor's report is a binary signal with calibrated true-positive and true-negative rates. The gate computes the preposterior value of each closed edge, opens the best positive one, updates, and repeats. Its precondition is the refinement theorem's: an edge has strict value only when the arm's belief aliases cases with different optimal actions and the report separates them. A finite check over {thm["cases"]:,} cases agreed in {thm["agree"]:,}; the {thm["ties"]} others are exact ties.</p>
</div></div>

<h2>2. Experimental design</h2>
<p><b>Design space.</b> An instance is (F, T, N, G, {{G<sub>i</sub>}}, Ω, c, Π<sub>fuse</sub>, Π<sub>gate</sub>, Φ): task family and horizon, number of specialists, task graph and subgraphs, overlap, cost, fusion rule, gate policy, verifier spec. Outcomes: compliant completion (drawer open, zero violations), violations, unnecessary stops, decision preservation against always-align, regret against the oracle gate, messages per decision.</p>
<div class="two"><div>
<h3>Stage A: synthetic POSCM, exact</h3>
<p>Force drives handle displacement, then the drawer or the cabinet moves; a force limit is exogenous. The arm's own evidence has a declared accuracy; the base and verifier report with accuracies 0.95 and 0.99; edge cost 0.1. Worlds, own evidence and all neighbor reports are enumerated, so every number is an expectation. Sweeps: own-evidence quality (overlap), cost, prior, and N up to 12 with redundant and irrelevant specialists.</p>
</div><div>
<h3>Stage B: physics drawer task</h3>
<p>A MuJoCo scene: cabinet on a slide joint (anchored or free), drawer with rail resistance, a heavy base whose plate braces the cabinet, a pull modelled as an external force pair, a stiff-start handle variant. Specialists fit class-balanced logistic models on {I["training_rows"]:,} steps from {I["training_configs"]} training configurations (limits vary so only the verifier knows the limit). Held-out: {I["held_out_configs"]} unseen configurations × 3 seeds. Report rates calibrated on training data: base {I["report_rates"]["base"]["tpr"]:.2f}/{I["report_rates"]["base"]["tnr"]:.2f}, verifier {I["report_rates"]["verifier"]["tpr"]:.2f}/{I["report_rates"]["verifier"]["tnr"]:.2f}; the arm alone reaches {I["arm_own_rates"]["s"]["acc"]:.2f} on sliding.</p>
</div></div>
<p><b>Controllers.</b> Centralized network (one classifier over all features, no messages); isolated arm; always-align with exact fusion or a fixed-weight CAN diffusion pool; event-triggered; budget-matched random (the gate's pilot open rates); the VOA gate; an oracle gate that opens an edge only if the true latent would change the action (a reference, since it still acts on noisy reports).</p>

<h2>3. Results</h2>
<h3>Stage A</h3>
<figure><img src="figures/a_overlap.svg" alt=""><figcaption>Utility, decision preservation and cost paid against the arm's own evidence accuracy. At 0.65 the gate pays {pct(a_cost_save65)} less than always-align for a utility gap of {a_util_gap65:.3f}; at 0.95 it pays {pct(a_cost_save95)} less. Dashed: the fixed-weight CAN pool, which loses decisions even with every edge open.</figcaption></figure>
<div class="two">
<figure><img src="figures/a_nscaling.svg" alt=""><figcaption>Adding redundant and irrelevant specialists: the gate opens {float(nA[("exact","voa","12")]["edges_opened"]):.2f} edges per decision at N=12 versus {float(nA[("exact","voa","3")]["edges_opened"]):.2f} at N=3; always-align grows to {float(nA[("exact","always","12")]["messages"]):.0f} messages (exact) or {float(nA[("can","always","12")]["messages"]):,.0f} (ring diffusion to consensus).</figcaption></figure>
<div><p><b>Milestone.</b> With the arm's own evidence saying "fine" (p = {ms["arm_prior_p"]:.2f}, q = {ms["arm_prior_q"]:.2f}) the arm alone would {ms["arm_alone_action"]}. The base edge is worth {ms["voa_base"]:.2f} and the verifier edge {ms["voa_verifier"]:.2f}, both above the cost of 0.1. A sliding report flips the action to {ms["action_if_base_reports_sliding"]}; a near-limit report to {ms["action_if_verifier_reports_near"]}. Complementary evidence changes the compliant action, which was the proposal's first deliverable.</p>
<p>Raising the declared cost closes edges smoothly. The budget-matched random control pays the same as the gate at every cost level and loses utility, because it opens the wrong edges.</p></div>
</div>
<h3>Stage B: held-out physics episodes ({n_eps} per controller)</h3>
<table><tr><th>Controller</th><th>Compliant</th><th>Violations</th><th>Stops / dec.</th><th>Preserved</th><th>Utility / dec.</th><th>Steps</th><th>Msgs / dec.</th></tr>{rows}</table>
<div class="two">
<figure><img src="figures/b_pareto.svg" alt=""><figcaption>Quality against cost. The gate sits beside always-align and the centralized network on utility at {pct(saving3)} fewer messages; CAN-pool variants are far right and below.</figcaption></figure>
<figure><img src="figures/b_nscaling.svg" alt=""><figcaption>From N=3 to N=5 always-align's cost grows {pct(growth_always)} while the gate's does not move; the always-align CAN pool collapses at N=5 (utility {N["always/can/N5"]["util"]:.2f}).</figcaption></figure>
</div>
<div class="two">
<div><table><tr><th>N</th><th>Always exact</th><th>Gate exact</th><th>Always CAN</th><th>Gate CAN</th><th>Util. exact A/G</th><th>Util. CAN A/G</th></tr>{nrows}</table>
<table><tr><th>Family</th>{"".join(f"<th>T={t}</th>" for t in grid["T"])}</tr>{grid_cells}</table><p style="font-size:8.4pt;color:var(--muted)">Δ compliant completion (gate − always) / message saving by horizon.</p></div>
<div><table><tr><th>Perturbation</th><th>Always</th><th>Gate</th><th>Event</th><th>Always msgs</th><th>Gate msgs</th></tr>{rrows}</table><p style="font-size:8.4pt;color:var(--muted)">Compliant completion and messages per decision under delayed reports, a dropped base and a corrupted base map. Nominal: always {pct(g("always/exact","compliant"))}, gate {pct(g("voa/exact","compliant"))}, event {pct(g("event/exact","compliant"))}.</p></div>
</div>

<h2>4. Interpretation</h2>
<ul>
<li><b>The gate keeps the decision and drops the traffic.</b> It matched always-align on compliant completion, utility ({g("voa/exact","util"):.3f} vs {g("always/exact","util"):.3f}) and stops, preserved {pct(g("voa/exact","preserved"),1)} of decisions and sent {pct(saving3)} fewer messages. The oracle reference ({g("oracle/exact","messages"):.2f} messages per decision) shows headroom: the gate is conservative because the arm's own evidence is weak, so the base edge is usually worth its cost.</li>
<li><b>Cost scales with decision-relevant edges, not agents.</b> The gate never consulted the redundant head camera or the irrelevant torso; its messages stayed at {N["voa/exact/N3"]["messages"]:.2f} per decision from N=3 to N=5 while always-align rose to {N["always/exact/N5"]["messages"]:.2f}. Stage A shows the same up to twelve specialists.</li>
<li><b>The fusion rule matters more than the gate.</b> The fixed-weight CAN pool cut utility by {pct(can_util_drop)} and preservation to {pct(g("always/can","preserved"))} with every edge open, cost {g("always/can","messages"):.0f} messages per decision, and collapsed at N=5. This reproduces the earlier grid-pursuit finding inside a robot task: the gate should sit in front of an exact or confidence-weighted fusion rule.</li>
<li><b>Isolation cost completion and time, not violations.</b> No controller violated a constraint on held-out configurations; the arm's own learned model braces early. Isolation showed up as stalls ({pct(g("never/exact","compliant"))} completion) and stops ({g("never/exact","stops"):.2f} per decision): an arm that releases without the verifier's limit backs off to the prior and never recovers.</li>
<li><b>A centralized network is as good on quality.</b> One classifier over all features matched always-align on every quality metric. The gate's advantage is cost and robustness to a corrupted neighbor ({pct(rob["corrupt_map/voa"]["compliant"])} vs {pct(rob["corrupt_map/always"]["compliant"])} compliant completion), not decision quality.</li>
<li><b>Limits.</b> Report accuracies are state-independent calibrations; the task has two binary latents; the horizon grid shows no family dependence because both cabinet types were braced early; the process-safety advantage did not materialise because no controller violated.</li>
</ul>

<h2>5. Next steps and research direction</h2>
<ul>
<li><b>State-dependent report accuracies</b> conditioned on the neighbor's own evidence strength, so the gate sees when a neighbor currently knows nothing; expected to close part of the gap to the oracle.</li>
<li><b>Confidence-weighted and convergence-matched fusion</b> behind the gate, so the fusion axis is a fair contest rather than a fixed-weight pool.</li>
<li><b>Learned subgraphs.</b> Discover each specialist's subgraph online and feed the discovered overlap to the gate; the cost claim predicts cost tracks the discovered number of decision-relevant edges.</li>
<li><b>Harder process constraints</b> the arm cannot anticipate (anchoring that changes mid-episode, a contact-dependent limit), so the safety value of alignment is measured rather than only its efficiency.</li>
<li><b>Shared harness.</b> Package the Task (environment, subgraphs, verifier spec, cost model) and Solver (gate, fusion) interfaces so the same gate runs on grid-pursuit and real-time-strategy environments and on a full mobile-manipulation simulator.</li>
<li><b>Sequential gate</b> that values an edge by its effect on later beliefs and probes.</li>
</ul>

<h2>References</h2>
<ol class="refs">{refs}</ol>
<p style="font-size:8.4pt;color:var(--muted)">Reproduction: <span style="font-family:'IBM Plex Mono',monospace">python gca/stage_a.py; python gca/stage_b.py; python gca/plots.py; python gca/build_report.py</span>. Every number in this report is read from the results files those scripts write.</p>
</body></html>"""
    path = os.path.join(REP, "report.html")
    with open(path, "w") as f:
        f.write(html)
    pdf = os.path.join(REP, "report.pdf")
    if os.path.exists(CHROME):
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        "--virtual-time-budget=8000", "file://" + path], check=False, capture_output=True, timeout=120)
        print("wrote", pdf, os.path.exists(pdf))
    print("wrote", path)


if __name__ == "__main__":
    main()
