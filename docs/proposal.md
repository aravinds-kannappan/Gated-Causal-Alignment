# Gated Causal Alignment for Multi-Specialist Robot Control

Proposal for team discussion. Builds on the earlier "Selective Causal Alignment for Process-Compliant Manipulation" draft, the causal-observability (COO) paper and its registered-baseline pilot, the Pursuit fusion benchmark, and the team discussion thread.

## 0. Where this sits relative to the other two workstreams

The three workstreams are three different layers of one loop:

| Layer | Question | Who | State |
|---|---|---|---|
| Agent ↔ environment | When should one agent pay to refine its observation or probe before acting? | Information-acquisition workstream (COO, Lean-checked VOI gate; registered-baseline harness; now adding online causal discovery and message passing across agents and time) | Theorem + one-step gate done; sequential version and StarCraft adapter in progress |
| Agent ↔ agent (rule) | Once agents share, how should beliefs be combined? | Fusion workstream (exact fusion vs fixed-weight CAN in Pursuit and the ball-catch robot) | Exact fusion won 5/6 settings; CAN did not converge in 100 rounds at N ≥ 8; prediction RMSE identical across methods yet capture differed |
| Agent ↔ agent (selection) | Which agents' evidence should be aligned, when, for this decision, under a process constraint? | Me | This proposal |

The fusion workstream's open question ("which agents should be part of the network, and how different tasks need different networks") and the information-acquisition workstream's remark ("an apples-to-apples comparison should take individual agent confidence into account") are the same question. The claim here is that the COO criterion answers it if you swap "refine my observation" for "pull in neighbor j's belief." That turns the sheaf/CAN machinery into a module whose edges are opened only when doing so has decision value.

## 1. Hypothesis

A gate that aligns only decision-relevant evidence from specialists preserves compliant task success within a preregistered tolerance δ while cutting alignment cost (bytes, rounds, latency) relative to always-align CAN, and matches or beats exact fusion when specialists hold genuinely different causal subgraphs.

Falsifier: the claim fails if compliant completion drops by more than δ relative to always-align, or if the cost savings vanish against the never-align and simple event-triggered controls (the analog of the "reject-all" control that matched dependency admission in the registered-baseline pilot).

## 2. The gate, formally

Setting (from the robotics draft): one Fetch robot, three specialists. Base proposes navigation/positioning, arm/gripper proposes pull/reposition/release, verifier checks force limits and prohibited cabinet motion against the explicit task spec rather than proxy reward. Each specialist i holds a local belief b_i over its causal subgraph G_i ⊂ G.

Value of alignment for edge (i, j) at decision t:

    VOA_ij(t) = E[ U(a*(b_i ⊕ b_j)) ] − E[ U(a*(b_i)) ] − c_ij

where ⊕ is the fusion rule (exact fusion or CAN restriction map), c_ij the declared alignment cost. Open edge (i, j) iff VOA_ij(t) > 0. This is the COO paper's Corollary 1 (ΔV(O_f, O_c) > C_obs) with O_f = "my belief plus j's belief" and O_c = "my belief alone." Theorem 1's precondition carries over: alignment has strict value only when b_i aliases two histories with incompatible continuation-optimal actions and b_j separates them. That gives a mechanistic prediction, not just an empirical one: edges to agents whose subgraph overlaps mine completely (Ω_ij = G_i) have VOA ≤ 0 and get pruned; edges to agents with unique decision-relevant variables get opened.

Arbitration: if the verifier's constraint belief predicts a violation for the proposed action, it is revised or blocked; if evidence is insufficient (no gated edge clears VOA but ambiguity remains), a predeclared recovery (release, back off) is invoked. Model agreement alone does not certify correctness.

## 3. Formal design space

An instance is a tuple

    D = (F, T, N, G, {G_i}, Ω, c, Π_fuse, Π_gate, Φ)

| Symbol | Meaning | Values in the first study |
|---|---|---|
| F | task family (width axis) | drawer-open; drawer-open with resistance/proxy exploit; pick-and-place; ball-catch (the leg/arm/head robot, for transfer) |
| T | horizon (length axis) | decisions per episode, swept 50 / 200 / 500 |
| N | number of specialists (parts) | 2 (arm, verifier), 3 (+base), 5 (+head camera, torso) |
| G, {G_i} | task causal graph and each specialist's subgraph | given, not discovered (discovery is the information-acquisition workstream's follow-on) |
| Ω | overlap structure {G_i ∩ G_j} | low / medium / high redundancy, set by subgraph assignment |
| c | cost vector per edge (bytes, rounds, compute, latency) | declared units, as in the Pursuit message model |
| Π_fuse | fusion rule | local-only, exact fusion, fixed CAN, confidence-weighted CAN |
| Π_gate | alignment gate | never, always, event-triggered (disagreement/force spike), budget-matched random, VOA gate, oracle |
| Φ | verifier spec | force limit, cabinet displacement = 0 |

Outcomes per cell: compliant completion rate, violation rate, unnecessary stops, decision-preservation rate (fraction of decisions where the gated action equals the always-align action), regret vs oracle gate, alignment cost. Primary estimand: ΔP_compliant(VOA gate − always-align) ≥ −δ and cost ratio ≤ ρ, with δ, ρ, cost weights and sample sizes locked after a pilot.

## 4. How the success criteria map to measurements

1. Baseline comparison. Six controllers, matched on sensor access, training, low-level skills and arbitration: (a) one centralized network over all joints with a process checker; (b) isolated specialists; (c) always-align CAN; (d) exact fusion; (e) event-triggered alignment; (f) VOA gate. Oracle gate as ceiling.
2. Measuring success. Decision-level metrics above. The Pursuit result that prediction RMSE was identical while capture differed is the reason we do not use model accuracy as the primary outcome.
3. Robustness. Partial failures: drop one specialist mid-episode; stale evidence with imposed delay d ∈ {0, 1, 3} (same delays as the registered-baseline pilot, so we can reuse the bounded-delay handoff contract); corrupted alignment map; noisy force sensor. Unseen environments: held-out resistance, friction and handle geometry; transfer to ball-catch.
4. Is requesting information useful? Leave-one-out marginal value per specialist; VOA histogram per edge; gate-open rate plotted against overlap Ω. Prediction: redundant agents get pruned, agents with unique decision-relevant variables get requested. An agent that is consistently low-VOA and low-confidence is excluded, which is the diagnostic for "one agent isn't providing good information."
5. Assumptions. Specialists are honest (no strategic misreporting; adversarial agents are out of scope). Low-level skills fixed or pretrained. G and subgraph assignment given. Verifier spec correct for the declared constraints. Reward hacking is tested only in the proxy variant (reward on handle displacement even when the cabinet moves) and is caught by the verifier, not assumed away. Beliefs are linear-Gaussian approximations, with the same caveat the Pursuit benchmark noted.
6. Scalability in parts. Sweep N ∈ {2, 3, 5}. Expectation from the Pursuit data: always-align CAN cost grows with N and the ring diameter and stops converging; gate cost grows with the number of decision-relevant edges, which depends on Ω, not N.
7. Scalability in task families. Report the F × T grid as a heatmap of cost saving and ΔP_compliant per cell, so the design space is read as width (family) by length (horizon).

## 5. Small-scale experiment

Stage A, synthetic. Encode the drawer task as a POSCM in the registered-baseline substrate: applied force f → handle displacement h → drawer position d; contact state k and cabinet displacement x; proxy reward on h, true reward on d subject to x = 0. Specialist subgraphs: arm sees {f, h, k}, base sees {x, pose}, verifier sees {f-limit, x}. Deterministic, no RL. Deliverable: the first milestone from the draft, "complementary evidence changes the compliant action," as a VOA table showing the arm alone pulls, the base's evidence of cabinet motion flips the action to reposition, and the gate opens exactly on those cases. This fits the registered-baseline format and could become a small Lean lemma as a corollary to Theorem 1.

Stage B, simulated robot. A physics drawer task with a mobile base and scripted skills. Three specialists with linear-Gaussian local models, reusing the Gaussian fusion and CAN diffusion code from the Pursuit benchmark's pursuit_models.py. Six controllers × 3 seeds × held-out configs (3 resistance levels × 2 noise levels × 2 handle variants). Report the six outcome metrics and cost. Confidence-weighted CAN and a convergence-matched CAN (run to the declared threshold rather than a 100-round budget) are included so the fusion workstream's open question is answered in the same run.

## 6. Literature: what we found, why it matters, what we take

| Source | What we found | Why it matters | Method we take |
|---|---|---|---|
| D'Acunto, Di Lorenzo, Barbarossa, Networks of Causal Abstractions (arXiv 2509.25236) | CAN aligns heterogeneous causal models via sheaf restriction maps; global sections exist under spectral conditions on the connection Laplacian; diffusion converges under stated conditions | Principled way to fuse specialists that model different subgraphs; convergence depends on the network, which is exactly what a gate changes | Treat CAN as the fusion black box; gate its edges; use the Laplacian spectral gap to predict when the gated subnetwork still converges |
| Causal Observability for Active RL (COO) | VOI gate for refine/probe/coarse, Lean-checked strict-value theorem, event-sourced replay substrate | Supplies the gate criterion and the proof pattern | VOA = ΔV − c; restate Theorem 1 for inter-agent edges; emit typed events so runs are replayable |
| RealtimeGym registered-baseline pilot | Reject-all control matched dependency admission; delayed advice needs a fallback; arrival-aware paths beat reusable tables on cost | Warns that sophisticated gating can be matched by trivial controls; stale evidence must be handled explicitly | Never-align and budget-matched random as mandatory controls; delay sweep with declared fallback; preregister thresholds and freeze protocol before held-out |
| Exact fusion paper (alphaxiv 2609.17384), the Pursuit benchmark | Pooled evidence counting the common prior once; beat fixed CAN 5/6, far cheaper; CAN non-convergent at N ≥ 8; RMSE identical, capture different | Alignment cost is real and accuracy does not predict task outcome | Exact fusion as the strong cheap baseline; decision-preservation as primary metric; reuse code |
| Hu et al., Causal Policy Gradient for Whole-Body Mobile Manipulation (RSS 2023) | Factorizes action dimensions by which reward components they causally affect | Shows whole-body manipulation benefits from causal structure and gives a principled way to assign subgraphs to parts | Use their factorization to define {G_i} and Ω |
| Wang et al., Context-Aware Sparse Deep Coordination Graphs (ICLR 2022) | Learns a sparse coordination topology per state | Closest MARL work to "who should talk"; a learned-sparsity baseline | Include as the event-triggered/learned baseline; show the VOA gate is value-based, not attention-based |
| Roy & Roy, Neuro-Symbolic Hypothesis Engine (NSHE) | Auditable hypothesis traces, evidence graph that CogTrace already ingests into | Keeps our evidence packets compatible with the knowledge-integration tooling | Emit claim_evidence artifacts in the same format |
| ManiSkill mobile manipulation; PettingZoo Pursuit; SMAC/PySC2 | Available envs with mobile-manipulation, grid pursuit, and RTS multi-agent control | One gate implementation should run in all three | Env-agnostic Task interface in the shared harness |

## 7. What the overall project should be

One shared evaluation harness (aira-dojo style: Task / Solver / Runner) where a Task supplies an environment, a subgraph assignment per agent, a verifier spec and a cost model; a Solver is a (Π_gate, Π_fuse) pair; and every run emits a registered evidence packet. COO handles vertical information acquisition, the gate handles horizontal alignment selection, the fusion rules handle combination, and the same three modules run on ManiSkill, Pursuit and StarCraft. The research claim of the project is that cost-gated causal alignment scales with the number of decision-relevant edges rather than the number of agents, and that this holds across task families.

## 8. Visuals

See site/gate-episode.html: an animated episode timeline showing the gate opening when the cabinet starts to move.

## 9. Repositories

- the Pursuit benchmark (pursuit_benchmark.py, pursuit_models.py): Gaussian belief, exact fusion, fixed CAN diffusion, PPO wrapper. Reuse directly.
- The registered-baseline / open research agent repo (access pending): event-sourced substrate, MLflow→NSHE evidence path, Lean contracts, POSCM generator.
- facebookresearch/aira-dojo: Task/Solver/Runner structure for the shared harness.
- haosulab/ManiSkill (v3): OpenCabinetDrawer with Fetch.
- Farama PettingZoo (Pursuit); SMAC/SMACv2 or PySC2 for StarCraft.
- CAN authors' MIXTURE-CALSEP code: to check whether released.
- Lean 4 for the gate corollary.

---

# Part 2: Takeaways and next steps (my version)

## Takeaways

- The three threads are layers of one loop: COO decides what to observe, the gate decides whom to align with, fusion decides how to combine. Selective alignment is the missing middle, and the COO criterion already gives it a formal footing.
- The Pursuit result changes the metric. Prediction accuracy was identical across methods while task outcome differed, so decision preservation and compliant completion have to be the primary outcomes, not model error.
- Alignment cost is real. CAN's non-convergence at N ≥ 8 and its byte cost are the strongest argument for gating edges rather than diffusing over all of them.
- The registered-baseline discipline should be adopted from day one: never-align and budget-matched random controls, preregistered δ, frozen protocol before held-out configs.
- Causal discovery stays out of scope for the first milestone. The graph is given; the causal-discovery work slots in later as the source of {G_i}.
- Visualization: do both. Synthetic POSCM first (deterministic, registrable, Lean-friendly), ManiSkill second (visual, falsifiable in a physical-looking setting).

## Next steps

- Stage A: synthetic drawer POSCM, VOA table, preregistration of δ, ρ and cost weights. Draft the shared-harness interface (Task, Gate, Fusion, Evidence emitter).
- Stage B: simulated-robot run with the six controllers.
- Ask the fusion workstream: share pursuit_models.py; rerun CAN to the declared convergence threshold and add confidence weighting, since that becomes my always-align baseline.
- Ask the information-acquisition workstream: repo access; whether the VOA criterion can be stated as a corollary to Theorem 1 in Lean; have the StarCraft adapter expose per-unit subgraphs so the gate can be tested there too.
- Industry robotics contact: ask which process constraints matter in real whole-body tasks, to make the verifier spec realistic.
- Meetings: brief check-ins; pair-programming session on the harness interface once Stage A exists.
