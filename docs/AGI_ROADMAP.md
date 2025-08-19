Below is a single, consolidated roadmap you can execute from current codebase and research direction. It knits together (1) the AGI-by-embodiment vision, (2) the mathematical formalism you now have, and (3) the affective/membrane agent design we discussed—grounded in pragmatic deliverables, tests, and success criteria.

Vision Statement — Embodied Field Intelligence (EFI)

Thesis. EFI treats fields in a cellular automaton (CA) as the native substrate of cognition. Instead of planning in a centralized world model, an EFI agent lives inside—and acts through—diffusing, interacting fields: attractive goals (A), aversive goals (B), repulsive trails (memory), novelty drives (curiosity), and slow schema biases (learned structure). The agent simply climbs a composed potential—like weather systems steering a balloon—turning global behavior into a sum of local dynamics.

Why now. CAs offer properties modern AI increasingly wants: strict locality, massive parallelism (GPU/TPU friendly), interpretability (every decision is a gradient in a visible field), and robustness (no brittle long-range backprop or replay buffers required). Our current EFI core already shows: (i) emergent spatial memory via trail repulsion; (ii) online valence learning that flips preferences from experience; (iii) novelty/frontier seeking; and (iv) schema fields learned with Oja/BCM + slowness that inject slow, task-agnostic priors.

What’s different. EFI is not a wrapper around deep nets; it is CA-native control and learning. We compose arbitrary drives through a channel-agnostic potential (a simple, extensible API), and we learn slow structure locally (schema tiles) without centralized training loops. Every component is inspectable and ablatable—and already is in code.

Where it can go. From robust 2D navigation → 3D robot sim control → multi-agent coordination via shared fields → field-level symbol grounding and binding. If fields can carry programs (policies + memory + symbols) with only local rules, that’s a new path to scalable intelligence—parallel, interpretable, and hardware-near.

Commitment. We will “stay CA-native” as far as the substrate allows, adding non-CA modules only when experiments force us. Success looks like an open ecosystem—benchmarks, libraries, viewers, and robot demos—that others can replicate, critique, and extend.

---

## 0) Executive intent

**North star:** build an *Embodied Field Intelligence (EFI)* stack where global behavior emerges from local field dynamics, semiring composition, slow schema memory, and affective control loops. “General” comes from: (i) **composability** (channels + semiring + schema), (ii) **self-preservation & curiosity** (affect + membranes), (iii) **structure learning** (spatiotemporal schema), (iv) **invariance** (group/semigroup–aware fields), and (v) **task binding** (goals/instructions turn into target fields).

**Key doctrine:** keep the agent simple and local (PDE/CA-like), push intelligence into **field composition + memory + symmetry**, and evaluate with strong **safety and robustness** metrics—not only reward.

---

## 1) Current foundation (what you already have)

From your code and last commit:

* **Local physics:** masked diffusion with CFL-stable updates; novelty and trail dynamics; corner hazard; wall proximity field.
* **Control:** potential composition, discrete action sampling with temperature, momentum, no‑backtrack.
* **Learning:**

  * **Valence learning** (A/B and novelty), now with **counterfactual, per-step credit**.
  * **SchemaField** with Oja/BCM + slowness, **signed schema bias** (reward‑aware) and **convolutional deposition**.
* **Composition algebra:** **semiring modes** (linear, log‑sum‑exp, max‑plus) with β temperatures.
* **Exploration/frontier:** **reachability-aware** frontier via flood‑fill.
* **Temperature schedule:** trail‑aware + **flatness-aware** (∥∇P∥).
* **Evaluation:** episode/experiment runners, metrics, plots/interactive tools, tests.

This is a strong, coherent base.

---

## 2) Roadmap at a glance

Four tracks run in parallel, with **phase gates** every 4–6 weeks:

1. **Affect & Protection (Weeks 0–6)** – implement nociception, emotion, peripersonal & brain membranes; wire to control/learning.
2. **Spatiotemporal Schema & World Modeling (Weeks 4–12)** – make schema predictive and temporally aware; turn prototypes into options.
3. **Invariance & Generalization (Weeks 8–16)** – bake in group/semigroup structure and multi-scale memory; stress-test transfer.
4. **Goals, Language hooks & Multi-agent (Weeks 12–24)** – bind abstract goals to fields; add instruction hooks; reason about other agents.

Each phase has concrete **PRs**, **tests**, and **acceptance criteria** below.

---

## 3) Phase 1 — Affect & Protective Membranes (Weeks 0–6)

### 3.1 Implement (agent-internal) nociception & affect

**Why:** safer behavior, robust escapes, and stabilized learning under stress.

**Code plan**

* `efi/core/affect.py`

  * `compute_nociception(bump, neg_reward, wall_prox_here, stuck) -> float`
  * `update_affect(state, nociception, surprise, reward) -> AffectState`
  * `AffectState = {valence v_t, arousal a_t, control c_t}` (EWMA with configurable ρ)
* Extend `AgentConfig` with:

  * `affect_rho_v/a/c`, `w_pain`, `pain_to_temp_gain`, `pain_semiring_threshold`.
* Runner hook:

  * After each step, compute nociception; update affect; **add** “Pain” channel as repulsor with weight `w_pain`.
  * If `a_t` or pain exceeds threshold, **flip** composition `mode` to `maxplus` and boost temperature.

**Tests & acceptance**

* New `tests/test_affect.py`:

  * Pain increases with bump and negative B pick-up; decays otherwise.
  * Temperature rises in flat regions *and* with pain; returns fall back.
  * Under high `bump_pen`, mean bumps/100 steps ↓ ≥30% at equal or better returns.

### 3.2 Peripersonal (body) membrane

**Why:** keep safe distance, reduce wall-kissing and cul‑de‑sac ping‑pong.

**Code plan**

* `efi/core/membrane.py`

  * `peripersonal_field(known_walls, seen, y, x, R_t) -> M(x,y)`
  * `R_t = R_min + k1 * arousal + k2 * pain` (smooth, clipped).
* Plug into runner as repulsor `"Membrane"` with `w_membrane`.

**Tests & acceptance**

* Narrow-corridor suite: average distance to wall ↑; bumps/100 steps ↓; A pick-ups unchanged (±5%).

### 3.3 Brain membrane (learning gate)

**Why:** keep plasticity stable under stress; prevent runaway negative learning.

**Code plan**

* In `SchemaField` and agent’s valence updates:

  * Scale learning rates by `g = clamp(g0 - k * pain, g_min, g0)`.
* Keep a **floor** so learning never fully halts.

**Tests & acceptance**

* Under adversarial B density, variance of valence weights over episodes ↓ ≥25% with no mean-return drop.

---

## 4) Phase 2 — Spatiotemporal Schema & Predictive Fields (Weeks 4–12)

### 4.1 Temporal credit in schema

**Why:** capture motifs that precede good/bad outcomes (not just concurrent).

**Code plan**

* `SchemaConfig`: add `eligibility_tau`.
* In `SchemaField.update`: maintain per‑prototype eligibility traces `e_k ← (1−λ)e_k + y_k`.
* In `update_valence(reward)`: use `Δq_k ∝ reward * e_k` (not just last winners).

**Tests & acceptance**

* In environments where B is typically **after** a visual motif, agents learn to avoid those motifs earlier (reduction in time-to-avoid).

### 4.2 Predict-next-P / predictive novelty

**Why:** shift from reactive novelty to **prediction error** on fields.

**Code plan**

* `efi/core/predict.py`

  * Lightweight linear conv predictor for `[GA, GB, Novel, Trail]` with 1–2 step horizon.
  * Predictive novelty: `N_pred = clamp(|F_t+1 − F̂_t+1|)`, deposited like current `Novel`.
* Replace 0.5*Δscent + 0.5*d\_obs with a learned weighted mixture.

**Tests & acceptance**

* Exploration efficiency ↑ (A per 100 steps ↑) in sparse‑target maps at equal bumps.

### 4.3 Schema→Options (motor primitives)

**Why:** let useful prototypes become short-horizon skills with call/return.

**Code plan**

* Maintain simple 6–12 step *trajectory sketches* associated with high‑valence schema prototypes; call as a proposal when the prototype is active.
* Action sampler mixes: gradient-follow (base), option proposal (schema), random (temperature).

**Tests & acceptance**

* In mazes, success rate after 50 steps ↑; option usage correlates with schema activation.

---

## 5) Phase 3 — Invariance & Generalization (Weeks 8–16)

### 5.1 Group/semigroup-aware fields

**Why:** better transfer across rotations/reflections/scales; this ties to your earlier question.

**Code plan**

* `efi/core/symmetry.py`

  * Discrete D4 group (rotations/reflections) augmentations of local patches and fields.
  * **Equivariant** schema: tie or pool prototype weights across D4 transforms within a tile.
* During diffusion/composition, ensure operations are **equivariant** (they already are for rotations/reflections; verify in tests).

**Tests & acceptance**

* Train on canonical orientation; evaluate on rotated/flipped maps: return drop < 10%.

### 5.2 Multi-scale memory

**Why:** reach across larger spaces without cranking diffusion steps.

**Code plan**

* Pyramid of fields: `{1×, 2× downsampled, 4× …}` with cross‑scale diffusion every K steps.
* Schema at 2 scales: fine (tile=5), coarse (tile=10). Bias fields upsample back.

**Tests & acceptance**

* Large maps (e.g., 35×35): same compute budget, time‑to‑first‑A ↓ ≥25%.

---

## 6) Phase 4 — Goals, Language Hooks & Multi‑Agent (Weeks 12–24)

### 6.1 Goal binding

**Why:** task generality = ability to follow instructions/goals.

**Code plan**

* `efi/agents/goals.py`:

  * Map symbolic goal descriptors `{seek:A, avoid:B, go:(y,x), patrol:region}` into **target fields** and **weight presets**.
* Runner can load a goal at episode start or mid‑episode; switch weights smoothly.

**Tests & acceptance**

* Curriculum with mixed goals: success rate ≥ 85% on held‑out goal mixes.

### 6.2 Language hook (optional first pass)

**Why:** practical interface—text to fields.

**Code plan**

* Simple parser mapping a tiny DSL to `goals.py`. (Full LLM integration later.)

**Tests & acceptance**

* Unit tests for DSL→field compilation; end‑to‑end run with two chained goals.

### 6.3 Multi-agent fields (preview)

**Why:** social reasoning as field composition.

**Code plan**

* Treat other agents as moving attractor/repulsor sources with learned valence; maintain predicted position fields via short‑horizon rollouts.

**Tests & acceptance**

* In shared corridors, collision rate ↓; passing behavior emerges in ≥50% trials.

---

## 7) Cross‑cutting engineering

### 7.1 Config & module additions

* `efi/configs/affect_config.py`, `efi/configs/membrane_config.py` (or extend `AgentConfig`).
* New modules proposed above: `affect.py`, `membrane.py`, `predict.py`, `symmetry.py`, `goals.py`.

### 7.2 Metrics you’ll track on every PR

* **Safety:** bumps/100 steps, mean `WallProx` at chosen cells, mean/max pain, distance-to-wall.
* **Control/Affect:** temp vs. pain correlation, semiring flip count, recovery latency after spikes.
* **Learning:** variance of valence weights; schema q-distribution entropy; option usage rate.
* **Task:** return, A collected, B collected, steps, efficiency; transfer (rot/flip, size).

Add plots: time series for pain, arousal, temperature, semiring mode.

### 7.3 Bench suites

* **Baseline:** current random‑walls ForageWorld.
* **Diagnostics:**

  * “NarrowCorridor”, “Culdesac”, “TrapRooms” (corner density), “SparseA‑DenseB”.
  * (Optional) “HotZones” (soft hazard) and “SoothingTiles” (healing) to stress affect/membrane—no change needed for core benefits.

---

## 8) Theory mileposts (kept pragmatic)

* **Semiring safety lemma:** show that with nonnegative repulsor weights and max‑plus flips under pain, the chosen action is **Pain‑monotone** (pain at next cell ≤ ε above minimum among neighbors). Empirically verify.
* **Barrier-like guarantee:** define a membrane‑modulated **forbidden set**; with sufficiently high `w_membrane`, the policy is a discrete **control barrier** that prevents entering it (except via stochastic temperature spikes). Tune caps to keep rare.
* **Equivariance checks:** diffusion, gradient, and frontier are D4‑equivariant; schema pooling maintains equivariance; add tests that transformed worlds produce transformed decisions up to symmetry.

---

## 9) Risks & brakes (with dials)

* **Over‑caution:** w\_membrane too high → misses tight A. Dial `w_membrane↓`, `R_gain↓`.
* **Thrashing in pain:** pain→temp too strong. Cap contribution; re‑enable small momentum after pain subsides.
* **Proto over‑binding:** temporal schema may suppress exploration. Keep `eligibility_tau` modest; use arousal to open exploration.

---

## 10) Milestones & acceptance gates

**M1 (Week 2–3): Affect + body membrane online**

* Safety metrics improve ≥30% bumps reduction; returns ≥ baseline; tests green.

**M2 (Week 6): Brain membrane + stable learning**

* Weight variance ↓ ≥25% under harsh B; no performance drop.

**M3 (Week 10–12): Spatiotemporal schema + predictive novelty**

* Exploration efficiency ↑ on sparse‑A; earlier avoidance of B‑precursors.

**M4 (Week 14–16): Invariance & multi‑scale**

* Rot/flip transfer gap <10%; large‑map time‑to‑A ↓ ≥25%.

**M5 (Week 20–24): Goal binding (+ optional DSL)**

* ≥85% success on mixed goals; smooth mid‑episode goal switch.

---

## 11) Concrete PR checklist (you can start now)

1. **PR: Affect & Nociception**

   * `core/affect.py`, config knobs, runner hooks, tests.
2. **PR: Peripersonal Membrane**

   * `core/membrane.py`, compose as repulsor, tests (corridor, cul‑de‑sac).
3. **PR: Brain Membrane**

   * Learning-rate gates in agent & schema; stress tests with dense B.
4. **PR: Predictive Novelty**

   * `core/predict.py`, replace novelty mixture; A‑sparse benchmark.
5. **PR: Temporal Schema**

   * Eligibility traces; temporal credit; regression tests.
6. **PR: Symmetry & Multi‑scale**

   * D4 pooling; pyramid fields; transfer tests.
7. **PR: Goals**

   * `agents/goals.py`, simple DSL parser, e2e tests.

Each PR adds: metrics, plots, ablations, and a short “what changed in behavior” note.

---

## 12) What this gets you (in AGI terms)

* **Self‑preserving curiosity:** the agent explores vigorously when safe, pulls back when risky, and learns motifs to seek/avoid—without global planning.
* **Compositionality:** semiring + schema + goals = a compact grammar of behavior.
* **Generalization:** invariance + multi‑scale memory + options provide the scaffolding for rapid transfer.
* **Interfaces:** goals/language hooks let you *name* behaviors and stitch them into tasks.

---

### Quick starters (default knobs)

* `w_pain=0.7`, `pain_to_temp_gain=0.6`, pain flip threshold ≈ 0.6.
* Membrane: `R_min=1.0`, `R_gain = 1.0*arousal + 1.5*pain`, `w_membrane=0.6`.
* Affect EMA: `ρ_v=0.02, ρ_a=0.05, ρ_c=0.05`.
* Temporal schema: `eligibility_tau=0.9` (per-step multiplier), winners K=1–2.
* Symmetry: D4 pooling on schema; multi‑scale with 2 levels to start.

---

If you follow this sequence, you’ll get visible safety and robustness gains in the first weeks (Phase 1), then genuine *structure learning* and *transfer* by Phase 2–3, and a clean path to goal‑conditioned behavior by Phase 4—while staying faithful to the field‑based, semiring‑compositional vision that makes EFI distinctive.

