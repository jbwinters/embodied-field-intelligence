# EFI Research Roadmap

## Vision

EFI treats fields in a cellular-automaton-style substrate as the native medium of control. Instead of planning in a centralized world model, an EFI agent lives inside — and acts through — diffusing, interacting fields: attractive goals, aversive goals, repulsive trails (memory), novelty drives (curiosity), and slow schema biases (learned structure). The agent climbs a composed potential — like weather systems steering a balloon — so that global behavior emerges from a sum of local dynamics.

Why this substrate is worth exploring: strict locality, massive parallelism (every update is a stencil or pointwise op), interpretability (every decision is a gradient in a visible field), and robustness (no long-range backprop or replay buffers required). The commitment is to stay CA-native as far as the substrate allows, adding non-local machinery only when experiments force it — and to evaluate with safety and robustness metrics, not only reward.

## Current state (Phase 1 complete)

The implemented core provides:

- **Local physics:** masked, CFL-stable diffusion; novelty and trail dynamics; corner hazard; wall-proximity field.
- **Control:** channel-agnostic potential composition with semiring modes (linear, log-sum-exp, max-plus); discrete action sampling with temperature, momentum, and no-backtrack.
- **Learning:** online valence (per-channel weight) learning with per-step counterfactual credit; a `SchemaField` with Oja/BCM + slowness, reward-signed deposition, and convolutional spreading.
- **Affect & membranes (Phase 1):** nociception → affect state (valence/arousal/control/pain); a peripersonal membrane repulsor whose radius expands with arousal and pain; a "brain membrane" that gates learning rates under stress; a semiring flip to max-plus under high pain.
- **Evaluation:** episode/experiment runners, ablation suite, safety metrics (bumps, pain, wall distance), interactive viewers.

Phase 1 status, honestly stated: the mechanisms are implemented and unit-tested, but not all behavioral acceptance criteria are met yet. In particular, the full affect stack currently *costs* return relative to baseline at default dials, and the membrane's effect on wall distance is within seed noise at small sample sizes (these are tracked as expected-failure benchmarks in the test suite). Quantifying and shrinking the safety/return trade-off is active work — see the safety bench suite below.

## Phase 2 — Spatiotemporal schema & predictive fields

**Temporal credit in the schema.** Add per-prototype eligibility traces so valence updates credit motifs that *precede* outcomes, not just coincide with them. Acceptance: in environments where hazards follow a visual motif, time-to-avoid drops.

**Predictive novelty.** Replace the hand-crafted novelty mixture with prediction error from a lightweight local predictor over the fast fields (1–2 step horizon). This unifies two components (novelty and schema) under one defined objective and gives the schema a measurable job: held-out prediction accuracy. Acceptance: exploration efficiency up on sparse-target maps at equal bump rates.

**Schema → options.** Associate short trajectory sketches with high-valence prototypes and let the action sampler mix gradient-following with option proposals. The first step from reactive to anticipatory behavior without leaving the field substrate.

## Phase 3 — Invariance & multi-scale

**Group-aware fields.** Diffusion, gradients, and frontier are already D4-equivariant; add tests that transformed worlds produce transformed decisions, and pool schema prototypes across D4 transforms. Acceptance: return drop < 10% on rotated/flipped held-out maps.

**Multi-scale memory.** Field pyramids (1×, 2×, 4× downsampled) with cross-scale coupling, and schema at two tile scales. This is the intended fix for the known scaling ceiling: single-scale diffusion gradients dilute beyond ~15×15, and performance decays with grid size. Acceptance: on large maps at equal compute, time-to-first-target down ≥ 25%.

## Phase 4 — Goals, instruction hooks, multi-agent

**Goal binding.** Map symbolic goal descriptors (`seek:A`, `avoid:B`, `go:(y,x)`, `patrol:region`) to target fields and weight presets; switch goals mid-episode by smoothly re-weighting channels.

**Instruction hook.** A tiny DSL compiled to goal fields — a practical text-to-fields interface before any LLM integration.

**Multi-agent fields.** Other agents as moving attractor/repulsor sources with learned valence; shared trail fields give stigmergic coordination with zero explicit communication. Benchmarks: coverage/dispersion and cooperative foraging vs. non-sharing agents.

## Cross-cutting

**Metrics tracked on every change:**

- *Safety:* bumps per 100 steps, mean/max pain, distance-to-wall.
- *Control/affect:* temperature–pain correlation, semiring flip count, recovery latency after pain spikes.
- *Learning:* valence-weight variance, schema valence entropy, option usage.
- *Task:* return, targets collected, efficiency, transfer gaps (rotation/flip, map size).

**Bench suites.** Beyond random-walls ForageWorld: NarrowCorridor, Culdesac, TrapRooms, SparseA-DenseB — adversarial layouts where the safety mechanisms have measurable work to do. The Phase-1 behavioral criteria (bump reduction, wall-distance increase, learning-variance reduction) get re-evaluated here with enough seeds for significance tests.

**Baselines.** Random walk, greedy-toward-visible-target, an oracle shortest-path collector (upper bound), and a small learned policy on the same observation window — so every EFI number can be reported as a normalized score.

**Theory mileposts.**

- *Pain-monotone choice:* with nonnegative repulsor weights and the max-plus flip under pain, the chosen action's pain is within ε of the neighborhood minimum (proved as a lemma; verified empirically along trajectories).
- *Barrier-like guarantee:* with sufficiently high membrane weight, the policy behaves as a discrete control barrier around a forbidden set, except under stochastic temperature spikes (kept rare by caps).
- *Equivariance:* transformed worlds produce transformed decisions up to symmetry.

## Known risks and dials

- **Over-caution:** membrane weight too high sacrifices reachable targets — dial `w_membrane` and radius gains down.
- **Thrashing under pain:** pain-to-temperature gain too strong — cap the contribution, restore momentum as pain subsides.
- **Schema over-binding:** temporal credit can suppress exploration — keep eligibility decay modest, let arousal reopen exploration.
- **Safety tax:** the central open empirical question. The affect stack must earn its return cost, or the dials must be found where safety is near-free. This is measured, not assumed.
