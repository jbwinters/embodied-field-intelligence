# EFI Theory: Value as a Local Fixed Point, Safety as Algebra

*This document restates EFI's control law and its two safety results in the
linearly-solvable-MDP (LMDP) frame introduced in `efi/core/desirability.py`.
It supersedes the heuristic potential-composition story and the two sketches
in `paper/efi_paper.tex` (Lemma 1, Proposition 1), both of which get
strictly stronger here.*

## 1. The control law

Let the agent's internal state supply, per tick:

- **Beliefs** `p_c(v) = sigmoid(L_c(v))` — a log-odds Bayes filter per target
  channel `c` (`efi/core/belief.py`): diffusion is the prediction step,
  window evidence (positive *and negative*) is the correction step.
- **State costs** `q(v) ≥ 0` (reward units per step): step effort, visit
  trail, hazards, membrane shells, pain, and *aversive* targets
  (`q += |valence_c| · p_c` when `valence_c < 0`).
- **Reward injection** `R(v)`: expected reward for arriving at `v`
  (`R += valence_c · p_c` when `valence_c ≥ 0`, plus the frontier/epistemic
  bonus).

Define the desirability `z(v) = exp(V(v)/λ)`. For an LMDP with uniform
passive dynamics over the up-to-4 passable neighbors and state cost `q`,
optimal value iteration is the **local, linear** fixed-point recursion

    z(v) ← exp(−q(v)/λ) · mean_{u ∈ N(v)} z(u),     z clamped up by exp(R/λ),

which we run in log space (`V = λ log z`, one `value_sweeps` sweep =
log-sum-exp neighbor average − q, then `V ← max(V, R)`). The optimal policy
is exactly the neighbor softmax

    π*(v → u) ∝ z(u) = exp(V(u)/λ),

which is what `pick_action_from_value` samples (Gumbel-max, scale 1).

**Why log space matters (the scaling result).** `z` attenuates
exponentially with distance; `V` attenuates *linearly*:
`V(d) ≈ R − (q + λ·log deg)·d` along a corridor (the `λ·log deg` term is
the KL control cost of steering the uniform passive walk). The gradient of
`V` is therefore range-independent — empirically this took 30×30 success
from a collapse to parity with 15×15 (98.9% both, `tests/test_desirability.py`).

**Reach budget.** A reward of magnitude `r` is visible to the planner out to
`d* ≈ r / (q_step + λ·log 4)` cells. λ must be small relative to reward
scale (default λ = 0.02 → `d* ≈ 26` per unit reward). This replaces the old
"diffusion dilutes past 15×15" ceiling with an explicit, tunable budget.

## 2. One dial: λ replaces temperature, the semiring flip, and Gumbel scale

The legacy controller had three coupled mechanisms: an action temperature
schedule, a pain-triggered linear→max-plus semiring flip, and Gumbel noise.
In the LMDP frame these are the *same object*:

- λ → 0: `log-sum-exp → max` — the **max-plus/tropical semiring** is the
  zero-temperature limit of the value recursion, and the softmax policy
  becomes greedy/worst-case. The "semiring flip" is not a mode switch; it is
  the continuous λ → 0 limit.
- λ large: diffusive value, high-entropy policy — exploration.

Affect sets λ (`affect_to_lambda`, `efi/core/affect.py`):

    λ_t = clip( λ_base · (1 + k_a·arousal − k_p·pain), λ_min, λ_max )

Pain lowers λ (hurt ⇒ plan worst-case, act decisively); arousal raises it
mildly. One value of λ_t drives **both** the value sweeps and the action
softmax each tick (single source of truth, `lam_current`). Continuity in
pain removes the behavioral cliff at the old flip threshold
(`tests/test_lambda_affect.py::test_no_behavioral_cliff...`).

## 3. Pain-monotone choice (strengthened Lemma 1)

**Claim.** Let pain enter the state cost as `q_pain(v) ≥ 0` with weight
`w > 0`, and let pain drive λ → λ_min. Then the chosen neighbor satisfies

    q_pain(u_chosen) ≤ min_{u ∈ N(v)} q_pain(u) + ε(λ_min),

where `ε(λ) = λ · (G₁ − G₂ + ΔV_other)/w → 0` as λ → 0 (G's are the Gumbel
draws, ΔV_other the non-pain value differences, both bounded).

**Derivation.** The softmax samples `argmax_u [V(u)/λ + G_u]`. Write
`V(u) = V₀(u) − w·q_pain(u)` (pain separated from the rest). As λ → λ_min,
the deterministic term dominates the bounded Gumbel noise, and among
neighbors with comparable `V₀` the ordering is exactly by `−q_pain`. The
legacy version needed three ad-hoc mechanisms (nonnegative repulsor weights,
backtrack penalty, capped noise) to sketch this; here it is one limit of
one formula. ∎

## 4. Exact barrier (strengthened Proposition 1)

**Claim.** Let `F` be the forbidden set (membrane ≥ threshold). Set
`q(v) = +∞` for `v ∈ F` — implemented as `V(v) = −VBIG`, excluded from
propagation. Then for **every** λ > 0 and every tick in which the agent has
at least one non-forbidden open neighbor:

    P(agent enters F) = 0.        (exactly, not "bounded by temperature")

**Derivation.** `π*(v→u) ∝ exp(V(u)/λ)`. With `V(u) = −VBIG`, the score
`V(u)/λ + G_u` is below every finite alternative with probability 1 (Gumbel
draws are a.s. finite). The legacy proposition could only bound violation
probability by the temperature's tail mass, because the membrane was a
*subtracted potential* that a strong attractor could overwhelm; as an
infinite path cost it cannot be outbid. Verified over 500-step adversarial
walks at λ ∈ {0.005, 0.02, 0.1}: zero entries
(`tests/test_lambda_affect.py::TestExactBarrier`). ∎

**Deadlock semantics.** If *all* open neighbors are forbidden, the softmax
degrades to least-bad (still no crash); the runner logs a
`barrier_deadlocks` event per tick (`EpisodeMetrics.barrier_deadlocks`).
Barrier design should keep `F` from enclosing the agent; the counter makes
violations of that design rule observable instead of silent.

## 5. Why repulsors must be costs, not negative attractors

The legacy composition `P = Σ w⁺·attractors − Σ w⁻·repulsors` diffuses a
repulsor and then subtracts it: geometry is counted twice (once by the
repulsor's own diffusion, once by the attractor's), and a strong attractor
can always outbid a finite subtracted penalty. In the LMDP the repulsor is
a *running cost along the path the planner itself computes*: the planner
routes around it exactly when the detour is worth it, and `q = ∞` is an
unoutbiddable wall. This is also why legacy weights (`w_trail = 0.6`,
producing up to 1.8/step) had to be re-scaled as costs (`q_trail = 0.08`):
in cost units, 1.8/step prices a corridor above the total available reward
and traps the agent behind its own trail.

## 6. Fixed-point tracking (the real-time story; see Task 4)

`value_sweeps` is a contraction on the passable region (positive `q` ⇒
per-sweep gain < 1 in z-space). Beliefs move slowly (≤ one observation
window of evidence per tick), so the optimal `V*_t` moves slowly, and K
warm-started sweeps per tick track it within
`ε·γ^K/(1−γ^K)` (ε = per-tick drift of the fixed point, γ = contraction
rate). Real-time intelligence, in this architecture, *is* amortized
tracking of a slowly-moving fixed point; `z_sweeps` (κ) is the thinking
rate. Diagnostics land in `EpisodeMetrics.{mean,p95}_residual` and
`gamma_hat_median`.
