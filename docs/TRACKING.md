# Real-Time Cognition as Fixed-Point Tracking

*Companion to [THEORY.md](THEORY.md) §6. Data: `docs/assets/data/kappa_curves.json`,
plot: `docs/assets/images/kappa_curves.png`. Reproduce:
`python scripts/exp_kappa.py --episodes 20 --seeds 3 --deep-verify`.*

## The claim

The value recursion in `efi/core/desirability.py` is a contraction on the
passable region. The agent's beliefs change slowly — at most one observation
window of evidence arrives per tick — so the optimal value field `V*_t`
moves slowly, and K warm-started sweeps per tick keep the agent's field
within the steady-state bound

    |V_t − V*_t|  ≤  ε · γ^K / (1 − γ^K)

where ε is the per-window drift of the fixed point and γ the per-sweep
contraction rate. **Real-time intelligence, in this architecture, is
amortized tracking of a slowly-moving fixed point.** The thinking rate κ
(`z_sweeps`, sweeps per world tick) is the dial.

## Measured: behavior vs thinking rate (κ)

20 episodes × 3 seeds per cell, `max_steps = 0.9·H·W`, schema off:

| κ | 15×15 success | 30×30 success |
|---|---|---|
| 0 | 5.0% | 8.3% |
| 1 | 96.7% | 100% |
| 2 | 95.0% | 100% |
| 3 | 98.3% | 96.7% |
| 5 | 98.3% | 98.3% |
| 8 | 98.3% | 98.3% |

Two honest readings:

1. **Thinking is load-bearing.** κ=0 (act on the stale field) collapses to
   near-floor. One sweep per tick already recovers the ceiling.
2. **This task saturates at κ=1.** After the initial orientation sweeps
   (H+W on episode start), warm-started tracking is good enough that extra
   per-tick sweeps buy nothing ForageWorld can measure — the curve is a step,
   not a slope. Monotone-up-to-saturation holds trivially; resolving κ
   beyond 1 needs harder worlds (larger maps with κ-scaled init budgets,
   non-stationary targets — see the non-stationarity benchmark).

The *residuals* meanwhile fall smoothly with κ exactly as contraction
predicts (15×15 mean final residual: 0.157 at κ=1 → 0.017 at κ=8), so the
instrument sees what the task cannot.

## Measured: the tracking bound

Instrumented episode (15×15, κ=3, snapshot every 10 ticks, `V*` from 200
extra sweeps on the frozen inputs):

- contraction rate γ̂ (median per-sweep residual ratio): **0.750**
- fixed-point drift ε (median, per 10-tick window): **0.082**
- steady-state bound ε·γ^K/(1−γ^K): **0.060**
- median measured tracking error: **0.0039** (≈15× inside the bound)
- **quiescent windows within bound: 100%** (11/19)

The other 8 windows are **transients**: a pickup or a new wall/target
discovery *jumps* the fixed point discontinuously (drift > 3× median), and
a steady-state bound does not apply during a jump — the field re-converges
at rate γ afterward (that re-convergence latency is the light-cone
experiment's subject). Claiming the bound over jump windows would be wrong;
excluding them is not a dodge but the bound's actual scope, and both counts
are reported.

## What follows

- **κ is a real cognitive dial with a measurable floor** (κ=0 fails) and a
  task-dependent saturation point. "More thought per tick" has a defined,
  falsifiable meaning.
- **The bound is predictive where it applies.** Between surprises, the
  agent's value field provably shadows the optimum it would compute with
  unlimited time.
- **Transients are where the interesting dynamics live**: reaction latency
  after a surprise is distance/(c·κ) — measured below.

## Measured: the speed of thought

One-tick belief injection at distance d, stationary agent, exact policy
distributions (deterministic — no sampling); latency = think-ticks until
KL(policy ‖ baseline) > 0.1 nats
(`scripts/exp_lightcone.py`, data `docs/assets/data/lightcone.json`):

| | d=5 | d=10 | d=15 | d=20 |
|---|---|---|---|---|
| κ=1 | 5 | 10 | 15 | 20 |
| κ=3 | 2 | 4 | 5 | 7 |
| κ=5 | 1 | 2 | 3 | 4 |

Fit: **latency = 0.99·(d/κ) + 0.17, R² = 0.999** — an internal speed of
thought of **1.01 cells per sweep**, exactly the radius-1 stencil's
theoretical light cone. At κ=1 the latency literally equals the distance.

This is the falsifiable signature of field intelligence: no other
architecture class produces a reaction time that is a *ruler* for stimulus
distance, and any non-local shortcut sneaking into the internals would
flatten this line (it is enforced as a regression test,
`tests/test_lightcone.py`).
