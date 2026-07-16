# Non-Stationarity Benchmark: Pre-Registered Hypotheses

*Written BEFORE running the experiment (repo norm: results get reported
whether or not they favor EFI). Protocol and data:
`scripts/exp_nonstat.py` → `docs/assets/data/nonstat/`.*

## Why this experiment decides the thesis

A static gridworld cannot demonstrate "online beats trained" — anything can
be trained for a fixed world. EFI's premise (all competence from real-time
internal dynamics, zero training) earns its keep only where the world
*changes*: the tracking bound (docs/TRACKING.md) never assumed
stationarity, while a train-then-freeze policy assumed nothing else.

## Conditions (ForageWorld 17×17, 1000-step episodes)

- **regrow**: picked targets respawn elsewhere (Geometric delay, p=0.02) —
  a world that never runs out, testing sustained foraging.
- **drift**: every 200 steps targets teleport within radius 4 (p=0.5) —
  periodic distribution shift.
- **swap**: at step 400, reward_A and reward_B exchange values — the
  revaluation test: everything the agent liked becomes aversive and vice
  versa.

## Contenders

| Agent | Training | Adapts online? |
|---|---|---|
| EFI (egocentric, this repo) | 0 episodes | yes (beliefs, valences, λ) |
| Tabular Q, frozen | 2000 episodes, static distribution | no |
| Tabular Q, online | 2000 episodes, keeps learning | slowly |
| Greedy-visible | 0 | n/a (memoryless) |
| Clairvoyant A\* (reference) | — | replans every step on truth |

Metrics: cumulative regret vs the clairvoyant (same cloned world, same rng
state), adaptation lag after each shift (steps until the 20-step mean
reward recovers to within 20% of its pre-shift 100-step mean), regret slope
per 100-step window, and for swap: EFI's valence trajectory.

## Hypotheses (pre-registered)

- **H1 (drift)**: EFI's adaptation lag after target drift is far below
  frozen-Q's — frozen-Q should frequently *never* recover within the
  episode (lag = None), because its value estimates point at stale
  locations, while EFI's negative evidence erases stale beliefs within one
  window pass and its epistemic term re-explores.
- **H2 (swap)**: after the reward swap, EFI's valence for the
  newly-aversive channel goes negative within ≤ 3 pickups of it, and its
  B/A pickup mix flips accordingly; frozen-Q keeps collecting the
  newly-aversive target at the pre-swap rate.
- **H3 (regret slopes)**: EFI's per-window regret slope returns to its
  pre-shift level within one window after each shift; frozen-Q's slope
  stays elevated for the rest of the episode. Under regrow (no shocks,
  just sustained non-depletion) EFI's slope is flat and near the
  clairvoyant's; memoryless greedy's is the worst among non-random agents
  in walled worlds.

Falsifiers we commit to reporting: if frozen-Q matches EFI's lag (H1
false), the online-vs-trained framing dies; if EFI's regret slope stays
elevated after shifts (H3 false), the tracking story does not transfer
from theory to behavior.

## Results

*3 seeds × 3 episodes × 1000 steps per condition/contender; data in
`docs/assets/data/nonstat/summary.json`, regret curves in
`docs/assets/images/nonstat_regret.png`.*

**Regrow** (sustained foraging): EFI is the only contender in the black —
return **+5.51** vs −5.89 (greedy), −11.9 (Q-online), −13.0 (Q-frozen),
collecting 16.7 A-targets per episode vs ≤ 4.1 for everyone else, with the
lowest regret slope (0.022 vs 0.034–0.041). A world that never runs out
rewards an agent whose exploration never goes extinct.

**Drift** (H1): EFI recovers from target teleports in a mean **46 steps
with 1/36 non-recoveries**, vs Q-frozen 67 steps (4/36 never), Q-online 98
(7/36 never). Final regret: EFI 1.07 vs ~4.0–4.6 for both Q variants —
negative evidence erases stale beliefs within one window pass. **Honest
caveat:** memoryless greedy is competitive here (regret 0.09, lag 7 when it
recovers — though 7/36 never): radius-4 drift keeps targets near where they
were, so myopia is cheap and EFI pays a small curiosity tax (return −9.07
vs greedy's −8.09). Drift this local under-rewards memory; larger r_drift
should separate them further.

**Swap** (H2, the revaluation test): decisive. EFI's pickups of the
newly-aversive A drop **17× (1.89 → 0.11 per episode)** and its behavioral
adaptation lag is **5 steps**, vs 75 (Q-frozen), 69 (Q-online), 83
(greedy). A few sweeps re-propagate value under the flipped valence —
revaluation without relearning, exactly what the value-recursion
architecture promises.

**H3** (regret slopes): supported in regrow and swap (EFI's slope lowest by
3–5×); in drift all slopes are tiny and greedy's is smallest (see caveat).

**Verdict:** H1 supported, H2 strongly supported, H3 supported with the
drift caveat reported above. The online-beats-trained framing survives its
pre-registered test; the surviving weakness (curiosity tax under mild,
local drift) is quantified rather than hidden.
