# EFI-02 protocol: returning response conditions

Version 1, September 8, 2026. Written before implementation/development runs.
Development amendments must be dated here before held-out evaluation.

## Claim and mechanism

Test whether a bounded bank of acquired response models reduces errors when
a condition returns, while retaining rapid adaptation to a new condition.
This is contextual memory over supplied categorical effects, not learned
concepts or condition-chain prediction. No response labels, switch signals,
episode countdowns, or condition sequence enter the agent.

The proposed pilot keeps three archived 16×5×25 response tables plus one
fast model. A continuous posterior weights alternatives from actual local
feedback. Slow archive updates receive posterior and prequential support;
the fast model continues updating. Predictively validated fast snapshots may
enter the bank when they differ from existing supported predictions. At
capacity, replace the least-used archive. Allocation is memory management,
not a behavior switch. Freeze the mixture during each hypothetical rollout;
information-seeking and hypothetical context-posterior updates are EFI-03.

Use the existing 5×5 observation, reliable body displacement, isolated block
channel, 31×31 map, two-pass rule transport, and bounded joint-effect field.
Only the mixed table is transported; archives remain bounded body-local
state. Queries use at most four components at the default capacity. Record
copies are correlated views of actual experience, never new observations.

## World and exposure

One continuous 13×13 room contains the body, one block, walls, and goal paint
under the block. Lateral contact commands anchor the body and move the
block according to a hidden response map; inward commands are blocked.
After collection, paint appears beneath the block again. Block and body
move at most one cell per tick. No physical or agent reset occurs at a
condition change. This extends EFI-01's supplied actuator physics.

Primary conditions are aligned lateral response, reversed response, and
outward response to both lateral commands. Randomly assign these to A/B/C
per seed. Run A–B–A–C–B with independently sampled segment lengths of
160–224 physical ticks. The agent receives no sequence model. The delayed
condition inserts three neutral (no block movement) contact responses at
each change; those outcomes do not distinguish which response will follow.
Report the whole transition, including that ambiguity, and the first eight
contacts after the neutral interval. No agent receives a neutral-interval flag.

Common prediction streams use an external, declared acquisition policy:
balanced random lateral attempts, occasional inward attempts, and local
physical detours to change approach after blockage. All detour moves and
failed attempts count. The source policy uses public local geometry and
feedback, never the hidden response or phase. Every observer receives the
same actual observations, commands, and displacement. Source observers use
one-step fields because commands are forced; behavior runs use two steps.

Behavior runs are separate uninterrupted lifetimes with each controller's
own actions, empty initial evidence, and the same scheduled physical worlds.
No common-stream records are copied into behavior agents. Count their full
acquisition, delays, and failed contacts in lifetime return/collections.

Development seeds: **51000–51009**. Held out: **61000–61039**. Behavioral
tests use 71000 and above; resource tests use 81000 and above. No development
choice may use held-out outcomes.

## Controls

- `bank`: fast adaptation plus three retained alternatives and history-based
  responsibilities; all empirical updates remain enabled.
- `fast`: existing single table, retention 0.95.
- `slow`: one table, retention 0.995.
- `no_reuse`: the bank's faster table (retention 0.5), with the same archive
  bookkeeping, but archived predictions disconnected from published output.
- `no_history`: same bank with uniform rather than history-based mixture
  weights at query time.

All receive identical observations/motor support and matching field work.
Report every control. The fast-only memory ablation separates retention from
the changed learning rate. No oracle is a primary comparison.

## Gates and statistics

1. **Retention:** at least 25% fewer wrong top-1 joint contact predictions
   over the first eight opportunities after a return, relative to `fast`,
   with positive lower paired 95% interval. Also require a positive paired
   error reduction versus `no_reuse`, and improved common-stream log loss
   versus `fast`. Score immutable predictions before each update. Exact ties
   use the declared lowest joint-effect index for all methods.
2. **Adaptation:** on the first eight contacts of each first-seen condition,
   bank error rate may exceed `no_reuse` by at most 5 percentage points and
   its mean log loss by at most 0.25 nats (upper paired 95% limits). Report
   the initial cold condition separately from the later new conditions.
3. **Behavior:** full-lifetime goal collection rate must be noninferior to
   `fast` within 2 percentage points per physical tick (lower paired limit).
   Report every mode, return, contacts, stalls, and delayed-feedback outcomes.
   Intervene on archive contents with the physical scene fixed and show a
   corresponding change in prediction and useful action preference.
4. **Mechanism/capacity:** no empirical updates during imagination, missing
   joint feedback, or frozen diagnostic calls. Retained models must be
   validated on later real outcomes. Test active mixture transport locality.
   Stress 2/3/5/8 response conditions against capacities 1/3/6, retaining all
   return windows and every eviction; this is a separate capacity experiment,
   not extra target training. No unlimited-retention claim.
5. **Resources/preservation:** default capacity ≤32 MiB peak allocation,
   ≤96 MiB incremental RSS, p95/p99 decision plus feedback/learning ≤50/100
   ms on a named CPU, one numerical worker, no GPU. Profile alone after
   benchmarks finish; include saturated rule caches and bank storage. Preserve
   existing executable defaults and tests. Hash all old agent/core/world/
   evaluation files before edits. If shared decision code changes, rerun all
   affected canonical archives; otherwise verify unchanged-source hashes and
   replay the full EFI-01 benchmark against its archived behavioral/model
   records. Distinguish old-path preservation from an integrated successor.

Primary statistics use the immediate-feedback common stream; the delayed
condition is a separately reported generalization/stress result. Bootstrap
10,000 paired seed-level resamples, RNG 29, keeping repeated windows within
their seed. Report absolute rates and relative reductions. Missing first-eight
windows are reported and fail coverage; they cannot silently disappear.
The 25% threshold is relative reduction, not percentage points.

## Deliverables

CLI experiment and resource profile, all settings/seeds and trials, model
allocation/eviction records, prequential source and behavior curves, capacity
curves, paired comparisons, and preservation record. Record the complete
first held-out bank behavior lifetime with the original viewer and existing
narration/guide contract. No success-based replay selection. Label evaluator
truth and show uncertainty, evidence, and actual feedback. Publish no broader
claim than the gates support; a failed gate remains an explicit result.

## Development amendment 1 · September 8, 2026

The initial ten development streams failed new-condition adaptation: the
bank's first-eight error excess over its equally fast ablation was 12.5–20
percentage points (paired 95% interval). Changing only mixture priors did
not fix it. We replayed the same immutable public records to test eight
hazard/prior pairs, then nine combinations of faster retention and priors.
Record replay reproduced the field observers' predictions exactly, including
top-1 ties. This is parameter development, not additional physical training.

The selected configuration uses **fast retention 0.1, hazard 0.2, fast prior
0.5**. Other settings and all acceptance margins remain unchanged. The
`no_reuse` control uses that same 0.1 retention. On the ten development
streams, returning-condition bank error was 22.5%; the new-condition error
excess interval was 0–3.125 points and loss excess 0.102–0.144 nats. The
configuration keeps a fast alternative available while preserving archives.
These deterministic-response results do not establish robustness to noisy
response laws. Final development also includes all autonomous controls and
three-contact neutral intervals before the held-out run.

Capacity uses development seeds 52000–52001 and held-out seeds
**62000–62009**, with every 2/3/5/8-condition × 1/3/6-capacity combination.
Present each condition once, then return to each in a seeded permutation;
rotate that permutation if it would repeat the immediately preceding
condition without a change. One actual source stream is recorded per
seed/load, and bounded schema observers replay its identical public records.
Capacity replays measure prediction/storage, not repeated field execution or
additional physical experience. Report incomplete return windows explicitly.
The primary experiment still uses the original five-segment sequence.

A separate archive-content intervention acquires 80 EFI-01 physical source
transitions (48 contacts), installs that acquired table into a diagnostic
archive with fixed applicability, and compares intact, command-swapped, and
erased contents in four rotations of the same test scene. It isolates the
archive-to-policy pathway. Recognition and admission must be established
by the continuous experiment and invariants; this intervention does not
establish them. None of its evidence enters a primary lifetime.

The full archive is gzip-compressed JSON; summaries remain plain JSON.
`informative_index` in records means contact number after the neutral
interval, not a guarantee that wall blockage makes that contact informative.
Resource measurement uses three continuous lifetimes, including initial
learning ticks, plus a fresh process with saturated archive/rule storage and
40 full decision/feedback ticks. Benchmarks run separately from profiling.

Independent seed lifetimes may be evaluated in parallel processes; random
state is private to each learner/world and output retains seed order. This
accelerates evaluation only. It changes neither per-agent compute budgets
nor the separately measured one-worker latency. The CLI defaults to one
process and records the requested evaluation worker count.

## Configuration lock · September 8, 2026

All ten final development seeds completed with all primary gates passing.
Immediate autonomous collection rate was 17.07% for the bank versus 16.33%
for the existing fast table; the paired difference interval was −1.54 to
+3.21 percentage points. The no-history control's strong first-seed result
did not become an aggregate advantage (16.83%); it made more contacts and
more blocked attempts. These are separate trajectories, so that observation
does not isolate a causal explanation for its individual wins.

Delayed autonomous behavior remained uncertain: bank-minus-fast interval
−3.59 to +4.30 points. As specified above, the delayed condition is a separate
stress result, not the primary retention/behavior acceptance condition.
Do not claim delayed behavioral noninferiority from this development result.
The held-out configuration and all primary margins are now locked. No
held-out outcome has been inspected. The final test suite passed 285 tests
with four expected failures, and two parallel seed streams exactly matched
serial field execution.
