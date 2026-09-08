# EFI-01 protocol: command-specific contact consequences

Protocol version 1, September 8, 2026. Written before development runs and
held-out evaluation. Any development amendment must be recorded here before
the held-out run, whose artifact will include this file's SHA-256.

## Question and supplied boundary

Does acquired command-specific evidence improve prediction and useful action
when two commands produce the same body displacement but different object
effects? The current contact report's passive control can outperform the
learner, so this experiment isolates a narrower missing comparison.

Use the existing `InteractionFieldController`, five command indices,
5×5 observation, 31×31 internal map, 16×5×25 count table, and two-step
planner. Introduce an opt-in environment with a lateral contact actuator:
when an object is cardinally adjacent, either lateral directional command
holds the body still and attempts to move the object one lateral cell. The
hidden response is either aligned with the command or reversed. A wall
blocks the object. An inward command cannot push this object; other movement
and waiting retain ordinary rigid exclusion. Outside adjacency, the object
is passive. These actuator capabilities are supplied task structure, not a
learned invention of a motor primitive.

The learner receives no response label, actuator branch, contact flag,
world coordinates, or object displacement from the evaluator. It receives
the existing local observation and reliable body displacement. Object
effects must be associated from successive sightings. The controller's
generic empirical support already permits these joint effects; no new
task-specific policy or motor-support rule is proposed inside the agent.

## Exposure and environments

- **Development:** seeds 11000–11009. **Held out:** seeds 21000–21039.
  Behavioral tests use 31000 and above; resource profiling uses 41000 and
  above. No held-out results may influence development choices.
- **Source:** each seed learns each of two response laws independently.
  Two balanced repetitions of eight adjacent-wall contexts × five commands:
  80 actual transitions per acquired model. Contexts never place a wall in
  the body's cell. All lateral, inward, backward, and waiting attempts count.
  Source scenes contain no reward. Order and orientation vary with the seed.
- **Common source stream:** conditioned and command-pooled learners plus an
  empty diagnostic model receive exactly the same forced actions and sensory
  feedback. Their saved distributions are scored before updating. The
  conditioned and pooled learners retain equal empirical counts; only use of
  the command dimension differs.
- **Held-out prediction diagnostic:** fresh frozen copies of the source
  records predict both lateral commands in each of four target contexts.
  These eight matched physical interventions per law/seed are scored on a
  common stream. They do not train the target agents. Record both joint log
  loss and object log loss conditional on the identical stationary-body
  feedback; both commands must actually leave the body stationary.
- **Target:** the body starts adjacent to goal paint beneath the object.
  Either its left or right side is blocked; an extra front wall alternates.
  The first useful command clears the object, then an inward move collects
  the goal. Source covers the local wall contexts. Eight trials per layout
  and law, with four rotations and room sizes 9/11/13. A seed-specific room
  decoration is independent of the response law. Two physical steps per
  trial, +1 collection, −0.01 per step. Every failure remains in the result.
  This tests command reuse across reward placement and scene arrangement,
  not unseen local geometry or long-horizon navigation.

## Controls and accounting

| Mode | Target initialization | Target learning |
|---|---|---|
| `conditioned` | Acquired command-conditioned records | On |
| `action_blind` | Identical records, averaged across commands when queried | On |
| `shuffled` | Swap only the two acquired lateral-command rows | On |
| `empty` | Erase source records | On |
| `frozen` | Acquired records | Off; diagnostic |
| `blind_frozen` | Acquired command-pooled records | Off; diagnostic |
| `reference` | Acquired records; independent scalar continuation reduction | On; numerical diagnostic |

All modes execute the same two-step outcome budget, priors, temperature,
rule transport, motor support, and trial order. Primary controls continue
learning. Source exposure is charged equally even when its records are
erased or shuffled. Diagnostics' extra physical interventions and computation
are reported separately; none are free target training. A scalar reducer
checks the computation and does not constitute an independent field-versus-
tabular architectural comparison.

## Gates fixed before held-out evaluation

1. **Behavior:** conditioned success must exceed both `action_blind` and
   `shuffled` by at least 10 percentage points pooled over the two target
   layouts/laws, with each lower paired 95% interval above zero. Report
   each layout/law, the first target trial, complete learning curves, and
   all other controls. An unexpectedly stronger relevant control must be
   discussed rather than omitted.
2. **Prediction:** improve held-out common-stream joint log loss and
   stationary-body conditional object log loss over both pooled and shuffled
   evidence, with positive paired intervals for loss reduction. Source
   prequential curves include cold-start failures and all 80 exposures.
3. **Mechanism:** matched starting observations and both actual body
   displacements are identical across the two lateral interventions. Swapping
   acquired command bindings reverses useful action preference in the same
   scene. No hidden law enters observation or control. Frozen diagnostics
   cannot update; imagination cannot change empirical evidence.
4. **Resources:** one numerical worker, no GPU; peak agent allocation ≤32
   MiB including scratch, incremental process RSS ≤96 MiB; decision plus
   feedback/learning p95 ≤50 ms and p99 ≤100 ms on a named CPU. Meter startup,
   acquisition and target latency, rule bytes, and outcome work. Measure
   timing separately from allocation tracing and other running benchmarks.
5. **Preservation:** preserve the existing unit suite and paired serialized
   foraging (212), crossing (4,800), transfer (7,680 target + 400 source),
   and original contact (20,160 target + 9,600 common-source records)
   results. Strip only nondeterministic timing fields for contact comparison.
   Capture baseline source hashes before implementation. This is an opt-in
   world/experiment extension, not an integrated successor replacing those
   controllers.

Bootstrap 10,000 paired **seed-level** differences with RNG 29; a seed's
two laws, layouts, and repeated trials remain one cluster. Use percentile
95% intervals. Numerical thresholds are research choices, not forecasts.
The locked run has 8,960 target trials, 6,400 source transitions, and 640
separate matched prediction interventions. Report these distinct populations.

## Recording and reporting

Reuse the original HTML viewer and its existing guide/narration contract.
Record the first source repetition's open-context lateral attempts and the
first two target trials of both layouts for conditioned and action-blind
agents, for both laws on the first held-out seed. Select these before seeing
their outcomes; identify omitted source interventions and scene resets.
Command arrows indicate attempted commands, not necessarily body motion.
Record the actual command, policy, displacement, object feedback, model
version, uncertainty, and before-update score. Show evaluator truth only as
labeled narration/world display. Archive all trials and a print-quality
figure of the complete comparisons.

Passing supports command-specific embodied learning under supplied motor,
geometry, and observation structure. It does not establish general causal
discovery, contextual retention, language, or superiority over other AI
architectures. A failed gate remains a reported result.

## Development lock, before held-out evaluation

The ten development seeds completed with 90.625% conditioned success,
15.3125% command-pooled success, 0% shuffled success, and 51.25% empty
success. The acquired frozen and scalar-reference diagnostics tied the
conditioned learner. The empty control is stronger than command pooling
and must also receive a paired comparison in the report. Both original
primary comparisons remain required; no thresholds or agent parameters
were changed. Development fixed an evaluator initialization error (reset
before observation), not a controller rule.

Target evidence carries through the left-blocked trials and then the
right-blocked trials within each law/mode; spatial memory resets each trial.
“First trial” means the first left-blocked trial, before any target learning.
Each layout's complete learning curve is also reported. Source prequential
loss includes all three common-stream observers; the empty source observer
is frozen. All agents, including source observers, use the two-step budget.

The only existing executable file changed is CLI registration. Agent,
schema, field, old world, old evaluation, and viewer implementations are
unchanged. The held-out artifact records source hashes and the base Git
revision so these boundaries can be checked independently.
