# EFI research stories: from online experience to shared meaning

**Status: EFI-01 demonstrated; EFI-02 implemented with a recall gain and an open behavioral gate; EFI-03–EFI-15 proposed. Updated September 8, 2026.**
Prepared September 5 against the merged contact-learning baseline, commit
`104c2eb`, and updated with [EFI-01 evidence](COMMAND_CONSEQUENCES.md).
These are experiments to
earn new capabilities, rather than a forecast that every approach will work.

The aim is a more capable embodied learner whose knowledge accumulates and
changes what it can accomplish on an ordinary CPU. EFI's grid remains its
control substrate: sensed evidence, memory, predictions, and motor
preferences interact through bounded field dynamics.

This is the **active research backlog**. The
[architecture design](ONLINE_INTELLIGENCE_DESIGN.md) specifies the contact,
memory, and composition contracts; the [review](ONLINE_INTELLIGENCE_REVIEW.md)
records important constraints. The [earlier roadmap](ROADMAP.md) is historical.
Stories EFI-09 through EFI-15 extend the design toward responsive agents,
shared signaling, and grounded human language.

## Starting point

We have evidence for online motion prediction, reuse of motion across
supplied object roles, and acquisition/reuse of local contact consequences.
The contact pilot has a two-step horizon, one isolated object, a supplied
effect vocabulary, and one fast empirical table. Its longer demo illustrates
response changes, but also stalls when it loses the object.

The [contact report](INTERACTION_LEARNING.md),
[crossing report](PREDICTIVE_CONTROL.md), and
[transfer report](PREDICTIVE_TRANSFER.md) distinguish the measured results
from limitations. The controllers remain separate opt-in capabilities.
Keeping the older controllers executable has not yet shown that one new
agent retains all their competence.

## Sequence at a glance

The numbers set the default work order. Dependencies identify the capability
an experiment needs; a later story can be explored earlier when those
dependencies are met. Human word grounding does not require first evolving
a complete private language between artificial agents.

| Story | Capability to earn | Depends on | Relationship to the design |
|---|---|---|---|
| [EFI-01](#efi-01-learn-what-a-command-changes) | Learn what a command changes | Current contact pilot | Strengthen the action-conditioning evidence for A |
| [EFI-02](#efi-02-recognize-and-reuse-a-returning-context) | Recognize and reuse a returning context | EFI-01 | Retention milestone B |
| [EFI-03](#efi-03-choose-an-action-to-find-something-out) | Choose an action to find something out | EFI-02 | Informative action within B |
| [EFI-04](#efi-04-combine-two-independently-learned-effects) | Combine two independently learned effects | EFI-01, EFI-02 | Primitive composition milestone C |
| [EFI-05](#efi-05-navigate-and-interact-as-one-agent) | Navigate and interact as one agent | EFI-04 | Integrated-successor gate |
| [EFI-06](#efi-06-reuse-a-learned-sequence-as-a-skill) | Reuse a learned sequence as a skill | EFI-04, EFI-05 | Temporal compression after primitive composition |
| [EFI-07](#efi-07-learn-which-distinctions-matter) | Learn which distinctions matter | EFI-02, EFI-05 | Bounded representation learning |
| [EFI-08](#efi-08-keep-track-of-more-than-one-entity) | Keep track of more than one entity | EFI-05, EFI-07 | Extend the current identity/observation contract |
| [EFI-09](#efi-09-learn-how-another-entity-responds) | Learn how another entity responds | EFI-01, EFI-02, EFI-08 | Responsive-dynamics milestone D |
| [EFI-10](#efi-10-invent-a-useful-shared-signal) | Invent a useful shared signal | EFI-05, EFI-09 | First two-learner communication experiment |
| [EFI-11](#efi-11-share-knowledge-beyond-the-immediate-scene) | Share knowledge beyond the immediate scene | EFI-07, EFI-10 | Reference and experience transfer |
| [EFI-12](#efi-12-adapt-communication-to-a-partner) | Adapt communication to a partner | EFI-02, EFI-10, EFI-11 | Shared conventions and communication repair |
| [EFI-13](#efi-13-combine-signals-to-express-something-new) | Combine signals to express something new | EFI-04, EFI-07, EFI-11, EFI-12 | Compositional communication |
| [EFI-14](#efi-14-ground-human-words-in-shared-activity) | Ground human words in shared activity | EFI-07, EFI-08, EFI-11 | Human participation in the signaling loop |
| [EFI-15](#efi-15-understand-a-new-grounded-instruction) | Understand a new grounded instruction | EFI-06, EFI-13, EFI-14 | Limited productive human language |

The immediate batch is **EFI-01 through EFI-04**. EFI-01 now isolates
command-specific knowledge under matched body feedback without adding new
agent machinery. The original contact world's stronger passive control
remains in its report; the new result concerns a different, declared task
family. EFI-02 now adds bounded memory: recall passes, but autonomous
behavioral acceptance remains open. See its [complete result](CONTEXT_MEMORY.md). The social
and language stories become progressively more provisional.

## Execute through Task Master

The stories are loaded into the local Task Master **`master` tag as tasks
1–15**, matching EFI-01 through EFI-15. Task 1 and its three subtasks are
complete. Task 2 is in progress: protocol and implementation subtasks are
complete; behavioral acceptance remains open in 2.3. Tasks 3–15 are pending. Each has three
ordered subtasks: establish the protocol and preservation baseline;
implement the bounded mechanism and checks; evaluate and document the
result. The main tasks include their experimental acceptance criteria and
the shared advancement contract below.

```bash
task-master list --with-subtasks --tag master
task-master next
task-master show 1.1
```

Task 2 is the next ready story. Task records live in
`.taskmaster/tasks/tasks.json`, which the existing repository ignore rules
exclude from Git. This document preserves the ordered specification in the
repository; changes to execution status belong in Task Master, and claims of
demonstrated capability require linked evidence here.

## EFI-01: Learn what a command changes

**Demonstrated September 8, 2026.** The [report](COMMAND_CONSEQUENCES.md)
and [locked protocol](EFI01_PROTOCOL.md) document 87.81% conditioned success
versus 17.89% command-blind, 0% swapped, and 48.28% empty evidence on 40
held-out seeds. Matched stationary-body prediction, mechanism, resource,
and full prior-path preservation gates passed. The existing controller
earned the result; no new agent machinery was necessary. See the
[original-viewer replay](assets/interactive/command_contact.html).

**Story.** From actual attempts, the agent learns that different commands can
have different consequences even when the visible starting situation and
its own displacement do not distinguish them.

**Experiment.** Extend the small contact world with distinguishable reactions
to two commands under matched starts. Declare any added motor primitives as
supplied. Acquire balanced physical experience, then rearrange the scene.
Compare action-conditioned, action-blind, empty, and shuffled-evidence models
with the same model capacity and planning budget. Score predictions before
updating on a common experience stream.

**Acceptance.** Command conditioning improves held-out joint-effect
prediction and changes useful approach/action preferences. Target at least
a 10-percentage-point success gain over the strongest action-blind or
shuffled control, with the lower paired 95% interval above zero. Report
acquisition failures and cost. An advantage explained entirely by different
body displacement does not pass this particular test.

**Scope.** Strengthen evidence for acquired action consequences. General
causal discovery and another agent's intentions remain outside this story.

## EFI-02: Recognize and reuse a returning context

**Implemented; not yet demonstrated as a complete capability.** The first
locked held-out study reduces return prediction errors by 64.0%, passes
new-condition adaptation and resources, but misses the autonomous behavior
noninferiority gate (−3.04 to +1.72 points; required lower limit ≥−2).
[Report and next hypothesis](CONTEXT_MEMORY.md) ·
[Protocol](EFI02_PROTOCOL.md) · [Full replay](assets/interactive/context_memory.html).
The failed gate remains open; dependent capability claims stay provisional.

**Story.** After learning two different response conditions, the agent can
recover useful old knowledge when an earlier condition returns while
continuing to learn new conditions.

**Experiment.** Run irregular A–B–A–C–B streams without resets or phase
labels. Start with contexts distinguishable through local cues or physical
feedback; then delay those cues. Introduce a bounded bank of alternatives,
contextual evidence, and an explicit replacement policy. Compare the current
fast table, a slow-only model, and bounded alternatives at matched exposure.
Increase the number of contexts through and beyond storage capacity.

**Acceptance.** Preserve design gate B: at least **25% fewer wrong contact
predictions in the first eight opportunities after a return** than the
single fast model, with the lower paired improvement interval above zero.
Predeclare an allowable new-context adaptation loss and report both curves,
evictions, memory, and update work. Retention bought by refusing to adapt
does not pass.

**Scope.** Reuse starts when evidence permits disambiguation. Identical
observations cannot reveal an unannounced hidden switch before any
diagnostic feedback; that unavoidable uncertainty belongs in the results.

## EFI-03: Choose an action to find something out

**Story.** The agent sometimes takes a costly diagnostic action because its
outcome helps it choose a better subsequent action.

**Experiment.** Give two plausible contact models that require different
approaches. A probe can distinguish them. Include a distractor that stays
novel but offers no useful predictive improvement. Compare useful probing,
no information contribution, raw novelty, and matched random probes. Charge
every diagnostic move, failed contact, and delay to the task.

**Acceptance.** Improve total return over the strongest non-oracle control,
with a positive paired interval, and target at least 20% fewer costly wrong
contacts at noninferior goal success. Show that changing the diagnostic
value of a probe changes its preference. Irreducible noise should not
attract sustained expenditure.

**Scope.** Hypothetical branches may evaluate possible information, but
only actual feedback can increase empirical support. Distinct hidden models
with indistinguishable feedback must share a future policy.

## EFI-04: Combine two independently learned effects

**Story.** Knowledge acquired in two separate situations helps accomplish a
new combination that the agent has never practiced.

**Experiment.** Independently acquire a command-to-body relation and a
contact relation. Transfer only those records into unfamiliar arrangements
that require both. Compare intact composition, disconnected effect matching,
empty records, and erasure of each ingredient separately. Use the same
physical exposure and bounded planning work.

**Acceptance.** Preserve design gate C: at least a **10-point success gain**
over disconnected composition, with the lower paired interval above zero.
Erasing either acquired ingredient must hurt. A supplied navigation law
cannot count as the second learned ingredient.

**Scope.** Begin with primitive two-step composition. Each intermediate
consequence must update the next local prediction; no supplied composite
routine, hidden scene label, or clairvoyant terminal value is allowed.
Longer sequences are a separate story.

## EFI-05: Navigate and interact as one agent

**Story.** The same learner can find an object, approach it, interact, lose
contact, recover, and continue toward a goal using its accumulated evidence.

**Experiment.** Integrate spatial approach, motion prediction, and contact
consequences through a common belief and continuation calculation. Run
continuous obstructed scenes containing contact and navigation, including
the current demo's object-loss failure. Also run the new controller itself
on the existing foraging, crossing, transfer, and contact tasks.

**Acceptance.** Pass the integrated-successor margins below on every prior
task. On a separately held-out recovery family, target at least a 10-point
gain over the current contact controller's uniform movement when the object
is absent. Account for any added terminal relaxations and scratch storage.

**Scope.** One continuous field computation must support these behaviors;
a task-name switch between existing controllers does not establish
integration. Remembered and hypothetical occupancy must remain distinct,
and terminal continuation must respect the agent's actual information.

## EFI-06: Reuse a learned sequence as a skill

**Story.** A reliable behavior acquired through primitive actions becomes
a reusable temporal unit, reducing the computation needed for longer tasks.

**Experiment.** After EFI-04 works, learn a short approach/contact sequence,
then reuse it with changed goals and layouts. Represent its expected effect,
duration, uncertainty, and applicability in a bounded record. Compare a
primitive-only planner at the same work budget; use a larger-work planner
only as a labeled diagnostic reference. Interrupt sequences with surprises.

**Acceptance.** Target a 25% reduction in measured planning work at
noninferior return, success, and collision rate, plus reuse on an untrained
layout. Remove the temporal record to isolate its contribution. Include
acquisition and interruption costs.

**Scope.** Initiation, continuation, and termination should arise through
learned support and field dynamics. A manually scripted option or discrete
task sequencer cannot earn the learned-skill claim. Compression must not
hide primitive physical steps or suppress new evidence.

## EFI-07: Learn which distinctions matter

**Story.** Experience changes how the agent distinguishes situations that
look similar now but predict different consequences.

**Experiment.** Start with a small supplied set of local features and
bounded recent history. Construct perceptually ambiguous cases in which
earlier contact or motion distinguishes the correct prediction. Let a
fixed-capacity representation learn which combinations matter. Compare
current-frame-only, fixed-history, and fixed-feature alternatives at matched
capacity; a hand-labeled sufficient-state model is a diagnostic ceiling.

**Acceptance.** Improve held-out pre-action prediction and target a 10-point
behavioral gain over the strongest matched fixed representation. Test new
combinations of familiar features and intervene on the learned distinction
to show that it controls the gain. Report unavailable/ambiguous cases.

**Scope.** This establishes a bounded learned predictive distinction. The
raw sensors, feature candidates, and capacity remain supplied; it does not
establish unrestricted concept discovery or a general predictive-state
representation learner.

## EFI-08: Keep track of more than one entity

**Story.** The agent can maintain separate, uncertain expectations about
two locally observed entities and avoid attributing one entity's effects
to the other.

**Experiment.** Begin with two objects in one observation channel, short
occlusions, crossings, and reappearance. Keep a fixed number of association
alternatives. Compare observation-only and forced single-association
controls. Hidden object identities are evaluator data; they must not enter
the agent's records. Test beyond the declared tracking capacity.

**Acceptance.** Improve association-aware prediction and target a 10-point
task gain over the strongest non-oracle control. During genuinely
unresolvable crossings, preserve uncertainty and avoid inventing complete
joint evidence. Record identity mistakes, association coverage, evictions,
latency, and memory, including failures under capacity stress.

**Scope.** Replace the current one-isolated-object-per-channel assumption
for a bounded scene. General visual recognition and unlimited tracking
remain separate problems. This story prepares the information contract
needed for objects and other agents to coexist.

## EFI-09: Learn how another entity responds

**Story.** The agent learns that its approach, waiting, or contact changes
another entity's subsequent behavior, then acts using that knowledge.

**Experiment.** Start with a locally observed entity that yields, follows,
or resists under a small supplied behavior family. Compare action-responsive
prediction with exogenous motion extrapolation using matched actual
interventions. Vary the response conditions and let them recur. The other
entity's policy, target, and internal state remain hidden.

**Acceptance.** Improve common-stream consequence prediction and target a
10-point success gain over exogenous prediction on held-out arrangements.
The effect must survive controls for simple proximity and observed motion;
retain the EFI-02 recurrence benefit under the declared resource budget.

**Scope.** A scripted partner is acceptable for isolating this first
mechanism, and must be labeled as such. The result would establish learned
responsive dynamics, not inferred intentions, empathy, or a theory of mind.
Two simultaneously learning partners enter in EFI-10.

## EFI-10: Invent a useful shared signal

**Story.** Two online learners acquire a signaling convention that lets
them coordinate better than their individual observations permit.

**Experiment.** One agent sees which passage is useful while another can
operate a gate. Supply a small arbitrary signal alphabet plus silence,
limited range, and a transmission cost. Learn both emission and response;
do not assign signals meanings. Compare silence, permuted messages,
matched random signaling, and a labeled fixed-protocol reference.

**Acceptance.** Target a 10-point joint-success gain over the strongest
noncommunicating or scrambled-message control after all acquisition and
signal costs. Intervene on delivered messages while holding the physical
scene fixed: a message must change the receiver's behavior and help the
task. Correlation between messages and outcomes alone is insufficient.

**Scope.** Each agent has private memory and locally available feedback.
No shared latent vectors, centralized training, hidden partner state, or
reward-timing channel may carry the answer. Count movement, timing, and
silence if they convey information. Report both per-agent and total compute.
This earns useful coordination, not a compositional language claim.

## EFI-11: Share knowledge beyond the immediate scene

**Story.** An agent benefits from something its partner experienced without
having to repeat the same physical discovery itself.

**Experiment.** A sender encounters an object's response or a route
condition while the receiver cannot observe it. Later they communicate
locally; the receiver must use that information after the sender or
referent is absent. Vary locations and delays so a signal cannot merely
mean “copy my current move.” Compare no message, shuffled experience, and
shuffled messages at equal exposure and message budgets.

**Acceptance.** Target a 10-point gain on the receiver's first relevant
attempt, before it obtains its own diagnostic feedback, with a positive
paired interval. Swap only the sender's acquired evidence and show the
corresponding change in receiver expectation and useful action. Measure
how delay and uncertainty affect the benefit.

**Scope.** The signal becomes fallible evidence in the receiver's fields.
The environment must not copy the sender's model or provide a referent ID
to the receiver. This is a bounded test of reference and experience
transfer, not general testimony or abstract conversation.

## EFI-12: Adapt communication to a partner

**Story.** A convention can be learned by another partner, and a sender can
adjust communication when its listener missed relevant information.

**Experiment.** Use a small population with separate memories, rotating
speaker/listener roles and limited encounters. Include partners that missed
an observation or a transmission, followed by one opportunity to repair the
exchange. Compare a fixed pair, a partner-insensitive policy, and matched
extra signaling. A repeat/acknowledgment meaning must also be learned.

**Acceptance.** Target 25% fewer failed exchanges or physical teaching
trials with a new partner at noninferior total return, with a positive
paired interval. Intervene on what the listener actually observed while
holding task difficulty constant. Extra messages should be useful
specifically when information is missing, including after roles reverse.

**Scope.** Report private conventions, partner adaptation, and failures
separately. New partners receive no copied vocabulary. Limited sensitivity
to a listener's observable history is not a claim to read its mind.
Population size must not become an unreported training or resource budget.

## EFI-13: Combine signals to express something new

**Story.** Message parts learned separately can express an unfamiliar
combination that the receiver acts on correctly.

**Experiment.** Begin with two independently varied factors, such as an
object property and a destination relation. Hold out combinations, rather
than merely new random seeds. Compare a capacity-matched holistic message
table, shuffled order/components, and removal of each learned component.
Keep the signal alphabet, sequence length, and turn budget explicit.

**Acceptance.** Target a 10-point success gain on unseen combinations over
the strongest matched holistic or disconnected control, with a positive
paired interval. Intervening on one message component must selectively
change the appropriate part of the receiver's behavior. Test reversed
roles and changed layouts.

**Scope.** List any supplied feature axes or binding operations. Longer
messages, human-readable translations, and high training accuracy do not
establish compositionality. This first experiment concerns two grounded
factors, not open-ended grammar.

## EFI-14: Ground human words in shared activity

**Story.** A human word becomes associated with an embodied distinction
through shared attention, demonstration, contrast, and feedback.

**Experiment.** Introduce a small text vocabulary during object/action
interactions: for example, contrast “blue” across several objects and
“push” across several arrangements. Treat text forms as initially opaque
inputs. Separate teaching examples from held-out referents and layouts.
Compare shuffled words, co-occurrence-only learning, and identical physical
experience without linguistic cues.

**Acceptance.** Target a 10-point gain in grounded selection or action on
unseen instances over the strongest shuffled/co-occurrence control. Change
the word while holding the scene fixed and verify the appropriate change
in behavior. Report teaching interactions, corrections, retention, and
mistakes on ambiguous references.

**Scope.** No pretrained embeddings or LLM may supply meanings for this
claim. Text avoids making speech processing a prerequisite. This experiment
can start after EFI-11 if its representation dependencies are ready; an
invented agent language is an optional bridge, not a required precursor to
learning human conventions. It establishes a small grounded vocabulary.

## EFI-15: Understand a new grounded instruction

**Story.** The agent follows a new combination of familiar words by binding
them to the current scene and its acquired skills.

**Experiment.** Combine independently learned words and behaviors in a
bounded instruction family. Start with object/action combinations, then
test role-sensitive relations or two ordered actions. Hold out combinations
and scene arrangements. Compare memorized whole-instruction mappings,
shuffled order, and erasure of a required learned word or physical skill.

**Acceptance.** Target a 10-point gain over the strongest matched control
on unseen combinations, with a positive paired interval. Reversing object
roles or action order must change behavior where the instruction requires
it. Physical surprises must still update predictions and interrupt an
unsupported execution. Recheck prior embodied and communication tasks.

**Scope.** Report the supplied tokenization, grammar/binding machinery, and
learned mappings separately. A hand-written DSL-to-goal compiler is an
interface feature, not evidence of learned instruction understanding. The
result would be limited productive grounded language; unrestricted dialogue,
abstract explanations, and conventional human language in general remain open.

## Rules for advancing a story

### Evidence and experiment contract

Before implementing each story, write its short protocol: the learned versus
supplied boundary, observable inputs, world generator, dependencies, primary
metric, controls, exposure budget, and resource limits. The mechanism is a
hypothesis and may change when the experiment exposes a failure.

The numerical targets above are **proposed research choices**, not observed
results. Keep the existing design gates for retention and composition. For
new stories, finalize thresholds and any noninferiority margins before
held-out evaluation. Start with 10 development seeds and 40 disjoint
evaluation seeds, as in the design; use independently trained populations
as units for population experiments. Bootstrap paired differences over
independent seeds/populations, not correlated ticks or pairs within one
population. A percentage reduction is relative; a percentage-point gain is
an absolute difference in success rates.

Select the primary non-oracle control using development evidence, report all
specified controls, and investigate an unexpectedly stronger control. Count
source acquisition, probes, rehearsal, partner encounters, and teaching.
Principal evaluations remain online. Frozen diagnostic controls must be
labeled, and identical scenario seeds do not imply identical visited states;
use common physical experience streams for isolated prediction comparisons.

Mark a story **demonstrated** only when behavior, mechanism, resources, and
preservation checks pass. A negative result can close an experiment without
completing the capability: retain it, revise the hypothesis explicitly, and
leave dependent claims provisional. Do not grow architecture merely to move
to the next number.

### Preserve competence and stay within the substrate

- **Existing paths:** run the relevant invariant tests and compare the
  deterministic foraging, crossing, transfer, and contact references when
  shared code changes. Preserve current defaults and retain all failures.
- **Integrated successor:** before replacing any controller, evaluate the
  replacement itself on every relevant prior task. Use the design's per-task
  margins: success lower confidence bound at least −2 percentage points,
  mean return lower bound at least −0.05 in that task's units, and collision
  increase upper bound at most +1 point. These are explicit practical
  tolerances, not proof of exactly zero behavioral regression. Later stories
  add their tasks to this ledger; define margins for new metrics in advance.
- **Local evidence and control:** retain local sensing, declared transport
  passes, and fixed-capacity working state. No hidden world coordinates,
  identity/phase labels, partner internals, global search, or task-specific
  controller switch enters the decision path. Declare supplied sensor and
  motor extensions. Test the full learning-enabled information cone.
- **Honest learning:** score predictions before updating. Imagination,
  recirculated records, missing feedback, and discarded hypotheses cannot
  create new empirical support. Shared feedback requires shared contingent
  policies, including at a planning horizon's boundary.
- **Resources:** begin with the design's 31×31 internal map, 5×5 sensing,
  ≤32 MiB peak agent allocation, ≤96 MiB incremental process RSS, and
  p95/p99 decision-plus-learning targets of ≤50/100 ms on a named CPU with
  one numerical worker. Retain declared record, history, horizon, and pass
  budgets. Any extension is a new measured condition. For multiple agents,
  predeclare aggregate memory and world-step latency as well as per-agent
  limits; count all partners and any offline preparation. No GPU dependence
  is introduced as an unreported prerequisite.

Keeping cognition small is an experimental constraint throughout. Temporal
skills, selective computation, and sparse caches must earn their complexity
on quality/cost curves. Fixed memory cannot retain arbitrarily many unrelated
contexts; measure its failure boundary instead of promising no forgetting.

### Deliverables for each demonstrated story

1. A CLI experiment with complete configuration, seeds, revision, and all
   trials; do not document a runnable command before it exists.
2. A report containing learning curves, held-out comparisons, failures,
   acquisition costs, mechanism interventions, and resource measurements.
3. A regression record distinguishing unchanged legacy paths from the
   competence of the actual successor.
4. A prospectively selected replay in the **existing EFI viewer**, with
   actual fields, observations, feedback, and action probabilities. Extend
   that viewer for agents/signals; keep evaluator truth visibly separate
   from what each learner could sense. Do not build a competing player.
5. An updated status entry here linking the implementation and evidence.

## Research anchors and boundaries

These sources inform mechanisms and tests; the backlog is our proposal.
They do not establish that EFI already has the corresponding abilities.

| Direction | Reading to revisit |
|---|---|
| Retention and informative experience | [Online RL in nonstationary contexts](https://arxiv.org/abs/2302.02182); [intrinsic motivation and learning progress](https://doi.org/10.1109/TEVC.2006.890271) |
| Temporal skills and reuse | [Temporal abstraction/options](https://doi.org/10.1016/S0004-3702(99)00052-1); [successor features](https://arxiv.org/abs/1606.05312); [skills to symbols](https://cs.brown.edu/people/gdk/pubs/orig_sym_jair.pdf) |
| Representations and field architecture | [Predictive representations of state](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html); [intentional embodied field agent](https://doi.org/10.1111/cogs.13491) |
| Shared signaling | [From Grunts to Lexicons](https://arxiv.org/abs/2505.12872); [mismatches between emergent protocols and human language](https://arxiv.org/abs/2204.10590) |
| Grounding words | [WOLVES](https://doi.org/10.1037/rev0000313); [shared human–machine vocabularies](https://doi.org/10.3389/frai.2022.886349) |

Successor-feature transfer does not automatically handle changed dynamics;
options do not automatically discover skills; emergent signaling does not
automatically become English. Each transition above needs its own evidence.

Richer physical simulation or a robot body can be introduced after the
relevant integration and sensing stories, with an explicit sensor/motor
adapter and a new resource measurement. It need not wait for language and
does not require a microcontroller. Broad dialogue, social intentions, and
open-ended abstract reasoning should receive new stories only when the
grounded experiments make their prerequisites concrete.
