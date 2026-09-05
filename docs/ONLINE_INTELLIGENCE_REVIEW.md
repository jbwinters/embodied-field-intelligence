# Independent architecture review and response

The user requested a fresh **GPT-6-Astra** review at high or xhigh effort
after the initial design and three revision passes. A new reviewer ran at
**xhigh**, with no inherited conversation history. It read `AGENTS.md`, the
[design](ONLINE_INTELLIGENCE_DESIGN.md), relevant source, and the recorded
validation artifact. The review was read-only; no historical benchmarks
were rerun. This is independent model review, not external experimental
validation.

## Material reviewed

The completed third-pass draft had SHA-256
`87250ddb55a9c168a97fd22e8e5b9eee88bf9075ad3bec6dc2cc54ebfb209846`.
It followed three actual edits of the initial document. The design's final
revision table records the purpose and changes of each pass. The amendments
below were made after the independent review, so the current design differs
from that reviewed snapshot.

## Reviewer's overall assessment

The reviewer recommended a small prototype centered on learned action
consequences, with explicit evidence and resource accounting. It identified
four specification problems to resolve before implementing the full solver.
It did not consider the document evidence of increased general intelligence,
successful continual retention, or an advantage attributable specifically
to field computation.

The reviewer considered the distinction between preserving existing code
paths and preserving competence in an integrated successor particularly
sound. It confirmed that merely wrapping the current anticipatory controller
would retain the exogenous dynamics and shared-table assumptions.

## Findings and changes

| Finding | Why it matters | Response in the design |
|---|---|---|
| Latent displacement effects were described as observable feedback slots | Distinct hidden effects can produce the same feedback; new visible geometry can produce observations outside the proposed alphabet | Section 6.2 now distinguishes effect, model, and observation-group axes; specifies a deterministic restricted feedback model; and lists changes that invalidate the term bound |
| Terminal value could optimize separately under hidden maps | Correct grouping of the first two actions does not prevent clairvoyant continuation at their boundary | Section 6.1 starts with local terminating outcomes and zero continuation; later terminal integration must use a common belief or shared continuation policy |
| Fractional counts could strengthen unseen effects | Unchanged predicted means can conceal unjustified growth in confidence | Section 5.1 requires complete joint-effect observations for parameter/support updates in A; partial feedback supports belief inference and marginal scoring only |
| Stencil routes were underspecified | Two passes cover a 5×5 window with diagonal communication, but not with cardinal-only communication | Section 9.2 defines both neighborhoods, source/destination support, stage order, and a revised 28-pass reservation; requires stage-level and nonvacuous full-chain tests |
| Composition could have only one learned ingredient | Supplied navigation could explain passage through an opening after a learned push | Section 8.1 now requires independently acquired command-to-body effects and interventions on each of the two learned ingredients |
| Conditional-term counts could obscure potentially dominant terminal work | Four 9×9 sweeps per term would add 43,804,800 cell-sweeps | Section 9.3 explicitly counts this risk, uses the zero boundary for the first pilot, and requires a new measured budget for later terminal integration |
| Model ablations alone cannot establish a field-specific advantage | Learning might help equally in another small model-based implementation | Section 10.3 now requires a comparable CPU tabular model-based control before an architectural advantage claim |

The reviewer checked the allocation arithmetic: approximately **8.56 MiB**
for the cache reservation and **10.28 MiB** for the hypothesis buffers. It
did not validate CPU latency. The design continues to label memory and
latency limits as engineering targets.

## Follow-up review

The same independent reviewer checked the amendments and concluded that
the narrowed pilot substantially resolved the four original findings. It
identified one remaining information-contract issue: missing model coverage
had been defined as nonphysical, yet still appeared to enter observable
feedback groups. Section 6.2 now keeps that unresolved mass outside ordinary
groups, carries finite remaining-return bounds, and prevents an imagined
policy from treating it as an observation. The pilot scores its lower bound
without renormalizing the remaining favorable outcomes.

The reviewer's three wording corrections were also applied: body-site
anchoring and no approach bonus in the pilot; a *potential* terminal oracle
leak rather than an implemented defect; and terminal work that the term
count could obscure, rather than work the original count claimed to include.
The reviewer considered the review record otherwise accurate and recommended
no further broad design pass before the small prototype. The final bounded
changes above implement its explicit follow-up recommendations; they do not
substitute for the required executable tests.

## What remains to establish

The routing table must become an executable dataflow whose stage tests
demonstrate the claimed offsets and passes. Observation grouping must be
checked with indistinguishable hidden effects, including at the terminal
boundary. None of those contracts is established by prose alone.

Retention still depends on experimental choices for context matching,
candidate validation, and eviction. Long-horizon terminal integration,
multi-object interaction, and representation discovery remain research
problems. The first pilot deliberately postpones them.

The smallest next implementation is one isolated contact factor, one fast
categorical model plus an uncertain background, complete-effect parameter
updates, immutable pre-action scoring, two-step control with shared actions
for indistinguishable feedback, and measured local transport. The larger
memory bank and temporal composition must earn their place in later tests.
