# Stage196-B2-B6P7 Full Counterfactual Forward Design Specification

## Scope and frozen authority

This stage is static source analysis and design only. It does not implement a
counterfactual forward, add a stability loss, change an objective, load a
checkpoint or model, execute a model, train, or evaluate.

The required upstream decisions are:

```text
P6 = STAGE196B2B6P6_FULL_COUNTERFACTUAL_FORWARD_REQUIRED
P5 = STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED
P4 = STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE
P2 = STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE
```

All four must have empty blocking reasons. P6 must have zero failed contracts
and the exact nine-file closure supplied beside its explicit analysis path.
No timestamp glob selects an upstream artifact.

P2/P4 final-composer recomposition remains authoritative diagnostic geometry.
P6 rejected it only as an exact training-gradient path because its donor
primitives come from a separately trained frame-local-only arm. This P7 stage
does not reinterpret the numerical P2/P4 diagnostics.

The seed183 checkpoints referenced by P6 retain the provenance role
“reconstructed seed183 checkpoints.” Their joint and frame-local-only roles,
original training command, and original source commit
`fa16787efa84bb15d832b6d9382fafd77016c4e2` remain authoritative. Reconstructed
checkpoint byte identity is not historical authority.

## CLI and publication

The analyzer requires:

```text
--repo-root
--stage196b2b6p6-analysis-json
--stage196b2b6p5-analysis-json
--stage196b2b6p4-analysis-json
--stage196b2b6p2-analysis-json
--current-git-commit
--output-dir
```

The output directory must not exist. It is created once, and each artifact is
published atomically without overwriting an existing file. The return code is
zero exactly when `blocking_reasons == []`; otherwise it is two.

## Source closure and native execution trace

The analysis reads the trainer, model, every directly invoked head, shared
masking/pooling code, the frozen-encoder cache implementation, P6 probe, P5
analyzer, and the authoritative B2-B6 primitive application source.

The native path is:

```text
tokenized input_ids + attention/claim/evidence masks
  -> frozen Mamba last_hidden_state
  -> FrameGate projection, masked pooling, pair projector
  -> frame_pair_repr + frame_logit/frame_prob
  -> PredicateCoverageHead projection, pooling, pair projector
  -> predicate_pair_repr + predicate_coverage_logit/prob
  -> SufficiencyGate
  -> sufficiency_repr + sufficiency_logit/prob
  -> PolarityEnergyHead
  -> polarity_features + positive/negative energies
  -> FinalEntitlementDecisionHead explicit-product entitlement
  -> base logits in REFUTE, NOT_ENTITLED, SUPPORT order
  -> temporal/predicate/optional recipient-native final modulation
  -> final logits used by native cross-entropy
```

There is no trainable router or selector in
`ContraMambaV6BMinimal.forward`. Candidate identity and its exact row action
are discrete P2 provenance inputs, not learned model features.

The execution audit records every state’s producer, source range, symbolic
shape, producer trainability, candidate and native-loss dependencies, detach
and serialization boundaries, and replay feasibility. Diagnostic JSON values
are never reused as replay tensors.

## Candidate semantics

The values

```text
00100000000000
01000000000000
10000000000000
```

are opaque 14-bit selector feature-subset identities. They are not primitive
bitmasks. For each seed and recipient, the P2 candidate CSV resolves each
identity to exactly one row-specific five-bit action in this authoritative
order:

```text
FRAME
PREDICATE
SUFFICIENCY
POSITIVE_ENERGY
NEGATIVE_ENERGY
```

The analyzer validates every observed action against `[01]{5}`, preserves all
three exact candidate strings in provenance, and exports the observed action
keys and counts per candidate. Therefore “which primitive action” is answered
by the P2 row mapping, never by interpreting a position in the 14-bit
candidate ID.

The first candidate-specific operation is the row-wise selection between live
joint-recipient and live frame-local-only-donor primitive tensors. The first
donor value that can differ is earlier: because the two arms were separately
trained, their downstream trainable parameters can diverge starting at the
first FrameGate and PredicateCoverageHead projections after the shared Mamba
state. All downstream representations, gate values, energies, entitlement,
decision logits, and final logits may consequently differ as permitted by the
row’s selected primitive subset.

Candidate identity changes neither tokenization nor frozen Mamba inputs or
hidden states. It does not require a new backbone forward.

## Earliest exact replay boundary

The unique earliest shared tensor is:

```text
encoder_hidden_states / Mamba last_hidden_state [B,T,H_backbone]
```

This boundary is exact because:

- candidate actions occur downstream;
- the authoritative encoder is fully frozen;
- candidate IDs do not change tokens or Mamba inputs;
- the existing trainer caches Mamba hidden states under `mamba.eval()` and
  `no_grad`;
- all downstream trainable modules are rerun as live production modules;
- no detached diagnostic value is reused;
- class order remains `REFUTE, NOT_ENTITLED, SUPPORT`.

Frozen parameters alone would not be enough. Sharing is authorized here
because source establishes unchanged tokens, inputs, cached hidden values,
and upstream stochastic realization. The permitted conclusion is:

```text
SHARED_FROZEN_BACKBONE_STATE_WITH_EXACT_DOWNSTREAM_REPLAY
```

The separately trained donor means no strict suffix smaller than the complete
trainable downstream path is exact. Thus the selected design is full
trainable-path replay from the shared frozen state, not partial downstream
replay.

## Stochastic-state contract

Four downstream production modules contain training-mode dropout:

- FrameGate pair projector;
- PredicateCoverageHead pair projector;
- SufficiencyGate projector;
- PolarityEnergyHead feature projector.

The Mamba cache is produced once in evaluation mode, as in the authoritative
frozen-encoder cache. Training mode is retained for downstream heads.

For a fair native/donor comparison, save RNG state immediately before the
native downstream replay, execute it, retain the post-native state, restore
the pre-native state for the donor replay, and restore the post-native state
afterward. Corresponding dropout calls then consume matched masks while the
global RNG advances exactly once. The design must cover CPU and every active
CUDA generator.

It is invalid to run one arm in evaluation mode, silently accept independent
arm masks, or flatten native and donor rows into one ordinary dropout call.
No stochastic operation occurs after primitive selection, so the three
candidate compositions may be vectorized deterministically.

No random masking, augmentation, sampling, stochastic router, BatchNorm
running statistic, or other stateful candidate-dependent operation appears in
the audited native forward.

## Counterfactual designs

### Design A — final-composer-only recomposition

Rejected for training-gradient use by P6. It remains valid for P2/P4
diagnostic geometry. It cannot generate the separately trained donor state.

### Design B — partial downstream replay

Sharing the frozen encoder state is exact, but a partial downstream suffix is
not. Separate arm training can change parameters at the first trainable head
operations. Choosing a later cut would silently substitute a shared state
that is not shared by the real arms.

### Design C — full trainable-path replay

Selected. Execute the native joint downstream path and one complete
frame-local-only downstream path from the same cached Mamba state. Preserve
the donor hook’s exact forward-value and gradient-detach semantics. Then form
the three row-specific candidate primitive selections, call the exact
decision-head production algebra, and retain recipient-native final
modulation.

Only two downstream arm executions are required. The three candidates do not
require three donor-head or Mamba executions.

### Design D — full model forward

Semantically possible with the correct two parameter arms and explicit RNG
control, but not minimal. Repeating frozen Mamba work cannot improve gradient
exactness because it is not a gradient recipient and candidate semantics do
not alter its inputs. Four full model forwards (native plus three candidates)
are specifically unnecessary.

## Candidate batching

Arm execution remains sequential so RNG can be matched exactly. After both
arm primitive sets exist, construct explicit tensors with candidate shape
`[B,C,5]`, where `C=3`. Candidate logits remain `[B,C,3]`.

Flattening to `[B*C,...]` is allowed only across deterministic decision-head
arithmetic and must restore `[B,C,...]` before pairing or loss construction.
It must not:

- mix rows;
- erase candidate identity;
- alter native/counterfactual pairing;
- use seed or stable row identity as a feature;
- reinterpret candidate masks;
- change class order;
- change randomness or normalization semantics.

Sequential candidate composition is semantically equivalent because no
stochastic operation follows primitive selection. Vectorization is selected
only for deterministic arithmetic, not as a reason to change semantics.

## Gradient paths and independent interventions

The first implementation stage remains three separate variants:

```text
baseline
direction-consistency only
candidate-order-consistency only
```

There is no combined variant.

The frozen Mamba receives no gradient and a trainable backbone is absent.
Both intervention families can reach the live frame, predicate, sufficiency,
and polarity paths in the two arm graphs. The frame-local-only hook preserves
the donor arm’s direct FrameGate ownership while detaching all downstream
FrameGate aliases. Decision-head bias/alpha and final comparator parameters
are coordinate-conditional recipients; for example, SUPPORT-minus-REFUTE
cancels the not-entitled bias/alpha algebraically. There is no selector
recipient.

Direction consistency uses the P5-precommitted four signed response
coordinates for all exact three candidate outputs. Candidate-order consistency
uses the same candidate outputs but forms the three unordered candidate pairs
for each coordinate. The flags and objective terms must remain independently
disableable.

## Compute and memory formulas

Let:

```text
F_M, A_M = frozen Mamba forward compute and retained activation memory
F_D, A_D = one complete downstream trainable-path forward compute/memory
F_C, A_C = one deterministic candidate composition compute/memory
B_D, B_C = corresponding backward compute
```

Baseline is one native downstream graph. With an online Mamba forward, either
intervention family has:

```text
forward multiplier =
  (F_M + 2 F_D + 3 F_C) / (F_M + F_D)

activation-memory multiplier =
  (A_M + 2 A_D + 3 A_C) / (A_M + A_D)

backward multiplier =
  (2 B_D + 3 B_C) / B_D
```

With the existing frozen-state dataset cache, the per-training-batch forms
are:

```text
Mamba forwards = 0
downstream arm forwards = 2
logical candidate outputs = 3
forward multiplier = 2 + 3 F_C/F_D
activation multiplier = 2 + 3 A_C/A_D
```

Direction loss arithmetic is `O(B*3*4)`. Candidate-order loss arithmetic is
`O(B*3_pairs*4)`. They use the same model forwards under the P5 precommitment
but are not claimed to have identical reduction cost.

These are source-derived symbolic estimates, not runtime or memory benchmark
claims. Gradient accumulation divides the complete microbatch loss once,
retains no graph across optimizer steps, and adds no optimizer step. If the
dual downstream graph does not fit, reducing the physical batch and
compensating accumulation would require an explicit later-stage semantics
audit. Static evidence does not prove the path resource-unsafe.

## Teacher-state design

For both intervention families the analyzer compares:

- stop-gradient same-step native state;
- stop-gradient candidate reference;
- EMA model;
- frozen pre-tail anchor;
- previous-epoch snapshot.

No teacher is selected. Same-step native state supplies no justified stable
candidate topology. A same-step candidate reference risks circular targets.
EMA is P5’s conceptual preference but requires two additional downstream
parameter-arm states, a decay choice, an update rule, and a no-gradient exact
teacher replay. A pre-tail anchor risks seed/epoch-specific leakage and
staleness. A previous-epoch snapshot introduces an arbitrary lag and
stepwise drift.

Every future teacher must report its additional paths, storage, drift,
seed-specific information, and must ignore exact response or pair-order ties.
Ease of implementation is not teacher authority.

## Data and leakage boundary

The only main classification dataset is:

```text
data/controlled_v5_v3_without_time_swap.jsonl
```

Its frozen SHA-256 is validated and `time_swap` must have zero main-data rows.
No external or OOD data, P3 safety labels, recovery/harm categories, stable
row feature, or seed feature is permitted. Candidate strings and stable
identities may appear only as provenance/pairing keys, never model inputs.

## Decision hierarchy

The analyzer derives rather than hardcodes the decision:

1. shared frozen-backbone replay ready when a strict downstream suffix is
   exact;
2. full trainable-path replay ready when the frozen state is shared but the
   complete downstream path must be rerun;
3. full model forward ready when candidate semantics alter backbone inputs or
   hidden state;
4. resource unsafe when an exact path is established but cannot fit the
   existing resource envelope without semantic change;
5. semantics unresolved when no unique exact candidate path exists.

Contract failures override these scientific outcomes with the blocked
decision. A negative semantic or resource result is not itself a contract
failure.

The source-derived P7 outcome is expected to be:

```text
decision =
STAGE196B2B6P7_FULL_TRAINABLE_PATH_REPLAY_READY

recommended_next_stage =
STAGE196B2B6P8_FULL_TRAINABLE_PATH_REPLAY_IMPLEMENTATION
```

This follows because frozen Mamba state is exactly shareable, while the
separately trained donor requires every downstream trainable production
module to be rerun.

## Outputs and contracts

Exactly ten files are declared:

```text
stage196b2b6p7_analysis.json
stage196b2b6p7_report.md
stage196b2b6p7_execution_boundary_audit.csv
stage196b2b6p7_candidate_semantic_trace.csv
stage196b2b6p7_stochastic_state_audit.csv
stage196b2b6p7_gradient_path_design.csv
stage196b2b6p7_counterfactual_forward_designs.csv
stage196b2b6p7_compute_memory_estimate.csv
stage196b2b6p7_decision_gate.csv
stage196b2b6p7_contract.csv
```

Contracts cover commit and upstream closure, exact P6 companions, source and
class-order trace, Mamba identity and freeze policy, clean data, exact
candidate mapping, unique replay boundary, stochastic control, gradient
recipients, intervention separation, symbolic resource formulas, batching,
teachers, non-implementation, non-execution, leakage exclusions, output
closure, and decision reachability.

## Static implementation status

Implementation of this design analyzer and specification was reviewed
statically. Python, compilation, `--help`, analyzer execution, trainer/probe
execution, model/checkpoint loading, smoke tests, training, evaluation,
commit, and push were not performed.
