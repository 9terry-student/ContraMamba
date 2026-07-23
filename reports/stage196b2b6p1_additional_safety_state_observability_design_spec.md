# Stage196-B2-B6P1 Additional Safety-State Observability Design

## Status and purpose

This specification defines a static artifact-and-source analyzer for
`STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN`.

Stage196-B2-B6P0 exhausted the authorized single-state, tail-trajectory, and
paired-delta feature families. B2-B6P1 therefore does not search another subset
of recipient, trajectory, or generic donor-recipient features. It asks a new,
precommitted question: how does a proposed selector action move the exact final
composer state relative to its native decision boundary?

The stage identifies authority only. It does not evaluate or promote a safety
gate, enumerate feature subsets, train a model, fit a threshold, load a model or
checkpoint, change training, change the classifier, change FrameGate, or change
the selector.

## Authoritative invocation

The CLI requires explicit paths:

```text
--repo-root
--stage196b2b6p0-analysis-json
--stage196b2b6-analysis-json
--stage196b2b5-analysis-json
--stage196b2b4-analysis-json
--stage196b2b3r1-analysis-json
--current-git-commit
--output-dir
```

Every path must be absolute and resolve under the supplied repository root.
Companion files are derived from the supplied analysis JSON directory; no
timestamp glob selection is permitted.

## Commit roles and closure

`a959097bd2b34302503dac19d45a8a113f6b139a` is the frozen
preimplementation authority baseline. It identifies the repository history
containing the committed Stage196 authority chain; it is not the analyzer
execution commit.

The analyzer execution commit is expected to be a descendant implementation
commit and is supplied through `--current-git-commit`. The current
implementation commit is never hardcoded.

Equality is required only between the CLI commit and actual repository `HEAD`.
The `current_commit_identity` contract records both values and fails closed
when they differ or `HEAD` cannot be resolved.

The separate `preimplementation_authority_commit_identity` contract requires
the frozen baseline to resolve as a commit object. The
`current_commit_descends_from_preimplementation_authority` contract then uses
exact Git ancestry semantics equivalent to:

```text
git merge-base --is-ancestor
    a959097bd2b34302503dac19d45a8a113f6b139a
    <repository HEAD>
```

Direct parentage is not required. Missing authority objects, unrelated or
divergent history, and indeterminate Git ancestry all block analysis.

These repository-level roles do not replace artifact-specific provenance.
Each supplied report retains and validates its own stored runtime or
implementation commit semantics where already implemented; an artifact is not
required to have been generated at the frozen baseline.

The B2-B5 companion closure deliberately excludes
`stage196b2b5_recipient_signature_rows.csv`. The analyzer neither opens nor
hashes that 407.89 MiB local-only file.

## Required source closure

The analyzer fail-closes unless all of the following hold.

### B2-B6P0

- Decision is
  `STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT`.
- Recommended next stage is
  `STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN`.
- `blocking_reasons` is empty.
- Inclusion-minimal feasible single-state gates are empty; tail and paired
  feasible-gate lists are also empty.
- Aggregate row targets are exactly 6,480: 21 MUST_ALLOW, 171 MUST_BLOCK, and
  6,288 OPTIONAL.
- Summary CSV cardinalities reproduce 49,149 single-state, 49,149
  tail-trajectory, and 3,069 paired-delta subsets per selector candidate.

P0 target labels are used only to close prior-stage provenance and counts. They
are not inspected against candidate quantities and do not select a state family.

### B2-B6

- Decision is `STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE`.
- The ordered nondominated masks are exactly `00100000000000`,
  `01000000000000`, and `10000000000000`.
- Stored decision-rule summaries assert both primary exactness and clean-dev
  unsafety.

### B2-B5

- Decision is `STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED`.
- Feature dictionary, recipient selector summary, paired-delta selector
  summary, paired-delta signature rows, and row-action sets are present.
- The large recipient-signature CSV is not a dependency.

### B2-B4

- Decision is `STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION`.
- Primitive coalition, primitive tail, primitive Möbius, residual coalition,
  residual Möbius, and localization files have their required schemas.
- Epoch-level primitive coalition authority remains exactly 20,480 rows and is
  distinct from the 1,024 tail primitive summaries.

### B2-B3-R1

- The stage has no blocking reasons.
- Native reconstruction covers 86,400 rows with prediction equality 1.0 and
  maximum final-logit error at most `1e-6`.
- Component swap rows, composer graph, native reconstruction, row swap summary,
  and group swap summary expose the required fields.

All five prior contracts must contain only passing rows with empty blocking
reasons.

## Precommitted candidate-state hierarchy

The hierarchy is fixed before any relationship to P0 targets is considered.

### Family A — native composer decision geometry

Audit the exact native class-score vector, top-1 and runner-up class, top1-minus-
runner-up margin, SUPPORT-minus-NOT_ENTITLED, SUPPORT-minus-REFUTE, and
REFUTE-minus-NOT_ENTITLED margins, plus entitlement and polarity branch scores.
A probability vector is secondary and cannot substitute for raw scores.

### Family B — candidate-action counterfactual composer geometry

For each of the three selector candidates and each exact proposed primitive
action, define the counterfactual final class-score vector and the same winner,
runner-up, and named-margin geometry. Simulation is inference-authorized because
it uses the frozen model and proposed action, not gold labels or outcomes.

### Family C — exact action-response delta

For row `x` and proposed action `a`:

```text
s_native(x) = exact native final composer class scores
s_a(x)      = exact final composer class scores after action a
Delta_a(x)  = s_a(x) - s_native(x)
```

The family includes the score delta vector, top-1 margin delta, three named
pairwise-margin deltas, prediction change, entitlement transition, polarity
transition, polarity-direction preservation, and a declared norm of the score
delta. `prediction changed` is a model-internal response and is not recovery,
harm, correctness, MUST_ALLOW, or MUST_BLOCK.

### Family D — primitive causal contribution geometry

Retain individual primitive contributions to every class score and decision
margin, coalition interaction contributions, Möbius terms, residual
contributions, dominant primitive, and dominant interaction order. Class
coordinates and interaction orders may not be collapsed into an opaque scalar.

### Family E — tail3 action-response stability

Audit sign stability, margin stability, counterfactual winner stability,
entitlement-transition stability, polarity-transition stability, and primitive-
contribution stability over epochs 18–20. Family E is diagnostic-only unless a
later stage separately authorizes it. Generic P0 tail features are not treated
as action-conditioned tail response.

## Observability statuses

Every dictionary row receives exactly one status:

```text
EXACT_EXISTING_ARTIFACT
EXACT_DETERMINISTIC_RECONSTRUCTION
SOURCE_AVAILABLE_EXPORT_MISSING
NEW_COMPOSER_INSTRUMENTATION_REQUIRED
NOT_INFERENCE_AUTHORIZED
NOT_MECHANISTICALLY_JUSTIFIED
```

Each row records family, name, formal definition, inference availability, gold
independence, action conditionality, mechanistic scope, source authority and
fields, status, integration status, missing requirement, and recommended export
location. A reconstructed prediction never establishes exact score or margin
authority.

## Static authority finding encoded by the analyzer

B2-B6/P0 broadly store native and selector predictions. B2-B4 and B2-B3-R1
store exact recipient/counterfactual logits, margins, swaps, and class-coordinate
Möbius terms for controlled primary identities. They do not materialize native
and candidate-action final score vectors for all 6,480 candidate-row
applications.

Current source contains an identifiable boundary:

- `FinalEntitlementDecisionHead` constructs a three-class score vector;
- `ContraMambaV6BMinimal.forward` assigns `base_logits`;
- composer modulation produces `final_logits`;
- the returned `logits` value is `final_logits`; and
- predictions are derived with `final_logits.argmax`.

Therefore the minimal missing authority is an action-aligned export, not a new
classifier. A future export should contain one record per seed, epoch, row,
selector candidate, and proposed primitive action with exact native and
counterfactual final score vectors. It must preserve class ordering, dtype,
checkpoint identity, action mask, and row alignment. Named margins and deltas
are deterministic derivatives of those vectors.

## Leakage authorization matrix

Potentially integration-authorized quantities are native internal scores,
candidate-action counterfactual scores, exact score and margin deltas, exact
primitive contributions, exact composer interactions, and label-free action-
response stability.

Tail3 aggregates, cross-seed recurrence, donor-recipient paired statistics, and
population-frequency features remain diagnostic-only.

The following are prohibited selector inputs: gold or stored target labels,
recovery/harm status, MUST_ALLOW/MUST_BLOCK/OPTIONAL, correctness,
false-entitlement status, stable row ID, raw identity, lexical identity,
primary-case membership, discovery membership, seed identity, and post-hoc
thresholds. Seed is permitted only for grouping, transfer audit, and provenance.

## Ordered decision hierarchy

The analyzer evaluates in this order:

1. All primary A–C state exists exactly in committed artifacts:
   `STAGE196B2B6P1_EXISTING_ACTION_RESPONSE_STATE_OBSERVABLE`, then
   `STAGE196B2B6P2_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC`.
2. Exact native and counterfactual scores are deterministically reconstructable
   but unmaterialized:
   `STAGE196B2B6P1_ACTION_RESPONSE_RECOMPOSITION_REQUIRED`, then
   `STAGE196B2B6P2_EXACT_ACTION_RESPONSE_RECOMPOSITION`.
3. Source computes the scores but full action-aligned export is missing:
   `STAGE196B2B6P1_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_REQUIRED`, then
   `STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT`.
4. No identifiable composer boundary exists:
   `STAGE196B2B6P1_COMPOSER_RESPONSE_INSTRUMENTATION_REQUIRED`, then
   `STAGE196B2B6P2_COMPOSER_RESPONSE_INSTRUMENTATION`.
5. Only prohibited outcome, identity, or gold state remains:
   `STAGE196B2B6P1_NO_INFERENCE_AUTHORIZED_SAFETY_STATE`, then
   `STAGE196B2B7_SELECTOR_INTERVENTION_RETHINK`.

Exactly one rule must be reached. The decision is derived from artifact and
source authority, not forced by a constant.

## Exact output contract

The analyzer writes exactly these nine files:

```text
stage196b2b6p1_analysis.json
stage196b2b6p1_report.md
stage196b2b6p1_source_closure.csv
stage196b2b6p1_existing_observability_inventory.csv
stage196b2b6p1_candidate_state_dictionary.csv
stage196b2b6p1_action_response_authority_matrix.csv
stage196b2b6p1_leakage_boundary.csv
stage196b2b6p1_decision_gate.csv
stage196b2b6p1_contract.csv
```

Writes are staged in a sibling temporary directory, flushed, checked for exact
nine-file closure, and atomically renamed. An existing output directory is never
overwritten. The output parent must already exist.

The contract schema is `scope, run, gate, required, observed, passed,
blocking_reason`. It separately closes CLI-to-HEAD identity, frozen authority
identity, authority-to-current ancestry, five-stage authorities, P0 decision/
zero-gate/target/enumeration facts, B2-B6 masks and summaries, B2-B5
localization, B2-B4 epoch/tail separation, B2-B3-R1 recomposition, large-file
nondependency, inventory/dictionary/status/leakage completion, decision
reachability, and output closure. Analysis provenance records
`preimplementation_authority_commit`, `current_implementation_commit`,
`repo_head`, and `authority_is_ancestor_of_current` separately in both the
analysis and source closure.

Unhandled analysis exceptions produce the same exact nine filenames with a
blocked analysis and whatever contract rows were accumulated, where output
creation itself remains possible. Exit status is 0 only when
`blocking_reasons == []`; otherwise it is 2.

## Interpretation limits

This stage authorizes only an observability design and the next authority-driven
stage. It makes no safety-separation claim, learns no threshold, evaluates no
gate, promotes no selector, and provides no external or OOD evidence. Primitive
and residual attribution currently has controlled-primary rather than full
candidate-population coverage; tail3 response remains diagnostic-only.
