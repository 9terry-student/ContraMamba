# Stage196-B2-B6P3 Action-Response Safety-Gate Diagnostic Specification

## Objective and authority

Stage196-B2-B6P3 determines whether exact, gold-independent action-response
geometry contains a mechanistically interpretable diagnostic gate that allows
every `MUST_ALLOW` row and blocks every `MUST_BLOCK` row. `OPTIONAL` rows are
descriptive only.

P2 is authoritative because its 2,160 native and 6,480 candidate-action rows
exactly reproduce the existing final composer with zero native,
counterfactual, and categorical-response disagreements. P3 is the first stage
that joins this geometry to P0 safety targets. P0 targets are evaluation labels
only and never enter a gate predicate.

The analyzer performs no training, classifier fitting, selector modification,
learned gating, production gating, model or checkpoint loading, external
evaluation, or promotion. It makes no deployability, external/OOD improvement,
formal causal-mediation, or unfrozen-Mamba claim.

## Required invocation

```text
--repo-root
--stage196b2b6p2-analysis-json
--stage196b2b6p0-analysis-json
--current-git-commit
--output-dir
```

Both analysis paths must be exact absolute paths. Their companion artifacts are
resolved only from their exact parent directories and exact filenames; no
timestamp glob or discovery fallback is used. The output directory must not
exist.

## Source closure

P2 must retain:

```text
decision = STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE
recommended_next_stage = STAGE196B2B6P3_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC
blocking_reasons = []
```

Its exact nine-file directory closure, fully passing contract, 2,160 native
rows, 6,480 candidate-action rows, three exact masks, seeds 183/184/185, and
zero reproduction disagreements are required. The analyzer loads its analysis,
native scores, candidate-action scores, response margins, reproduction
summary, and contract.

P0 must retain:

```text
decision = STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT
recommended_next_stage = STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN
blocking_reasons = []
```

Its exact nine-file closure, fully passing contract, and target counts of 21
`MUST_ALLOW`, 171 `MUST_BLOCK`, and 6,288 `OPTIONAL` rows are required. The
analyzer loads its analysis, row-safety targets, and contract.

## Exact semantic join

The join authority is:

```text
P0 candidate_feature_subset_mask == P2 candidate_mask
P0 seed                          == P2 seed
P0 stable_row_id                 == P2 stable_row_id
canonical(P0 id, source_row_id, dev_position) == P2 data_identity
```

All four components are independently validated. P0 and P2 must each have
6,480 unique keys, the join must contain 6,480 rows, and unmatched, duplicate,
and identity-disagreement counts must all be zero. Row order is never an
authority and `dev_position` is never the primary join key.

## Feature authorization

Primary mechanistic state consists of native, counterfactual, and delta signed
class margins; native and counterfactual top-1 margins and their delta; all
three class-score deltas; the four transition flags; and native and
counterfactual prediction states.

Raw native/counterfactual class scores and L1/L2/L-infinity response norms are
secondary explicit diagnostics. Raw offsets and norms cannot replace signed
class-margin geometry.

Seed, stable row, data identity, candidate mask, action key, text, gold label,
population, primary-case and transition-role metadata, safety targets and
reasons, correctness and transition outcomes, discovery membership, and
signature support are prohibited gate inputs. Seed and mask are grouping,
transfer, and provenance fields only.

## Natural predicate language

Every signed numeric field has the precommitted exact predicates `< 0`, `== 0`,
and `> 0`. Positive and negative serialized zero both map consistently to exact
numeric zero; no epsilon is introduced. Response norms have only `== 0` and
`> 0`. Boolean flags have exact false and true predicates. Prediction state has
the three named class predicates for native and counterfactual output plus
native/counterfactual equality and inequality.

The search evaluates single predicates and conjunctions of exactly two.
Two-predicate rules exclude duplicate or same-underlying-field predicates and
permit at most one raw absolute-score predicate. Canonical ordering and
human-readable formulas are deterministic. No conjunction larger than two,
disjunction, whole-expression negation, arbitrary threshold, or opaque
accumulation is evaluated.

Logically equivalent rules are deduplicated by their allow/block vector, with
the simplest canonical representative retained. A feasible conjunction is
inclusion-minimal only when neither constituent is independently feasible in
the same scope. Ranking is feasible first, then predicate count, constrained
error, block leakage, allow blocking, optional activation, and formula.

## Shared and candidate-specific semantics

A shared gate uses one identical formula for all masks; mask is not in its
predicate. Shared success requires feasibility for every candidate and every
constrained seed.

A candidate-specific gate may use a separately interpretable predeclared rule
for each mask. The mask selects which rule is audited but is not treated as a
numeric or lexical feature. Candidate-specific success requires all three
candidates to have an inclusion-minimal feasible rule across all their
constrained seeds and is never reported as shared.

For every rule:

```text
allowed = predicate is true
blocked = predicate is false
constrained_error_count = must_allow_blocked + must_block_allowed
feasible = constrained_error_count == 0
```

Seed 183, 184, and 185 feasibility is reported independently. One-seed success
cannot hide failure on another seed.

## Exact conflict audits

The categorical signature is precommitted from exact signs of authorized
margin/delta state, transition flags, native prediction, and counterfactual
prediction. Shared and candidate-specific audits count signatures containing
each constrained target and signatures containing both.

A second audit groups the exact full authorized numeric vectors using the
unrounded serialized P2 strings. It records identical numeric state carrying
both `MUST_ALLOW` and `MUST_BLOCK`. No rounding precedes duplicate analysis.

## Continuous threshold envelope

For every continuous authorized field and both orientations, the analyzer
audits one-dimensional `field <= threshold` and `field >= threshold` rules.
Feasible intervals are bounded by exact observed constrained values and include
strict midpoint boundaries between adjacent distinct values. An exact observed
closed endpoint is the canonical representative because, for these non-strict
one-dimensional orientations, it induces the same row partition as the
adjacent strict midpoint. No multi-feature threshold, threshold conjunction,
weighted score, or classifier is evaluated.

Pooled separation is reported. Every label-inspected threshold is marked
`POSTHOC_DIAGNOSTIC_ONLY` and is never integration-authorized.

Seed-184 feasible intervals are tested unchanged on seed 185, and seed-185
intervals are tested unchanged on seed 184. Bidirectional transfer requires the
same field, the same orientation, and a nonempty interval intersection. Seed
183 is reported separately and never tunes a threshold.

## Optional rows

Optional activation count and rate are reported overall, by candidate, and by
seed for every natural rule. Threshold summaries also report optional
activation for feasible representatives. `OPTIONAL` rows never determine
feasibility or constitute positive activation evidence. Conservative optional
activation is only a descriptive tie-break after constrained safety.

## Ordered decisions

The analyzer applies exactly this hierarchy:

1. shared inclusion-minimal natural boundary gate;
2. one candidate-specific inclusion-minimal natural gate for every candidate;
3. bidirectionally transferable post-hoc one-dimensional threshold signal;
4. pooled or seed-specific signal without bidirectional transfer;
5. current action-response state insufficient.

These map respectively to:

```text
STAGE196B2B6P3_SHARED_NATURAL_BOUNDARY_GATE_IDENTIFIED
STAGE196B2B6P3_CANDIDATE_SPECIFIC_NATURAL_BOUNDARY_GATES_IDENTIFIED
STAGE196B2B6P3_POSTHOC_THRESHOLD_SIGNAL_ONLY
STAGE196B2B6P3_CROSS_SEED_UNSTABLE_ACTION_RESPONSE_SIGNAL
STAGE196B2B6P3_CURRENT_ACTION_RESPONSE_STATE_INSUFFICIENT
```

Source, schema, join, numerical, or contract failure maps to
`STAGE196B2B6P3_BLOCKED_CONTRACT_FAILURE` and recommends
`STAGE196B2B6P3_REPAIR_CONTRACT`. Scientific negative results are not blockers.
The next stage follows only from the precommitted hierarchy.

## Outputs, atomicity, and contract

Exactly eleven files are written:

```text
stage196b2b6p3_analysis.json
stage196b2b6p3_report.md
stage196b2b6p3_source_closure.csv
stage196b2b6p3_joined_action_response_safety_rows.csv
stage196b2b6p3_feature_dictionary.csv
stage196b2b6p3_natural_boundary_gate_summary.csv
stage196b2b6p3_state_conflict_audit.csv
stage196b2b6p3_continuous_threshold_envelope.csv
stage196b2b6p3_cross_seed_transfer_audit.csv
stage196b2b6p3_decision_gate.csv
stage196b2b6p3_contract.csv
```

Files are staged, flushed, fsynced, checked for exact closure, and atomically
published by renaming a new sibling directory. Existing output directories are
never overwritten. Where output publication remains possible, unhandled
exceptions produce the same eleven-file blocked shape.

The contract columns are `scope`, `run`, `gate`, `required`, `observed`,
`passed`, and `blocking_reason`. It covers commit and upstream decision
identity; exact P2/P0 closures and counts; masks and seeds; unique keys and the
exact join; feature authorization and leakage exclusion; finite values and
margin/delta arithmetic; natural-predicate precommit, canonicalization, and
two-predicate complexity; conflict and threshold audits; bidirectional
transfer; hierarchy reachability; and exact output declaration.

Exit status is zero exactly when `blocking_reasons == []`, including an
ordinary `CURRENT_ACTION_RESPONSE_STATE_INSUFFICIENT` result, and two
otherwise.
