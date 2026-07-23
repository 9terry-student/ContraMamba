# Stage196-B2-B6P4 Action-Response Stability-State Design Specification

## Objective

Stage196-B2-B6P4 reconstructs exact action-conditioned final-composer response
trajectories at epochs 18, 19, and 20. It is a label-blind mechanism and state
design stage. It does not load P0 safety targets, search a safety gate or
threshold, train a model, load a checkpoint, change the selector, or make an
integration or safety decision.

P3 established that no natural-boundary gate was feasible and no
one-dimensional threshold transferred bidirectionally across seeds. P4
therefore distinguishes absolute endpoint shift from within-tail trajectory
instability, action-response sign or winner-topology changes, and instability
in the relative ordering of the three frozen candidate actions.

## Required invocation

```text
--repo-root
--stage196b2b6p3-analysis-json
--stage196b2b6p2-analysis-json
--stage196b2b6-analysis-json
--stage196b2b5-analysis-json
--stage196b2b4-analysis-json
--stage196b2b3p0-run-root
--stage196b2b3p0-runtime-git-commit
--stage196b2b5-recipient-signature-rows-csv
--current-git-commit
--output-dir
```

Every source path is explicit. The external B2-B5 recipient-signature CSV is
used only from its supplied path. Its actual SHA256 is always computed, but an
upstream digest is byte-level authority only when it normalizes to exactly 64
lowercase hexadecimal characters. Timestamp-based artifact discovery is
prohibited. The output directory must not already exist.

## Frozen authorities

P3 must retain decision
`STAGE196B2B6P3_CROSS_SEED_UNSTABLE_ACTION_RESPONSE_SIGNAL`, recommendation
`STAGE196B2B6P4_ACTION_RESPONSE_STABILITY_STATE_DESIGN`, no blockers, zero
feasible natural gates, and zero successful bidirectional threshold transfers.
P4 reads the P3 analysis, relevant audit summaries, and contract for closure.
It never reads P3 joined safety rows or safety-target fields.

P2 must retain decision
`STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE`, no
blockers, 2,160 native rows, 6,480 candidate-action rows, and zero native,
counterfactual, and categorical-response disagreements. Its epoch-20 score,
margin, prediction, and response CSVs are exact endpoint reproduction
authority.

B2-B6 must retain decision `STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE` and
the exact masks `00100000000000`, `01000000000000`, and
`10000000000000`. Its clean-dev signature audit is the deterministic
authority for one singleton primitive action at each of the 6,480 candidate,
seed, and data-identity applications. Mask bits are never reinterpreted.

P2 independently reproduces that full-population mapping through
`candidate_mask`, `seed`, `data_identity`, `candidate_action_key`, and
`stable_row_id`. P4 requires 6,480 B2-B6 rows, 6,480 P2 rows, 6,480 exact
matches, zero action disagreements, and zero stable-row disagreements.

B2-B5 must retain decision
`STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED`. The explicit
recipient-signature CSV supplies contextual acceptable-action sets and
primitive state semantics without contributing outcome, recovery, harm,
correctness, or safety-target state. It is not the deterministic action
authority for the complete clean-dev population.

## B2-B5 signature-evaluation contexts

B2-B5 contains both pooled signature rows and transfer-specific signature
rows. A `signature_action_intersection` is an acceptable-action set for its
exact evaluation context. Candidate mask plus canonical signature alone is
therefore not a complete key. P4 keys the authority by
`feature_subset_mask`, canonical signature, `transfer_status`, normalized
`source_seed`, and normalized `target_seed`.

`POOLED` requires null source and target seeds. Transfer rows require one of
the exact directions 184-to-185 or 185-to-184 and the frozen B2-B5 status
vocabulary `UNSEEN`, `COMPATIBLE`, or `INCOMPATIBLE`; the row seed must equal
the target seed. Within one qualified context, repeated rows must expose one
identical canonical intersection. Every intersection must be a list, duplicate
serialized actions are rejected, and every action must match `[01]{5}`. Empty
intersections remain valid infeasibility evidence but never assign an action.

Differences among pooled and transfer-specific intersections for the same
unqualified mask/signature are expected contextual variation. P4 counts and
reports them without treating a positive count as corruption. Transfer rows
remain diagnostic and are never used to assign full-population actions.

Only nonempty pooled `recipient_local` intersections support membership
checking. The frozen B2-B5 primary selector population contains seeds 184 and
185, 16 identities, and three masks, so P4 requires exactly 48 matched B2-B6
candidate rows and zero assigned-action membership disagreements. Seed 183 and
nondiscovery clean-dev rows do not require B2-B5 membership authority.

## Recipient-signature authority resolution

P4 resolves byte-level authority in this exact order:

1. the consumed-artifact record in P2 analysis,
   `stage196b2b6p2_source_closure.csv`, and the P2 contract;
2. a valid nonempty B2-B5 analysis or companion-closure digest;
3. exact fail-closed semantic closure when neither upstream stage records a
   valid digest.

P2 is preferred because it consumed the explicit external artifact while
materializing the exact 2,160 native and 6,480 candidate-action endpoint rows
and completed composer reproduction. P4 validates every available P2
recipient-artifact datum independently: semantic filename, normalized digest,
row count, source role, and required/loaded external-artifact status.
Machine-specific Windows and Kaggle root prefixes are not required to match.

A digest normalizer accepts only a string which, after surrounding whitespace
is stripped and uppercase hexadecimal is normalized to lowercase, has exactly
64 characters all in `[0-9a-f]`. Empty strings, null, NaN, `None`, `unknown`,
malformed text, and non-string values are `HASH_AUTHORITY_UNAVAILABLE`. They
are never compared with the computed artifact hash. `HASH_AUTHORITY_MISMATCH`
exists only when both expected and actual values are valid SHA256 digests and
differ.

The actual explicit-file SHA256 is computed and recorded in every successful
run regardless of authority mode. When a valid P2 digest exists, authority mode
is `P2_BYTE_HASH`. Otherwise a valid B2-B5 digest selects
`B2B5_BYTE_HASH`. When neither exists, `SEMANTIC_CLOSURE` is used and
`expected_sha256` is null rather than an empty string.

Semantic fallback is not permissive. It requires the explicit path to exist as
a regular file, a valid computed SHA256, exactly 524,256 rows, the complete
schema, seeds 183/184/185, unique semantic row/action keys, exact six-run and
candidate provenance, context-qualified B2-B5 action-set consistency, exact
48-row pooled-primary membership closure, exact reproduction of
P2's 6,480 candidate-action mapping, and exclusion of outcome or safety fields
from reconstruction inputs. The same schema, population, key, and mapping
checks remain mandatory when byte-level authority is available.

The `explicit_recipient_signature_authority` contract records the selected
authority mode and source, resolved explicit path, actual and expected digests,
expected-digest availability, byte-match result, row count, and
semantic-closure result. The analysis records the same facts under
`recipient_signature_authority`.

B2-B4 must retain decision
`STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION`. Its controlled coalition
coordinates provide exact semantic and numeric cross-checks at tail epochs.

The B2-B3P0 runtime commit must be exactly
`fa16787efa84bb15d832b6d9382fafd77016c4e2`. The run root must contain the six
named seed/mode runs, 120 exact trajectory sidecars, 120 exact composer
sidecars, and six manifests. Runs and sidecars are never regenerated.

## Exact composer reconstruction

P4 imports and reuses the deterministic source boundary already used by P2:
the P2 semantic score-coordinate mapping and the frozen B2-B6 `reconstruct`
and `apply_mask` functions. It does not implement a second composer formula,
infer scores from predictions, or infer scores from rounded summaries.

The internal coordinate order is:

```text
REFUTE
NOT_ENTITLED
SUPPORT
```

Semantic score names are mapped explicitly. Native coordinates retain the
source float32 serialization. Counterfactual application retains binary64
arithmetic. CSV numeric values use 17 significant digits and must be finite.

For every epoch, row, and candidate:

```text
delta_score_c = counterfactual_score_c - native_score_c
delta_margin = counterfactual_margin - native_margin
```

Named margins are SUPPORT-minus-NOT_ENTITLED, SUPPORT-minus-REFUTE,
REFUTE-minus-NOT_ENTITLED, and the state-dependent top1-minus-runner-up
margin. Exact margin and delta arithmetic must close.

At epoch 20, every native score, counterfactual score, signed response margin,
native/counterfactual prediction, and categorical transition must reproduce
P2 exactly. Any disagreement blocks P4.

## Tail population and identity

Tail epochs are exactly 18, 19, and 20. Required closure is:

```text
3 seeds x 720 rows x 3 epochs = 6,480 native epoch rows
3 seeds x 720 rows x 3 candidates x 3 epochs = 19,440 candidate epoch rows
6,480 unique seed-row-candidate trajectories
3 exact observations per trajectory
```

The identity join uses the structured triple `(id, source_row_id,
dev_position)`, serialized canonically as `data_identity`. Row order is never
an authority. Every seed/epoch must contain the same 720-identity set; every
candidate must be present exactly once for every seed, identity, and tail
epoch. Seed, epoch, stable row ID, and data identity are
`PROVENANCE_ONLY_NOT_FEATURE_AUTHORIZED`.

## Precommitted state hierarchy

### Family A — absolute tail response

For each signed score or margin response, P4 records epoch-18, epoch-19, and
epoch-20 values, minimum, maximum, range, mean, both adjacent differences, and
epoch-20 minus tail mean. Family A is diagnostic-only because P3 rejected
cross-seed stability of absolute endpoint thresholds.

### Family B — trajectory topology

For each signed response, P4 records exact-zero signs, sign persistence,
zero-crossing count, sign-reversal count, monotonic direction, and final-step
direction. No epsilon is used.

The monotonic categories are defined exactly:

```text
STRICTLY_INCREASING: x18 < x19 < x20
NONDECREASING:       x18 <= x19 <= x20 and not strictly increasing
STRICTLY_DECREASING: x18 > x19 > x20
NONINCREASING:       x18 >= x19 >= x20 and not strictly decreasing
CONSTANT:            x18 == x19 == x20
NON_MONOTONIC:       none of the above
```

Final-step direction is INCREASING, DECREASING, or CONSTANT from epoch 19 to
20. A sign reversal counts changes between consecutive nonzero signs; exact
zero is not silently assigned a polarity. A zero crossing counts every
adjacent exact sign-state change.

### Family C — categorical action-response persistence

For every trajectory, P4 records native and counterfactual prediction
sequences, counterfactual-winner persistence, prediction-changed sequence and
persistence, entitlement-transition sequence and persistence,
polarity-transition sequence and persistence, polarity-direction-preserved
sequence and persistence, and decision-boundary crossing count.

The entitled branch is `{SUPPORT, REFUTE}`. Entitlement transition means
movement between that set and `NOT_ENTITLED`. Polarity transition means a
SUPPORT/REFUTE change when both outputs are entitled. Polarity direction is
preserved when both outputs are entitled and their SUPPORT/REFUTE class is
unchanged. These definitions use model outputs only.

### Family D — candidate-relative action ordering

For each seed, identity, epoch, and signed response coordinate, P4 compares the
three exact candidate values. Descending ranks are represented as ordered tie
groups. Each precommitted candidate pair is recorded as GREATER, LESS, or TIE.
Ties are explicit and never broken lexically.

The audit records ranks and pairwise order at all three epochs, within-tail
rank persistence, within-tail pairwise persistence, cross-seed rank agreement,
cross-seed pairwise agreement, and full three-seed exact candidate-order
agreement. A stable relative ordering is distinct from a stable absolute
threshold.

### Family E — native-relative action geometry

P4 records counterfactual-minus-native signed margin changes, entitlement
reserve change, polarity reserve change, and candidate-centered signed deltas.
Entitlement reserve is
`max(SUPPORT, REFUTE) - NOT_ENTITLED`. Polarity reserve is
`SUPPORT - REFUTE`.

For response coordinate \(d\) and candidate \(a\):

```text
centered_delta(a) = delta(a) - mean(delta over the three candidates)
```

The mean is an explicit within-row comparison. No weights are learned and no
opaque composite score is created.

### Family F — cross-seed mechanism agreement

For the same data identity, candidate mask, and response coordinate, P4
records exact agreement of sign sequences, monotonic direction,
counterfactual-winner sequences, transition sequences, candidate ranks, and
pairwise order. Seed remains a grouping/provenance dimension and is never an
integration feature.

## Mechanism audits

The endpoint-shift audit counts exact endpoint inequality conditional on equal
sign sequence, equal monotonic direction, or equal categorical sequence. Exact
serialized numeric equality is used; no arbitrary tolerance defines endpoint
shift.

The topology-instability audit counts within-tail sign reversal, winner
change, transition-flag change, and cross-seed sign-sequence, winner-sequence,
monotonic-direction, and candidate-order disagreements.

The candidate-relative audit counts within-tail rank stability, within-tail
pairwise-order stability, cross-seed rank agreement, cross-seed pairwise-order
agreement, and full three-way exact candidate-order agreement.

The endpoint-authority audit requires all four P2 epoch-20 disagreement counts
to equal zero.

## Authorization boundary

Potential inference-time candidate state is limited to quantities available
from one model checkpoint and deterministic simulation of the three frozen
actions:

```text
epoch-20 signed action deltas
epoch-20 native-relative reserve changes
epoch-20 candidate-centered deltas
epoch-20 candidate ranks
epoch-20 pairwise candidate ordering
model-output transition flags
```

This is state-definition authorization only. P4 evaluates, promotes, and
integrates no gate.

Epoch-18/19 values, tail summaries, trajectory sequences and topology,
persistence, cross-seed agreement, runtime seed, and training epoch are
diagnostic-only.

Gold labels, correctness, recovery, harm, safety-target categories,
transition role, primary/discovery membership, raw text, identity, and seed as
a feature are prohibited. Projected authority reads retain only necessary
source fields. P0 is never opened. Output schemas exclude prohibited label and
outcome fields.

## Ordered decisions

The hierarchy is:

1. `STAGE196B2B6P4_CANDIDATE_RELATIVE_INVARIANT_STATE_IDENTIFIED` when at least
   one precommitted signed coordinate has exact candidate ordering stable over
   all tail epochs, all three seeds, and the full eligible population;
2. `STAGE196B2B6P4_STABLE_TRAJECTORY_TOPOLOGY_WITH_ENDPOINT_SHIFT` when no
   relative invariant exists, all topology is stable, and endpoints differ;
3. `STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE` when sign, winner,
   transition, monotonic, or candidate-order topology changes within tail or
   across seeds;
4. `STAGE196B2B6P4_EXACT_TAIL_ACTION_RESPONSE_EXPORT_REQUIRED` when existing
   source state is present but cannot materialize the exact full tail export;
5. `STAGE196B2B6P4_TAIL_COMPOSER_INSTRUMENTATION_REQUIRED` when no existing
   source boundary exposes the required coordinates.

Contract, source, schema, identity, arithmetic, or reproduction failure yields
`STAGE196B2B6P4_BLOCKED_CONTRACT_FAILURE` and recommends
`STAGE196B2B6P4_REPAIR_CONTRACT`. Ordinary topology instability is a
successful scientific result and exits zero when all contracts pass.

## Exact outputs and atomicity

Exactly eleven files are emitted:

```text
stage196b2b6p4_analysis.json
stage196b2b6p4_report.md
stage196b2b6p4_source_closure.csv
stage196b2b6p4_tail3_action_response_rows.csv
stage196b2b6p4_stability_state_dictionary.csv
stage196b2b6p4_trajectory_topology_audit.csv
stage196b2b6p4_candidate_relative_order_audit.csv
stage196b2b6p4_endpoint_shift_audit.csv
stage196b2b6p4_leakage_boundary.csv
stage196b2b6p4_decision_gate.csv
stage196b2b6p4_contract.csv
```

All files are rendered into a staging directory, each staged file is
atomically renamed, and the staging directory is atomically published.
Existing output directories are never overwritten. Where non-overwrite
semantics permit, unhandled exceptions still emit the exact eleven blocked
artifacts.

## Contract and exit behavior

Contract columns are:

```text
scope
run
gate
required
observed
passed
blocking_reason
```

The contract covers commit and frozen decision identity; P3 zero-gate and
zero-transfer closure; P2 source counts and endpoint authority; independent P2
recipient-artifact provenance; valid-digest normalization and ordered authority
selection; the exact 524,256-row external schema, seed, semantic-key, six-run,
B2-B5 context-key separation, 48-row pooled membership, transfer-context
diagnostics, and B2-B6/P2 6,480-row mapping closure; exact B2-B6 masks; B2-B4
controlled numeric agreement; runtime commit, 120/120 sidecar, and six-manifest
closure; exact tail populations and identities; class order, finite scores,
margin/delta arithmetic; exact P2 epoch-20 reproduction; dictionary,
authorization, nondependency, prohibited-field, topology, ordering, endpoint,
decision, and eleven-file closure.

Exit status is zero exactly when `blocking_reasons == []`, including an
ordinary `ACTION_RESPONSE_TOPOLOGY_UNSTABLE` result. It is two otherwise.
