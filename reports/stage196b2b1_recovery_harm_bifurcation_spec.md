# Stage196-B2-B1 Frame Recovery vs Preservation Harm Bifurcation Audit Specification

## Purpose and causal scope

Stage196-B2-B1 is an artifact-only descriptive bifurcation audit over the completed
Stage196-B2-A result `STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION`. It asks whether,
before observing the `frame_local_only` outcome, later recovery rows and later
preservation-harm rows occupy distinct native joint-arm local-channel states under
the frozen-Mamba Stage196-B2-P0 paired reruns.

The audit does not design, implement, train, or promote an intervention. It does not
claim formal causal mediation, deployable sample routing, open-world failure
detection, unfrozen-encoder behavior, external/OOD improvement, architecture
superiority, selective-intervention safety, or that a final prediction/status is a
mechanistic separator.

## Required command line

The analyzer requires exactly five arguments, all without defaults:

```text
--repo-root
--stage196b2a-analysis-json
--stage196b2a-analyzer-git-commit
--current-git-commit
--output-dir
```

`--stage196b2a-analyzer-git-commit` must equal
`833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6` and the authoritative source-contract
value. `--current-git-commit` must be lowercase full 40-hex and equal repository
HEAD. The output must be a new or empty directory below the repository and must not
be the frozen source directory.

## Exact Stage196-B2-A source closure

The parent directory of `--stage196b2a-analysis-json` must contain exactly these
nine files and no other directory entry:

```text
stage196b2a_analysis.json
stage196b2a_report.md
stage196b2a_seed_summary.csv
stage196b2a_support_transition_rows.csv
stage196b2a_channel_transition_summary.csv
stage196b2a_recurrent_position_propagation.csv
stage196b2a_harm_rescue_rows.csv
stage196b2a_epoch_propagation.csv
stage196b2a_contract.csv
```

The source JSON must close with decision
`STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION`, recommended next stage
`STAGE196B2B_NO_PROMOTION_MINIMAL_SEED_SPECIFIC_FOLLOWUP`, and empty blocking
reasons. Every source-contract row must pass with an empty blocking reason.

The analyzer requires three unique seed-summary rows for seeds 183, 184, and 185;
the exact 60-row seed-by-epoch grid (three seeds by epochs 1 through 20); and the
exact 4,320-row support-transition closure (three seeds by 720 rows by selected and
tail-three views). It loads every companion even when a table is not a primary
decision input.

Positive seeds are read from the source JSON's `positive_frame_shift_seeds` and
cross-checked against the seed-summary direction labels. They must reproduce 184
and 185. The negative direction-visible contrast is read from
`negative_frame_shift_contrast_seeds`, cross-checked the same way, and must
reproduce 183. Seed numbers are expected closure values, not the derivation rule.

The source seed summary must independently reproduce:

| Seed | Primary decision population | Largest blocker | Rate | Rescue | Harm |
|---|---:|---|---:|---:|---:|
| 184 | 12 | `FRAME_REMAINS_SUBTHRESHOLD` | 1.0 | 5 | 6 |
| 185 | 18 | `FRAME_REMAINS_SUBTHRESHOLD` | 1.0 | 2 | 3 |

Any disagreement fails closed as
`STAGE196B2B1_ANALYSIS_INCOMPLETE`; it is never converted into scientific
evidence.

## Provenance-role normalization

The following roles remain distinct:

```text
Stage196-B2-A analyzer:
833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6

Stage196-B2-P0 training runtime:
e9aaff24054f1d409119b70df13b94159a34a8e4

Original Stage196-B1 training runtime:
9835cbbf86d83aca0964821669e63f7f6deb1c59

FrameGate implementation origin:
5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8
```

Authoritative values come from role-specific passed rows in
`stage196b2a_contract.csv`: `analysis_runtime_commit_equals_head`,
`stage196b1_runtime_commit_format`, six `p0_runtime_commit` rows, and
`framegate_implementation_origin_commit_preserved`. The SUPPORT-versus-
NOT_ENTITLED margin source is normalized to
`support_logit - not_entitled_logit` only when all six passed `p0_contract` rows
require both native logits and all six passed `p0_sidecar_alignment` rows close at
`20 x 720`.

For each normalized value, a null or absent top-level source JSON value plus valid
contract evidence emits a nonblocking `SOURCE_SCHEMA_WARNING`. A populated value
must equal the contract. Disagreement, or absence of authoritative contract
evidence, fails closed. The new JSON records normalized roles and the margin source
at top level; frozen P0 provenance is never rewritten to the new analyzer commit.

## Primary analysis view and populations

The only primary view is `tail3`, meaning epochs 18, 19, and 20. Selected-checkpoint
rows supply traceability-only supporting evidence and never enter a primary
denominator.

Primary rows are loaded from `stage196b2a_harm_rescue_rows.csv`, not reconstructed
from aggregate metrics. A recovery must have source role `RESCUE`, joint tail-three
status `PERSISTENT_NOT_ENTITLED`, intervention tail-three status `STABLE_SUPPORT`,
paired transition `RESCUE_NE_TO_STABLE_SUPPORT`, and gold label `SUPPORT` in the
aligned support-transition row. A preservation harm must have source role
`INTERVENTION_INDUCED_HARM`, joint tail-three status `STABLE_SUPPORT`, intervention
tail-three status other than `STABLE_SUPPORT`, and aligned gold label `SUPPORT`.

Expected counts are recovery/harm 5/6 for seed184 and 2/3 for seed185, for exactly
16 primary rows. Seed183 remains visible only as a contrast summary and contributes
zero primary decision rows.

## Stable identity and alignment

Stage196-B2-A certified `stable_row_id` as selected-output `id` cross-checked with
trajectory `source_row_id`. Its CSV schema serializes that certified identity as
`stable_row_id`; therefore Stage196-B2-B1 preserves both `id` and `source_row_id`
from that certified value and records the mapping in `source_row_provenance`.

The analyzer requires stable identity `(id, source_row_id, dev_position, seed)` and
checks:

- unique seed × analysis-view × transition-role rows;
- identical position and intervention type across selected, tail-three, harm/rescue,
  and recurrent tables wherever a row appears;
- identical tail statuses and selected-view transitions;
- scalar equality between harm/rescue and tail-three support-transition rows;
- no recovery/harm identity overlap in any seed;
- every primary row in both selected and tail-three support-transition views;
- every recurrent-table member in both transition views;
- every recurrent membership flag in every support-transition row against the
  recurrent-position table;
- primary harm/rescue membership flags against both sources.

Disagreement is an alignment failure, never a scientific outcome.

## Native-state primary feature policy

Primary separator logic may use only these native joint-arm values, observed before
the intervention outcome:

- Frame probability;
- predicate-coverage probability;
- sufficiency probability;
- polarity SUPPORT-facing margin;
- entitlement probability;
- their fixed native pass/fail states.

Intervention type is retained as a separate descriptive grouping only. The primary
decision excludes intervention-arm values, deltas, threshold crossings,
transition role as an input feature, rescue/harm label as an input feature, gold
label, row identity, dev position, raw text, final prediction or tail status,
SUPPORT-versus-NOT_ENTITLED final margin, SUPPORT probability, and NOT_ENTITLED
probability. Traceability columns do not enter rules, signatures, or decisions.

## Fixed native thresholds and headroom

There is no search, calibration, optimized cut point, normalization, or rescaling.
Pass states are fixed as:

```text
frame probability >= 0.5
predicate probability >= 0.5
sufficiency probability >= 0.5
polarity SUPPORT-facing margin >= 0
entitlement probability >= 0.5
```

Signed headrooms subtract 0.5 for the four probability channels and use the raw
polarity SUPPORT-facing margin. Per seed and role, each channel reports ordered raw
headrooms, median, minimum, maximum, and counts on the fail (`< 0`) and pass
(`>= 0`) sides.

## Signatures and mechanistic state classes

The primary signature is exactly five bits in this order:

```text
frame|predicate|sufficiency|polarity|entitlement
```

It excludes final composition. Each row belongs to exactly one exhaustive,
mutually exclusive class:

- `ISOLATED_FRAME_DEFICIT`: Frame fails and all four downstream local channels pass;
- `FRAME_PLUS_DOWNSTREAM_DEFICIT`: Frame fails and at least one downstream local
  channel fails;
- `ALL_LOCAL_CHANNELS_PASS`: all five local channels pass;
- `DOWNSTREAM_DEFICIT_WITH_FRAME_PASS`: Frame passes and at least one downstream
  local channel fails.

For each positive seed and role, the group table reports row count; class and exact
signature counts/rates; every channel pass rate; every headroom distribution and
boundary-side count; intervention-type counts/rates; all recurrent-set membership
counts; and universal-all-six membership count. Seeds are never pooled away.

## Composition-augmented non-authorizing view

A distinct secondary view, labeled exactly
`composition_augmented_non_authorizing`, adds joint SUPPORT-versus-NOT_ENTITLED
margin, joint SUPPORT probability, and joint NOT_ENTITLED probability for
diagnosis. `final_composition_headroom` is the unscaled native logit difference.
The separate six-bit signature appends a final-composition pass bit (margin at
least zero) to the five local bits.

Composition-only separation cannot authorize local-channel selective routing. No
final composition value enters Rule A, Rule B, or the five-bit signature.

## Signature overlap and deterministic transfer

For both five-bit and six-bit signatures, each positive seed reports recovery and
harm signature sets, shared and role-exclusive sets, set Jaccard overlap,
role-exclusive row coverage, and collisions where one exact signature occurs in
both roles.

Deterministic set transfer runs in both fixed directions, seed184 to seed185 and
seed185 to seed184. Source recovery-exclusive and harm-exclusive signature sets are
applied unchanged to the target. The audit reports target coverage, purity,
collision counts, uncovered recovery/harm/total rows, and whether transfer is exact
(complete role coverage, zero cross-role collisions, and no uncovered row). It does
not fit a classifier, optimize a subset, or alter source signatures with target
data. The six-bit transfer is separately labeled non-authorizing.

## Fixed candidate rules

Exactly two local-state rule pairs exist.

Rule A recovery is `ISOLATED_FRAME_DEFICIT`; Rule A harm preservation is
`ALL_LOCAL_CHANNELS_PASS`. Rule B recovery is native Frame fail; Rule B harm
preservation is native Frame pass.

For each evaluated rule and positive seed, the analyzer reports exact numerators
and denominators for recovery coverage, recovery-rule contamination among harms,
harm coverage, and harm-rule contamination among recoveries, plus balanced coverage
mean. A rule passes only when every positive seed independently has recovery
coverage at least 0.75, recovery-rule harm contamination at most 0.25, harm coverage
at least 0.75, harm-rule recovery contamination at most 0.25, at least two recovery
rows, and at least three harm rows, with complete source closure.

Rule A is evaluated first. Rule B is evaluated only if Rule A fails. No additional
rule, rule search, classifier, clustering, dimensionality reduction, or threshold
optimization exists.

## Intervention-type audit

Intervention type remains descriptive and non-authorizing. Without merging sparse
types or searching text, the audit reports seed × intervention-type recovery and
harm counts/rates, types present only in recovery, only in harm, or in both, and
cross-seed persistence of role-exclusive types. It cannot itself authorize a
selective mechanism.

## Precommitted decision order

Use `STAGE196B2B1_ISOLATED_FRAME_DEFICIT_BIFURCATION` only when Rule A passes in
every positive seed. This authorizes design work only and maps to
`STAGE196B2B2_ERROR_CONDITIONED_FRAME_GRADIENT_ROUTING_DESIGN`.

Use `STAGE196B2B1_FRAME_BOUNDARY_BIFURCATION` only when Rule A fails and Rule B
passes in every positive seed. This authorizes design work only and maps to
`STAGE196B2B2_FRAME_BOUNDARY_CONDITIONED_INTERVENTION_DESIGN`.

When both rules fail, use
`STAGE196B2B1_LOCAL_STATE_OVERLAP_COMPOSITION_ONLY_SEPARATION` only when local
signatures overlap or lack exact transfer, six-bit signatures are role-exclusive
in both seeds or reduce collision rows, and six-bit source-exclusive signatures
transfer exactly in both directions. The distinction then depends on excluded
final-composition state and maps to
`STAGE196B2B2_FRAME_COMPOSITION_JOINT_MICROINTERVENTION_DESIGN`, without authorizing
Frame routing.

Otherwise use `STAGE196B2B1_SEED_SPECIFIC_NO_STABLE_BIFURCATION`, including when
one seed's role-exclusive signatures collide in the other or composition transfer
is not exact. This maps to
`STAGE196B2B2_NO_PROMOTION_ROW_LEVEL_CAUSAL_PROBE`.

Missing source files, decision/provenance/contract disagreement, unresolved schema,
alignment failure, unexpected populations, overlap/duplicates, or internal output
closure failure uses only `STAGE196B2B1_ANALYSIS_INCOMPLETE`, mapped to
`STAGE196B2B1_REPAIR_ANALYSIS_INPUTS`.

No outcome automatically authorizes a new loss, trainer or model modification,
global detachment, checkpoint-selection changes, full retraining, checkpoint/model
loading, or external evaluation.

## Exact eight-file output closure

The analyzer creates exactly:

```text
stage196b2b1_analysis.json
stage196b2b1_report.md
stage196b2b1_group_summary.csv
stage196b2b1_row_profiles.csv
stage196b2b1_signature_summary.csv
stage196b2b1_cross_seed_transfer.csv
stage196b2b1_intervention_type_summary.csv
stage196b2b1_contract.csv
```

The row-profile table contains exactly one row per positive-seed primary tail-three
recovery/harm row (16 complete rows). It includes certified stable identity,
intervention type, recurrent memberships, native joint values/passes/headrooms,
five-bit signature and state class, traceability-only intervention values/deltas and
selected transition, secondary composition fields/signature, and explicit source
provenance.

On incomplete analysis, all scientific CSVs retain headers with zero data rows,
the contract records gates reached plus the terminal failure, and the JSON and
Markdown report only the incomplete result. The writer uses exclusive creation and
checks the exact eight names after writing; no ninth output is permitted.

## Analysis JSON and activity closure

The JSON records the decision, next stage, blockers, all five commit/source roles,
source-schema warnings, normalized margin source, positive/contrast seeds, exact
population counts, both rule evaluations (Rule B may be explicitly skipped), both
signature views, both transfer views, intervention-type summary, authorized and
prohibited interpretations, and output count.

Required activity flags are fixed:

```text
training_performed = false
checkpoint_loaded = false
model_loaded = false
external_evaluation_performed = false
artifact_only_analysis = true
threshold_search_performed = false
classifier_fitted = false
```

## Markdown report contract

The generated report contains exactly these twenty sections in order:

1. Executive decision
2. Authorized interpretation
3. Stage196-B2-A source closure
4. Provenance normalization
5. Primary seeds and populations
6. Native-state feature policy
7. Recovery population
8. Preservation-harm population
9. Local-state classes
10. Fixed Rule A evaluation
11. Fixed Rule B evaluation
12. Signature overlap
13. Cross-seed signature transfer
14. Composition-augmented non-authorizing view
15. Intervention-type audit
16. Seed183 contrast
17. Decision-rule evaluation
18. Remaining uncertainty
19. Prohibited claims
20. Recommended next stage

## Historical preservation and static-review boundary

The analyzer reads only the supplied nine-file B2-A directory and writes only a new
eight-file output directory. It never modifies or overwrites Stage196-B2-P0,
Stage196-B2-A, Stage196-B1-C, or Stage196-A artifacts.

Implementation review is static only. It confirms two new repository files, exact
source/output closures, contract-based provenance normalization, a 16-row disjoint
tail-three primary population, source-derived primary seeds, primary native-state
feature restriction, fixed thresholds/classes/rules, visible seed directions, and
absence of classifier/threshold-search code. No Python, analyzer, compilation,
smoke run, training, checkpoint/model loading, Kaggle command, commit, or push is
part of this implementation stage.

## Remaining risk

Because execution is prohibited here, source serialization details and runtime
behavior remain untested until the new analyzer commit and completed B2-A directory
are supplied in an authorized artifact-only run. Any such mismatch must yield the
incomplete decision rather than being repaired by relaxing scientific contracts.
