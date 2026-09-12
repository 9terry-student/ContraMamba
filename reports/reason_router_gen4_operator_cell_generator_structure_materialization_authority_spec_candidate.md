# ContraMamba Gen4 Operator-Cell Generator-Structure Materialization Authority Specification — Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_MINIMAL_OUTCOME_BLIND_GENERATOR_STRUCTURE_MATERIALIZATION
- Authority type: static specification candidate
- Implementation authorized: NO
- Materialization execution authorized: NO
- Training/Evaluation authorized: NO
- Kaggle authorized: NO
- Automatic Commit/Push: NO

This candidate freezes the narrow provenance boundary required before implementation.

## 2. Upstream scientific authority

Frozen source authority:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Relevant source:

scripts/build_controlled_v5.py

Target operators:

- entity_swap
- role_swap
- title_name_swap
- predicate_swap

No other intervention family is in scope.

## 3. Static-audit conclusion

The frozen generator explicitly defines:

- entity_swap
  - intended_changed_axes = ["name"]
  - generator source = name := alternate_name

- role_swap
  - intended_changed_axes = ["role"]
  - generator source = role := alternate_role

- title_name_swap
  - intended_changed_axes = ["title", "name"]
  - generator source = title := alternate_title
  - generator source = name := alternate_name

- predicate_swap
  - intended_changed_axes = ["predicate"]
  - generator source = predicate := alternate_predicate

These are generator-side structural facts.

They require no inspection of rendered text, labels, predictions, evaluator outputs,
training outcomes, or evaluation outcomes.

The canonical controlled_v5 datasets do not persist explicit generator-side
semantic-slot provenance.

They contain identity fields, rendered text, intervention type, and supervision
fields, but no explicit intended_changed_axes, source_slot,
generator_source_fields, or equivalent generator structure.

Therefore:

DIRECT_FEATURE_DESIGN = BLOCKED

## 4. Stage182a exclusion

Stage182a is not valid Gen4 feature provenance for observed semantic-slot identity.

Its semantic_state() function reconstructs slot state from rendered evidence by
checking whether original and alternate values occur in the evidence string.

Its changed_axes() function therefore derives observed slot changes through
post-hoc text reconstruction.

The following Stage182a fields are consequently forbidden as Gen4 semantic-cell
feature provenance:

- observed_changed_axes
- entity_changed
- title_name_changed
- role_changed
- predicate_changed
- other changed-axis fields derived from rendered evidence

Therefore:

STAGE182A_OBSERVED_AXIS_REUSE = FORBIDDEN

Stage182a remains historical integrity-audit evidence only.

Stage182a INTENDED_AXES is consistent with the generator structure, but the Gen4
materialization must use the frozen generator/source mapping directly rather
than text-derived observed state.

## 5. Required next object

The next authorized scientific object is a minimal outcome-blind and text-blind
generator-structure sidecar.

Its sole purpose is to persist semantic structure already explicit in the frozen
generator.

It MUST NOT discover, infer, classify, or reinterpret semantic structure.

## 6. Permitted inputs

A future materializer may read only these canonical controlled-row identity fields:

- id
- pair_id
- intervention_type

Semantic-cell definition must come only from the frozen generator structure at:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea:scripts/build_controlled_v5.py

Authorized mapping:

entity_swap:
  intended_changed_axes = ["name"]
  generator_source_fields = {"name":"alternate_name"}

role_swap:
  intended_changed_axes = ["role"]
  generator_source_fields = {"role":"alternate_role"}

title_name_swap:
  intended_changed_axes = ["title","name"]
  generator_source_fields = {"title":"alternate_title","name":"alternate_name"}

predicate_swap:
  intended_changed_axes = ["predicate"]
  generator_source_fields = {"predicate":"alternate_predicate"}

No semantic information may be recovered by inspecting rendered text.

## 7. Explicitly forbidden inputs

A future implementation MUST NOT use any of the following to determine
operator-cell identity:

- claim
- evidence
- final_label
- frame_compatible_label
- predicate_covered_label
- sufficiency_label
- polarity_label
- primary_failure_type
- model predictions
- model logits
- model probabilities
- evaluator outputs
- error cohorts
- training outcomes
- evaluation outcomes
- Stage182a observed_changed_axes
- Stage182a changed-axis booleans derived from rendered evidence
- manual semantic interpretation of rendered natural language
- tokenizer-derived semantic reconstruction

Even if one of these signals reproduces the correct slot assignment, it is
invalid provenance for this phase.

## 8. Minimal future sidecar schema

Required fields:

- schema_version
- authority_commit
- row_id
- pair_id
- intervention_type
- intended_changed_axes
- generator_source_fields
- operator_cells

authority_commit must be exactly:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Example operator_cells:

- entity_swap:name
- role_swap:role
- title_name_swap:title
- title_name_swap:name
- predicate_swap:predicate

No observed text-derived state belongs in this schema.

## 9. Deterministic serialization requirements

Canonical axis order for this phase:

1. title
2. name
3. role
4. predicate

Within each row, intended_changed_axes MUST follow that canonical order.

operator_cells MUST follow the same axis order.

No source dataset row may be modified.

The canonical controlled dataset must remain byte-preserved.

The sidecar MUST be a separate artifact.

## 10. Fail-closed requirements

A future implementation must fail closed if:

- the source authority commit differs from the frozen commit;
- a target row lacks id, pair_id, or intervention_type;
- an intervention outside the four authorized operators is admitted;
- an authorized intervention has an unknown semantic mapping;
- claim or evidence must be inspected to determine a semantic cell;
- any label or model-derived field is consulted;
- duplicate row identities are encountered;
- the materialized intervention_type disagrees with the source row;
- serialization is nondeterministic;
- the canonical controlled dataset is modified.

## 11. Scientific interpretation boundary

This materialization does not establish:

- predictive value
- causal relevance
- representation localization
- mechanistic interaction
- performance improvement
- error reduction
- superiority of one operator-cell over another

It establishes only deterministic, provenance-valid operator-cell identity from
pre-existing generator structure.

Feature design and scientific testing require later authority.

## 12. Expected future implementation delta

If a later implementation authority is frozen, the expected delta should be:

- one generator-structure materialization script
- one focused test module
- no model changes
- no loss changes
- no tokenizer changes
- no canonical dataset rewriting
- no training code changes
- no evaluation code changes

Artifact generation itself remains a separate execution boundary unless
explicitly authorized.

## 13. Current decision

DIRECT_FEATURE_DESIGN = BLOCKED

Reason:
Canonical controlled rows do not persist explicit semantic-slot provenance.

STAGE182A_OBSERVED_AXIS_REUSE = FORBIDDEN

Reason:
Observed axes are reconstructed from rendered evidence text.

MINIMAL_GENERATOR_STRUCTURE_MATERIALIZATION = REQUIRED

Reason:
The frozen generator already contains deterministic outcome-blind semantic
structure sufficient to persist operator-cell identities without semantic
inference.

## 14. Stop condition

Stop after this authority candidate is reviewed.

Do not implement the materializer.

Do not generate the sidecar.

Do not train or evaluate.

Do not run Kaggle.

Do not promote any Gen4 feature hypothesis.

A later frozen authority must explicitly authorize the next boundary.