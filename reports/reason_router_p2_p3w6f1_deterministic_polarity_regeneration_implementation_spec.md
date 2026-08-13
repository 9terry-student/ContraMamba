# P3-W6-F1 Deterministic Polarity Regeneration Implementation Specification

Decision: `P3W6F1_IMPLEMENTATION_SPEC_READY_FOR_STATIC_REVIEW`

Schema version: `reason_router_p2_p3w6f1_deterministic_polarity_regeneration_implementation_manifest_v4`

This is a static implementation specification authority only. It does not establish implementation execution authority. It does not implement code, modify datasets, regenerate rows, run analyzers, run generators, run tests, perform manual review, train, evaluate, or perform Git actions.

## Authority

- P3-W5 specification authority commit: `01d983f8d09cacf0eddefd2014fc81a28771cf5e`
- P3-W5 authority files:
  - `reports/reason_router_p2_p3w5_separate_remediation_spec.md`
  - `reports/reason_router_p2_p3w5_separate_remediation_manifest.json`
- P3-W4 result authority commit: `f0a9afddc5b93c54aa72b0335c5a1a2f517cf934`
- P3-W4 authority artifacts:
  - `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json`
  - `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl`
  - `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_f2_manual_review.csv`

P3-W5 F1/F2 partition, release state, Git-preservation contract, semantic-authority contract, and non-authority wording are preserved.
## Revision History

- `v1_static_review`: `P3W6F1_IMPLEMENTATION_SPEC_V1_STATIC_REVIEW_BLOCKED_REVISION_REQUIRED`
- `v2_revision`: preserves the v1 source trace and authority findings while blocking generic negative `_statement` repair, selecting one base-form implementation authority, and preserving the full P3-W5 artifact field/accounting contract.
- `v2_static_review`: `P3W6F1_IMPLEMENTATION_SPEC_V2_STATIC_REVIEW_BLOCKED_SURGICAL_REVISION_REQUIRED`
- `v3_revision`: separates current base-form coverage state from future required result, restores exact P3-W5 full-output field names and row identity equality, preserves exact P3-W5 base-form provenance names, restores semantic status mapping, and fixes one approved repair API.
- `v3_implementation_review`: `P3W6F1_TARGET_SCOPE_SPECIFICATION_CONTRADICTION_DISCOVERED`
- `v4_revision`: replaces the invalid negative-polarity-flip equality contract with target-scope membership and full-output isolation semantics, narrows the repair consumer to authorized F1 row IDs, corrects generated-candidate accounting, and hardens Stage185 and execution provenance requirements.
- `v4_phase_order_review`: `P3W6F1_SPEC_V4_PHASE_ORDER_CONTRADICTION_CORRECTED`
- `v5_execution_totality_spec_repair`: adds the contract-only dedicated regeneration wrapper, deterministic execution/output directories, repaired Stage185 builder invocation, complete analyzer input producer map, and frozen base-form authority raw-file SHA256. This revision does not implement or execute the wrapper.

## Static Source Trace

Static audit used repository source and P3-W4/P3-W5 authority artifacts only.

| item | source_path | function_or_symbol | relevant responsibility |
|---|---|---|---|
| dataset generator source file | `scripts/build_controlled_v5.py` | module `build_controlled_v5` | Production controlled-v5 dataset generator. P3-W4 sidecar records `generator_source_path = /kaggle/working/ContraMamba/scripts/build_controlled_v5.py` and `generator_source_sha256 = c41e6a52401bd8c83970286b176950fc751509bee6d797d5da9aea4262c72802`. Current local SHA256 matches this value. |
| relevant generator function | `scripts/build_controlled_v5.py` | `_build_records(templates)` at line 380; `build_controlled_records(num_pairs)` at line 470 | Iterates facts, constructs canonical/paraphrase/intervention rows, and calls `_statement(fact, negative=not base_refute)` for `polarity_flip`. |
| polarity_flip construction path | `scripts/build_controlled_v5.py` | `_build_records`, `polarity_flip` record construction at line 453 | Creates the polarity member with intervention `polarity_flip`, `flipped_final`, labels `(frame=1, predicate=1, sufficiency=1)`, `polarity_label = flipped_final`, `primary_failure_type = polarity`, and the shared canonical claim. |
| predicate representation source | `scripts/build_controlled_v5.py` | `FACT_TEMPLATES`, `_ADDITIONAL_FACT_ROWS`, `_GENERATED_PREDICATES`, `_generated_fact_template` | Stores `predicate` and `alternate_predicate` as already inflected past-tense surface forms, for example `approved`, `opened`, `released`, `won`. No reusable upstream `predicate_base` or lemma field was found in the audited generator source; status `NOT_FOUND_IN_AUDITED_SOURCE`. |
| predicate surface-form construction | `scripts/build_controlled_v5.py` | `_statement(fact, negative=False, **overrides)` at lines 333-341 | Reads `values["predicate"]` and renders it directly into the sentence template. |
| negation construction | `scripts/build_controlled_v5.py` | `_statement(..., negative=True)` at lines 333-341 | Prefixes the same predicate surface with `did not ` when `negative` is true. |
| `"did not"` construction | `scripts/build_controlled_v5.py` | line 337, `predicate = f"did not {predicate}"` | Exact mechanism that creates `did not <inflected predicate>`. |
| inflected predicate construction | `scripts/build_controlled_v5.py` | `FACT_TEMPLATES`, `_ADDITIONAL_FACT_ROWS`, `_GENERATED_PREDICATES`, `_generated_fact_template` | Selects past-tense predicate surfaces before `_statement` is called. `_statement` does not distinguish base and inflected forms. |
| evidence rendering path | `scripts/build_controlled_v5.py` | `_statement`, `_paraphrase`, `_record`, `_build_records` | `_statement` renders canonical and non-paraphrase evidence; `_paraphrase` renders paraphrase evidence. F1 authorized polarity rows use `_statement`, not `_paraphrase`. |
| row id construction | `scripts/build_controlled_v5.py` | `_record` at lines 353-377 | Constructs `id` as `f"{fact['pair_id']}__{intervention}"`. |
| pair/canonical linkage construction | `scripts/build_controlled_v5.py`; `scripts/build_stage185a_controlled_train_integrity_sidecar.py` | `_record` stores `pair_id`; Stage185 `build_sidecar` links one `none` row per pair as canonical | The generator assigns common `pair_id` to every intervention. Stage185 groups by `pair_id`, selects `intervention_type == "none"` as canonical, and records `canonical_row_id`. |
| grammar validator path | `scripts/build_stage185a_controlled_train_integrity_sidecar.py` | `grammar_anomaly(row, fact)` at lines 343-346; `build_sidecar` at lines 479-513 | Detects `did not` followed by either `fact["predicate"]` or `fact["alternate_predicate"]`; assigns `DID_NOT_INFLECTED_PREDICATE` and `GRAMMAR_TEMPLATE_FAIL`. |

## Root-Cause Trace

P3-W4 `p3w4_canonical_grammar_authority_pairs.jsonl` statically proves:

- F1 selector count is exactly 121 records with `family == F1`, `automatic_root_cause_class == F1_TRUE_POLARITY_GENERATION_DEFECT`, and `remediation_state == REGENERATION_REQUIRED`.
- All 121 F1 polarity members have `intervention_type == polarity_flip`, `final_label == REFUTE`, and `polarity_label == REFUTE`.
- All 121 F1 canonical members have `final_label == SUPPORT`.
- All 121 F1 polarity members have `grammar_rule_reproduction.production_rule_reproduction_result == true`.
- All 121 F1 polarity members have reason codes exactly `DID_NOT_INFLECTED_PREDICATE,GRAMMAR_TEMPLATE_FAIL`.
- All 121 F1 polarity members have `generator_source_sha_matches == true`.
- All 121 F1 polarity members cite validator function `grammar_anomaly`.
- Representative matched spans include `did not released`, `did not opened`, `did not reopened`, `did not inspected`, `did not launched`, `did not selected`, `did not mapped`, `did not received`, and `did not approved`.

Answers:

- Q1. Yes. The 121 authorized F1 rows all trace to the same generator rule/path: `_build_records` creates a `polarity_flip` row using `_statement(fact, negative=not base_refute)`, and `_statement` renders `did not {values["predicate"]}` without converting the predicate to base form.
- Q2. The defect is surface rendering/morphology selection in the negated `_statement` branch. Static source shows input predicate metadata is already inflected; no separate base-form metadata exists. It is not proven that the source fact intent is wrong.
- Q3. No reusable upstream base predicate representation was found in the audited source. Status: `NOT_FOUND_IN_AUDITED_SOURCE` for `scripts/build_controlled_v5.py` symbols audited in this specification. The v3 target-scope equality between generated negative `polarity_flip` IDs and authorized F1 IDs is invalid because preserved authority establishes 121 authorized F1 rows within a 150-row structurally negative `polarity_flip` universe. The v4 closure requirement is membership plus non-authorized preservation.
- Q4. Current inflected predicate is selected from `fact["predicate"]`, populated by `FACT_TEMPLATES`, `_ADDITIONAL_FACT_ROWS`, or `_GENERATED_PREDICATES` via `_generated_fact_template`.
- Q5. `_statement` assigns `predicate = values["predicate"]`; if `negative`, it assigns `predicate = f"did not {predicate}"`; the final f-string renders that phrase as evidence.

Exact root cause:

`F1_TRUE_POLARITY_GENERATION_DEFECT` is caused by `_statement` reusing the inflected predicate surface after do-support negation. A generic repair to every `_statement(..., negative=True)` output is not authorized because that caller surface includes F2 canonical REFUTE rows. The repair must be scoped to the F1-authorized `polarity_flip` construction path and must prove complete-output isolation.

## Authorized Target Contract

Authorized F1 target set:

- `F1_target_pair_count == 121`
- `F1_target_row_count == 121`
- `authorized_F1_row_ids` are derived only from the P3-W4 pair artifact polarity member `source_row.id` values satisfying:
  - `family == F1`
  - `automatic_root_cause_class == F1_TRUE_POLARITY_GENERATION_DEFECT`
  - `remediation_state == REGENERATION_REQUIRED`
  - `pair_id` is in the P3-W5 `decision_supporting_pair_ids`

Source identity:

- comparison source identity field: `id`
- normalized artifact identity: `row_id = source_row["id"]`
- F2 and unaffected rows must not be added to the target set.

## Negative `_statement` Caller Inventory

`global_negative_statement_repair_authorized = false`

Forbidden implementation: changing every `_statement(..., negative=True)` output.

| source_path | caller_symbol | intervention/member type | negative condition | possible final label | F1/F2/unaffected membership relationship | repair effect if generic negative branch changes |
|---|---|---|---|---|---|---|
| `scripts/build_controlled_v5.py` | `_build_records` canonical `none` row calls `_statement(fact, negative=base_refute)` | canonical `none` | `base_refute == true` | `REFUTE` | P3-W4 F2 includes 119 canonical REFUTE grammar-defect rows under manual-review authority; other canonical REFUTE rows may be outside the F1 authorized set. | Generic negative repair would change F2 canonical REFUTE evidence and violate `F2_changed_row_ids == []`. |
| `scripts/build_controlled_v5.py` | `_build_records` canonical `none` row calls `_statement(fact, negative=base_refute)` | canonical `none` | `base_refute == false` | `SUPPORT` | Includes F1 canonical SUPPORT rows and unaffected canonical rows. | No negative rendering occurs; generic negative repair has no direct text effect here. |
| `scripts/build_controlled_v5.py` | `_build_records` `polarity_flip` row calls `_statement(fact, negative=not base_refute)` | `polarity_flip` | `base_refute == false` | `REFUTE` | P3-W4/P3-W5 authorized F1 target rows are a 121-row subset of the structurally negative `polarity_flip` universe; pre-generation membership must be verified before repair generation consumes any override, and non-authorized preservation is verified post-generation after repaired output exists and before candidate acceptance. | This is the only future repair branch allowed to request/use base predicate surface, and only when the row ID is in `authorized_F1_row_ids`. |
| `scripts/build_controlled_v5.py` | `_build_records` `polarity_flip` row calls `_statement(fact, negative=not base_refute)` | `polarity_flip` | `base_refute == true` | `SUPPORT` | Includes positive `polarity_flip` rows associated with canonical REFUTE pairs, including F2 triple members. | No negative rendering occurs; repair must preserve existing behavior. |
| `scripts/build_controlled_v5.py` | `_build_records` `entity_swap`, `event_swap`, `time_swap`, `location_swap`, `role_swap`, `title_name_swap`, `predicate_swap` calls to `_statement(...)` | other intervention callers | omitted, therefore default `negative == false` | `NOT_ENTITLED` | Unaffected by F1 authorization; may include frame/predicate/sufficiency rows. | No negative rendering occurs; repair must preserve existing behavior. |
| `scripts/build_controlled_v5.py` | `_build_records` `evidence_deletion`, `evidence_truncation`, `irrelevant_evidence` | content/non-`_statement` evidence | not applicable | `NOT_ENTITLED` | Unaffected by F1 authorization. | No `_statement` negative rendering occurs. |
| `scripts/build_controlled_v5.py` | `_build_records` paraphrase row calls `_paraphrase(fact, negative=base_refute)` | paraphrase | `base_refute == true` | `REFUTE` | P3-W4 F2 includes paraphrase REFUTE members under manual-review authority. | Not an `_statement` caller; F1 `_statement` repair must not alter `_paraphrase`. |

## Target-Scope Membership Contract

The v3 equality contract is deprecated and removed as an acceptance requirement:

```text
negative_polarity_flip_generated_ids == authorized_F1_row_ids
```

Reason invalid: preserved repository authority establishes 121 authorized F1 target rows, 300 full controlled dataset unique pairs, and 150 structurally negative `polarity_flip` rows. Stage185 authority shows `DID_NOT_INFLECTED_PREDICATE` on 150 `none`, 150 `paraphrase`, and 150 `polarity_flip` rows. Therefore the 121 F1 rows are an authorized subset of the structurally negative `polarity_flip` universe, not the entire universe.

Definitions:

- `structural_negative_polarity_flip_row_ids`: the complete set of generator rows whose `polarity_flip` construction requests negative rendering.
- `authorized_F1_row_ids`: IDs derived from P3-W4/P3-W5 authority: `family == F1`, `automatic_root_cause_class == F1_TRUE_POLARITY_GENERATION_DEFECT`, `remediation_state == REGENERATION_REQUIRED`, `pair_id` in `decision_supporting_pair_ids`, polarity member `source_row.id`. Its length is exactly 121.
- `non_authorized_structural_negative_polarity_flip_row_ids = structural_negative_polarity_flip_row_ids - authorized_F1_row_ids`.
- `repair_consumed_row_ids`: row IDs for which the future repair path actually consumes the base-predicate override.

Required phase order:

```text
1. authority extraction
2. pre-generation target-scope membership preflight
3. authorized repair generation
4. repair-consumption audit
5. full-output isolation
6. Stage185 provenance/state validation
7. semantic candidate validation
8. execution decision
```

This order is mandatory. Full-output isolation is a post-generation acceptance gate; it must not be used as a prerequisite for invoking the authorized repair because it cannot be computed until repaired output exists.

Pre-generation membership preflight:

Required before repaired generation may consume any base-form override.

PASS requires:

```text
authorized_F1_row_ids subset_of structural_negative_polarity_flip_row_ids
len(authorized_F1_row_ids) == 121
every authorized F1 row structurally requests negative polarity_flip rendering
```

On failure:

```text
TARGET_SCOPE_MEMBERSHIP_UNRESOLVED
```

Repair generation must fail closed before consuming any base-form override.

The pre-generation membership preflight must not require any of:

```text
repair_consumed_row_ids
changed_ids
evidence_changed_row_ids
non_authorized_structural_negative_polarity_flip_changed_row_ids
full-output isolation
Stage185 repaired state
```

because none of those can exist until repaired output has been generated.

Post-generation scope acceptance:

Required after repaired output exists and before candidate acceptance.

PASS requires:

```text
repair_consumed_row_ids == authorized_F1_row_ids
non_authorized_structural_negative_polarity_flip_changed_row_ids == []
changed_ids == authorized_F1_row_ids
evidence_changed_row_ids == authorized_F1_row_ids
full-output isolation PASS
```

A failure here blocks candidate acceptance and the all-accepted execution decision. Use the existing full-output isolation failure authority for post-generation isolation failures; this specification does not invent a second overlapping acceptance mechanism.

Machine-readable contract:

```text
membership_required = true
pre_generation_membership_preflight
post_generation_scope_acceptance
future_verification_required_before_execution = true
future_verification_required_before_execution_scope = pre_generation_membership_preflight_only
future_verification_required_during_execution = true
future_verification_required_during_execution_scope = post_generation_scope_acceptance_after_repaired_output_exists_before_candidate_acceptance
on_mismatch = TARGET_SCOPE_MEMBERSHIP_UNRESOLVED
```

Legacy future-verification boolean scope:

```text
future_verification_required_before_execution
does NOT require repair_consumed_row_ids, changed_ids,
full-output isolation, or repaired Stage185 state before repair generation.

future_verification_required_during_execution
refers to the post-generation acceptance checks performed only
after repaired output exists.
```

Current v4 static status: `TARGET_SCOPE_MEMBERSHIP_UNRESOLVED` for pre-generation membership preflight. The P3-W4 artifact proves that all 121 authorized F1 rows are negative `polarity_flip` REFUTE rows, but this specification does not execute the generator, construct the complete structural-negative polarity-flip set, or prove post-generation isolation. Future implementation must verify membership before repair generation consumes any override and must verify scope acceptance after repaired output exists. `TARGET_SCOPE_EQUIVALENCE_UNRESOLVED` may appear only as a deprecated v3/revision-history token, not as the active v4 acceptance contract.

Generation authority:

```text
Only the 121 authorized F1 polarity_flip rows may consume the
base-predicate override after membership preflight passes.
```

Acceptance authority:

```text
The generated change is not accepted until repair-consumption,
full-output isolation, Stage185, semantic, and provenance gates pass.
```

Required blast-radius contract:

```text
changed_ids == authorized_F1_row_ids
evidence_changed_row_ids == authorized_F1_row_ids
repair_consumed_row_ids == authorized_F1_row_ids
non_authorized_structural_negative_polarity_flip_changed_row_ids == []
F2_changed_row_ids == []
canonical_changed_row_ids == []
paraphrase_changed_row_ids == []
unaffected_changed_row_ids == []
```

F2 grammar defects may share the same source bug, but F2 remains under P3-W5 manual-review authority and has no automatic repair authority in this workstream.

## Proposed Minimal Code Change

Changed file:

- `source_path`: `scripts/build_controlled_v5.py`
- `symbol/function`: `_build_records` `polarity_flip` branch plus one approved callable interface: `_statement(fact, negative=False, predicate_surface_override=None, **overrides)`. A global `_statement(..., negative=True)` behavior change and any `or equivalent` API alternative are forbidden.
- `current behavior`: `_build_records` calls `_statement(fact, negative=not base_refute)` for `polarity_flip`; `_statement` uses `values["predicate"]` for negative rendering and creates `did not <inflected predicate>`.
- `root cause`: the F1 negative `polarity_flip` construction path lacks a base-form predicate surface and therefore carries past-tense inflection across `did not`.
- `proposed behavior`: in the `_build_records` `polarity_flip` branch only, if the polarity flip is negative and `polarity_row_id in authorized_F1_row_ids` after the pre-generation target-scope membership preflight has passed, obtain `base_predicate = _BASE_PREDICATE_BY_INFLECTED[fact["predicate"]]` and call `_statement(..., negative=True, predicate_surface_override=base_predicate)`. Otherwise preserve existing behavior exactly. Do not change canonical `none`, `_paraphrase`, positive `polarity_flip`, structurally negative but unauthorized `polarity_flip`, or other intervention rendering.
- `why this is minimal`: it repairs the source-level F1 polarity construction path without changing every negative `_statement` caller, thereby preserving F2 canonical REFUTE rows for the separate P3-W5 manual-review workstream.
- `generation authority`: only the 121 authorized F1 `polarity_flip` rows may consume the base-predicate override after pre-generation membership preflight passes.
- `acceptance authority`: generated changes are not accepted until repair-consumption audit, full-output isolation, Stage185 provenance/state validation, semantic candidate validation, and execution provenance gates pass.
- `forbidden effect`: changes to `claim`, labels, IDs, pair IDs, intervention types, row order, canonical rows, paraphrase rows, F2 rows, time-swap state, or any unaffected row.

Implementation preference: source repair in the `_build_records` `polarity_flip` construction branch, not post-hoc text repair and not generic negative `_statement` repair. Explicit mapping must be complete for required F1 inflected predicate surfaces, deterministic, local to generator provenance, and audited by tests. It must not be an unbounded regex cleanup, dataset-wide final-label heuristic, external dictionary lookup, LLM correction, or broad grammar cleanup.
## Exact Repair API Contract

One and only one approved callable interface is selected:

```text
_statement(
    fact,
    negative=False,
    predicate_surface_override=None,
    **overrides
)
```

Contract:

```text
predicate_surface_override is None
-> current behavior exactly preserved

predicate_surface_override is not None AND negative is true
-> use that exact supplied predicate surface after "did not"

predicate_surface_override is not None AND negative is false
-> fail closed / reject invalid invocation
```

Only authorized consumer:

```text
_build_records polarity_flip branch
AND polarity flip requests negative rendering
AND row_id in authorized_F1_row_ids
AND pre-generation target-scope membership preflight has passed
```

The branch obtains:

```text
base_predicate = _BASE_PREDICATE_BY_INFLECTED[fact["predicate"]]
```

Canonical `none`, `_paraphrase`, other interventions, and positive `polarity_flip` must never provide `predicate_surface_override`.

Pseudo-contract:

```text
polarity_row_id = f"{fact['pair_id']}__polarity_flip"

if (
    repair_mode
    and polarity_flip_negative
    and polarity_row_id in authorized_F1_row_ids
):
    use generator-owned base predicate override
else:
    preserve existing rendering exactly
```

Forbidden implementation:

```text
repair every negative polarity_flip row
repair every _statement(..., negative=True) row
```

## Base-Form Authority Contract

Current static finding:

- Existing reusable upstream base predicate metadata in audited generator source: `NOT_FOUND_IN_AUDITED_SOURCE`.
- Existing deterministic generator base morphology rule in audited generator source: `NOT_FOUND_IN_AUDITED_SOURCE`.
- Available generator provenance: inflected predicate surfaces are present in `fact["predicate"]` and `fact["alternate_predicate"]`.

Single approved implementation authority:

```text
symbol name = _BASE_PREDICATE_BY_INFLECTED
source path = scripts/build_controlled_v5.py
key representation = exact generator-owned inflected predicate surface string
value representation = exact generator-owned base predicate surface string suitable after "did not"
coverage universe = required_F1_inflected_predicate_surfaces, with optional superset coverage for generator-known predicate and alternate_predicate surfaces
duplicate-key policy = duplicate keys forbidden; conflicting base values fail validation
missing-key behavior = base form missing -> MANUAL_REVIEW_REQUIRED -> candidate_accepted = false
alternate-predicate handling = alternate_predicate surfaces may be included for table completeness and Stage185 validator parity, but the F1 repair consumer must use only the fact["predicate"] surface for authorized polarity_flip rows
validation behavior = future implementation validates coverage before candidate acceptance and fails closed on missing or ambiguous mapping
```

Fact metadata fields such as `predicate_base` and `alternate_predicate_base` are not selected for v3. The selected authority is a generator-owned explicit mapping because audited source stores predicate surfaces centrally and no reusable base-form fields exist.

Coverage closure contract separates current static state from future required result:

```text
current_static_status = BASE_FORM_COVERAGE_NOT_EVALUATED
current_mapping_implemented = false
current_coverage_evaluated = false
required_F1_inflected_predicate_surfaces = derive during authorized implementation/audit

required_result:
  missing_base_form_surfaces == []
  ambiguous_base_form_surfaces == []
  mapping_or_metadata_covered_surfaces superset_of required_F1_inflected_predicate_surfaces
```

This static specification does not claim coverage PASS. Mapping coverage must be evaluated by the future authorized implementation/audit. Mapping coverage does not authorize broad repair consumption. Even if `_BASE_PREDICATE_BY_INFLECTED` covers the whole generator predicate universe or additional predicates, only authorized F1 repair surfaces may consume it for automatic regeneration.

Future implementation must record for every accepted row:

- `inflected_predicate_surface`
- `expected_base_predicate`
- `base_form_derivation_method = generator_owned_explicit_mapping`
- `base_form_derivation_source_path = scripts/build_controlled_v5.py`
- `base_form_derivation_source_sha256`
- `base_form_source_symbol = _BASE_PREDICATE_BY_INFLECTED`
- `authorized_replacement_span`
- `outside_span_byte_identity`

Forbidden base-form sources:

- LLM inference
- free-form NLP model inference
- label-derived guess
- dictionary/network lookup at generation time
- heuristic unrelated to generator provenance

Fail closed rule:

```text
base form missing -> MANUAL_REVIEW_REQUIRED -> candidate_accepted = false
```
## Deterministic Semantic Repair Algorithm

Status enum:

- `DETERMINISTIC_POLARITY_REPAIR_PASS`
- `MANUAL_REVIEW_REQUIRED`
- `REJECTED`

For each authorized row, the positive path must prove:

1. The baseline row ID is in `authorized_F1_row_ids`.
2. The baseline `source_row.evidence` contains exactly one `did not <inflected_predicate_surface>` span reproduced by P3-W4 `matched_surface_span` or by the same Stage185 `grammar_anomaly` predicate source.
3. The replacement span is uniquely identified.
4. The replacement renders the same generator-authorized predicate in required base form after `did not`.
5. `claim` is byte-identical.
6. Evidence outside the authorized replacement span is byte-identical.
7. Exact negation semantics are preserved: `did not` remains present exactly as the polarity marker; no negation marker is added or removed outside the span.
8. Entity, role, title, object, location, and time tokens are unchanged.
9. `final_label == REFUTE` and `polarity_label == REFUTE`.
10. Stage185 post-regeneration contract passes.
11. Full-output isolation passes.

`MANUAL_REVIEW_REQUIRED` applies when the replacement span is absent, duplicated, or base form derivation is unavailable but no explicit contradiction is proven.

`REJECTED` applies when the candidate changes claim, labels, non-evidence fields, protected slots, polarity semantics, row identity, row ordering, or contradicts Stage185/final-label semantics.
P3-W5 semantic status mapping is preserved exactly:

```text
Positive acceptance:
semantic_validation_status = DETERMINISTIC_POLARITY_REPAIR_PASS
semantic_polarity_preserved = true
candidate_accepted = true
regenerated evidence preserves exactly one "did" and exactly one "not"

Unresolved:
semantic_validation_status = MANUAL_REVIEW_REQUIRED
semantic_polarity_preserved = null
candidate_accepted = false
ordered_rejection_codes includes SEMANTIC_AUTHORITY_UNRESOLVED

Explicit contradiction:
semantic_validation_status = REJECTED
semantic_polarity_preserved = false
candidate_accepted = false
```

## Full-Output Isolation Contract

Complete generated baseline and repaired outputs must be compared by `id`.

Allowed source-row field change:

```text
authorized_changed_source_fields == ["evidence"]
```

P3-W5 canonical full-output field names and required results:

```text
structural_negative_polarity_flip_row_ids
authorized_F1_row_ids
non_authorized_structural_negative_polarity_flip_row_ids
repair_consumed_row_ids
non_authorized_structural_negative_polarity_flip_changed_row_ids

changed_ids == authorized_F1_row_ids
repair_consumed_row_ids == authorized_F1_row_ids
non_authorized_structural_negative_polarity_flip_changed_row_ids == []

missing_ids == []
added_ids == []
duplicate_ids == []

unauthorized_changed_row_ids == []
F2_changed_row_ids == []
unaffected_changed_row_ids == []
canonical_changed_row_ids == []
paraphrase_changed_row_ids == []

evidence_changed_row_ids == authorized_F1_row_ids
claim_changed_row_ids == []
non_text_field_changed_row_ids == []
```

Shortened v2 names such as `unauthorized_changed_ids`, `F2_changed_ids`, `evidence_changed_ids`, and `non_evidence_field_changed_ids` are not canonical replacement fields. They may exist only as compatibility aliases if future tooling needs them; P3-W5 canonical fields above must exist in artifact schemas and manifest contracts.

Row identity equality:

```text
original_row_id == source_original["id"]
regenerated_row_id == source_regenerated["id"]
original_row_id == regenerated_row_id
```

Row order:

```text
baseline_id_sequence == repaired_id_sequence
baseline_id_sequence_sha256 == repaired_id_sequence_sha256
row_order_changed == false
```

Any shared generator change that alters F2 or unaffected rows fails F1 execution.

Structurally negative but unauthorized polarity-flip rows remain protected. Their presence is diagnostic/scope provenance, not repair authority.
## Stage185 Transition Contract

Baseline F1 polarity row:

```text
grammar_status == FAIL
integrity_status == INELIGIBLE
canonical_status == PASS
```

Accepted regenerated row:

```text
dataset_source_status == PASS
schema_status == PASS
intervention_contract_status == PASS
grammar_status == PASS
integrity_status == ELIGIBLE
canonical_status == PASS
polarity_contamination_status == PASS
time_swap_status == PASS
audit_expected_axes == ["polarity"]
audit_changed_axes == ["polarity"]
audit_pair_failure_scope == "none"
```

Transition token:

```text
F1_integrity_transition = INELIGIBLE_TO_ELIGIBLE
```

## Total Future Execution Contract

This section is a specification-only repair. It reserves a future dedicated regeneration wrapper path but does not create or authorize execution of that script in this task.

Reserved future wrapper path:

```text
scripts/regenerate_reason_router_p3w6f1_deterministic_polarity.py
```

Wrapper implementation state:

```text
CONTRACT_ONLY
NOT_YET_IMPLEMENTED
```

The existing `scripts/build_controlled_v5.py` CLI remains authoritative as currently observed:

```text
--dataset-version {seed,v1}
```

The P3-W6-F1 repaired workflow must not request a generator dataset-version choice named `v3`; the artifact name `controlled_v5_v3_without_time_swap` does not imply such a generator CLI choice exists.

The future wrapper must reuse the existing approved Python-level repair symbols rather than reimplementing repair semantics:

```text
scripts.build_controlled_v5.build_controlled_records_with_f1_polarity_repair_audit
scripts.analyze_reason_router_p3w6f1_deterministic_polarity_regeneration.project_replay_to_baseline_topology
```

The future wrapper may reuse the analyzer's authority-extraction helpers so the target definition cannot diverge:

```text
extract_decision_supporting_pair_ids
extract_authorized_f1_targets
baseline_pair_count
id_sequence_sha256
canonical_sha256
structural_negative_polarity_flip_row_ids_for_pair_count
```

The one authoritative `authorized_F1_row_ids` derivation is:

1. Load `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json`.
2. Load `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl`.
3. Load `reports/reason_router_p2_p3w5_separate_remediation_manifest.json`, whose authority commit is `01d983f8d09cacf0eddefd2014fc81a28771cf5e`.
4. Derive `decision_supporting_pair_ids` with `extract_decision_supporting_pair_ids(p3w4_summary, p3w5_manifest)`.
5. Derive targets with `extract_authorized_f1_targets(pair_records, decision_supporting_pair_ids)`.
6. The only authorized repaired row set is `targets["authorized_F1_row_ids"]`, requiring exactly 121 row IDs, each ending in `__polarity_flip`, each from a P3-W4 record with `family == F1`, `automatic_root_cause_class == F1_TRUE_POLARITY_GENERATION_DEFECT`, `remediation_state == REGENERATION_REQUIRED`, and `pair_id` in `decision_supporting_pair_ids`.

No other F1 row discovery mechanism, latest artifact, compatibility output, or manual row list may define the repair target set.

Define `F1_EXECUTION_COMMIT` as the full 40-character Git commit checked out at future execution. Future execution requires:

```text
repository branch/identity checked
tracked worktree clean
future wrapper present in F1_EXECUTION_COMMIT
approved repaired generator behavior present in F1_EXECUTION_COMMIT
no semantic behavior supplied by uncommitted tracked changes
```

The deterministic regeneration directory is:

```text
reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_<F1_EXECUTION_COMMIT>/
```

Here `<F1_EXECUTION_COMMIT>` is the exact full 40-hex execution commit, not a branch name, not `HEAD` as a moving reference, and not `latest`.

The future wrapper command contract is:

```text
python scripts/regenerate_reason_router_p3w6f1_deterministic_polarity.py \
  --repo-root . \
  --baseline-jsonl data/controlled_v5_v3_without_time_swap.jsonl \
  --baseline-jsonl-sha256 f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640 \
  --p3w4-summary-json reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json \
  --p3w4-pairs-jsonl reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl \
  --p3w5-manifest-json reports/reason_router_p2_p3w5_separate_remediation_manifest.json \
  --baseline-sidecar-jsonl reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl \
  --base-form-authority-commit 11102ea05b28f6638fdead205b4a9ee0f35ca0de \
  --base-form-authority-path scripts/build_controlled_v5.py \
  --base-form-authority-symbol _BASE_PREDICATE_BY_INFLECTED \
  --base-form-authority-sha256 37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684 \
  --f1-execution-commit <F1_EXECUTION_COMMIT> \
  --output-dir reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_<F1_EXECUTION_COMMIT>
```

The wrapper writes exactly these regeneration artifacts inside that output directory:

| artifact | producer | consumer |
| --- | --- | --- |
| `controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl` | deterministic wrapper after calling `build_controlled_records_with_f1_polarity_repair_audit(pair_count, authorized_F1_row_ids)` and projecting the replay to baseline ID sequence | analyzer `--repaired-jsonl`; dedicated P3-W6-F1 repaired Stage185 materializer source rows |
| `p3w6f1_deterministic_generator_invocation.json` | future wrapper | analyzer `--deterministic-generator-invocation-json` |
| `p3w6f1_generator_configuration_identity.json` | future wrapper | analyzer `--generator-configuration-identity-json` |
| `p3w6f1_regeneration_execution_manifest.json` | future wrapper | later static/provenance review; source of `--f1-output-sha256` |

The wrapper must emit `p3w6f1_deterministic_generator_invocation.json` with this canonical schema, matching the analyzer's `actual_repaired_generator_replay` identity object:

```json
{
  "authorized_F1_row_ids_sha256": "<canonical_sha256(sorted_authorized_F1_row_ids)>",
  "baseline_id_sequence_sha256": "<id_sequence_sha256(baseline_id_sequence)>",
  "pair_count": 300,
  "projection_policy": "baseline_id_sequence",
  "repair_api": "build_controlled_records_with_f1_polarity_repair_audit",
  "repair_mode": "f1_authorized_polarity_negative_only"
}
```

The wrapper must emit `p3w6f1_generator_configuration_identity.json` with this canonical schema, matching the analyzer's `actual_repaired_generator_replay` configuration object:

```json
{
  "authorized_F1_row_count": 121,
  "baseline_id_sequence_sha256": "<id_sequence_sha256(baseline_id_sequence)>",
  "baseline_topology_row_count": 3600,
  "generator_source_path": "scripts/build_controlled_v5.py",
  "pair_count": 300,
  "structural_negative_polarity_flip_row_count": 150
}
```

The wrapper execution manifest must pin at minimum:

```text
F1_execution_commit
baseline_input_path
baseline_input_sha256
repaired_output_path
repaired_output_sha256
repaired_generator_source_path = scripts/build_controlled_v5.py
repaired_generator_source_sha256
base_form_authority_commit = 11102ea05b28f6638fdead205b4a9ee0f35ca0de
base_form_authority_path = scripts/build_controlled_v5.py
base_form_authority_symbol = _BASE_PREDICATE_BY_INFLECTED
base_form_authority_sha256 = 37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684
deterministic_generator_invocation_json
generator_configuration_identity_json
```

The historical Stage185-v1 builder command formerly specified here is not executable for repaired data under the current repository-backed builder contract. The builder requires `--data` to resolve to `data/controlled_v5_v3_without_time_swap.jsonl`, requires the baseline SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`, reconstructs exact ordinary baseline generator rows, and enforces historical Stage182 overlap expectations. The repaired JSONL SHA-256 is `d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833` and intentionally differs in 121 authorized F1 evidence rows, so raw historical Stage185-v1 must remain immutable and must not be used as the repaired sidecar producer.

The replacement repaired Stage185 sidecar directory is:

```text
reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<MATERIALIZER_EXECUTION_COMMIT>/
```

The repaired sidecar path supplied to the analyzer is:

```text
reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<MATERIALIZER_EXECUTION_COMMIT>/stage185a_controlled_train_integrity_sidecar.jsonl
```

The exact historical builder command below is superseded for repaired data and must not be executed:

```text
python scripts/build_stage185a_controlled_train_integrity_sidecar.py \
  --repo-root . \
  --data reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_<F1_EXECUTION_COMMIT>/controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl \
  --stage184a-dir reports/stage184a_controlled_train_integrity_mask_spec_20260715_134538 \
  --stage182a-dir reports/stage182a_controlled_intervention_integrity_20260713_113314 \
  --output-dir reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<F1_EXECUTION_COMMIT> \
  --generator-source scripts/build_controlled_v5.py \
  --stage182a-analyzer-source scripts/analyze_stage182a_controlled_intervention_integrity.py \
  --split-seed 174 \
  --dev-ratio 0.2 \
  --rule-version stage185a_v1
```

The approved replacement producer is `scripts/materialize_reason_router_p3w6f1_repaired_stage185_sidecar.py`. It must reuse Stage185 contract functions and the analyzer's semantic derivation helpers, consume the repaired JSONL above plus the repaired generator replay rows, preserve the 2880/720 split, write the same JSONL schema required by analyzer `--repaired-sidecar-jsonl`, and leave historical Stage185-v1 constants, fail-closed checks, sidecar rows, SHA values, and Stage182 regression expectations unchanged. Expected split accounting remains:

```text
train rows = 2880
dev rows = 720
split seed = 174
dev ratio = 0.2
```

The exact reserved future materializer source path is:

```text
scripts/materialize_reason_router_p3w6f1_repaired_stage185_sidecar.py
```

This file is not created by this specification repair. The exact reserved future test path is:

```text
tests/test_reason_router_p3w6f1_repaired_stage185_sidecar_materializer.py
```

This test file is not created by this specification repair.

The exact active future materializer command contract is:

```text
python scripts/materialize_reason_router_p3w6f1_repaired_stage185_sidecar.py \
  --repo-root . \
  --repaired-jsonl reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl \
  --repaired-jsonl-sha256 d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833 \
  --regeneration-execution-manifest-json reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/p3w6f1_regeneration_execution_manifest.json \
  --deterministic-generator-invocation-json reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/p3w6f1_deterministic_generator_invocation.json \
  --generator-configuration-identity-json reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/p3w6f1_generator_configuration_identity.json \
  --p3w4-summary-json reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json \
  --p3w4-pairs-jsonl reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl \
  --p3w5-manifest-json reports/reason_router_p2_p3w5_separate_remediation_manifest.json \
  --f1-execution-commit dc8179e45f7c10416026acdadcbe5cbd8a78d37e \
  --materializer-execution-commit <MATERIALIZER_EXECUTION_COMMIT> \
  --generator-source scripts/build_controlled_v5.py \
  --stage184-contract-matrix reports/stage184a_controlled_train_integrity_mask_spec_20260715_134538/stage184a_family_contract_matrix.csv \
  --stage185-builder-source scripts/build_stage185a_controlled_train_integrity_sidecar.py \
  --split-seed 174 \
  --dev-ratio 0.2 \
  --rule-version stage185a_v1 \
  --output-dir reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<MATERIALIZER_EXECUTION_COMMIT>
```

No other materializer CLI inputs are authorized. In particular, the materializer must not accept free-form row IDs, alternate repair modes, implicit latest/current discovery, `--dataset-version`, or a historical Stage182 directory input.

Frozen materializer input identities:

```text
F1_EXECUTION_COMMIT = dc8179e45f7c10416026acdadcbe5cbd8a78d37e
repaired_jsonl_sha256 = d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833
authorized_F1_row_ids_sha256 = 386c0c5d5ed80e699f1607f94c2f8ba2861fa0cb1216d5d421f66e62c03d8c64
baseline_id_sequence_sha256 = 898070cd6718f9c677ba68442ee8ed9010200363df01d147528779306917c0eb
repaired_generator_source_sha256 = 37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684
stage184_contract_matrix_sha256 = 4287bf1ca7f1f2b08e5de53d24ad4019ca5ddff8a16db2dbb65727a5189e96fa
historical_stage185_builder_source_sha256 = 11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc
```

Corrective portability repair note:

```text
Runtime execution on Kaggle established that the previous Stage184 freeze
e5f61ac8d0ca3de3dd43767b83bec8c2c171a1635d419466c98d8d32ec2f38e5
identified a Windows CRLF worktree rendering of the tracked contract matrix, not
the repository Git-object raw bytes. The canonical authority for
reports/stage184a_controlled_train_integrity_mask_spec_20260715_134538/stage184a_family_contract_matrix.csv
is now the raw Git-object SHA256
4287bf1ca7f1f2b08e5de53d24ad4019ca5ddff8a16db2dbb65727a5189e96fa.
The old e5f61... value is historical diagnostic evidence only and must not be
accepted by executable authority validation. This repair changes no Stage184 CSV
content, no Stage184 semantics, no F1 authorization or repaired JSONL semantics,
no Stage185 semantics, and no analyzer semantics.
```

`F1_EXECUTION_COMMIT` identifies the already completed regeneration, generator replay authority, repaired JSONL production, analyzer `--f1-execution-commit`, and analyzer `--repaired-generator-commit`. `MATERIALIZER_EXECUTION_COMMIT` is a future full 40-hex commit created only after this specification, the materializer implementation, and focused tests are approved and committed. The materializer runtime must require `HEAD == MATERIALIZER_EXECUTION_COMMIT` and a clean tracked worktree. It must separately require `--f1-execution-commit == dc8179e45f7c10416026acdadcbe5cbd8a78d37e`; it must not require `HEAD == F1_EXECUTION_COMMIT`.

The reviewed materializer commit `90235a339714e40aff6ee0ead2256173891df685`
is now classified as `IMPLEMENTATION_REVIEWED_BUT_RUNTIME_PORTABILITY_BLOCKED`.
It is not the final executable materializer authority. After this corrective
repair is reviewed and committed, a new full commit becomes
`MATERIALIZER_EXECUTION_COMMIT`; `F1_EXECUTION_COMMIT` remains
`dc8179e45f7c10416026acdadcbe5cbd8a78d37e`.

The materializer writes exactly two final artifacts:

```text
reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<MATERIALIZER_EXECUTION_COMMIT>/stage185a_controlled_train_integrity_sidecar.jsonl
reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<MATERIALIZER_EXECUTION_COMMIT>/p3w6f1_repaired_stage185_materialization_manifest.json
```

No historical Stage185 report bundle and no historical Stage182 overlap-regression report are synthesized. The unique producer of analyzer `--repaired-sidecar-jsonl` is `scripts/materialize_reason_router_p3w6f1_repaired_stage185_sidecar.py`, and the unique artifact is the JSONL path above.

Deterministic output policy:

```text
all validation and semantic derivation complete before final artifacts appear
JSON/JSONL are UTF-8 without BOM
stable key order
final newline required
source row order preserved exactly
write temporary files first, then atomically replace into the final names
if neither final artifact exists, materialization may proceed
if both final artifacts already exist and byte-for-byte match newly derived outputs, return deterministic idempotent success
if either final artifact conflicts, is missing while the other exists, or contains different bytes, fail closed
never overwrite a conflicting artifact set
emit no accepted partial artifact set
```

The materializer must derive authorized F1 row IDs from the frozen P3-W4/P3-W5 authority chain, not from user-provided row IDs, using `extract_decision_supporting_pair_ids` and `extract_authorized_f1_targets`. It must require exactly 121 unique authorized row IDs, no F2 membership, and SHA-256 `386c0c5d5ed80e699f1607f94c2f8ba2861fa0cb1216d5d421f66e62c03d8c64`.

The materializer must use the repaired replay dependency, not ordinary `build_controlled_records(...)`, by calling `actual_repaired_generator_replay(...)`. The resulting `expected_generator_rows` are the projected repaired replay rows. The materializer must verify them against `p3w6f1_deterministic_generator_invocation.json` and `p3w6f1_generator_configuration_identity.json`, including pair count 300, authorized F1 row count 121, structural negative polarity_flip row count 150, baseline topology row count 3600, and baseline ID-sequence SHA-256 `898070cd6718f9c677ba68442ee8ed9010200363df01d147528779306917c0eb`.

Exact semantic helper dependency graph:

```text
frozen P3-W4/P3-W5 authority
-> extract_decision_supporting_pair_ids
-> extract_authorized_f1_targets
-> actual_repaired_generator_replay
-> projected repaired expected_generator_rows
-> generator_expected_rows_and_facts_for_source
-> load_authoritative_stage185_contracts
-> derive_stage185_dev_ids
-> historical stage185_builder.build_sidecar
-> candidate repaired sidecar
-> validate_stage185_sidecar_provenance
-> materialize exact output bytes
```

The materializer may import those pure deterministic helpers from `scripts/analyze_reason_router_p3w6f1_deterministic_polarity_regeneration.py` and `build_sidecar` from `scripts/build_stage185a_controlled_train_integrity_sidecar.py`. It must not invoke analyzer `main`, analyzer final-decision logic, analyzer output artifacts, or result-review artifacts. It must not reimplement Stage185 grammar classification, changed-axis logic, label matching, eligibility logic, split semantics, or F1 repair semantics.

For each repaired sidecar row, provenance fields are frozen as:

```text
source_dataset_path = reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl
source_dataset_sha256 = d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833
generator_source_path = scripts/build_controlled_v5.py
generator_source_sha256 = 37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684
stage184a_report_sha256 = ""
stage182a_report_sha256 = ""
integrity_builder_sha256 = 11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc
created_at = ""
```

The empty Stage182/Stage184 report SHA fields and empty `created_at` are intentional deterministic representations matching current analyzer semantic derivation behavior: the fields must exist for schema compatibility, but current repaired-sidecar provenance validation does not use them for semantic identity. `created_at` is ignored by semantic sidecar identity and must remain the empty string for deterministic materialization.

The dedicated materializer does not execute `scripts/build_stage185a_controlled_train_integrity_sidecar.py` as a binary against repaired input. `integrity_builder_sha256` identifies the historical Stage185 semantic-contract source used by imported pure functions; it does not claim historical binary execution. `historical_stage185_binary_executed` in the materialization manifest must be `false`. The materializer must not write `NOT_RUN` into raw Stage185 binary observation fields. Current analyzer compatibility accepts materializer sidecars through provenance and signature validation when raw fields contain concrete `PASS`/`FAIL` values and match the expected repaired raw signature.

Expected authorized-F1 repaired raw Stage185 fields remain:

```text
grammar_status = PASS
intervention_contract_status = FAIL
integrity_status = INELIGIBLE
canonical_status = PASS
polarity_contamination_status = PASS
dataset_source_status = PASS
schema_status = PASS
time_swap_status = PASS
audit_expected_axes = ["polarity"]
audit_changed_axes = ["polarity", "predicate"]
```

The materializer must not convert this raw state into compatibility PASS; compatibility remains a later analyzer concern.

The materialization manifest is a deterministic JSON object with at least:

```text
schema_version = p3w6f1_repaired_stage185_materialization_manifest_v1
status = P3W6F1_REPAIRED_STAGE185_MATERIALIZATION_PASS
F1_execution_commit
materializer_execution_commit
materializer_source_path
materializer_source_sha256
repaired_source_path
repaired_source_sha256
repaired_source_row_count
repaired_source_id_sequence_sha256
authorized_F1_row_count
authorized_F1_row_ids_sha256
repaired_generator_source_path
repaired_generator_source_sha256
regeneration_execution_manifest_path
regeneration_execution_manifest_sha256
deterministic_generator_invocation_path
deterministic_generator_invocation_sha256
generator_configuration_identity_path
generator_configuration_identity_sha256
stage184_contract_matrix_path
stage184_contract_matrix_sha256
historical_stage185_builder_source_path
historical_stage185_builder_source_sha256
historical_stage185_binary_executed = false
split_seed
dev_ratio
rule_version
train_row_count
dev_row_count
total_row_count
sidecar_path
sidecar_sha256
sidecar_semantic_sha256
provenance_validation_status = PASS
```

The future focused test module `tests/test_reason_router_p3w6f1_repaired_stage185_sidecar_materializer.py` must cover at least: exact CLI contract; direct-script importability if repository-package imports are used; invalid materializer commit format; HEAD/materializer commit mismatch; tracked dirty worktree rejection; repaired JSONL path mismatch; repaired JSONL SHA mismatch; regeneration manifest identity mismatch; invocation identity mismatch; configuration identity mismatch; wrong F1 execution commit; malformed/missing/duplicate/wrong-count authorized F1 IDs; F2 inclusion; wrong repaired generator replay; replay-consumption mismatch; generator source SHA mismatch; Stage184 matrix path/SHA mismatch; historical Stage185 builder source path/SHA mismatch; split/accounting mismatch; sidecar schema mismatch; source path/SHA provenance spoof; historical baseline SHA spoof; `historical_stage185_binary_executed` remains false; non-circular dependency with no analyzer output consumed; expected repaired raw Stage185 signature; `validate_stage185_sidecar_provenance` accepts a valid synthetic artifact; `validate_stage185_sidecar_provenance` rejects a tampered sidecar; exact deterministic output names; conflicting existing artifact rejection; partial-existing artifact rejection; deterministic idempotent identical-output success; exact 3600/2880/720 accounting.

Analyzer output directory:

```text
reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/analyzer/
```

Complete analyzer argument producer map:

| analyzer argument | exact upstream producer | exact artifact or authority | when value becomes known |
| --- | --- | --- | --- |
| `--p3w4-summary-json` | frozen P3-W4 canonical grammar authority | `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json` | before execution |
| `--p3w4-pairs-jsonl` | frozen P3-W4 canonical grammar authority | `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl` | before execution |
| `--p3w5-manifest-json` | frozen P3-W5 authority at commit `01d983f8d09cacf0eddefd2014fc81a28771cf5e` | `reports/reason_router_p2_p3w5_separate_remediation_manifest.json` | before execution |
| `--baseline-jsonl` | frozen baseline dataset authority | `data/controlled_v5_v3_without_time_swap.jsonl` | before execution |
| `--repaired-jsonl` | completed deterministic regeneration | `reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl` | known now |
| `--baseline-sidecar-jsonl` | frozen historical Stage185-v1 baseline sidecar | `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl` | before execution |
| `--repaired-sidecar-jsonl` | `scripts/materialize_reason_router_p3w6f1_repaired_stage185_sidecar.py` | `reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_<MATERIALIZER_EXECUTION_COMMIT>/stage185a_controlled_train_integrity_sidecar.jsonl` | after repaired sidecar materialization |
| `--baseline-generator-commit` | frozen baseline provenance authority | historical generator commit recorded by the baseline Stage185/P3-W4 provenance for `scripts/build_controlled_v5.py` | before execution |
| `--baseline-generator-source-path` | frozen baseline provenance authority | `scripts/build_controlled_v5.py` | before execution |
| `--baseline-generator-source-sha256` | frozen baseline Stage185/P3-W4 provenance | `c41e6a52401bd8c83970286b176950fc751509bee6d797d5da9aea4262c72802` | before execution |
| `--repaired-generator-commit` | completed F1 execution authority | `dc8179e45f7c10416026acdadcbe5cbd8a78d37e` | known now |
| `--repaired-generator-source-path` | analyzer generator-source identity contract | `scripts/build_controlled_v5.py` | known now |
| `--repaired-generator-source-sha256` | Git raw-file SHA256 of `dc8179e45f7c10416026acdadcbe5cbd8a78d37e:scripts/build_controlled_v5.py` | `37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684` | known now |
| `--deterministic-generator-invocation-json` | completed deterministic regeneration | `reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/p3w6f1_deterministic_generator_invocation.json` | known now |
| `--generator-configuration-identity-json` | completed deterministic regeneration | `reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/p3w6f1_generator_configuration_identity.json` | known now |
| `--f1-input-sha256` | frozen baseline dataset authority and wrapper manifest | `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640` | before execution |
| `--f1-execution-commit` | completed F1 execution authority | `dc8179e45f7c10416026acdadcbe5cbd8a78d37e` | known now |
| `--f1-output-sha256` | completed deterministic regeneration | `d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833` | known now |
| `--output-dir` | deterministic analyzer output contract | `reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/analyzer/` | known now |

Active continuation ordering is exactly:

1. final static review PASS
2. successful deterministic regeneration
3. materializer specification repair
4. independent materializer specification rereview
5. dedicated materializer implementation + focused tests
6. independent materializer implementation review
7. commit exact approved spec/materializer/test candidate
8. resolve `MATERIALIZER_EXECUTION_COMMIT` from new `HEAD`
9. Kaggle exact-commit sync
10. materializer runtime tests
11. reconfirm successful F1 regeneration artifacts by path/SHA
12. execute dedicated repaired Stage185 materializer
13. independent sidecar accounting/provenance verification
14. execute existing 19-argument P3-W6-F1 analyzer
15. result/audit review

No F1 regeneration rerun is required merely because the materializer is added.

Specification-level blockers resolved by this section:

```text
REGENERATION_PROCEDURE_UNSPECIFIED
REPAIRED_STAGE185_INVOCATION_UNSPECIFIED
ANALYZER_UPSTREAM_IDENTITY_UNSPECIFIED
BASE_FORM_AUTHORITY_SHA256_UNPINNED
```

Execution/implementation blockers intentionally remain unresolved:

```text
REPAIRED_STAGE185_NOT_EXECUTED
ANALYZER_NOT_EXECUTED
RESULT_REVIEW_NOT_EXECUTED
```

`REPAIRED_STAGE185_EXECUTION_CONTRACT_INCOMPATIBLE` is specification-resolved by the exact dedicated materializer contract above. It is not a runtime PASS and must not be confused with `REPAIRED_STAGE185_NOT_EXECUTED`.

## Future Implementation Components

A. Production generator repair:

- Reuse `scripts/build_controlled_v5.py`.
- Repair only the `_build_records` `polarity_flip` negative construction path at the source; generic `_statement(..., negative=True)` repair is forbidden.
- Add only the minimal deterministic base-form authority needed for generator-known predicates.

B. F1 authority target extractor:

- May be a new small utility or analyzer helper.
- Reads P3-W4 pair artifact and P3-W5 manifest/spec authority.
- Emits exactly the 121 authorized `source_row.id` values.
- Must not expand into F2.

C. Full-output isolation / semantic audit analyzer:

- New analyzer is likely needed because P3-W5/P3-W6 require complete-output diff, semantic-span proof, and artifact schemas.
- It may call existing production generator and Stage185 integrity logic.
- It must not reimplement production generator behavior to manufacture PASS.

D. Tests:

- Add focused generator, guard, full-output, and semantic-status tests.
- Reuse existing schema/label validation where possible.
- Do not duplicate Stage185 grammar/integrity semantics when direct invocation is available.

## Future Tests Specification

Generator unit tests:

- authorized defect example becomes grammatical
- predicate is base form after `did not`
- negation preserved
- claim unchanged
- labels unchanged
- non-target construction unchanged

Guard tests:

- ordinary affirmative rows unchanged
- ordinary REFUTE rows unchanged unless explicitly authorized and isolated
- F2 rows unchanged
- canonical rows unchanged
- paraphrase rows unchanged
- time_swap not admitted

Full-output tests:

- same row count
- same ID set
- same ID sequence
- exactly 121 changed IDs
- all changed IDs authorized
- only evidence changes

Semantic-status tests:

- valid deterministic repair -> `DETERMINISTIC_POLARITY_REPAIR_PASS`
- ambiguous/non-unique repair -> `MANUAL_REVIEW_REQUIRED`
- explicit semantic contradiction -> `REJECTED`

## Future Artifact Contract

Future execution may use the `p3w6f1_*` namespace, but it must preserve and include all P3-W5 v5 accounting and required record fields. Nested `baseline_source_row` or `regenerated_source_row` objects may be present, but they do not replace the explicit authority fields below.

Future execution must generate at least:

- `p3w6f1_regeneration_summary.json`
- `p3w6f1_regenerated_rows.jsonl`
- `p3w6f1_regeneration_audit.jsonl`
- recommended: `p3w6f1_full_output_isolation.json`

`p3w6f1_regeneration_summary.json` must include count and pair-ID array fields:

- `F1_target_pair_count`
- `F1_target_pair_ids`
- `F1_generated_candidate_count`
- `F1_generated_candidate_pair_ids`
- `F1_accepted_candidate_count`
- `F1_accepted_candidate_pair_ids`
- `F1_manual_review_required_count`
- `F1_manual_review_required_pair_ids`
- `F1_rejected_candidate_count`
- `F1_rejected_candidate_pair_ids`
- `F1_missing_candidate_count`
- `F1_missing_candidate_pair_ids`
- `F1_unauthorized_candidate_count`
- `F1_unauthorized_candidate_pair_ids`

Each count must equal the length of its array. Accepted, manual-review-required, and rejected arrays must be pairwise disjoint, and their union must equal generated authorized candidate pair IDs.

Generated candidate accounting:

```text
generated authorized candidate rows =
authorized_F1_row_ids that are present in the regenerated candidate output
```

or an equivalent exact authority-grounded representation. `F1_generated_candidate_pair_ids` must represent generated authorized F1 candidates, not the complete structural-negative `polarity_flip` universe. The 150-row structural universe is diagnostic/scope provenance, not `F1_generated_candidate_count`.

`ALL_ACCEPTED` still requires:

```text
F1_target_pair_count == 121
F1_generated_candidate_count == 121
F1_accepted_candidate_count == 121
```

The summary must also preserve execution namespace fields:

- `F1_execution_status`
- `F1_execution_decision`
- `F1_artifact_paths`
- `F1_input_sha256`
- `F1_execution_commit`
- `F1_output_sha256`

Execution provenance hardening:

```text
F1_input_sha256 == actual baseline JSONL sha256
F1_output_sha256 == actual repaired JSONL sha256
```

Non-empty strings are not sufficient provenance validation. Repaired generator source identity must be verified against the actual repaired generator source. Baseline generator commit/path/SHA must be tied to the baseline generator authority, preferably through repository/Git blob identity or an equivalent deterministic source snapshot proof. Arbitrary caller-supplied non-empty SHA/commit strings must not create execution PASS.

Fail closed token:

```text
PROVENANCE_IDENTITY_MISMATCH
```

`p3w6f1_regenerated_rows.jsonl` and `p3w6f1_regeneration_audit.jsonl` records must explicitly include at minimum:

- `pair_id`
- `original_row_id`
- `regenerated_row_id`
- `intervention_type`
- `original_text`
- `regenerated_text`
- `original_final_label`
- `regenerated_final_label`
- `canonical_row_id`
- `generator_source_path`
- `generator_source_sha256`
- `generator_commit`
- `fact_identity`
- `grammar_validator_source`
- `grammar_validator_sha256`
- `grammar_before`
- `grammar_after`
- `sidecar_before`
- `sidecar_after`
- `lineage_preserved`
- `semantic_validation_status`
- `semantic_polarity_preserved`
- `candidate_accepted`
- `ordered_rejection_codes`
- `inflected_predicate_surface`
- `expected_base_predicate`
- `base_form_derivation_method`
- `base_form_derivation_source_path`
- `base_form_derivation_source_sha256`
- `base_form_source_symbol`
- `authorized_replacement_span`
- `outside_span_byte_identity`

Semantic transformation evidence must include:

- `canonical_text`
- `original_defective_text`
- `regenerated_text`
- `normalized_changed_span`
- `negation_markers_added`
- `negation_markers_removed`
- `auxiliary_verb_changes`
- `predicate_inflection_changes`
- `duplicate_or_missing_tokens`
- `semantic_validation_method`
- `semantic_validation_evidence`

`p3w6f1_full_output_isolation.json` and the summary isolation block must include generator identity:

- `baseline_generator_commit`
- `baseline_generator_source_path`
- `baseline_generator_source_sha256`
- `repaired_generator_commit`
- `repaired_generator_source_path`
- `repaired_generator_source_sha256`
- `deterministic_generator_invocation`
- `generator_configuration_identity`
- `baseline_complete_output_sha256`
- `repaired_complete_output_sha256`
- `baseline_row_count`
- `repaired_row_count`
- `baseline_id_sequence`
- `repaired_id_sequence`
- `baseline_id_sequence_sha256`
- `repaired_id_sequence_sha256`
- `row_order_changed`

The full-output isolation artifact must also preserve existing diff arrays:

- `changed_ids`
- `structural_negative_polarity_flip_row_ids`
- `authorized_F1_row_ids`
- `non_authorized_structural_negative_polarity_flip_row_ids`
- `repair_consumed_row_ids`
- `non_authorized_structural_negative_polarity_flip_changed_row_ids`
- `evidence_changed_row_ids`
- `claim_changed_row_ids`
- `non_text_field_changed_row_ids`
- `missing_ids`
- `added_ids`
- `duplicate_ids`
- `unauthorized_changed_row_ids`
- `F2_changed_row_ids`
- `unaffected_changed_row_ids`
- `canonical_changed_row_ids`
- `paraphrase_changed_row_ids`

Required isolation results:

```text
repair_consumed_row_ids == authorized_F1_row_ids
non_authorized_structural_negative_polarity_flip_changed_row_ids == []
changed_ids == authorized_F1_row_ids
evidence_changed_row_ids == authorized_F1_row_ids
```

## Stage185 Provenance Contract

Because Stage185 state is an acceptance condition, externally supplied baseline/repaired sidecars must not be trusted by field values alone. Future implementation must verify at minimum:

```text
exact one-to-one row_id join with corresponding source dataset
sidecar pair_id == source row pair_id
canonical_row_id lineage is valid
source_dataset_sha256 matches the dataset used to build that sidecar
integrity_builder_sha256 matches the authorized Stage185 builder source
required Stage185 schema fields exist
```

If the existing production Stage185 semantic sidecar hash machinery is reusable, future implementation should prefer reuse over reimplementation. Synthetic unit-test sidecars are allowed only for unit tests; execution PASS must use provenance-validated Stage185 artifacts.

Fail closed token:

```text
STAGE185_PROVENANCE_UNRESOLVED
```

## Future Decisions

Future F1 execution decision must be one of:

- `P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW`
- `P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW`

Future result-review decision must be one of:

- `P3W5_F1_RESULT_REVIEW_PASS_ALL_ACCEPTED`
- `P3W5_F1_AUDIT_ARTIFACT_REVIEW_PASS_WITH_BLOCKERS`
- `P3W5_F1_RESULT_REVIEW_BLOCKED`

This P3-W6-F1 specification asserts none of those result-review decisions.

## F2 Isolation

F2 is completely deferred. This specification does not perform F2 manual review, design F2 regeneration expansion, mutate F2 source rows, change F2 decisions, or authorize automatic F1 repair application to F2. If shared generator code is repaired, future full-output isolation must prove:

```text
F2_changed_row_ids == []
```

## Release State

Current post-regeneration audit state:

```text
implementation_executed = true
generator_executed = true
regeneration_executed = true
analyzer_executed = false
tests_executed = false
manual_review_executed = false
training_executed = false
evaluation_executed = false

F1 regeneration executed = true
F1 candidate authority established = false
production dataset repair authority not established
training admission authority not established
polarity supervision released = false
A1/A2/A3 released = false
F2 review executed = false
```

Post-regeneration execution-gap audit state:

```text
F1_EXECUTION_COMMIT = dc8179e45f7c10416026acdadcbe5cbd8a78d37e
repaired_jsonl = reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_dc8179e45f7c10416026acdadcbe5cbd8a78d37e/controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl
repaired_jsonl_sha256 = d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833
TOTAL_ROWS = 3600
CHANGED_ROWS = 121
authorized_F1_row_ids_sha256 = 386c0c5d5ed80e699f1607f94c2f8ba2861fa0cb1216d5d421f66e62c03d8c64
baseline_id_sequence_sha256 = 898070cd6718f9c677ba68442ee8ed9010200363df01d147528779306917c0eb

GENERATOR_REPAIR_IMPLEMENTED_AND_VALIDATED
REGENERATION_EXECUTED
REPAIRED_STAGE185_NOT_EXECUTED
ANALYZER_NOT_EXECUTED
RESULT_REVIEW_NOT_EXECUTED
```

## Static Blockers

- `P3W6F1_IMPLEMENTATION_SPEC_NOT_STATICALLY_REVIEWED`
- `P3W6F1_REPAIRED_STAGE185_NOT_EXECUTED`
- `P3W6F1_ANALYZER_NOT_EXECUTED`
- `P3W6F1_RESULT_REVIEW_NOT_EXECUTED`
- `P3W5_F2_MANUAL_REVIEW_NOT_EXECUTED`
- `P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY`
- `A1_A3_NOT_RELEASED`

## Forbidden Claims

This specification does not claim:

- `P3W6F1_IMPLEMENTATION_COMPLETE`
- `P3W6F1_EXECUTION_PASS`
- `F1_REPAIR_COMPLETE`
- `F1_CANDIDATE_AUTHORITY_ESTABLISHED`
- `PRODUCTION_DATASET_REPAIRED`
- `TRAINING_ADMISSION_READY`
- `POLARITY_SUPERVISION_RELEASED`
- `F2_REVIEW_COMPLETE`
- `A1_READY`
- `A2_READY`
- `A3_READY`
- `P3_PASS`
