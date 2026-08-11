# P3-W6-F1 Stage185 Predicate-Realization Compatibility Audit

Decision: `P3W6F1_STAGE185_COMPATIBILITY_AUDIT_PASS_SPEC_REVISION_REQUIRED`

This is a report-only/static-audit artifact. It does not modify or supersede existing authority, does not authorize regeneration, does not authorize training, does not authorize evaluation, and does not establish P3-W6-F1 execution readiness.

## Executive Decision

The current P3-W6-F1 accepted repaired Stage185 post-state is not reachable under the currently pinned Stage185 v1 builder semantics when the approved deterministic F1 repair changes:

```text
did not <inflected predicate>
->
did not <generator-authorized base predicate>
```

The conflict is primarily:

```text
B. Stage185 representation defect for grammatical allomorph/base-form equivalence
```

with a P3-W6 integration/specification defect:

```text
C. P3-W6 currently requires a Stage185 v1 accepted post-state that the pinned Stage185 v1 semantic representation cannot derive for the approved repair.
```

It is not a generator repair defect. The generator repair is the approved English-realization correction for do-support negation. It is also not acceptable to change the generator to preserve the inflected predicate, because that would preserve the original P3-W4 F1 root cause.

Recommended minimal mechanism: Candidate B, a new versioned P3-W6 repaired-row semantic compatibility layer / Stage185-v2 authority, preserving historical Stage185 baseline sidecar identity and the trusted Stage185 v1 builder blob. The new layer must recognize only generator-proven authorized F1 predicate realization equivalence for the exact repair span.

## Authority Hierarchy

1. Current task instruction: report-only/static-audit stage; create only this report and its manifest.
2. P3-W6-F1 implementation authority: `11102ea05b28f6638fdead205b4a9ee0f35ca0de`.
3. Trusted Stage185 dependency commit: `ff6929bf33693fb4e70bd9528551053f4402fe1c`.
4. P3-W6-F1 specification authority files:
   - `reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_implementation_spec.md`
   - `reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_manifest.json`
5. P3-W5 authority:
   - `reports/reason_router_p2_p3w5_separate_remediation_spec.md`
   - `reports/reason_router_p2_p3w5_separate_remediation_manifest.json`
6. P3-W4 authority:
   - `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json`
   - `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl`
7. Repository contract: `AGENTS.md`.

## Exact Code-Path Trace

### Generator Predicate Realization

In `scripts/build_controlled_v5.py`, `_statement(...)` renders `values["predicate"]` directly into evidence. In the negative branch it prefixes the same surface:

```text
predicate = f"did not {predicate}"
```

The v4 implementation adds generator-owned base-form authority:

```text
_BASE_PREDICATE_BY_INFLECTED[inflected] = base
```

and consumes it only inside the authorized negative `polarity_flip` branch. This creates the approved repair:

```text
did not released -> did not release
did not opened -> did not open
did not approved -> did not approve
```

### Stage185 Semantic State

In `scripts/build_stage185a_controlled_train_integrity_sidecar.py`, `semantic_state(fact, evidence)` compares literal evidence containment against:

```text
fact["predicate"]
fact["alternate_predicate"]
```

For each slot axis, including `predicate`, it emits one of:

```text
original:<surface>
alternate:<surface>
ambiguous_both:<original>|<alternate>
absent
```

There is no allomorph/base-form equivalence layer. Therefore a repaired evidence string containing `did not release` does not contain the literal original predicate `released`.

### Stage185 Changed Axes and Contract

For non-content operations, `changed_axes(...)` computes:

```text
changed = {axis for axis in base if base[axis] != current[axis]}
```

The Stage184 family contract matrix states:

```text
polarity_flip intended_changed_axes = ["polarity"]
polarity_flip intended_preserved_axes includes "predicate"
```

Then `build_sidecar(...)` computes:

```text
unexpected = observed - intended
missing = intended - observed
contract_bad = bool(unexpected or missing or not labels_match(row, expected[row_id]))
intervention_contract_status = FAIL if contract_bad else PASS
integrity_status = INELIGIBLE if any criterion status is FAIL else ELIGIBLE
```

Thus a repaired row whose observed axes are:

```text
["polarity", "predicate"]
```

has:

```text
unexpected = {"predicate"}
intervention_contract_status = FAIL
reason_codes includes INTERVENTION_CONTRACT_FAIL
integrity_status = INELIGIBLE
```

even when:

```text
grammar_status = PASS
dataset_source_status = PASS
schema_status = PASS
canonical_status = PASS
time_swap_status = PASS
```

## Concrete Authorized Row Trace

Representative authorized F1 row:

```text
pair_id = archive_release
row_id = archive_release__polarity_flip
fact["predicate"] = released
fact["alternate_predicate"] = catalogued
generator base mapping = released -> release
```

Canonical evidence:

```text
Director Omar Haddad, the archivist, released the coastal records in Amman during January.
```

Defective polarity evidence:

```text
Director Omar Haddad, the archivist, did not released the coastal records in Amman during January.
```

Approved repaired polarity evidence:

```text
Director Omar Haddad, the archivist, did not release the coastal records in Amman during January.
```

Static Stage185 semantic trace:

| state | predicate axis | polarity axis |
|---|---|---|
| canonical | `original:released` | `positive` |
| defective polarity | `original:released` | `negative` |
| repaired polarity | `absent` | `negative` |

Baseline defective Stage185 sidecar state from P3-W4 authority:

```text
grammar_status = FAIL
intervention_contract_status = PASS
integrity_status = INELIGIBLE
audit_expected_axes = ["polarity"]
audit_changed_axes = ["polarity"]
reason_codes = ["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"]
```

Predicted repaired Stage185 v1 state:

```text
grammar_status = PASS
audit_expected_axes = ["polarity"]
audit_changed_axes = ["polarity", "predicate"]
intervention_contract_status = FAIL
reason_codes includes INTERVENTION_CONTRACT_FAIL
integrity_status = INELIGIBLE
```

This matches the observed focused pytest behavior:

```text
grammar_status == PASS
integrity_status == INELIGIBLE
```

## 121-Row Scope Implication

Static P3-W4 artifact inspection finds:

```text
F1_TRUE_POLARITY_GENERATION_DEFECT rows = 121
polarity_flip target rows = 121
unique authorized inflected predicate surfaces = 18
```

Authorized predicate surfaces and counts:

```text
approved:13
delivered:18
digitized:1
inspected:1
launched:15
mapped:1
opened:18
premiered:1
published:13
received:1
released:2
renovated:1
reopened:1
restored:16
selected:14
signed:2
upgraded:1
won:2
```

Each authorized row is a negative `polarity_flip` row whose P3-W4 grammar proof cites a `did not <inflected predicate>` matched span. The generator-owned base-form mapping converts these surfaces to different base forms. Under the pinned Stage185 literal predicate state, the same mechanism applies to all 121 authorized repaired rows: the repaired evidence no longer contains the literal inflected `fact["predicate"]`, so the predicate axis is observed as changed in addition to polarity.

## Root-Cause Classification

### A. Generator repair defect

Rejected as primary root cause.

The approved P3-W6 repair is explicitly to render the predicate governed by `did not` in generator-authorized base form. Keeping `did not <inflected>` would preserve the known P3-W4 F1 defect. The generator repair is correctly scoped to 121 authorized F1 rows and has generator-owned base-form provenance.

### B. Stage185 representation defect

Accepted as primary mechanism root cause.

Stage185 v1 conflates semantic predicate identity with literal surface realization. It treats:

```text
released
release
```

as different predicate states, even when the latter is the grammatically required do-support realization of the former and is generator-authorized for the exact F1 repair span.

### C. P3-W6 specification defect

Accepted as integration defect.

P3-W6 requires:

```text
grammar_status = PASS
intervention_contract_status = PASS
integrity_status = ELIGIBLE
audit_changed_axes = ["polarity"]
```

but also requires:

```text
did not <inflected predicate> -> did not <base predicate>
```

while pinning Stage185 v1 semantics at trusted commit `ff6929bf33693fb4e70bd9528551053f4402fe1c`. These requirements are jointly incompatible without a new compatibility authority.

### D. Some combination

The final classification is B plus C: a Stage185 v1 representation limitation exposed by a P3-W6 requirement that assumes Stage185 can distinguish semantic predicate identity from grammatical surface realization.

## Reachability of Current Required Post-State

Under the pinned Stage185 v1 builder, the current required repaired post-state is unreachable for the authorized base-form repair.

Reason:

```text
semantic_state(predicate) compares literal fact["predicate"] presence.
repaired evidence contains base predicate, not inflected fact["predicate"].
current predicate state becomes absent.
changed_axes includes predicate.
polarity_flip intended axes are only ["polarity"].
intervention_contract_status becomes FAIL.
integrity_status becomes INELIGIBLE.
```

The row can become grammatical, but it cannot simultaneously satisfy Stage185 v1 predicate preservation.

## Candidate Comparison

### Candidate A: Modify historical Stage185 builder

Description: Change `scripts/build_stage185a_controlled_train_integrity_sidecar.py` so `semantic_state` recognizes generator-authorized base/inflected predicate forms as equivalent.

Assessment: unsafe under current authority.

Risks:

- P3-W6 pins the trusted Stage185 dependency commit `ff6929bf33693fb4e70bd9528551053f4402fe1c`.
- The frozen baseline Stage185 sidecar semantic SHA must remain immutable.
- Changing historical Stage185 would change the meaning of baseline evidence and could invalidate Stage184/Stage185 provenance.
- A broad equivalence rule could weaken predicate-swap detection beyond the 121 authorized F1 repair rows.

Conclusion: not recommended without a new historical-stage migration authority, which is broader than this task.

### Candidate B: Versioned P3-W6 compatibility layer / Stage185-v2 authority

Description: Preserve historical Stage185 v1 and baseline sidecar identity, then add a P3-W6-specific repaired-row compatibility layer that treats only the generator-owned mapping:

```text
inflected_predicate_surface <-> expected_base_predicate
```

as the same semantic predicate identity for the exact authorized F1 repair span.

Assessment: recommended.

Required constraints:

- applies only to `authorized_F1_row_ids`;
- applies only to the byte-local replacement span;
- requires generator-owned `_BASE_PREDICATE_BY_INFLECTED` provenance;
- requires exact replay/full-output isolation;
- does not lemmatize arbitrary text;
- does not use an LLM or broad NLP heuristic;
- does not alter Stage185 v1 baseline sidecar identity;
- preserves predicate-swap detection outside the authorized realization equivalence.

Conclusion: minimal viable direction, but requires a P3-W6 spec revision before implementation readiness can be claimed.

### Candidate C: Relax P3-W6 required repaired state

Description: Permit:

```text
intervention_contract_status = FAIL
integrity_status = INELIGIBLE
audit_changed_axes includes predicate
```

Assessment: not recommended.

Risks:

- contradicts P3-W5/P3-W6 accepted post-state requirements;
- weakens predicate preservation and accepted Stage185 integrity;
- would make candidate acceptance tolerate the same Stage185 failure signal used for true predicate changes;
- blurs the distinction between successful grammatical realization and unresolved predicate semantics.

Conclusion: unacceptable as a final acceptance mechanism.

### Candidate D: Change generator repair to avoid base-form replacement

Description: Preserve `did not <inflected predicate>`.

Assessment: rejected.

Risks:

- preserves the exact P3-W4 F1 root cause;
- violates English `did not + base form`;
- contradicts the P3-W5/P3-W6 span-replacement contract;
- leaves `grammar_status = FAIL`.

Conclusion: not viable.

## Recommended Minimal Mechanism

Specify a versioned P3-W6 repaired-row compatibility authority:

```text
P3W6F1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1
```

It should sit after:

```text
exact repaired generator replay identity
full-output isolation
Stage185 v1 provenance reconstruction
```

and before final candidate acceptance.

It should reinterpret only the Stage185 v1 predicate-axis mismatch caused by the exact authorized replacement:

```text
did not <inflected_predicate_surface>
->
did not <expected_base_predicate>
```

as:

```text
semantic predicate identity preserved
surface grammatical realization changed
```

The mechanism must not alter the historical Stage185 sidecar or its semantic SHA. It should produce a new P3-W6 compatibility audit field rather than rewriting the Stage185 v1 evidence.

## Required Provenance Contract

Any future implementation of Candidate B must prove:

```text
authorized_F1_row_ids count == 121
row_id in authorized_F1_row_ids
intervention_type == polarity_flip
baseline evidence contains exactly one did not <inflected_predicate_surface>
repaired evidence contains exactly one did not <expected_base_predicate>
inflected_predicate_surface == fact["predicate"] from P3-W4 grammar proof
expected_base_predicate == _BASE_PREDICATE_BY_INFLECTED[inflected_predicate_surface]
base-form mapping source path == scripts/build_controlled_v5.py
generator replay identity PASS
repair_consumed_row_ids == authorized_F1_row_ids
full-output isolation PASS
all non-evidence source fields unchanged
evidence outside authorized replacement span byte-identical
labels unchanged
Stage185 v1 runtime dependency identity still pinned
historical baseline Stage185 sidecar semantic SHA unchanged
```

The compatibility output should record at minimum:

```text
compatibility_rule_id
compatibility_status
row_id
inflected_predicate_surface
expected_base_predicate
authorized_replacement_span
predicate_semantic_identity_preserved
surface_realization_changed
stage185_v1_observed_changed_axes
stage185_v1_intervention_contract_status
stage185_v1_integrity_status
```

## Second Failing Test Audit

The failing test:

```text
test_stage185_expected_generator_label_mismatch_fails_semantic_identity
```

expects Stage185 provenance to fail after tampering `expected_generator_rows[*].polarity_label`.

This expectation is not reliable under the current repaired-row conflict. Stage185 `labels_match(row, expected[row_id])` contributes to:

```text
contract_bad
intervention_contract_status
reason_codes
integrity_status
```

but P3-W6 `validate_stage185_sidecar_provenance` compares derived sidecar semantic identity fields, not raw expected row labels. If the repaired row is already contract-failing because the predicate axis is unexpected, then a label mismatch may produce the same derived coarse state:

```text
intervention_contract_status = FAIL
reason_codes includes INTERVENTION_CONTRACT_FAIL
integrity_status = INELIGIBLE
```

Therefore the observed sidecar and the expected sidecar derived from label-tampered expected rows can still match on the compared Stage185 semantic identity fields. This is downstream of the already-FAIL repaired intervention contract and also demonstrates that Stage185 provenance is not a raw expected-label identity oracle.

Raw repaired generator/output identity remains owned by:

```text
validate_repaired_output_replay_identity
full-output isolation
execution provenance
```

Stage185 should not be prescribed to own exact evidence replay identity that is already owned by those gates.

## Tests Needed Later

Future tests, after a specification revision, should include:

1. A unit test showing Stage185 v1 still reports the repaired authorized F1 row as `grammar_status = PASS` and `integrity_status = INELIGIBLE` with `audit_changed_axes` including `predicate`.
2. A compatibility-layer positive test for `archive_release__polarity_flip` proving `released -> release` is accepted only as generator-authorized predicate realization equivalence.
3. A 121-row aggregate test proving every authorized F1 row has exactly one approved inflected/base replacement span.
4. A negative test where the repaired predicate is a different base-form-like surface not present in `_BASE_PREDICATE_BY_INFLECTED`; must fail.
5. A negative test where the same replacement appears on a non-authorized structural polarity row; must fail.
6. A predicate-swap preservation test proving alternate predicate or unrelated predicate substitutions are still detected.
7. A full-output isolation test proving F2, canonical, paraphrase, row order, labels, and non-evidence fields remain unchanged.
8. A provenance test proving historical Stage185 v1 baseline sidecar semantic SHA remains frozen.
9. A test proving label mutation is still caught by full-output isolation / replay identity even if Stage185 derived status is already failed.

## Explicit Non-Authorizations

This audit does not authorize:

```text
production code modification
test modification
data modification
report supersession beyond these two new audit artifacts
checkpoint mutation
training
evaluation
regeneration
Kaggle execution
Git commit/push/pull
historical Stage185 builder modification
historical Stage185 baseline sidecar mutation
P3-W6-F1 execution PASS
F1 candidate authority establishment
production dataset repair
polarity supervision release
A1/A2/A3 release
F2 automatic repair
LLM semantic judgment
arbitrary morphology heuristic
generic lemmatization
weakening predicate-swap detection
```

## Final Status

```text
P3W6F1_STAGE185_COMPATIBILITY_AUDIT_PASS_SPEC_REVISION_REQUIRED
```
