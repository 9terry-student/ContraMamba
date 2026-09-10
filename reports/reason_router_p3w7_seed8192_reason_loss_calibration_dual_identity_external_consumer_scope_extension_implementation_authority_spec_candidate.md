# Seed8192 Dual-Identity External-Consumer Scope-Extension Implementation Authority Candidate

## Verdict and governing authority

```text
verdict = PASS_READY_FOR_INDEPENDENT_EXTERNAL_CONSUMER_SCOPE_EXTENSION_VERIFICATION
phase = REPORT-ONLY / SCOPE-EXTENSION AUTHORITY DESIGN
current_HEAD = 184723b680cd011493803d7fdc2f024f0f0e3280
frozen_original_dual_identity_authority_blob = fb35df48bf09a25627fb6d6e3e186160ce4597dd
```

This is a narrow implementation-whitelist correction for one verified external
tracked production consumer.  It supplements, and does not replace, the
frozen dual-identity schema-recovery authority.  The current uncommitted
three-file v2 implementation remains otherwise valid and is preserved.

The root cause is a compatibility/scope omission, not a defect in the
dual-identity schema: the schema-v2 trainer helper renamed the existing P3-W1
row+pair+normalized-label semantic identity from the overloaded v1 return key
`ordered_train_row_identity_hash` to the explicit return key
`p3w1_ordered_train_row_label_sha256`.  The v2 schema must remain fail-closed;
the v1 name must not be restored as an alias or fallback.

## Sole external consumer and exact bounded repair

Static audit of tracked, non-report source finds exactly one production-script
consumer of the removed helper return key:

```text
scripts/analyze_reason_router_p3w3_polarity_authority.py
```

Its sole relevant wrapper is `ordered_train_identity_hash(records)`.  It uses
the trainer helper only to obtain the historical P3-W1 ordered
row+pair+normalized-label identity.  The current broken expression is exactly:

```python
return trainer._p3w1_ordered_train_identity(records)[
    "ordered_train_row_identity_hash"
]
```

The only newly authorized production change is the exact semantic-preserving
key transition:

```python
return trainer._p3w1_ordered_train_identity(records)[
    "p3w1_ordered_train_row_label_sha256"
]
```

Equivalent formatting is permitted; no other behavioral change is.  The
wrapper continues to mean the historical P3-W1 row+pair+normalized-label
serialization, explicitly **not** P4-X.  It must fail closed when the v2 key
is absent.  Do not introduce an alternate identity algorithm, P4-X identity,
v1 alias, or any `.get(...)` fallback.

## Exact later implementation whitelist

After this extension, the complete and exclusive implementation set is:

1. `scripts/train_controlled_v6b_minimal.py`
2. `scripts/aggregate_reason_router_p3w1_calibration.py`
3. `tests/test_reason_router_p3w1_calibration.py`
4. `scripts/analyze_reason_router_p3w3_polarity_authority.py`

No fifth source, test, or implementation file is authorized.  In particular,
`tests/test_reason_router_p3w3_polarity_authority.py` is not authorized for
modification and static inspection found no need to modify it: its existing
coverage exercises the wrapper through split-contract validation and has no
dependency on preserving the removed v1 field name.

## P3-W3 semantic invariance

The compatibility repair changes none of the P3-W3 authority, data, split,
eligibility, exclusion, counterfactual, canonical-lineage, sidecar,
remediation, production-supervision-audit, decision/claim, or execution
isolation semantics.  The following constants and their meanings are frozen:

```text
SCHEMA_VERSION = reason_router_p3w3_polarity_authority_audit_v3
EXPECTED_SPLIT_SEED = 174
EXPECTED_DEV_RATIO = 0.2
EXPECTED_TRAIN_ROWS = 2880
EXPECTED_TRAIN_IDENTITY = cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784
EXPECTED_DATA_SHA256 = f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640
EXPECTED_SIDECAR_SEMANTIC_SHA256 = 5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc
```

CPU-only static recomputation over the historical split174 ordered train
records returned 2,880 train records, 720 dev records, and:

```text
trainer._p3w1_ordered_train_identity(train_records)[
  "p3w1_ordered_train_row_label_sha256"
] = cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784
```

This equals `EXPECTED_TRAIN_IDENTITY`.  No model, tokenizer, CUDA, training,
evaluation, or calibration execution was used.  Later implementation must
repeat a direct CPU-only source-backed assertion that
`audit.ordered_train_identity_hash(authoritative_split174_train_records) ==
audit.EXPECTED_TRAIN_IDENTITY`, or its exact equivalent.

## Preserved dual-identity contracts

The existing three-file implementation remains untouched.  Its frozen v2
contracts remain:

```text
unit schema = reason_router_p3w1_calibration_unit_v2
aggregate schema = reason_router_p3w1_calibration_aggregate_v2
p4x_ordered_train_row_sha256 = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
p3w1_ordered_train_row_label_sha256 = 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
```

`ordered_train_row_identity_hash` must not be reintroduced as a v2 artifact
field, helper alias, or fallback.

## Required later validation and boundaries

After the later one-line compatibility implementation, independently run:

```text
python -m pytest tests/test_reason_router_p3w1_calibration.py -q
python -m pytest tests/test_reason_router_p3w3_polarity_authority.py -q
```

The P3-W3 suite must be used unchanged.  Run `git diff --check` and confirm
the fourth-file edit is only the specified dict-key transition.

This authority authorizes no calibration execution, training, evaluation,
model/tokenizer loading, CUDA, Kaggle work, seed180 retry3, seeds181/182,
aggregation execution, or scientific interpretation.  Historical retry2
remains provenance-invalid for corrected evidence.  The separate collector
blocker remains `COLLECTOR_START_MARKER_LIFECYCLE / PROVENANCE_DISCOVERY_DEFECT`;
the `FILES=0` ZIP remains forbidden and future Kaggle execution remains
blocked.  No code change to the fourth file is authorized until this candidate
is independently verified and manually frozen; thereafter this freeze and the
original dual-identity authority jointly govern the bounded implementation.

## This report-only task record

```text
report_created = true
existing_files_modified_by_this_task = false
existing_three_file_implementation_preserved = true
training_executed = false
evaluation_executed = false
calibration_executed = false
model_loaded = false
tokenizer_loaded = false
cuda_used = false
kaggle_used = false
staged = false
commit = false
push = false
```
