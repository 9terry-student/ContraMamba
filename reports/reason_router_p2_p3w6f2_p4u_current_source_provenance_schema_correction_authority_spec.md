# P3-W6-F2-P4-U Current-Source Provenance Schema Correction Authority Candidate

Authority/version:

`P3W6F2P4U_CURRENT_SOURCE_PROVENANCE_SCHEMA_CORRECTION_AUTHORITY_CANDIDATE_V1`

This document is an authority-spec candidate only. It authorizes no Python
execution, pytest, runtime validation, Kaggle execution, training, evaluation,
artifact mutation, commit, push, or retroactive relabeling of historical P4-T.

## 1. Authority Basis

Creation authority:

- Current controller instruction.
- P4-T freeze: `89cc8ad374b1b9656e2a7333a4fa412916e007c9`.
- P4-T observed execution failure:
  `P4T_FAIL:PROVENANCE_SOURCE_PHYSICAL_SHA256_MISMATCH`.
- Frozen P4-L: `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- Exact builder: `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Frozen P4-S: `2faa789c35f7ff9258fb7b005a92890da17d04be`.
- `AGENTS.md`.

Phase:

`P4-U CURRENT-SOURCE PROVENANCE SCHEMA RECOVERY VALIDATION CORRECTION
AUTHORITY CANDIDATE CREATION ONLY`

## 2. Historical Disposition

P4-T remains FAIL. This candidate does not retroactively relabel P4-T PASS and
does not assert that the canonical P4-L artifact is valid.

The P4-T failure is classified here as a validator-schema failure only:

`P4T_FAIL:PROVENANCE_SOURCE_PHYSICAL_SHA256_MISMATCH`

The failure does not establish an artifact defect because frozen builder/P4-L
provenance does not emit the alias field `source_physical_sha256`.

## 3. Root Cause

Frozen builder/P4-L provenance uses the current-source field names:

- `source_dataset_path`
- `source_dataset_sha256`
- `source_dataset_semantic_sha256`

It does not emit these alias fields:

- `source_physical_sha256`
- `source_semantic_sha256`

Frozen P4-S required the alias fields as provenance keys, and P4-T inherited
that validator surface while correcting only the P4-B schema assertion. The
P4-T observed failure therefore came from requiring a nonexistent alias field,
not from a demonstrated current-source identity mismatch.

## 4. Required P4-U Correction

A future P4-U validator must require exactly these current-source provenance
bindings:

```python
provenance["source_dataset_path"] == (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
)
provenance["source_dataset_sha256"] == (
    "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
)
provenance["source_dataset_semantic_sha256"] == (
    "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
)
```

A future P4-U validator must not require:

- `source_physical_sha256`
- `source_semantic_sha256`

Absence of either alias field must not be treated as a provenance failure.

## 5. Preserved P4-T Requirements

This candidate preserves all P4-T requirements unrelated to the inherited
current-source alias defect:

- corrected P4-B checks from P4-T;
- P4-B physical SHA checks;
- semantic-hash algorithms;
- sidecar physical/semantic hashes;
- provenance physical hash, except that it must not imply alias-field presence;
- read-only boundaries;
- execution HEAD `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- CPU-only execution constraints for any future authorized validation;
- P4-T historical dispositions and non-retroactive FAIL status.

No training, evaluation, Kaggle execution, artifact mutation, commit, or push is
authorized by this candidate.

## 6. Evidence Pointers

Builder/P4-L field schema:

- `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
  defines `SOURCE_DATASET_PATH`, `SOURCE_DATASET_SHA256`, and
  `SOURCE_DATASET_SEMANTIC_SHA256` at lines 43-49.
- The builder writes per-row `source_dataset_path`,
  `source_dataset_sha256`, and `source_dataset_semantic_sha256` at lines
  682-684.
- The builder writes provenance `source_dataset_path`,
  `source_dataset_sha256`, and `source_dataset_semantic_sha256` at lines
  790-792.
- In those builder evidence blocks, `source_physical_sha256` and
  `source_semantic_sha256` are not emitted.

P4-S inherited bad alias assertion:

- `reports/reason_router_p2_p3w6f2_p4s_source_semantic_recovery_validation_correction_execution_authority_spec.md`
  includes `source_physical_sha256` and `source_semantic_sha256` in
  `required_provenance` at lines 664-686.
- The same P4-S spec repeats those alias bindings as required provenance source
  bindings at lines 825-837.

P4-T inheritance of that alias validator surface:

- `reports/reason_router_p2_p3w6f2_p4t_p4b_schema_recovery_validation_correction_execution_authority_spec.md`
  states that its only semantic correction from frozen P4-S is the P4-B
  summary/provenance schema assertion correction at lines 12-16.
- The same P4-T spec scopes P4-T to correcting only P4-S's incorrect P4-B
  summary and P4-B provenance schema assertions at lines 71-74.
- The same P4-T spec preserves source provenance bindings, sidecar hashes, and
  provenance physical hash checks at lines 381-385 and 428-429.
- The controller-observed P4-T execution failure token was
  `P4T_FAIL:PROVENANCE_SOURCE_PHYSICAL_SHA256_MISMATCH`.

P4-B/current-source identity preserved:

- P4-T requires P4-B `regenerated_dataset_path` and
  `regenerated_dataset_sha256` at
  `reports/reason_router_p2_p3w6f2_p4t_p4b_schema_recovery_validation_correction_execution_authority_spec.md`
  lines 223-228 and 306-311.
- P4-T records the corrected source semantic SHA
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
  at line 335.

## 7. Stop Conditions

Future verification must fail closed if any implementation or validator:

- requires `source_physical_sha256`;
- requires `source_semantic_sha256`;
- weakens or removes required `source_dataset_*` provenance checks;
- changes P4-B physical checks, semantic-hash algorithms, read-only boundaries,
  execution HEAD, CPU-only constraints, or historical P4-T dispositions;
- relabels P4-T PASS.

Final token:

`P4U_CURRENT_SOURCE_PROVENANCE_SCHEMA_CORRECTION_CANDIDATE_READY_FOR_VERIFICATION`
