# P3-W7-A0 Seed180 Provenance Recovery

## Forensic Attestation Candidate

Status: immutable-candidate forensic attestation.

This document records **operator-returned live forensic inspection evidence** from an exact, anchored, read-only inspection of the frozen seed180 `run_provenance.json` source artifact. It is not standard cm wrapper provenance, historical wrapper evidence, independently regenerated provenance, scientific evidence, or proof of model quality.

This is a forensic record only. It does not amend authority, implement a mechanism, execute recovery, or establish a scientific result.

## Recorded source artifact

Source file inspected:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json`

Exact source identity:

- Size: `68429` bytes
- SHA256: `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b`

## Canonical dataset identities

- Canonical dataset physical SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- Canonical dataset semantic SHA256: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

## Exact observed semantic SHA matches

The live read-only traversal of the exact anchored `run_provenance` object found the canonical semantic SHA256 at exactly these reported paths. All six returned exactly `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`:

1. `compatible_positive_margin.authoritative_dataset_semantic_sha256`
2. `compatible_positive_margin.run_activity.single.sidecar_contract.authoritative_dataset_semantic_sha256`
3. `compatible_positive_margin.sidecar_contract.authoritative_dataset_semantic_sha256`
4. `resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256`
5. `resolved_runtime_config.compatible_positive_margin.sidecar_validation.authoritative_dataset_semantic_sha256`
6. `resolved_runtime_config.reason_router_p2_metadata_integrity_source.source_dataset_semantic_sha256`

No additional semantic path is recorded here.

## Exact observed physical SHA matches

The live read-only traversal found the canonical physical SHA256 at exactly these reported paths. All seven returned exactly `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`:

1. `compatible_positive_margin.authoritative_dataset_sha256`
2. `compatible_positive_margin.run_activity.single.sidecar_contract.authoritative_dataset_sha256`
3. `compatible_positive_margin.sidecar_contract.authoritative_dataset_sha256`
4. `data_provenance.main_data.sha256`
5. `resolved_runtime_config.compatible_positive_margin.authoritative_dataset_sha256`
6. `resolved_runtime_config.compatible_positive_margin.sidecar_validation.authoritative_dataset_sha256`
7. `resolved_runtime_config.reason_router_p2_metadata_integrity_source.source_dataset_physical_sha256`

## Exact observed `data_provenance.main_data`

The supplied read-only JSON representation, recorded in semantic content, was:

```json
{
  "byte_size": 1879593,
  "configured": true,
  "error": null,
  "exists": true,
  "expected": true,
  "mode": "main_clean_classification",
  "path": "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl",
  "resolved_path": "/kaggle/working/ContraMamba/reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl",
  "row_count": 3600,
  "sha256": "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
}
```

`data_provenance.main_data.semantic_sha256` was **not present** in the returned object. This attestation does not infer that its absence means semantic identity was absent elsewhere.

## Recovery failure and package absence

The immediately preceding collect blocker under implementation `9a9b11a3212fb0073d3f3678875bc2a3ae003501` was:

```text
PROVENANCE_RECOVERY_BLOCKER:
data_provenance.main_data.semantic_sha256 expected
'3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b'
got None

Exit status:
64
```

Subsequent read-only package inspection returned:

```text
PACKAGE_DIR_EXISTS=False
TARGET_EXISTS=False
```

Therefore no successful recovery ZIP was published by that attempt. This is not reinterpreted as artifact invalidity, training failure, evaluation failure, or scientific failure.

## Provenance boundary

This attestation:

- records a live forensic observation;
- does not modify or replace `run_provenance.json`;
- does not fabricate missing historical fields;
- does not create standard cm wrapper provenance;
- does not alter seed180 attempt disposition;
- does not establish scientific conclusions;
- does not itself authorize the semantic-SHA reconciliation candidate;
- does not authorize implementation or a collect retry.

Preserved recovery state:

- seed = `180`
- attempt = `CONSUMED`
- execution success = `OBSERVED`
- standard cm wrapper provenance = `INCOMPLETE / MISSING`
- artifact/provenance validity = `NOT YET FORMALLY RECOVERED`
- scientific conclusion = `NOT_ESTABLISHED`

## Non-fabrication statement

The six semantic paths, seven physical paths, source identity, `main_data` representation, blocker, exit status, and package-absence observations above are recorded only as supplied operator-returned live forensic inspection evidence. No missing `data_provenance.main_data.semantic_sha256` field has been synthesized, and no claim of independently regenerated provenance or scientific validity is made.
