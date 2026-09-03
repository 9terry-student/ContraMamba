# P3-W7 A0 Seed180 Reference-Recovery Helper Sidecar Semantic Correction Authority Spec Candidate

Status: READY candidate for a narrowly bounded future implementation correction.

Phase: static implementation-correction authority only.

This candidate authorizes only a future correction to the canonical sidecar semantic SHA algorithm in the P3-W7 A0 seed180 reference-recovery helper. It does not authorize source or test modification in this task, helper execution, artifact materialization, recovery execution, training, evaluation, model or checkpoint loading, Kaggle execution, commit, or push.

## Authority

- Defective frozen helper implementation commit: `98dfe3ee25c266ad0e12e2215f8ca68ea499fdda`
- Helper path: `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`
- Helper test path: `tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py`
- Frozen corrected helper implementation authority: `df1cba2ed0833026d7e2293b22f6ab47687229cb`
- Frozen upstream retained-artifact recovery authority: `ceaee6236340ef7006f7004d910f388ec565db0e`
- P4-L authority commit: `80cb034792f03226cf6e22c196c1229ed4e6dd62`
- Exact P4-L canonical sidecar builder source commit: `2f9e6076791358922e3ebd70e89533d9cb83b458`
- Exact P4-L builder source path: `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
- Frozen builder source SHA256 recorded by canonical provenance: `b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`
- Canonical sidecar provenance path: `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

The prior draft object `2f9e6076791358922e3d6bf89f8b4eb5cec463458` is VOID and is not an authority for this task or any future correction authorized by this candidate.

## Defect

The helper implementation at `98dfe3ee25c266ad0e12e2215f8ca68ea499fdda` defines `semantic_sidecar_sha256(path)` by loading JSONL rows, serializing each complete row as sorted compact JSON, joining those row strings with `\n`, appending a final `\n`, and hashing those bytes.

On the canonical P4-L sidecar, that algorithm returns:

`2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`

That value is the frozen sidecar physical SHA256 for the LF-normalized canonical JSONL payload. It is not the frozen P4-L semantic SHA256. This is a helper semantic-hash algorithm defect, not a sidecar, dataset, label, split, or artifact drift issue.

The required frozen sidecar semantic SHA256 is:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

The canonical provenance distinguishes both identities:

- `sidecar_physical_sha256 = 2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- `sidecar_semantic_sha256 = 0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

Neither frozen identity may be reinterpreted or weakened.

## Normative Algorithm

The exact builder source at commit `2f9e6076791358922e3ebd70e89533d9cb83b458` implements the P4-L semantic sidecar contract as:

```python
def semantic_sidecar_sha256(rows):
    canonical = [
        {key: row[key] for key in sorted(row) if key != "created_at"}
        for row in rows
    ]
    return canonical_sha256(canonical)
```

where `canonical_sha256` hashes:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

Required future helper semantics:

- preserve JSONL row order;
- exclude exactly `created_at` from each row;
- preserve every other field and value;
- hash the canonical JSON array of rows;
- do not hash newline-joined canonical JSONL;
- do not use physical-file SHA256 as semantic identity;
- do not modify the sidecar.

The builder source SHA256 at the valid builder commit is `b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`, matching canonical provenance.

## Authorized Future Delta

A later implementation task may modify only:

- `scripts/reason_router_p3w7_a0_seed180_reference_recovery_helper.py`
- `tests/test_reason_router_p3w7_a0_seed180_reference_recovery_helper.py`

No third file is authorized. No trainer modification, existing recovery-script modification, dataset modification, sidecar modification, checkpoint mutation, or artifact mutation is authorized.

## Required Future Source Correction

Future implementation must correct helper `semantic_sidecar_sha256()` to be semantically identical to the frozen P4-L builder algorithm.

The canonical sidecar must produce exactly:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

Future implementation must not change:

```python
EXPECTED_SIDECAR_SEMANTIC_SHA256 = (
    "0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08"
)
```

No CLI override is authorized. No fallback may accept `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1` as semantic identity.

## Required Future Test Matrix

Focused future tests must cover:

1. Real canonical sidecar semantic regression: observed semantic SHA256 equals `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`.
2. Physical/semantic distinction: physical SHA256 equals `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`, while semantic SHA256 equals `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`.
3. `created_at` invariance.
4. Nonvolatile-field sensitivity.
5. Row-order sensitivity.
6. Correct canonical JSON-array semantics versus newline-joined JSONL.
7. Existing helper materialization tests remain passing.

## Preserved Helper Contracts

Future correction must preserve all other helper contracts, including:

- production CLI;
- runtime execution-authority binding;
- retained ZIP identity;
- ZIP validation;
- duplicate-key rejection;
- manifest and run-provenance gates;
- dataset physical SHA;
- split and dev identity;
- destination namespace;
- no-overwrite semantics;
- transaction and publication semantics;
- persisted audit exact reread validation;
- recovery provenance fields;
- checkpoint opaque-byte treatment;
- forbidden imports.

No scientific semantics change is authorized.

## Execution-Authority Consequence

The uncommitted execution-authority candidate in the other worktree remains BLOCKED and must not be frozen against helper commit `98dfe3ee25c266ad0e12e2215f8ca68ea499fdda`.

Required order:

1. Freeze this correction authority.
2. Implement helper and helper-test correction.
3. Independently verify corrected implementation.
4. Commit and push corrected helper implementation.
5. Create a new recovery execution-authority candidate bound to that corrected implementation commit.

The later execution-authority candidate must explicitly require:

`provenance_disposition = RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE`

This task does not authorize modifying the blocked execution-authority draft.

## Non-Authorizations

This candidate does not authorize:

- source or test modification in this task;
- helper execution;
- ZIP extraction or materialization;
- `A0_REFERENCE_AUDIT` creation;
- training or evaluation;
- model or checkpoint load or deserialization;
- Kaggle;
- A1, A2, or A3;
- scientific interpretation;
- commit;
- push.

## Stop Conditions

Return BLOCKED rather than implementing if:

- the valid builder source contradicts the normative algorithm;
- correction requires more than the helper plus helper-test future delta;
- any frozen identity would need weakening;
- dataset or sidecar modification is required;
- another authority conflict remains.

## Static Verification Summary

Static verification for this candidate established:

- valid builder commit `2f9e6076791358922e3ebd70e89533d9cb83b458` exists locally;
- invalid prior draft object `2f9e6076791358922e3d6bf89f8b4eb5cec463458` is not a valid local object and is not used as authority;
- builder source at the valid commit implements the normative `created_at`-excluding canonical JSON-array semantic algorithm;
- builder source SHA256 is `b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`;
- canonical provenance records physical SHA256 `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
- canonical provenance records semantic SHA256 `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- independent reconstruction of the current helper algorithm on the canonical sidecar yields `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
- independent reconstruction of the normative P4-L algorithm on the canonical sidecar yields `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`;
- the checked-out Windows worktree sidecar has CRLF-expanded bytes, but the tracked LF-normalized Git object has physical SHA256 `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`; this is not semantic sidecar drift.

Candidate end state required after this task:

- exactly one new untracked Markdown file, this candidate;
- no tracked diff;
- no staged diff;
- `git diff --check` passes;
- no source/test modification, helper execution, materialization, training, evaluation, model load, Kaggle, commit, or push.
