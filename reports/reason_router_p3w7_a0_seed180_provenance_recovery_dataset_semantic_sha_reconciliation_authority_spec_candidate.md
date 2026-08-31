# P3-W7-A0 Seed180 Provenance Recovery Dataset Semantic SHA Reconciliation Authority Specification Candidate

## 1. Status and scope

**Authority/version:** `P3W7_A0_SEED180_PROVENANCE_RECOVERY_DATASET_SEMANTIC_SHA_RECONCILIATION_AUTHORITY_V1`

**Status:** CANDIDATE ONLY.

This is a narrow SPEC / AUTHORITY RECONCILIATION candidate. It reconciles the
required dataset semantic identity with the actual immutable `stage174a_v1`
`run_provenance.json` schema. It authorizes no implementation, collection,
audit-import, training, evaluation, Kaggle execution, dataset regeneration,
prediction regeneration, checkpoint mutation, artifact mutation, staging,
commit, or push.

The authority basis is the current user instruction, followed by the frozen
seed180 recovery implementation authority and the preserved authorities listed
below. No scientific contents of unrelated O0b work are interpreted.

## 2. Repository preconditions and descendant independence

The operative repository precondition for verification and freeze is current
HEAD:

`7ce4e0cd05d87118c29526a53ab5178dc722db27`

The tracked worktree and index must be clean before candidate materialization.
Protected unrelated untracked files are not to be changed, staged, or removed.

The historical repository state
`2ed4439e511f7534186cbd5df9110e45fdc1d66c` is retained only as the historical
creation-state reference for the evidence-availability gap. It is not an
operative current repository precondition and does not control verification or
freeze.

The immutable forensic attestation freeze is commit
`3de16c2215fe50e6f17aabe5ae33da3eab3f8540`, carrying
`reports/reason_router_p3w7_a0_seed180_provenance_recovery_dataset_semantic_sha_forensic_attestation_candidate.md`
with identity `5578` bytes and SHA256
`f736e424d27de3e821b511592dabaddbb2e6020cfffe76e36b7c579e25e89d44`.
The attestation is the stable evidence carrier for the historical six-path
observations only; it is not recovery authority, implementation authority,
execution authority, standard cm provenance, or scientific evidence.

The attestation freeze commit must be an ancestor of current HEAD. The tracked
delta from that commit to current HEAD is required to contain exactly:

- `data/longterm_o0b_matched_controls_v1.jsonl`;
- `reports/longterm_o0b_matched_controls_v1_validation.json`;
- `scripts/validate_longterm_o0b_matched_controls.py`;
- `tests/test_validate_longterm_o0b_matched_controls.py`.

The preceding delta from `2ed4439e511f7534186cbd5df9110e45fdc1d66c` to the
attestation freeze commit added exactly the frozen forensic attestation above.
The current descendant delta is unrelated O0b parallel-lane work only and must
not modify any seed180 recovery authority, the semantic-SHA reconciliation
candidate, the frozen forensic attestation, the recovery script/test,
seed180-relevant trainer/model code, or the original A0 authority. This
path-level descendant independence preserves seed180 recovery semantics. No
scientific contents of the unrelated O0b work are interpreted.

## 3. Preserved authority chain

This candidate preserves, without reinterpretation or weakening:

- frozen seed180 recovery implementation authority:
  `9a9b11a3212fb0073d3f3678875bc2a3ae003501`;
- recovery execution authority:
  `233ed0be080e1d30dd47de2e66136475ec2ede76`;
- tooling implementation authority:
  `cdd71ea4f556392eab594ebb5df8258355610e01`;
- external-cm reconciliation authority:
  `1b6516d16596d1169ff2fa4fd8d8c8f8adb80450`;
- prediction-row-count reconciliation authority:
  `8752646b106eb5b11d2de5241fce874edae75087`;
- original A0 execution authority:
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`.

The actual immutable historical provenance object remains `stage174a_v1`, size
`68429`, SHA256
`4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b`.

## 4. Observed blocker and root-cause classification

The frozen implementation failed closed with exit `64`:

`PROVENANCE_RECOVERY_BLOCKER: data_provenance.main_data.semantic_sha256 expected '3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b' got None`

After failure, `PACKAGE_DIR_EXISTS=False` and `TARGET_EXISTS=False`; therefore
no recovery ZIP was published.

The immutable historical provenance contains
`data_provenance.main_data.sha256`, equal to the canonical physical SHA, but it
does not contain `data_provenance.main_data.semantic_sha256`. The canonical
semantic SHA is genuinely present at the six historical paths named in the
request. The six values equal:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

The frozen authorities require the dataset semantic identity and its exact
value; they do not require that identity to occur at the nonexistent literal
`data_provenance.main_data.semantic_sha256` path. The recovery implementation
added a mandatory check for that absent path. The root cause is therefore:

**AUTHORITY / SCHEMA RECONCILIATION ISSUE**

This is a provenance/schema blocker only. It is not a training, evaluation,
artifact-hash, or scientific failure. Another collect under unchanged
`9a9b11a` is not authorized.

## 5. Narrow supersession

This candidate supersedes only the interpretation that
`data_provenance.main_data.semantic_sha256` must exist or independently carry
the semantic dataset identity.

It does **not** supersede the mandatory requirement that dataset semantic SHA256
equal exactly:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

No recovery requirement concerning artifact identity, physical identity,
provenance identity, command semantics, row counts, seed, split, A0 settings,
wrapper provenance, or scientific status is weakened.

## 6. Replacement semantic identity rule

Future remediation, if separately authorized, must require all of the following
paths, with exact string equality to the canonical semantic SHA above:

1. `compatible_positive_margin.authoritative_dataset_semantic_sha256`
2. `resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256`
3. `resolved_runtime_config.reason_router_p2_metadata_integrity_source.source_dataset_semantic_sha256`

These are an AND conjunction. All three are required; no fallback, default,
OR, path inference, or physical-SHA substitution is allowed.

The six genuine historical copies are:

- `compatible_positive_margin.authoritative_dataset_semantic_sha256`;
- `compatible_positive_margin.run_activity.single.sidecar_contract.authoritative_dataset_semantic_sha256`;
- `compatible_positive_margin.sidecar_contract.authoritative_dataset_semantic_sha256`;
- `resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256`;
- `resolved_runtime_config.compatible_positive_margin.sidecar_validation.authoritative_dataset_semantic_sha256`;
- `resolved_runtime_config.reason_router_p2_metadata_integrity_source.source_dataset_semantic_sha256`.

The frozen writer/schema inspection establishes that the three additional
copies are structurally emitted members of the `stage174a_v1` provenance
record: `sidecar_contract`, `run_activity.single.sidecar_contract`, and
`sidecar_validation`. Accordingly, they are required and each must equal the
canonical semantic SHA. If a future compatible schema omits one of those
additional structural copies, the implementation authority must explicitly
classify that schema before acting; absent such structural mandate, any present
copy must equal the canonical value and no present contradiction may pass.

The absent path `data_provenance.main_data.semantic_sha256` must not be
required. If it appears in synthetic or future input, it is only an optional
contradiction check and must equal the same canonical semantic SHA.

## 7. Physical / semantic independence

Preserve the authoritative physical identity:

`data_provenance.main_data.sha256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

All existing authoritative physical SHA checks remain mandatory and exact.
Physical identity alone is insufficient for semantic identity. No semantic
identity may be inferred from physical SHA, path, filename, or row count.

## 8. Preserved state and non-fabrication

The following state remains authoritative:

- seed: `180`;
- attempt: `CONSUMED`;
- execution success: `OBSERVED`;
- standard cm wrapper provenance: `INCOMPLETE / MISSING`;
- artifact/provenance validity: `NOT YET FORMALLY RECOVERED`;
- scientific conclusion: `NOT_ESTABLISHED`.

The prediction-row-count reconciliation is unchanged. Its exact historical
clean-dev cardinality evidence and all required row-count checks remain in
force.

It is explicitly prohibited to rewrite `run_provenance.json`, add the missing
semantic field, change frozen hashes, create replacement provenance, regenerate
the dataset or predictions, or mutate any seed180 artifact. No historical
wrapper metadata may be fabricated, backfilled, renamed into place, or
backdated.

## 9. Future implementation boundary

This candidate does not authorize implementation. After independent verifier
PASS, immutable spec freeze, and an explicit controller transition, the only
future implementation files are:

- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`.

The expected future delta is limited to removing the mandatory nonexistent
semantic path, validating the genuine historical semantic paths above, and
updating synthetic/adversarial tests. No third file is authorized. Existing
physical identity, artifact, command, row-count, provenance, and external-cm
checks remain unchanged unless separately authorized by a higher-priority
authority.

## 10. Execution boundary and next action

No collect retry is authorized until remediation is implemented,
independently verified, frozen, and passes post-freeze gates with a new exact
command/hash materialized under explicit execution authority. This candidate
does not authorize training, evaluation, Kaggle, collect, audit-import, or
scientific interpretation.

The exact recommended next action is: independently verify this candidate for
scope, schema-path correctness, narrow supersession, and non-fabrication; if it
passes, freeze it as a new immutable authority commit only after explicit
controller authorization. Do not implement or execute in this phase.
