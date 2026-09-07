# Seed8192 Revised P4-L Phase-III Post-Materialization Consumer-Rebind Implementation Authority Candidate

## 1. Verdict and lifecycle

`PASS_READY_FOR_FRESH_INDEPENDENT_VERIFICATION`

```
PHASE_III_CONSUMER_REBIND_IMPLEMENTATION_DELTA_REQUIRED = YES
P4L_FROZEN_ARTIFACT_SCHEMA_MUTATION_REQUIRED = NO
P4L_CONSUMER_SCHEMA_REBIND_REQUIRED = YES
TRAINING_EVALUATION_SEMANTICS_CHANGE_REQUIRED = NO
WINDOWS_CHECKOUT_COMPATIBILITY_DELTA_REQUIRED = YES
HISTORICAL_CONSUMER_DUAL_RUNTIME_MODE_REQUIRED = NO
EXECUTION_RECORD_RUNTIME_DEPENDENCY_REQUIRED = NO
PHASE_IV_REQUIRED_AFTER_PHASE_III = YES

ACTIVE_SEED8192_REVISED_P4L_PHASE3_CONSUMER_REBIND_IMPLEMENTATION_AUTHORITY = NONE_YET
PHASE_III_IMPLEMENTATION = NOT_AUTHORIZED_BY_THIS_CANDIDATE
```

This is a report-only candidate. It establishes a bounded future implementation
delta, not permission to make it. Independent verification, staging, a local
commit, or a push alone does not activate it.

## 2. Opening repository state and authority chain

Read-only inspection found branch
`p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`, HEAD
`ef26310f3532368b9de6cb96a19cb26e7626716d`, upstream
`origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`, and ahead/behind
`0/0`. `git status --short`, cached diff, and working diff emitted no path
entries before authoring (Git emitted only permission warnings for pre-existing
inaccessible pytest-cache directories). There were no staged or tracked
modifications and no reported untracked paths.

The authenticated authority chain is:

| Role | Immutable commit |
| --- | --- |
| Phase-II evidence freeze | `ef26310f3532368b9de6cb96a19cb26e7626716d` |
| Phase-II activation / its exact parent | `cb6f4482b463d5f85331e2a6ddfbbd34499c930a` |
| frozen revised producer implementation | `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` |
| revised P4-L implementation delta | `1f05ae3aca63138c482101633412690be213b36d` |
| revised P4-L reconstruction/rebinding/provenance | `ff181f565cefa0a28280c084246862286daf1f2d` |
| revised split | `b4fbb5666d796161f95ae23612ce2448c25063ee` |
| split-contract remedy | `c82a164ac460599c68318a3b29180303f12cbc1a` |

All seven identifiers resolve to commit objects. The evidence-freeze parent is
exactly the listed Phase-II activation commit. Its tree delta has exactly three
additions: the sidecar, its provenance, and the Phase-II execution record.

## 3. Immutable Phase-II evidence authentication

The sidecar and provenance frozen at
`ef26310f3532368b9de6cb96a19cb26e7626716d` are immutable. Their frozen
schemas are, respectively,
`P3W7_SEED8192_REVISED_P4L_EFFECTIVE_INTEGRITY_SIDECAR_V1` and
`P3W7_SEED8192_REVISED_P4L_INTEGRITY_SIDECAR_PROVENANCE_V1`.
`FROZEN_ARTIFACT_SCHEMA = IMMUTABLE_EXISTING_REVISED_SCHEMA`:
Phase III does not authorize changing either schema definition, editing,
regenerating, materializing replacements for, renaming/relabeling schema
versions of, or creating alternate revised evidence for these artifacts.

`CONSUMER_SCHEMA_BINDING = REBOUND_TO_EXISTING_REVISED_SCHEMA`:
the only future implementation change is consumer-side, from the historical
seed174 schema/path/hash/authority bindings to these existing frozen revised
seed8192 schema/path/hash/authority bindings. Trainer validators and tests must
accept and validate these existing revised schemas exactly; they must not
transform or reinterpret either artifact schema.

The canonical frozen directory is
`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/`.

| Artifact | Git blob | canonical Git/LF SHA256 | bytes |
| --- | --- | --- | ---: |
| `p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl` | `83d119e327acacda7cff6b4e24c6502898294e03` | `9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d` | 8,390,518 |
| `p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json` | `6c970033fae82286452f6d635b94f441d0f3d048` | `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8` | 5,459 |
| `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase2_execution_record_cb6f4482b463d5f85331e2a6ddfbbd34499c930a.json` | `0d07e52dd4240a84c60805a4495b18a727e289d3` | `b36e2415d2f773305693ae3261b05b512218d6b567bb52e21efb3610f7f0eb9b` | physical SHA256 measured; byte count not required by the freeze |

The sidecar semantic SHA256 is
`2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9`.
These are frozen evidence values, never values to regenerate.

The provenance itself authenticates `lineage_mode = revised-seed8192`, sidecar
schema `P3W7_SEED8192_REVISED_P4L_EFFECTIVE_INTEGRITY_SIDECAR_V1`, provenance
schema `P3W7_SEED8192_REVISED_P4L_INTEGRITY_SIDECAR_PROVENANCE_V1`, P4-L
authority `ff181f565cefa0a28280c084246862286daf1f2d`, split authority
`b4fbb5666d796161f95ae23612ce2448c25063ee`, and builder source
`149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b`. It gives source dataset Git/LF
SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
and semantic SHA256
`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.

The frozen split identities are: pair universe
`41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2`,
shuffled pairs `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55`,
train pairs `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049`,
dev pairs `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4`,
ordered train rows `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`,
and ordered dev rows `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`.
The split is seed 8192, 300 pairs / 3,600 rows, 240 train pairs / 2,880 rows,
60 dev pairs / 720 rows, and pair leakage zero.

## 4. Current consumer authentication and historical contract

At HEAD, `scripts/train_controlled_v6b_minimal.py` is blob
`3dcc0864b85bde5fb8090c3b7bdbd04de02025e0`; its canonical Git/LF content is
1,335,154 bytes, SHA256 `b67a905bff87c3ae730e2e330bddd428dfd6b4d7ae859b7187bf07355eec3d72`,
28,438 LF, zero CR/CRLF, no BOM, final LF present, and 45 trailing-whitespace
lines. Its Windows working copy is 1,363,592 bytes, 28,438 CRLF / CR, and SHA256
`5f82ddbbc0506d74f3c92b37614d7d3f4eb55094cee49c2cc2148c700efb2562`.

`tests/test_reason_router_p4x_trainer_rebind.py` is blob
`b7e8c6eb4e5ab0e7b5bef57cf0d74c2ca6411728`; canonical Git/LF is 14,518 bytes,
SHA256 `0fd55b34fbd9b2e221db90f5fd563454d70707fae275d341150f322c6c4d3c81`,
290 LF, no CR/CRLF or BOM, final LF, and no trailing whitespace. Its Windows
working copy is 14,808 bytes, 290 CRLF / CR, and SHA256
`f54570e80bc2fe7fad1d55cd9292757f97917c959880594af4243c3c951ffc6a`.

The current P4-X contract is not path-only. It binds the historical directory
`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`, sidecar
`p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`, and matching
provenance; it requires dataset SHA above, source semantic SHA above, sidecar
physical `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`,
provenance physical `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`,
sidecar semantic `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`,
historical schemas, P4-L authority `80cb034792f03226cf6e22c196c1229ed4e6dd62`,
and builder `2f9e6076791358922e3ebd70e89533d9cb83b458`.

`_p4x_validate_canonical_integrity_binding`, `_p4x_validate_provenance`,
`_p4x_validate_sidecar_rows`, `_p4x_validate_stable_join`,
`_stage187_load_integrity_sidecar`, `_p2_load_reason_integrity_sidecar`, and
`_p2_checkpoint_metadata_from_args` load, validate, join, report, and write
this binding. The current targeted test is the only runtime-focused test file
that references these P4-X constants; repository-wide hits outside it and the
trainer are historical builder/test/report artifacts or unrelated seed-174
calibration/replay code. Thus the minimum future implementation fileset is
exactly the trainer and this focused test.

## 5. Exact future rebinding table

| Binding | Historical value | Revised seed8192 value | Change required |
| --- | --- | --- | --- |
| canonical directory | historical `...2f9e6076...` directory | canonical frozen directory in section 3 | YES |
| sidecar / provenance names | historical `p3w6f2...current_lineage...` | `p3w7_seed8192_revised_p4l_effective_integrity_sidecar.{jsonl,provenance.json}` | YES |
| sidecar physical / semantic | `2b8cffdf...` / `0e652c80...` | `9bbbb48a...` / `2528a05e...` | YES |
| provenance physical | `9d248df...` | `170647d...` | YES |
| consumer schema binding | historical P3W6F2 current-lineage schemas | existing frozen P3W7 seed8192 revised schemas in section 3 | YES: rebind only; no frozen-schema mutation |
| P4-L authority / builder | `80cb0347...` / `2f9e6076...` | `ff181f56...` / `149adf32...` | YES |
| split authority / seed | absent / historical seed174 behavior | `b4fbb566...` / 8192 plus all frozen identities | YES |
| source dataset Git/LF and semantic identities | same values | same values | NO, retain and revalidate |
| Phase-II lineage metadata | absent | activation `cb6f4482...`, evidence freeze `ef26310f...`, execution blob `0d07e52d...` | YES, metadata/static binding only |

No future implementation commit SHA is invented here.

## 6. Independently recomputed revised counts

Rows were parsed directly from the frozen sidecar. Counts below are independent
of its stated provenance values.

| Consumed field/count | Historical | Revised total | train | dev | Change required / why |
| --- | ---: | ---: | ---: | ---: | --- |
| total rows | 3600 | 3600 | 2880 | 720 | NO |
| reason eligible True | 1769 | 1769 | 1409 | 360 | NO total; split bind must be recorded |
| reason eligible False | 1831 | 1831 | 1471 | 360 | NO total; split bind must be recorded |
| integrity ELIGIBLE | 1769 | 1769 | 1409 | 360 | NO total; split bind must be recorded |
| integrity INELIGIBLE | 1562 | 1562 | 1250 | 312 | NO total; split bind must be recorded |
| integrity UNRESOLVED | 269 | 269 | 221 | 48 | NO total; split bind must be recorded |
| positive margin True | 724 | 695 | 695 | 0 | YES: split-bound change |
| positive margin False | 2876 | 2905 | 2185 | 720 | YES: split-bound change |

Applicable reason-eligible primary reasons are: all/train/dev respectively
AUTHORIZED 419/338/81, FRAME 900/714/186, PREDICATE 150/119/31, and
SUFFICIENCY 300/238/62. The all-row primary-reason distribution is AUTHORIZED
900, FRAME 1800, PREDICATE 300, SUFFICIENCY 600. Therefore
`_STAGE187_EXPECTED_ELIGIBLE_ROWS` must become 695 and
`_P4X_EXPECTED_POSITIVE_MARGIN_INELIGIBLE_ROWS` 2905; the other listed current
total constants happen to retain their values, but must be rebound to revised
evidence and protected by revised split/count tests.

## 7. Windows/Git byte and consumption policy

The source dataset blob is `2b6829bf04a1333446aac6f7c603d9178b339f36` and
canonical Git/LF bytes are 1,879,593 with the required dataset SHA. This Windows
checkout has 3,600 CRLF/CR source rows, 1,883,193 bytes, and working hash
`eedbf93cf7fc3e141c4a49511750cbe4d8b0443e7de3463ea7e77696aca2c572`.
The current physical-file comparison therefore fails after a normal CRLF
checkout. The new sidecar and provenance currently happen to match canonical
bytes (respectively 3,600 and 74 LF, no CR/CRLF, BOM, or trailing whitespace),
but that observation is not a portable identity policy.

The future implementation must (1) reject symlinks and substituted paths, (2)
require clean tracked status for the canonical sidecar/provenance path against
HEAD, (3) obtain `HEAD:<path>` bytes, verify the frozen blob and SHA256, and
parse those exact authenticated canonical bytes rather than reopening a
working-tree file. This rejects staged and unstaged dirt and closes
validate-A/consume-B. It must not normalize arbitrary bytes.

For the source dataset, verify the expected `HEAD:<path>` blob and canonical
Git/LF SHA, reject path/symlink substitution and dirty canonical source, and
retain semantic hashing of the actual parsed records consumed by the trainer.
Do not require its CRLF working-tree physical SHA to equal the Git/LF SHA. This
authenticates immutable source identity and actual consumed semantics without
accepting arbitrary byte normalization.

## 8. Required provenance and lineage delta

Fail closed on the existing exact provenance `schema_version` and
`sidecar_schema_version` (validate them exactly; do not transform or
reinterpret either frozen schema), `lineage_mode = revised-seed8192`,
P4-L/split/builder commits, source dataset
physical and semantic identities, sidecar path/physical/semantic identity, row
count, every frozen split identity/count, and
`provenance_physical_sha256_self_certified = false`. Require false for
`training_admission_released`, `artifact_materialization_authorized_by_p4l`,
`a0_execution_authorized`, `training_authorized`, `evaluation_authorized`,
`kaggle_authorized`, and `gpu_authorized`; retain the frozen provenance's exact
`implementation_authorized = true` requirement. The false P4-L materialization
flag does not invalidate already separately authorized and frozen Phase-II
evidence.

The execution record is not a runtime training input. Its activation commit,
evidence-freeze commit, and execution-record blob belong in static constants and
checkpoint/run metadata, so the run remains reconstructible without unnecessary
runtime coupling. Include all three immutable identities in the P2 sidecar
audit/checkpoint metadata.

## 9. Seed174 classification and preserved semantics

`P3W1_CALIBRATION_SPLIT_SEED = 174` and the calibration check are
`FUTURE_CALIBRATION_SCOPE_NOT_PHASE_III`. The Stage191/193/195/196-B1 explicit
seed-174 replay/observability guards are `HISTORICAL_ONLY_PRESERVE`. P4-X
consumer split-bound positive-margin counts and canonical P4-L provenance are
`REBOUND_IN_PHASE_III`. Historical reports, historical builder tests, and
historical artifact strings are `HISTORICAL_ONLY_PRESERVE`; unrelated seed174
mentions are `UNRELATED`. The revised binding is the canonical future P4-X
consumer binding; historical artifacts remain historical evidence and must not
be accepted or relabeled as revised. Calibration/history references outside
this consumer rebind remain unchanged. No selectable historical/revised
dual-runtime consumer mode is authorized. Nothing is silently relabeled.

The implementation must not alter dataset rows/order, labels, eligibility or P2
reason derivation, reason order/secondary semantics, router/gradient/loss/EMA
semantics, A0--A3 definitions, architecture, optimizer, schedule, selection,
evaluation, or promotion criteria. It must not edit producer/evidence files,
split authority, execution record, historical artifacts, calibration code, or
any file outside the two-file implementation scope.

## 10. Future focused validation contract

The future implementation must add focused tests for revised path success;
sidecar blob/hash; exact existing frozen provenance/sidecar schema validation
without schema transformation or reinterpretation; lineage/split/builder bindings; all false
execution flags; exact seed8192 identities/counts; dataset Git/LF plus consumed
semantic identity; Windows CRLF safety; dirty sidecar/provenance and staged-dirty
rejection; symlink substitution; canonical authenticated bytes consumed;
physical and semantic mismatch failure; historical sidecar rejection; seed174
historical references retained; required checkpoint/run metadata; and unchanged
A0/A1/A2/A3 semantic tests.

The later, authorized narrow command is:

```powershell
pytest -q tests/test_reason_router_p4x_trainer_rebind.py
git diff --check
```

Do not run it in this phase.

## 11. Activation and execution boundary

Activation requires exact candidate verification, exact byte/blob freeze,
single-file staging, dedicated authority commit, push, remote tip and parent
verification, and remote blob/body verification. Only then may the exact bounded
two-file implementation delta be authored.

Even an activated Phase-III implementation authority and implementation freeze
authorize no A0, calibration, A1/A2/A3, training, evaluation, CUDA/GPU, or
Kaggle work. The next dependency is:

```
PHASE_III_POST_MATERIALIZATION_CONSUMER_REBIND
    -> PHASE_IV_PRELAUNCH_STATIC_CONTROL
```

Candidate identity is this one Markdown file only. No source/test modification,
materialization, producer invocation, trainer invocation, pytest, execution,
staging, commit, or push occurred. Exact next authorized action:
`FRESH_INDEPENDENT_VERIFICATION_OF_PHASE_III_CONSUMER_REBIND_AUTHORITY_CANDIDATE`.
