# ContraMamba Gen5 Phase 2 Fresh Ownership Assay Implementation Authority

## 0. Status

PHASE =
`GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_IMPLEMENTATION`

STATUS =
`CANDIDATE_BECOMES_ACTIVE_ON_COMMIT`

IMPLEMENTATION_ALLOWED_AFTER_FREEZE =
`YES_EXACT_SCOPE_BELOW`

SCIENTIFIC_EXECUTION_ALLOWED =
`NO`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

TASK_EVALUATION_ALLOWED =
`NO`

KAGGLE_SCIENTIFIC_EXECUTION_ALLOWED =
`NO`

CODEX_ALLOWED =
`NO`

This authority permits only implementation and non-scientific verification of the
already-prespecified Gen5 Phase 2 fresh ownership assay.

It does not authorize opening the fresh cohort for scientific model execution,
computing the confirmatory scientific p-value, interpreting the ownership hypothesis,
or modifying any completed training artifact.

---

## 1. Frozen scientific design

PHASE2_DESIGN_COMMIT =
`d9b84bca8871464807d2dccf6380a6e911d6dbf8`

The following scientific semantics are frozen and may not be changed by implementation.

Primary ownership dimension:

`STATE_UPDATE_AUTHORITY`

Owner basis:

`R22`

Matched control basis:

`C22`

Training arms:

- `G5-C0`
- `G5-C1`
- `G5-M1`

Training seeds:

- `5201`
- `5202`
- `5203`

Primary comparison:

`G5-M1 versus G5-C1`

C0 remains descriptive and does not enter the confirmatory test.

No rank, layer, projector, arm, seed, cohort, loss, checkpoint, or endpoint search
is permitted.

---

## 2. Completed training evidence input

Frozen scientific-training execution commit:

`acdf2ee8940070a6e60190a1c2fcbede8933e419`

Imported run:

`gen5-phase2-dualt4-matrix-acdf2ee-r1`

Validated matrix facts:

- nine cells completed;
- exactly 20 optimizer steps per cell;
- exactly 180 optimizer steps total;
- parent and R22/C22 immutable;
- final correction checkpoints authenticated against per-cell reports and matrix manifest;
- fresh XG1 was not loaded during training;
- scientific p-value count during training was zero.

Known artifact packaging defect:

`SHA256SUMS.txt` contains one stale entry for `matrix_provenance.json` because the
runner originally rewrote final provenance after checksum generation.

Independent post-import validation established:

- the stale entry is exactly `matrix_provenance.json` and no other file;
- all other internal checksums pass;
- matrix manifest and both worker manifests pass;
- all nine cell reports and run-provenance files pass;
- all nine final correction payload identities and tensor hashes pass.

Prospective runner correction commit:

`e1f03e36d83a8e2a1ee6c21d87fbd555eadca427`

The packaging defect does not authorize mutation or regeneration of the completed
scientific training checkpoints and does not require retraining.

---

## 3. Frozen parent and basis identities

Representative parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Native backbone signature SHA256:

`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

R22 SHA256:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22 SHA256:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

R22/C22 shape:

`[24576, 2]`

No parent or basis reconstruction, replacement, rotation, fitting, or mutation is
authorized.

---

## 4. Frozen fresh primary cohort

DATA_ROOT =
`data/reason_router_gen5_phase2_xg1_ownership_assay_v1`

PAIR_RANGE =
`xg1_fact_8701..xg1_fact_9000`

PAIR_COUNT =
`300`

ROWS_PER_PAIR =
`6`

ROW_COUNT =
`1800`

Frozen file identities:

- `structured_source_facts.jsonl`
  SHA256 =
  `f316d7e2a187e90451ff0743829d6c615c063969b66ca51f91e15158bcc1ee06`
- `synthetic_reason_router_six_cell.jsonl`
  SHA256 =
  `529027ba79d3bd9fc4e300fbf317f0d84668152ed4fa0593c55af601e7934976`
- `tokenizer_anchor_manifest.jsonl`
  SHA256 =
  `880586e043e50e960c05482f3edc6b953b358e50ed0eba5fc279c7785cc33c00`
- `tokenizer_eligibility_summary.json`
  SHA256 =
  `ff64b60e3651cf78fe6cb9e9757446593471781cc5977d427eb4c1a6bffd0229`

Static preparation established:

- exact 300-pair cohort;
- no overlap with prior XG1 ranges;
- response fields absent;
- labels absent;
- endpoint values absent;
- tokenizer anchor eligibility PASS for all 300 pairs;
- no row filtering or cohort replacement allowed.

Tokenizer semantics:

- tokenizers `0.22.2`
- maximum length `128`
- claim budget `63`
- evidence budget `64`
- serialization `claim[:63]+EOS(0)+evidence[:64]`
- effective pad token id `0`

---

## 5. Frozen restoration semantics

The implementation must reuse the already frozen Phase1B restoration semantics.

Each item begins from:

`PP3_NEUTRALIZED_AT_LAYER17`

For each trained arm/seed, the learned Phase2 correction remains active.

The native donor write is captured before adding the Phase2 correction.

Define:

`a_native = R22^T w22_native`

For arm A, corrected PP3-neutralized background:

`w22_B^A = w22_B_native + DeltaW_eff^A`

Then:

`w22_RR^A = w22_B^A + R22 a_native`

`w22_RC^A = w22_B^A + C22 a_native`

The exact same `a_native` must be used for RR and RC.

The same trained correction must remain identically active across B, RR, and RC.

No response-dependent fitting or coefficient selection is allowed.

---

## 6. Frozen endpoints

For each arm A, training seed s, and fresh item i, compute:

- `Q_B`
- `Q_RR`
- `Q_RC`
- `S_R = Q_RR - Q_B`
- `S_C = Q_RC - Q_B`
- `I_A = Q_RR - Q_RC`

The Q definition remains the frozen Phase1B definition:

`Q = E_XG2_22 - E_XG4_22`

No alternative mechanistic endpoint may be introduced.

---

## 7. Frozen seed aggregation and confirmatory statistic

Training seeds are not independent item replicates.

For each fresh item i:

`Ibar_M1_i = mean_seed(I_M1_i)`

`Ibar_C1_i = mean_seed(I_C1_i)`

Primary contrast:

`D_OWN_i = Ibar_M1_i - Ibar_C1_i`

Confirmatory sample size:

`n = 300`

Exactly one scientific confirmatory p-value is permitted in the later execution
stage:

`one-sided one-sample Student t-test(D_OWN, alternative mean > 0)`

No seed-specific confirmatory tests are permitted.

No C0 confirmatory test is permitted.

No task-performance p-value is permitted.

Implementation tests may exercise statistical functions only on synthetic fixtures;
such tests are not scientific execution.

---

## 8. Frozen scientific decision rule

The later scientific execution may label ownership supported only if all provenance
and manipulation gates pass and all of the following hold:

1. `mean(Ibar_M1) > 0`
2. seed-averaged `mean(Q_RR_M1) > 0`
3. seed-averaged `mean(S_R_M1) > 0`
4. `mean(D_OWN) > 0`
5. the single one-sided Student t-test gives `p < 0.05`

Positive label:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_PRESERVES_CAUSAL_ROLE_INTEGRITY_OVER_MATCHED_C22_CONTROL`

Otherwise:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

A valid negative result is terminal for this exact Phase2 hypothesis.

No rescue sweep or cohort replacement is authorized.

---

## 9. Authorized implementation files

After this authority is committed, implementation may add only:

1. `scripts/reason_router_gen5_phase2_fresh_ownership_assay.py`
2. `tests/test_reason_router_gen5_phase2_fresh_ownership_assay.py`

No existing scientific implementation file may be modified under this authority.

In particular, do not modify:

- `src/contramamba/gen5_phase2_state_update_ownership.py`
- `scripts/train_reason_router_gen5_phase2_state_update_ownership.py`
- Phase1B restoration implementation files
- R22/C22 artifacts
- fresh XG1 files
- imported Phase2 training artifacts

The new runner should reuse frozen existing machinery by import rather than copy or
alter its semantics where technically possible.

---

## 10. Required implementation behavior

The implementation must fail closed on:

- repository/head mismatch;
- frozen source identity mismatch;
- XG1 file hash/count/order mismatch;
- matrix/cell manifest mismatch;
- missing or altered correction checkpoint;
- wrong seed/arm assignment;
- parent checkpoint mismatch;
- R22/C22 mismatch;
- correction tensor shape/hash mismatch;
- model or basis mutation;
- row drop/reorder;
- non-finite mechanistic values;
- unexpected statistical test count.

The implementation must support a two-worker 2×T4 scientific topology without DDP.

Each physical GPU worker must expose only one logical CUDA device.

No scientific CPU fallback is allowed.

Exact execution-head binding, runtime/binary identities, forward budgets, output
directory, and run name will be frozen only in the later execution authority after
implementation verification.

---

## 11. Artifact design requirements

The future scientific output must preserve enough scalar evidence to independently
recompute the frozen aggregation and decision.

At minimum it must support reconstruction of, for every item/seed/arm:

- `Q_B`
- `Q_RR`
- `Q_RC`
- `S_R`
- `S_C`
- `I_A`

It must also preserve:

- pair identity and canonical order;
- seed and arm identity;
- checkpoint/correction hashes;
- model/basis identities;
- forward accounting;
- parameter immutability signatures;
- tokenizer/input provenance;
- worker provenance.

The coordinator must produce seed-averaged item values and `D_OWN`.

Raw full recurrent-state, native-write, or POST_STATE22 vectors must not be persisted.

Final artifact hashes must be generated only after all mutable provenance files have
been finalized.

---

## 12. Implementation verification gates

Before scientific execution authority may be created, dedicated tests must establish
at least:

1. exact XG1 pair range/count/order contract;
2. exact seed/arm matrix contract;
3. exact correction-checkpoint authentication;
4. native donor coefficient computed before correction;
5. correction applied identically across B/RR/RC;
6. RR and RC use the same `a_native`;
7. M1/C1 projector semantics remain frozen;
8. endpoint algebra;
9. seed averaging before M1-C1 contrast;
10. exactly one confirmatory test in coordinator logic;
11. C0 excluded from confirmatory test;
12. no seed pseudo-replication;
13. negative outcome is terminal/not-established, not rescued;
14. output finalization/checksum ordering;
15. no training/backward/task-evaluation path.

Implementation verification must use synthetic fixtures or static identities only.

It must not perform scientific model forward execution on
`xg1_fact_8701..xg1_fact_9000`.

---

## 13. Prohibited actions during this authority

Do not:

- execute the fresh scientific assay;
- load the fresh cohort into a scientific model forward;
- compute a scientific Phase2 p-value;
- interpret the ownership result;
- retrain any correction;
- change any completed correction checkpoint;
- inspect task performance for selection;
- add seeds;
- drop or replace rows;
- alter owner/control bases;
- alter rank or layer;
- tune correction architecture or hyperparameters;
- perform rescue experimentation;
- commit unrelated changes.

---

## 14. Success boundary

This implementation stage succeeds when:

- the exact two new implementation/test files are present;
- dedicated tests pass;
- static/non-scientific verification passes;
- no XG1 scientific model forward occurred;
- no training/backward/task evaluation occurred;
- no scientific p-value was computed;
- source scope is clean and reviewable.

The next stage is:

`GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_EXECUTION_AUTHORITY`

That later authority must bind the exact implementation commit before any fresh
scientific execution.
