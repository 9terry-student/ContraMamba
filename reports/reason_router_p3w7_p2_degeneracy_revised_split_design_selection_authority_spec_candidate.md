# P3-W7 P2 Degeneracy Revised Split Design Selection Authority Finalized Content

## 1. Verdict

`FINAL_REVISED_SPLIT_DESIGN_SELECTION_AUTHORITY_CONTENT`

`PASS_READY_FOR_FREEZE`

`INDEPENDENT_VERIFICATION = PASS_READY_FOR_FINALIZATION`

`ACTIVATION_CONDITION = ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

`ACTIVE_REVISED_SPLIT_DESIGN_SELECTION_AUTHORITY = NONE_YET`

`COMMIT_MESSAGE_DOES_NOT_OVERRIDE_BODY_LEVEL_AUTHORITY_STATUS`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

`Training/Evaluation/Kaggle/CUDA = NOT_AUTHORIZED`

This finalized lifecycle content preserves exactly one deterministic revised split selection. It does not generate or write a sidecar, modify the dataset, modify labels, modify eligibility/applicability, change guard semantics, run A0, recalibrate, train/evaluate, authorize Kaggle, implement execution/prelaunch code, or authorize A1/A2/A3 execution.

Commit subject text alone cannot activate this authority. The body-level lifecycle/status markers in this file remain controlling.

## 2. Authority Used

Highest authority for this candidate:

1. Active split-contract remedy decision authority: `c82a164ac460599c68318a3b29180303f12cbc1a`.
2. Active P2 root-cause authority: `eea0714904ea1f95c42da48e85cd1af4bad23123`.
3. Active authority-lineage reconciliation: `1bb08179adb38637e9391491ba72cfd7e9bff3b3`.
4. Active unauthorized-execution incident correction: `0f6e00642fb6126ec86d7b7dde4b84626befca67`.
5. Final verified decision-support evidence: `reports/reason_router_p3w7_p2_degeneracy_scientific_contract_remedy_decision_spec_candidate.md`.
6. Explicit controller selection rule in the task card.
7. Repository contract: `AGENTS.md`.

The active split-contract remedy decision authority was authenticated at current HEAD `c82a164ac460599c68318a3b29180303f12cbc1a`. It records:

- `SELECTED_REMEDY_CLASS = SPLIT_CONTRACT_CHANGE`
- `EXACT_REPLACEMENT_SPLIT_SEED = NOT_YET_SELECTED`
- `CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`
- `Training/Evaluation/Kaggle/CUDA = NOT_AUTHORIZED`

## 3. Preconditions

Pre-materialization git state:

| Field | Required | Observed | Result |
|---|---|---|---|
| Branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | PASS |
| HEAD | `c82a164ac460599c68318a3b29180303f12cbc1a` | `c82a164ac460599c68318a3b29180303f12cbc1a` | PASS |
| `git status --short` | empty | empty | PASS |
| `git diff --name-status` | empty | empty | PASS |
| `git diff --cached --name-status` | empty | empty | PASS |
| `git diff --check` | no output, exit 0 | no output, exit 0 | PASS |

Frozen dataset:

- path: `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
- tracked blob SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- semantic SHA256 from current lineage authorities: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- working-tree byte SHA256 observed on Windows checkout: `eedbf93cf7fc3e141c4a49511750cbe4d8b0443e7de3463ea7e77696aca2c572`
- working-tree/tracked-byte note: the tracked blob hash matches the authoritative physical identity; the working-tree byte hash differs due checkout line-ending representation while `git status --short` remains clean.

Dataset content used for selection was read only. No temporary repository artifact was created.

## 4. Exact Source Split/Readiness Semantics

Split implementation cited:

- runtime split source: `scripts/build_controlled_v5.py::split_by_pair_id`
- trainer invocation: `scripts/train_controlled_v6b_minimal.py` calls `v5.split_by_pair_id(records, dev_ratio=args.dev_ratio, seed=resolved_split_seed)`
- integrity replay source: `scripts/validate_reason_router_p3w6f2_p4d_controlled_data_integrity_gate.py::replay_stage185_split`

Current split semantics used:

1. Validate/materialize records.
2. Require `0.0 < dev_ratio < 1.0`.
3. Obtain sorted unique `pair_id` values.
4. Initialize `random.Random(split_seed)`.
5. Shuffle that sorted pair list using Python `random.Random.shuffle`.
6. Compute `dev_count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * dev_ratio)))`.
7. With 300 pairs and `dev_ratio=0.2`, `dev_count = 60`.
8. Assign all rows whose `pair_id` is in the first 60 shuffled pair IDs to dev; all remaining rows to train.
9. Fail if any pair appears in both train and dev.

Python/random reproducibility semantics:

- Python `random.Random(seed)` uses the Python standard-library deterministic Mersenne Twister implementation for integer seeds.
- The input list to shuffle is the canonical sorted string list of all 300 unique `pair_id` values.
- The selected split is fully reproducible from the tracked dataset blob, `dev_ratio=0.2`, Python integer seed `8192`, and the source algorithm above.

Current P2 applicability/readiness semantics used:

- primary reason order: `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`
- frame applicable: P2 eligible
- predicate applicable: P2 eligible and frame compatible
- sufficiency applicable: P2 eligible, frame compatible, and predicate covered
- polarity applicable: P2 eligible, frame compatible, predicate covered, sufficient, and final label in `{REFUTE, SUPPORT}`
- polarity target mapping: `REFUTE -> 0`, `SUPPORT -> 1`
- A1/A3 primary reason minimum counts: train `>= 50`, dev `>= 20`
- applicable binary cohort readiness: each of frame, predicate, sufficiency, and polarity must contain class `0` and class `1` with count `>= 1` in both train and dev
- fail-closed exception semantics: a missing class in any required applicable binary cohort raises `P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE`; an undersized primary reason raises `P2_REASON_MIN_CLASS_COUNT_FAILED`

The scan used the current frozen source rows plus the current P4-L row-level P2 eligibility/applicability fields. No split-bound sidecar was regenerated or rebound.

## 5. Fresh 0..9999 Feasibility Reproduction

Candidate split seeds:

- domain: `0..9999` inclusive
- total candidates: `10000`
- search policy: complete fresh scan, no early stop
- dev ratio: `0.2`
- total pair count per candidate: `300`
- dev pair count per candidate: `60`
- train pair count per candidate: `240`

Fresh reproduction result:

| Metric | Value | Required | Result |
|---|---:|---:|---|
| TOTAL | `10000` | `10000` | PASS |
| FEASIBLE | `9991` | `9991` | PASS |
| INFEASIBLE | `9` | `9` | PASS |

Infeasible seeds:

| Seed | Failure |
|---:|---|
| `174` | `dev:binary:polarity:{0: 0, 1: 58}` |
| `1617` | `dev:primary:PREDICATE:18<20` |
| `1746` | `dev:primary:PREDICATE:17<20` |
| `2929` | `dev:primary:PREDICATE:19<20` |
| `4751` | `dev:primary:PREDICATE:17<20` |
| `6125` | `dev:primary:PREDICATE:19<20` |
| `6185` | `dev:primary:PREDICATE:19<20` |
| `7907` | `dev:primary:PREDICATE:19<20` |
| `9753` | `dev:primary:PREDICATE:18<20` |

Required special reproduction:

- seed `174` is the sole dev-polarity-degenerate seed: PASS
- other infeasible seeds exactly `1617, 1746, 2929, 4751, 6125, 6185, 7907, 9753`: PASS

## 6. Pre-Registered Selection Rule

Among feasible seeds only:

1. Maximize the number of dev `pair_id` values shared with the frozen seed174 dev pair set.
2. Equivalently, because every valid split has exactly 60 dev pairs, minimize the symmetric difference between the candidate dev pair set and the seed174 dev pair set.
3. If multiple feasible seeds achieve the same maximum overlap, select the smallest integer split seed.

No model outcome, final macro F1, accuracy, training loss, reason loss, calibration loss, A0/A1/A2/A3 metric, label-balance score beyond pass/fail readiness, ideal class balance distance, or future promotion performance participates in the objective.

Rationale: the criterion makes the minimum pair-membership perturbation to the previously frozen seed174 split while satisfying the unchanged scientific readiness contract. It preserves the chosen split-contract-remedy principle: change split membership only as much as necessary under a deterministic outcome-independent criterion. This criterion is not claimed to be universally optimal; it is the controller-pre-registered preservation criterion for this revised split authority.

## 7. Exact Selected Revised Split Seed

`SELECTED_REVISED_SPLIT_SEED = 8192`

`MAX_DEV_PAIR_OVERLAP_WITH_SEED174 = 23`

`DEV_PAIR_SYMMETRIC_DIFFERENCE_SIZE = 74`

`NUMBER_OF_FEASIBLE_SEEDS_AT_MAX_OVERLAP = 1`

All feasible seeds tied at maximum overlap, ascending:

```text
8192
```

Tie-break proof: the maximum-overlap tied set contains exactly one feasible seed, `8192`. Therefore `8192` is trivially the smallest integer in the tied set and is selected deterministically.

Removed/added counts:

| Quantity | Value |
|---|---:|
| `SEED174_DEV_PAIRS_RETAINED` | `23` |
| `SEED174_DEV_PAIRS_REMOVED` | `37` |
| `NEW_DEV_PAIRS_ADDED` | `37` |

Because both dev sets contain exactly 60 pairs, removed and added counts are identical. Verified: `37 == 37`.

No model execution was used to establish this result.

## 8. Exact Selected Split Identity

| Field | Value |
|---|---:|
| split seed | `8192` |
| dev ratio | `0.2` |
| total pair count | `300` |
| train pair count | `240` |
| dev pair count | `60` |
| total row count | `3600` |
| train row count | `2880` |
| dev row count | `720` |
| pair leakage count | `0` |
| exact split implementation | `scripts/build_controlled_v5.py::split_by_pair_id` |

Canonical hash convention for pair identity lists in this report:

- Sort pair IDs lexicographically unless the field explicitly says shuffled sequence.
- Serialize as UTF-8 text with exactly one pair ID per line and a final LF.
- Hash the resulting bytes with SHA256.

Canonical hash convention for ordered row identities:

- Use trainer `_p2_row_identity_hash` semantics from `scripts/train_controlled_v6b_minimal.py`: iterate records in dataset order within the selected split and update SHA256 with `"{id}\t{pair_id}\n"` encoded as UTF-8 for each row.

Hashes:

| Identity | SHA256 |
|---|---|
| complete pair universe, sorted LF list | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| selected dev pair identities, sorted LF list | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| selected train pair identities, sorted LF list | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| selected seed8192 shuffled pair sequence, LF list | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| seed174 dev pair identities, sorted LF list | `259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d` |
| ordered train row identity | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| ordered dev row identity | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

## 9. Exact Dev Pair Identities

Selected seed8192 dev pair IDs, sorted canonical order:

```text
clinic_expansion
forest_mapping
garden_award
generated_fact_034
generated_fact_042
generated_fact_045
generated_fact_048
generated_fact_051
generated_fact_056
generated_fact_062
generated_fact_073
generated_fact_076
generated_fact_078
generated_fact_085
generated_fact_087
generated_fact_089
generated_fact_090
generated_fact_091
generated_fact_096
generated_fact_102
generated_fact_108
generated_fact_118
generated_fact_133
generated_fact_136
generated_fact_138
generated_fact_139
generated_fact_144
generated_fact_152
generated_fact_157
generated_fact_165
generated_fact_166
generated_fact_167
generated_fact_174
generated_fact_179
generated_fact_181
generated_fact_192
generated_fact_193
generated_fact_195
generated_fact_205
generated_fact_225
generated_fact_227
generated_fact_240
generated_fact_241
generated_fact_242
generated_fact_243
generated_fact_248
generated_fact_249
generated_fact_257
generated_fact_258
generated_fact_259
generated_fact_261
generated_fact_272
generated_fact_278
generated_fact_282
generated_fact_285
generated_fact_286
jazz_archive
museum_purchase
railway_restoration
satellite_launch
```

## 10. Exact Train-Pair Reconstruction

Train pair identities are exactly:

`TRAIN_PAIR_SET = COMPLETE_FROZEN_300_PAIR_UNIVERSE - SELECTED_SEED8192_DEV_PAIR_SET`

The complete frozen 300-pair universe, sorted canonical order:

```text
archive_release
bridge_opening
canal_reopening
clinic_expansion
dam_inspection
ferry_launch
festival_selection
forest_mapping
garden_award
generated_fact_031
generated_fact_032
generated_fact_033
generated_fact_034
generated_fact_035
generated_fact_036
generated_fact_037
generated_fact_038
generated_fact_039
generated_fact_040
generated_fact_041
generated_fact_042
generated_fact_043
generated_fact_044
generated_fact_045
generated_fact_046
generated_fact_047
generated_fact_048
generated_fact_049
generated_fact_050
generated_fact_051
generated_fact_052
generated_fact_053
generated_fact_054
generated_fact_055
generated_fact_056
generated_fact_057
generated_fact_058
generated_fact_059
generated_fact_060
generated_fact_061
generated_fact_062
generated_fact_063
generated_fact_064
generated_fact_065
generated_fact_066
generated_fact_067
generated_fact_068
generated_fact_069
generated_fact_070
generated_fact_071
generated_fact_072
generated_fact_073
generated_fact_074
generated_fact_075
generated_fact_076
generated_fact_077
generated_fact_078
generated_fact_079
generated_fact_080
generated_fact_081
generated_fact_082
generated_fact_083
generated_fact_084
generated_fact_085
generated_fact_086
generated_fact_087
generated_fact_088
generated_fact_089
generated_fact_090
generated_fact_091
generated_fact_092
generated_fact_093
generated_fact_094
generated_fact_095
generated_fact_096
generated_fact_097
generated_fact_098
generated_fact_099
generated_fact_100
generated_fact_101
generated_fact_102
generated_fact_103
generated_fact_104
generated_fact_105
generated_fact_106
generated_fact_107
generated_fact_108
generated_fact_109
generated_fact_110
generated_fact_111
generated_fact_112
generated_fact_113
generated_fact_114
generated_fact_115
generated_fact_116
generated_fact_117
generated_fact_118
generated_fact_119
generated_fact_120
generated_fact_121
generated_fact_122
generated_fact_123
generated_fact_124
generated_fact_125
generated_fact_126
generated_fact_127
generated_fact_128
generated_fact_129
generated_fact_130
generated_fact_131
generated_fact_132
generated_fact_133
generated_fact_134
generated_fact_135
generated_fact_136
generated_fact_137
generated_fact_138
generated_fact_139
generated_fact_140
generated_fact_141
generated_fact_142
generated_fact_143
generated_fact_144
generated_fact_145
generated_fact_146
generated_fact_147
generated_fact_148
generated_fact_149
generated_fact_150
generated_fact_151
generated_fact_152
generated_fact_153
generated_fact_154
generated_fact_155
generated_fact_156
generated_fact_157
generated_fact_158
generated_fact_159
generated_fact_160
generated_fact_161
generated_fact_162
generated_fact_163
generated_fact_164
generated_fact_165
generated_fact_166
generated_fact_167
generated_fact_168
generated_fact_169
generated_fact_170
generated_fact_171
generated_fact_172
generated_fact_173
generated_fact_174
generated_fact_175
generated_fact_176
generated_fact_177
generated_fact_178
generated_fact_179
generated_fact_180
generated_fact_181
generated_fact_182
generated_fact_183
generated_fact_184
generated_fact_185
generated_fact_186
generated_fact_187
generated_fact_188
generated_fact_189
generated_fact_190
generated_fact_191
generated_fact_192
generated_fact_193
generated_fact_194
generated_fact_195
generated_fact_196
generated_fact_197
generated_fact_198
generated_fact_199
generated_fact_200
generated_fact_201
generated_fact_202
generated_fact_203
generated_fact_204
generated_fact_205
generated_fact_206
generated_fact_207
generated_fact_208
generated_fact_209
generated_fact_210
generated_fact_211
generated_fact_212
generated_fact_213
generated_fact_214
generated_fact_215
generated_fact_216
generated_fact_217
generated_fact_218
generated_fact_219
generated_fact_220
generated_fact_221
generated_fact_222
generated_fact_223
generated_fact_224
generated_fact_225
generated_fact_226
generated_fact_227
generated_fact_228
generated_fact_229
generated_fact_230
generated_fact_231
generated_fact_232
generated_fact_233
generated_fact_234
generated_fact_235
generated_fact_236
generated_fact_237
generated_fact_238
generated_fact_239
generated_fact_240
generated_fact_241
generated_fact_242
generated_fact_243
generated_fact_244
generated_fact_245
generated_fact_246
generated_fact_247
generated_fact_248
generated_fact_249
generated_fact_250
generated_fact_251
generated_fact_252
generated_fact_253
generated_fact_254
generated_fact_255
generated_fact_256
generated_fact_257
generated_fact_258
generated_fact_259
generated_fact_260
generated_fact_261
generated_fact_262
generated_fact_263
generated_fact_264
generated_fact_265
generated_fact_266
generated_fact_267
generated_fact_268
generated_fact_269
generated_fact_270
generated_fact_271
generated_fact_272
generated_fact_273
generated_fact_274
generated_fact_275
generated_fact_276
generated_fact_277
generated_fact_278
generated_fact_279
generated_fact_280
generated_fact_281
generated_fact_282
generated_fact_283
generated_fact_284
generated_fact_285
generated_fact_286
generated_fact_287
generated_fact_288
generated_fact_289
generated_fact_290
generated_fact_291
generated_fact_292
generated_fact_293
generated_fact_294
generated_fact_295
generated_fact_296
generated_fact_297
generated_fact_298
generated_fact_299
generated_fact_300
glacier_study
harbor_upgrade
jazz_archive
language_program
library_digitization
market_renovation
museum_purchase
opera_premiere
orion_approval
poetry_prize
railway_restoration
reef_protection
robotics_championship
satellite_launch
school_opening
solar_farm_contract
theater_restoration
treaty_signature
vaccine_delivery
volcano_observatory
wetland_restoration
```

This representation is fully reconstructable because the dev set is explicitly listed and the complete pair universe is explicitly listed.

## 11. Exact Row Identities And Integrity

Ordered row identity hashes use `scripts/train_controlled_v6b_minimal.py::_p2_row_identity_hash` semantics:

- ordered train row count: `2880`
- ordered train row identity hash: `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8`
- ordered dev row count: `720`
- ordered dev row identity hash: `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4`

Integrity checks:

| Check | Result |
|---|---|
| every dataset row occurs exactly once in train or dev | PASS |
| no row omitted | PASS |
| no row duplicated | PASS |
| no pair crosses train/dev | PASS |
| unique row IDs in partition | `3600` |
| total dataset rows | `3600` |
| pair leakage count | `0` |

## 12. Selected Split Primary Reason Counts

| Split | FRAME | PREDICATE | SUFFICIENCY | AUTHORIZED |
|---|---:|---:|---:|---:|
| train | `714` | `119` | `238` | `338` |
| dev | `186` | `31` | `62` | `81` |

Threshold checks:

- train FRAME `714 >= 50`: PASS
- train PREDICATE `119 >= 50`: PASS
- train SUFFICIENCY `238 >= 50`: PASS
- train AUTHORIZED `338 >= 50`: PASS
- dev FRAME `186 >= 20`: PASS
- dev PREDICATE `31 >= 20`: PASS
- dev SUFFICIENCY `62 >= 20`: PASS
- dev AUTHORIZED `81 >= 20`: PASS

## 13. Selected Split Applicable Binary Counts

| Split | Cohort | Class 0 | Class 1 | Readiness |
|---|---|---:|---:|---|
| train | frame | `714` | `695` | PASS |
| train | predicate | `119` | `576` | PASS |
| train | sufficiency | `238` | `338` | PASS |
| train | polarity | `100` | `238` | PASS |
| dev | frame | `186` | `174` | PASS |
| dev | predicate | `31` | `143` | PASS |
| dev | sufficiency | `62` | `81` | PASS |
| dev | polarity | `19` | `62` | PASS |

Applicable polarity labels:

| Split | REFUTE(0) | SUPPORT(1) |
|---|---:|---:|
| train | `100` | `238` |
| dev | `19` | `62` |

Direct polarity labels:

| Split | REFUTE | SUPPORT |
|---|---:|---:|
| train | `361` | `359` |
| dev | `89` | `91` |

The selected split contains no A1/A3 readiness degeneracy.

## 14. Dataset/Pair/Row Integrity

Dataset and split identity:

- tracked dataset blob SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- dataset semantic SHA256: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- rows: `3600`
- pairs: `300`
- rows per pair: `12` for every pair
- selected train rows: `2880`
- selected dev rows: `720`
- selected train pairs: `240`
- selected dev pairs: `60`
- pair leakage: `0`

No dataset row content, labels, pair IDs, eligibility, or applicability fields were modified.

## 15. Scientific Invariants

This selection authority preserves exactly:

- dataset identity/content
- labels
- eligibility
- P2 applicability
- polarity mapping `REFUTE -> 0`, `SUPPORT -> 1`
- pair-group semantics
- dev ratio `0.2`
- training seeds `180/181/182`
- A1/A3 reason supervision enabled
- A2 reason supervision disabled
- current fail-closed readiness guard semantics
- current dev reason-validation semantics
- router semantics
- gradient ownership
- final CE ownership
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`
- secondary reasons diagnostic-only

This authority changes only exact split selection.

## 16. Seed174 Historical Boundary

Seed174 remains historically valid for its original evidence lineage.

This candidate does not retroactively invalidate:

- historical A0 evidence
- historical calibration
- historical reports

Seed174 must not be used for the future revised-split factorial. The newly selected split supersedes seed174 only for the future revised-split lineage after this selection authority itself becomes active and downstream split-bound artifacts are reconstructed.

## 17. Downstream Artifact Boundary

Selecting the split does not itself make downstream artifacts valid.

Still required:

1. revised P4-L reconstruction/rebinding/provenance
2. revised-split A0 baseline evidence
3. revised-split reason-loss recalibration plus acceptance
4. revised factorial contract
5. required implementation/prelaunch control
6. validation/freeze
7. new explicit execution authority
8. only then Kaggle/training

Existing P4-L sidecar, A0 same-seed references, weight `0.6202430063306562`, and old factorial execution authority must not be rebound automatically merely because an exact new split is selected.

## 18. Prelaunch-Control Boundary

Future fail-closed static P2 applicable-cohort feasibility validation remains required before:

`TRAINER_PROCESS_LAUNCH_BEGIN`

For exact selected split seed `8192`, the selection-time static readiness computation passes now. This is selection-time evidence only and does not substitute for the future execution-time prelaunch check. This candidate does not implement the check.

## 19. Execution Boundary

`ACTIVE_REVISED_SPLIT_DESIGN_SELECTION_AUTHORITY = NONE_YET`

`CURRENT_EXECUTION_STATE = BLOCKED_PENDING_NEW_EXPLICIT_EXECUTION_AUTHORITY`

| Item | Status |
|---|---|
| A0 revised-split execution | `NOT_AUTHORIZED` |
| A1/A2/A3 | `BLOCKED` |
| Training | `NOT_AUTHORIZED` |
| Evaluation | `NOT_AUTHORIZED` |
| CUDA | `NOT_AUTHORIZED` |
| Kaggle | `NOT_AUTHORIZED` |
| Calibration execution | `NOT_AUTHORIZED` |
| P4-L regeneration | `NOT_AUTHORIZED_BY_THIS_DOCUMENT` |

No run name. No recovery4.

## 20. Non-Inheritance

This exact seed was not inherited from premature factorial-v2 artifacts, historical non-activated split proposals, root-cause candidates, or any result not authorized by `c82a164ac460599c68318a3b29180303f12cbc1a`.

The exact seed selected here arises solely from:

1. active `c82a164ac460599c68318a3b29180303f12cbc1a` split-contract remedy decision authority;
2. this task's pre-registered candidate domain;
3. this task's pre-registered max-overlap/minimum-perturbation criterion;
4. exact current frozen data/readiness semantics.

## 21. Finalized Candidate Path

`reports/reason_router_p3w7_p2_degeneracy_revised_split_design_selection_authority_spec_candidate.md`

This finalized report content is `PASS_READY_FOR_FREEZE`, but it is not active because it exists. It is not `ACTIVE` or `ACTIVATED`. It does not itself authorize staging, commit, or push.

`ACTIVE_REVISED_SPLIT_DESIGN_SELECTION_AUTHORITY = NONE_YET`

`ACTIVATION_CONDITION = ON_EXACT_COMMIT_PUSH_REMOTE_VERIFICATION`

This exact finalized authority content may be recognized as active only after all of the following lifecycle controls have been fulfilled:

1. this exact finalized file is explicitly staged;
2. the exact staged Git blob is independently byte-verified against the finalized file identity;
3. the staged delta contains exactly this one intended file and no other file;
4. a dedicated freeze/activation commit is created;
5. its exact full commit SHA is obtained;
6. its parent is verified to be the expected current authority HEAD;
7. that exact commit is pushed;
8. the remote branch tip is verified to equal that exact commit;
9. the remote blob identity is verified against the exact staged/finalized identity;
10. the remote body-level lifecycle/status and exact scientific selection are independently verified.

Before all conditions are fulfilled:

`ACTIVE_REVISED_SPLIT_DESIGN_SELECTION_AUTHORITY = NONE_YET`

After all conditions are fulfilled, the controller may recognize the dedicated commit as the active revised split design/selection authority.

`COMMIT_MESSAGE_DOES_NOT_OVERRIDE_BODY_LEVEL_AUTHORITY_STATUS`

Commit subject alone cannot activate this authority.

## 22. Git State After Intended Materialization

Expected final state after finalizing this one candidate:

- `git branch --show-current`: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- `git rev-parse HEAD`: `c82a164ac460599c68318a3b29180303f12cbc1a`
- `git status --short`: exactly one untracked file, this report
- `git diff --name-status`: empty
- `git diff --cached --name-status`: empty
- `git diff --check`: no tracked diff errors

## 23. Blocker/Mismatch

No selection blocker remains in this candidate body.

Line-ending note: direct working-tree hashing of the frozen JSONL on this Windows checkout produced a different byte hash from the authoritative tracked blob. The tracked Git blob byte hash matches the authority (`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`), the worktree is clean, and the parsed dataset content reproduces the verified row/pair counts and feasibility aggregate exactly.

## 24. Exact Next Authorized Action

Final independent verification of this finalized candidate only:

1. Verify the finalized candidate file is the only untracked file and no tracked/staged changes exist.
2. Verify the finalized candidate file byte identity and lifecycle/status markers.
3. Verify the scientific selection remains seed `8192`.
4. Verify pair, row, hash, and readiness counts in this report remain unchanged.
5. Verify execution remains blocked and no downstream phase is authorized.

No commit, push, staging, training, evaluation, Kaggle, CUDA, A0, A1/A2/A3, P4-L regeneration, or calibration execution is authorized by this finalized candidate.
