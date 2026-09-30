# ContraMamba Gen5 Phase 1B — R22 Local Necessity Full CUDA Execution Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_R22_LOCAL_NECESSITY_FULL_CUDA_EXECUTION`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED_AFTER_FREEZE =
`YES_BOUNDED_THIN_CUDA_RUNNER_ONLY`

SCIENTIFIC_EXECUTION_ALLOWED_AFTER_ZERO_FORWARD_VERIFICATION =
`YES_ONE_FULL_300_PAIR_RUN`

KAGGLE_ALLOWED_AFTER_ZERO_FORWARD_VERIFICATION =
`YES_T4_ONLY`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This authority advances the already-frozen Phase 1B local-necessity protocol
from a qualified one-pair CUDA backend to one full 300-pair confirmatory run.

It does not change the scientific question, cohort, intervention, endpoint,
confirmatory statistic, decision threshold, or advancement rule.

The only newly permitted implementation is a thin full-cohort CUDA execution
runner that imports and reuses the exact already-qualified backend.

---

## 1. Frozen parent authorities and implementations

PHASE1B_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

NECESSITY_IMPLEMENTATION_AUTHORITY_COMMIT =
`eed071f4dc93f49973da8b2313d7d1b390e0e5d1`

Q22_CAUSAL_ORDER_CORRECTION_COMMIT =
`d31b9df1fddfb719d95a6d486ebb8f18b397f6ad`

NECESSITY_SEMANTIC_IMPLEMENTATION_FREEZE_COMMIT =
`130a3474cf83d1ba6080561afbf886614dde4786`

CUDA_EQUIVALENCE_AUTHORITY_COMMIT =
`9770a2f0588353b0708b760e7fcce6083f4c7e06`

CUDA_EQUIVALENCE_IMPLEMENTATION_FREEZE_COMMIT =
`5075d862c69ca406b12c30f9ca438c3203aef2a2`

CUDA_EQUIVALENCE_ARTIFACT_FREEZE_COMMIT =
`96f8a9a8385d71175db6c0d52a86f16c5ea75040`

R22_C22_ARTIFACT_FREEZE_COMMIT =
`1d3542013934870aa9181d1bbaf565ff4724112c`

The frozen semantic-reference necessity runner remains:

`scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py`

GIT_BLOB_SHA1 =
`50be219720b2809969b4f3caac25b2c25598a81b`

It must not be modified.

The qualified CUDA backend implementation remains:

`scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py`

GIT_BLOB_SHA1 =
`421d30f00cf71690ed41c983ccf0540808e1de1c`

It must not be modified.

---

## 2. Qualified CUDA backend evidence

QUALIFIED_BACKEND =
`FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY`

EQUIVALENCE_RESULT =
`PASS_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE`

EQUIVALENCE_PAIR =
`xg1_fact_7801`

EQUIVALENCE_EXECUTION_HEAD =
`5075d862c69ca406b12c30f9ca438c3203aef2a2`

EQUIVALENCE_ARTIFACT_ROOT =
`reports/reason_router_gen5_phase1b_q22_cuda_equivalence_5075d86_r3`

EQUIVALENCE_SUMMARY_SHA256 =
`fde0ca28dbc22e5ab953bcd6dc68c46ba32be9c25cf5f587cfd5f608ae7ff528`

EQUIVALENCE_ITEMS_SHA256 =
`24f25de8bb2c87b45b2a4e6f8e3969195b821596b422cdaa6ab15cb43542dd84`

EQUIVALENCE_MANIFEST_SHA256 =
`a0c6944e35246ac81f57a5b642a9eb20921a2798ba51de03e8c77e472701cff2`

The imported equivalence artifact was validated after handoff import.

Observed maxima from the frozen summary:

- WRITE22 max absolute difference:
  `9.5367431640625e-07`
- POST_STATE22 max absolute difference:
  `2.384185791015625e-06`
- PE22 max absolute difference:
  `5.324092411385095e-08`
- F22 max absolute difference:
  `8.928854655643192e-08`
- J22 max absolute difference:
  `2.6729579361006728e-06`
- coefficient max absolute difference:
  `8.843943248848518e-07`

All were inside the prospectively frozen tolerances.

The full CUDA necessity implementation must import and call the frozen backend
implementation. Copying, rewriting, approximating, substituting, or tuning the
qualified backend is forbidden.

---

## 3. Frozen necessity-confirmation population

DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_necessity_confirmation_v1`

PAIR_RANGE =
`xg1_fact_8101..xg1_fact_8400`

PAIR_COUNT =
`300`

CHECKSUMS_SHA256 =
`c2413ead1bb0e55ab079c6f8832e5338ad14cc5fe7394ba4814b37040d93aaed`

STRUCTURED_SOURCE_FACTS_SHA256 =
`813534ffa5753dcf84637c43e177a6f96dac45230ceb3326747a528330e5b285`

SIX_CELL_ROWS_SHA256 =
`cf27ab041f77302562fd4cdd3fa4f2fc24e4b3f8f7367460512476b77efb1d14`

STRUCTURAL_MANIFEST_SHA256 =
`043fb36890b2663e430fc018b7a72fc1d9872b7bed15c019e2c9e8c626d98b58`

TOKENIZER_ANCHOR_MANIFEST_SHA256 =
`5ed5dc57b95df7121b301db2414c886655771dc77a53e0c6cbaad210d4683f88`

TOKENIZER_ELIGIBILITY_SUMMARY_SHA256 =
`752361c5367a68ea460f10dd554fd385d325d2c9434e998714366723c3004373`

No row may be removed, reordered, replaced, filtered, or selected using
responses.

The restoration-confirmation cohort must remain scientifically unread.

The prior construction/equivalence pair `xg1_fact_7801` is not part of the
confirmatory necessity population.

---

## 4. Frozen R22/C22 ownership objects

R22_PATH =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/r22_basis.f64le`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_PATH =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/c22_basis.f64le`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

RANK =
`2`

STATE_WIDTH =
`24576`

No reconstruction, refit, reranking, sign change, rank change, or alternate
basis selection is authorized.

---

## 5. Frozen model and tokenizer identity

REPRESENTATIVE_CHECKPOINT =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

CANONICAL_MODEL_TOKENIZER_REVISION =
`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

MODEL_CONFIG_SHA256 =
`784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`

TOKENIZER_JSON_SHA256 =
`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

TOKENIZER_CONFIG_SHA256 =
`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

SPECIAL_TOKENS_MAP_SHA256 =
`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

No checkpoint, model revision, or tokenizer substitution is permitted.

---

## 6. Frozen CUDA runtime

EXPECTED_RUNTIME =

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- CUDA runtime `12.8`
- device `Tesla T4`
- compute capability `7.5`

MAMBA_KERNEL_REVISION =
`c8ffc584c147878a6eb978ae0e8db4d116c93a8c`

CAUSAL_CONV1D_KERNEL_REVISION =
`f2651e776f66069cdcf842840db637583def1223`

BUILD_VARIANT =
`torch210-cxx11-cu128-x86_64-linux`

MAMBA_BINARY_SHA256 =
`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

CAUSAL_CONV1D_BINARY_SHA256 =
`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

CPU scientific model forwards are forbidden.

---

## 7. Frozen scientific coordinates

UPSTREAM_LAYER17_CONDITION =
`NATIVE`

UPSTREAM_SIGNED_PROBE_SITE =
`FROZEN_LAYER17_TARGET_TOKEN_STRONG_CHANNEL_COORDINATE`

TARGET_LAYER =
`22`

TARGET_OBJECT =
`WRITE22`

POST_STATE_OBJECT =
`POST_STATE22`

CONDITIONS =
`NATIVE22 / R22_NEUTRALIZED / C22_COEFFICIENT_CONTROL`

BASIS_FAMILIES =
`XG2 / XG4`

DIRECTIONS_PER_FAMILY =
`5`

EPSILON =
`0.025`

ORIENTATIONS =
`+1 / -1`

BRANCHES =
`TP / TM`

Q22_READOUT =
`core.post4_path_efficiency@layer22`

For native write vector `w`:

`a = R22^T w`

`N_R(w) = w - R22 a`

`N_C(w) = w - C22 a`

The exact same R22-derived coefficient vector `a` must be used for both
interventions.

The matched correction-norm equality remains mandatory.

---

## 8. Full-run CUDA computation

For every branch, the model executes once on CUDA using the frozen fast
upstream.

The frozen layer-17 signed probe is applied exactly as in the semantic
necessity implementation.

At layer 22 the runner must use the qualified backend imported from:

`scripts.reason_router_gen5_phase1b_q22_cuda_equivalence`

The backend semantics are fixed:

1. capture exact layer-22 `selective_scan_fn` inputs;
2. obtain the state before the five-token Q22 window with the exact frozen
   `selective_scan_fn`;
3. replay ordinary Q22-window tokens with the exact frozen
   `selective_state_update`;
4. at the target token recover
   `WRITE22_fast = STATE_NATIVE_UPDATE - STATE_DECAY_ONLY`;
5. apply the exact frozen NATIVE22/R22/C22 rule;
6. continue state replay with the exact frozen kernel;
7. compute Q22 only from the resulting layer-22 POST_STATE22 window.

No hand-written recurrence replacement is authorized.

No full-model continuation after the local Q22 readout is scientifically
required or authorized.

---

## 9. Exact forward budget

Per pair:

- 3 conditions
- 10 directions
- 2 orientations
- 2 TP/TM branches

Therefore:

`FORWARDS_PER_CONDITION = 40`

`FORWARDS_PER_PAIR = 120`

`PAIR_COUNT = 300`

`FULL_CUDA_MODEL_FORWARD_BUDGET = 36000`

Exactly 36,000 scientific model forwards are authorized for the completed full
run.

A run with fewer or more scientific model forwards is not a valid confirmatory
run.

No CPU scientific model forward is authorized.

---

## 10. Frozen item endpoints

For every pair compute:

`Q0 = Q22(NATIVE22)`

`QR = Q22(R22_NEUTRALIZED)`

`QC = Q22(C22_COEFFICIENT_CONTROL)`

Then use the frozen semantic-reference function:

`necessity.endpoint(Q0, QR, QC)`

which defines:

`A_R = Q0 - QR`

`A_C = Q0 - QC`

`D_NEC22 = QC - QR`

The full CUDA runner must not implement a second independent version of these
endpoint identities.

---

## 11. Frozen confirmatory decision

Only after all 300 pair items have completed, the exact full forward budget has
been asserted, and parameter immutability has been verified, the runner must
call:

`necessity.confirmatory_decision(items)`

This frozen function supplies the sole confirmatory statistic and decision
rule.

Exactly one confirmatory p-value is authorized:

one-sided one-sample Student t-test on `D_NEC22`.

Positive necessity requires all of:

1. all provenance/manipulation gates pass;
2. `mean(Q0) > 0`;
3. `mean(A_R) > 0`;
4. `mean(D_NEC22) > 0`;
5. `p_one_sided_greater < 0.05`.

Positive label:

`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

Otherwise:

`GEN5_R22_LOCAL_NECESSITY_NOT_ESTABLISHED`

No additional p-value, alternative endpoint, subgroup confirmatory test,
multiple-testing procedure, threshold change, or post-hoc exclusion is
authorized.

---

## 12. Manipulation and provenance gates

The full CUDA runner must fail closed on at least:

- branch/head/authority identity mismatch;
- qualified-backend source blob mismatch;
- frozen equivalence artifact identity/result mismatch;
- checkpoint SHA256 mismatch;
- model/tokenizer revision or file-hash mismatch;
- CUDA runtime/device/kernel identity mismatch;
- necessity population hash/order/count mismatch;
- R22/C22 hash/shape/orthonormality/cross-orthogonality mismatch;
- target-token mismatch;
- layer-17 probe sign or magnitude mismatch;
- native WRITE22 inconsistency across the three conditions;
- R22/C22 coefficient-transfer mismatch;
- correction-norm inequality;
- nonfinite PE22/F22/J22/E/Q values;
- forward-budget mismatch;
- model parameter mutation;
- any training/backward/task-head optimization/logit use;
- response-guided row dropping;
- restoration-cohort scientific access.

---

## 13. Allowed CUDA-runner implementation files

Exactly these new files may be created:

`scripts/reason_router_gen5_phase1b_r22_local_necessity_fast_cuda.py`

`scripts/verify_reason_router_gen5_phase1b_r22_local_necessity_fast_cuda.py`

`tests/test_reason_router_gen5_phase1b_r22_local_necessity_fast_cuda.py`

Existing production files must not be modified.

In particular, do not modify:

- the frozen CPU semantic-reference necessity runner;
- the frozen CUDA equivalence runner;
- Gen4 production files;
- data files;
- frozen artifacts;
- README.

The new CUDA runner should be a thin composition layer over frozen functions,
not a copied scientific implementation.

---

## 14. Zero-forward verification before execution

Before the 36,000-forward run, all of the following must pass:

1. `git diff --check`;
2. narrow pytest for the new CUDA-runner test file;
3. independent static verifier;
4. exact qualified-backend source identity check;
5. exact frozen semantic necessity source identity check;
6. exact equivalence artifact identity/result check;
7. exact necessity population static identity check;
8. synthetic tests of endpoint/control plumbing only;
9. model loaded = false;
10. checkpoint loaded = false;
11. model forward count = 0;
12. CUDA scientific execution = false;
13. scientific p-value count = 0.

Passing this verification authorizes the one full run under this same authority.

It does not itself establish necessity.

---

## 15. Output artifact boundary

The completed full run may persist only:

`r22_local_necessity_items.jsonl`

`r22_local_necessity_summary.json`

`artifact_manifest.json`

`SHA256SUMS.txt`

The implementation should reuse the frozen semantic necessity artifact
serialization where compatible.

No raw full WRITE22 tensor or POST_STATE22 tensor may be persisted.

Hashes, scalar diagnostics, norms, residuals, per-direction scalar endpoints,
and per-pair scalar endpoints are allowed.

The summary must record at least:

- exact execution HEAD;
- this authority commit;
- qualified CUDA-equivalence artifact freeze commit;
- exact backend identity;
- exact runtime/kernel identity;
- exact checkpoint/tokenizer identities;
- source pair count/range;
- 36,000 CUDA scientific forwards;
- zero CPU scientific forwards;
- model parameter signature before/after;
- exactly one confirmatory p-value;
- final frozen decision label;
- zero restoration execution;
- zero ownership implementation.

---

## 16. Kaggle and run provenance

Execution is permitted only after:

- this authority is frozen;
- the bounded CUDA runner is implemented and frozen;
- zero-forward verification passes;
- commit/push completes;
- `cm kaggle` authenticates the exact execution commit;
- the exact shell command is registered with `cm run save`;
- the Kaggle run is pinned to the full commit SHA and command SHA256.

The GPU must be Tesla T4 under the frozen runtime.

The run must be collected with `cm collect` and imported with `cm import`
before scientific interpretation.

A successful Kaggle process exit alone is not sufficient evidence.

---

## 17. Failure and retry boundary

If the run fails before all 300 pairs and the exact 36,000-forward budget are
completed:

- do not compute or interpret a confirmatory p-value from partial data;
- preserve run/log provenance;
- do not merge partial items with a later run unless a separately authorized
  restartable protocol explicitly permits it;
- use a new run name for any retry;
- do not relax tolerances, alter the backend, change the cohort, or substitute
  runtime/model assets.

An implementation defect requires a correction authority and a new frozen
implementation commit before retry.

---

## 18. Advancement

After successful collect/import and artifact validation:

If the frozen decision label is

`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

then the next permissible scientific stage is preparation of the frozen
Phase 1B restoration-confirmation implementation/authority.

If the label is

`GEN5_R22_LOCAL_NECESSITY_NOT_ESTABLISHED`

then the Phase 1B bridge is not established and restoration must not be promoted
as if necessity had passed.

Neither outcome by itself establishes the later Gen5 ownership-benefit
hypothesis.

---

## 19. Success boundary

Freezing this authority establishes only permission to:

1. implement the exact thin CUDA full-cohort runner in the three-file whitelist;
2. perform zero-forward independent verification;
3. freeze that implementation;
4. execute exactly one full 300-pair / 36,000-forward CUDA necessity run;
5. collect and import the resulting four artifacts.

It does not establish the necessity result in advance.

It does not authorize restoration execution.

It does not authorize ownership implementation or training.

STATUS =
`FROZEN_ON_COMMIT`
