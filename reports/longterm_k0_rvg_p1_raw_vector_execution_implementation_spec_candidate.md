# K0-RVG-P1 Scientific Raw-Vector Execution Implementation Specification Candidate

**Status:** scientific execution implementation specification candidate only.

**Immediate parent authority:**

`265d060e96f98f73e4a40a9c54a88c3de482a312`

**Frozen K0-RVG-P preregistration commit:**

`cdad87acf664cd61e48406f9d4568b6ab206da24`

**Frozen K0-RVG-P0 implementation commit:**

`421d79815045938ea45ddcbc37a870a14218d133`

**Frozen K0-RVG-P0 archive commit:**

`265d060e96f98f73e4a40a9c54a88c3de482a312`

**Validated raw recurrence observer commit:**

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

This document specifies the implementation of a future scientific raw-vector execution runner.

It does **not** authorize execution on the frozen 336-item scientific population.

It does **not** authorize scientific model forward, scientific recurrent-state read, scientific endpoint computation, causal intervention, or K4.

After this exact specification is frozen, bounded implementation and fabricated synthetic validation may proceed under the scope below.

## 1. Objective

Implement a fail-closed scientific runner that can, under a later separate execution-authority artifact:

1. authenticate the exact frozen P0 scientific-input artifacts;
2. authenticate the exact validated raw recurrence observer and runtime;
3. reconstruct the four preregistered branches per item;
4. capture native Mamba recurrence tuples at the exact preregistered token coordinates;
5. compute the two preregistered primary raw-vector endpoints at layer 23;
6. compute exact sign tests and Holm correction across exactly two primary endpoints;
7. emit deterministic scientific result/provenance artifacts without persisting raw recurrent tensors.

The implementation must be complete before any scientific recurrent state from the frozen P0 population is read.

## 2. Scientific branch state

The branch state remains unchanged:

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

The P1 implementation itself does not activate either branch.

## 3. Exact implementation scope after P1 freeze

After this exact specification is frozen, implementation may create exactly two new files:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

No existing repository file may be modified.

Historical untracked K1 files remain untouched:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

Implementation validation must stop after:

- the exact two-file delta;
- focused tests;
- fabricated synthetic integration validation;
- independent implementation review.

No scientific execution is authorized at P1 implementation time.

## 4. Frozen P0 archive dependency

Archive directory:

`reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1`

The runner must authenticate these exact paths and SHA256 identities.

### candidate_pool.jsonl

`743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

### generated_source.jsonl

`8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

### phase_pair_mapping.json

`c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

### token_contracts.jsonl

`6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

### provisioning_manifest.json

`feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

### validation_report_candidate.md

`ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

Any mismatch blocks both synthetic integration and later scientific execution.

The runner must consume the archived artifacts.

It must not regenerate the scientific population from the generator.

## 5. Frozen P0 scientific-input contract

Expected:

- item count: `336`;
- phase block count: `168`;
- source-row count: `4368`;
- correction-source counts:
  - `none = 168`;
  - `polarity_flip = 168`;
- prior-overlap counts: all `12` values equal `0`;
- matched divergence-offset histogram:
  - offset 1: `168`;
  - offset 2: `168`;
- swapped divergence-offset histogram:
  - offset 1: `168`;
  - offset 2: `168`;
- minimum post-divergence token availability: `20`.

The runner must fail closed if the archived manifest or parsed artifact content disagrees with these frozen values.

## 6. Frozen observer dependency

Observer path:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

Required SHA256:

`12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

Required Git blob:

`f2dbdfe52661eca384897578ab272e602e36deac`

Validated observer commit:

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

The runner may import and use only the already-frozen public measurement objects/functions needed for scientific execution, including:

- `resolve_source_binding`;
- `registered_mamba_layers`;
- `RawRecurrenceCollector`;
- `validate_recurrence_record`;
- `RecurrenceRecord`.

The observer file must not be modified.

## 7. Frozen historical runtime bridge

Historical K2S helper:

`scripts/longterm_k2s_pair_specific_event_dynamics.py`

Required SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

Required Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

The P1 runner may reuse the already-validated K2S utilities for:

- handoff authentication;
- checkpoint loading;
- encoder fingerprint;
- frozen HF snapshot resolution;
- A0 model construction;
- task-mask bundle construction;
- ordinary full model forward.

The P1 runner must not use historical scientific population loaders or historical scientific endpoint logic.

## 8. Frozen model/runtime identity

Model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Installed Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Upstream Mamba source Git blob:

`87987e3e6646d8d0f9f0048bdd8a155d99c845db`

Scientific measurement runtime is restricted to the validated sequential CPU slow path.

The runner must fail if captured recurrence tensor metadata is not:

- original device `cpu`;
- dtype `torch.float32`;
- shape `(1,1536,16)` at frozen layer 23.

No CUDA fused-kernel execution is authorized for this study.

## 9. Frozen checkpoint/handoff identity

Seed180 handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Common encoder canonical SHA256:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Common encoder raw-concat SHA256:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

The future execution authority must bind the exact handoff path/bytes used at runtime.

P1 implementation validation may load this checkpoint only for **fabricated synthetic integration**.

## 10. Primary layer and tensor coordinate

Frozen primary layer:

`23`

The scientific object is the exact raw native recurrent state:

`S_t ∈ R^(1536×16)`

captured post-consumption at the validated recurrence timing.

The primary native geometry is the raw Frobenius geometry.

No learned/tuned metric is allowed.

No whitening, PCA projection, probe, semantic direction, outcome-selected subspace, channel selection, or state-mode selection may enter primary inference.

## 11. Scientific branch reconstruction

For candidate item `i`, let `mate(i)` be the phase mate recorded by the frozen P0 mapping.

Construct exactly:

`M_corr(i) = prefix_i + correction_i`

`M_ctrl(i) = prefix_i + control_i`

`S_corr(i) = prefix_i + correction_mate(i)`

`S_ctrl(i) = prefix_i + control_mate(i)`

No text normalization, whitespace repair, retokenization-based rewrite, or regeneration is permitted.

Branch text must be reconstructed exactly from the archived candidate rows.

## 12. Token-contract revalidation before model forward

Before any future scientific model forward, the runner must:

1. authenticate the P0 archive hashes;
2. reconstruct the four exact branch texts;
3. tokenize with the frozen tokenizer and `add_special_tokens=False`;
4. require exact prefix-token identity;
5. recompute matched and swapped divergence anchors;
6. require exact equality to the archived token-contract values;
7. require exact prefix token SHA256 equality;
8. require W=8 availability;
9. require `t_e >= 1`.

Any mismatch blocks the entire scientific run before the first model forward.

No item replacement is permitted.

## 13. Execution order

Future scientific execution order is frozen as:

ascending `local_template_index`:

`0..335`

Within one item:

1. matched correction;
2. matched control;
3. swapped correction;
4. swapped control.

No outcome-dependent reordering is allowed.

No adaptive stopping is allowed.

A partial run is not scientifically interpretable.

## 14. Exact recurrence coordinates per branch pair

For a pair-specific divergence anchor `t_e`, target token indices are exactly:

`t_e - 1`

and:

`t_e, t_e+1, ..., t_e+7`

for a total of:

`9`

target coordinates.

The `t_e - 1` record defines the incoming velocity.

The next 8 records define the frozen post-divergence window.

Matched and swapped pairs may have different `t_e` values.

## 15. Complete observer capture requirement

For every branch forward:

- instantiate a fresh single-use `RawRecurrenceCollector`;
- target exactly the 9 frozen token coordinates for that pair;
- require capture completeness across all 24 registered Mamba layers:
  `24 × 9 = 216` records;
- extract scientific metrics from layer 23 only;
- discard nonprimary-layer tensors immediately after completeness validation.

Nonprimary-layer records may not enter scientific inference or layer selection.

## 16. Recurrence validation during scientific execution

For every layer-23 target record used in a scientific metric:

1. require the exact source recurrence:
   `torch.equal(G_t ⊙ S_(t-1) + W_t, S_t)`;
2. require the frozen velocity rearrangement tolerance:
   - `atol = 1e-6`;
   - `rtol = 1e-5`.

Any layer-23 recurrence failure invalidates the whole run.

The runner must aggregate and report:

- maximum velocity absolute residual;
- maximum relative Frobenius residual;
- maximum scaled-tolerance residual;
- count of exact recurrence checks.

## 17. Incoming-velocity common-state check

Within each correction-control pair, the runner must require exact equality at `t_e-1` between correction and control for all four recurrence roles:

- `S_prev`;
- `G`;
- `W`;
- `S_post`.

This proves the frozen incoming velocity is common to the pair before divergence.

Failure blocks the entire run.

The incoming velocity is then defined as:

`U = S_post(t_e-1) - S_prev(t_e-1)`.

## 18. Raw velocity

For each post-divergence token:

`tau = 0,...,7`

define:

`V_tau^(corr) = S_post^(corr)(t_e+tau) - S_prev^(corr)(t_e+tau)`

`V_tau^(ctrl) = S_post^(ctrl)(t_e+tau) - S_prev^(ctrl)(t_e+tau)`

All tensor arithmetic is performed on the exact float32 snapshots.

All inner products, squared norms, sums, means, Gram matrices, and scalar endpoint accumulation use float64.

## 19. Frobenius cosine

For tensors `X,Y`:

`dot_F(X,Y) = sum(float64(X) * float64(Y))`

`norm_F(X) = sqrt(dot_F(X,X))`

`cos_F(X,Y) = dot_F(X,Y) / (norm_F(X) * norm_F(Y))`

A scientific cosine is defined only if both norms are strictly greater than zero.

No epsilon may be added to a scientific cosine denominator.

Undefined cosines propagate endpoint invalidity according to the preregistered support rule.

## 20. Primary P1 turning endpoint

For one correction-control pair:

`A_corr = mean_tau cos_F(V_tau^(corr), U)`

`A_ctrl = mean_tau cos_F(V_tau^(ctrl), U)`

for `tau=0..7`.

Turning contrast:

`T = A_ctrl - A_corr`

For candidate item `i`:

`T_M(i)` = matched turning contrast.

`T_S(i)` = swapped turning contrast.

Primary item turning score:

`X_turn(i) = T_M(i) - T_S(i)`.

No alternative incoming reference direction is permitted.

## 21. Primary P2 response-coherence endpoint

Response vector:

`R_tau = V_tau^(corr) - V_tau^(ctrl)`.

Pair response-direction coherence:

`C = (1/7) * sum_(tau=1..7) cos_F(R_tau, R_(tau-1))`.

For candidate item `i`:

`C_M(i)` = matched coherence.

`C_S(i)` = swapped coherence.

Primary item coherence score:

`X_coh(i) = C_M(i) - C_S(i)`.

No alternative lag, smoothing, weighting, or window is permitted.

## 22. Phase-block aggregation

For block index:

`p = 0..167`

items are exactly:

`i = p`

`j = p + 168`.

Block turning:

`B_turn(p) = (X_turn(i) + X_turn(j)) / 2`.

Block coherence:

`B_coh(p) = (X_coh(i) + X_coh(j)) / 2`.

A block is invalid for one endpoint if either reciprocal item is invalid for that endpoint.

No valid item may be paired with a different mate.

## 23. Zero-vector support rule

For each endpoint independently:

- valid block count is the number of blocks with all required scientific cosines defined;
- if:
  `valid_block_count < 160`
  then endpoint status is:
  `NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE`.

No epsilon rescue is allowed.

No replacement item is allowed.

No layer/window/metric rescue is allowed.

## 24. Sign classification

For every valid block score `B`:

- positive if `B > 0.0`;
- negative if `B < 0.0`;
- zero if `B == 0.0`.

No tolerance band is used for scientific sign classification.

Effective n:

`n_eff = positive + negative`.

Zeros are reported but excluded from the exact sign-test denominator.

## 25. Exact two-sided sign test

For endpoint counts:

`k = min(positive, negative)`

`n = positive + negative`

the raw p-value is:

`p = min(1, 2 * sum_(r=0..k) C(n,r) / 2^n)`.

The implementation must use exact integer binomial coefficients.

No asymptotic normal approximation is permitted.

If `n == 0`, the endpoint is not evaluable.

## 26. Holm correction

There are exactly two primary p-values:

- turning;
- response coherence.

Sort raw p-values ascending:

`p_(1) <= p_(2)`.

Adjusted values are:

`adj_(1) = min(1, 2*p_(1))`

`adj_(2) = min(1, max(adj_(1), p_(2)))`.

Map adjusted values back to their named endpoints.

No secondary diagnostic enters this multiplicity family.

## 27. Sign effect

For each evaluable endpoint:

`sign_effect = (positive - negative) / n_eff`.

Endpoint verdict:

If Holm-adjusted p `< 0.05` and sign effect `> 0`:

`<ENDPOINT>_POSITIVE_DIRECTIONAL_SIGNAL`

If Holm-adjusted p `< 0.05` and sign effect `< 0`:

`<ENDPOINT>_REVERSED_DIRECTIONAL_SIGNAL`

Otherwise:

`<ENDPOINT>_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED`.

Named endpoint prefixes are exactly:

- `TURNING`;
- `RESPONSE_COHERENCE`.

## 28. Overall verdict

If either primary endpoint is not evaluable:

`RAW_NATIVE_VECTOR_ORGANIZATION_NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE`

Otherwise apply the preregistered endpoint combination:

- both positive:
  `RAW_NATIVE_VECTOR_ORGANIZATION_CONVERGENT`;
- exactly one positive and the other not established:
  `RAW_NATIVE_VECTOR_ORGANIZATION_ENDPOINT_SPECIFIC`;
- neither significant:
  `RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED`;
- reversed without a positive counterpart:
  `RAW_NATIVE_VECTOR_ORGANIZATION_DIRECTIONALLY_CONTRADICTED_OR_MIXED`;
- one positive and one reversed:
  `RAW_NATIVE_VECTOR_ORGANIZATION_MIXED_NOT_PROMOTABLE`.

This machine label is not interpreted scientifically until execution artifacts are independently validated.

## 29. Exact carry/write observational diagnostics

For every layer-23 post-divergence record:

`V_t^(carry) = (G_t - 1) ⊙ S_(t-1)`

`V_t^(write) = W_t`

and:

`V_t = V_t^(carry) + V_t^(write)`.

For correction-control response:

`R_tau^(carry) = V_tau^(carry,corr) - V_tau^(carry,ctrl)`

`R_tau^(write) = V_tau^(write,corr) - V_tau^(write,ctrl)`.

The implementation may report, as descriptive diagnostics:

- Frobenius norms of total/carry/write response;
- `cos_F(R_write,R_total)`;
- `cos_F(R_carry,R_total)`;
- `cos_F(R_write,R_carry)`.

These diagnostics are observational only.

They do not enter primary inference.

No causal retain/write claim is permitted.

## 30. Singular-spectrum descriptive implementation

If singular-spectrum diagnostics are implemented, they must use the raw 8-token response matrix:

`M = [vec(R_0)^T; ...; vec(R_7)^T]`.

For numerical efficiency, the implementation may form the float64 `8×8` Gram matrix:

`K = M M^T`

and recover singular-value squares from the nonnegative eigenvalues of `K`.

This is descriptive only.

Singular vectors may not be used as confirmatory projection directions.

The singular spectrum may not rescue a failed primary endpoint.

P1 implementation correctness does not require singular-spectrum diagnostics if the runner omits them entirely.

## 31. Historical scalar diagnostics

Historical scalar quantities such as speed, displacement, path length, path efficiency, or correction-control separation magnitude may be omitted from the first P1 implementation.

If included, they must be explicitly marked secondary/descriptive.

No scalar p-value may enter the primary Holm family.

No scalar result may change the overall raw-vector verdict.

## 32. Raw tensor persistence policy

Raw recurrent tensors are **ephemeral execution memory only**.

The scientific runner must not persist full raw `S_prev`, `G`, `W`, `S_post`, `V`, or `R` tensors in repository artifacts.

For auditability, it must persist SHA256 hashes of each layer-23 recurrence role used by the primary endpoints.

Per coordinate/branch role hashes may be written to a compact audit artifact.

This preserves exact state identity without creating multi-gigabyte tensor artifacts.

## 33. Scientific result artifact set

A successful future scientific execution must emit exactly these result files in a fresh output directory:

`item_metrics.jsonl`

`block_metrics.jsonl`

`endpoint_summary.json`

`recurrence_audit.json`

`state_hash_audit.jsonl`

`execution_manifest.json`

No result file may be written until all 336 items have completed successfully.

The writer must use temporary files/directories and atomic finalization so a crashed partial run cannot masquerade as a complete scientific result.

## 34. item_metrics.jsonl

Exactly 336 rows in ascending local-template order.

Each row must contain at least:

- schema version;
- local template index;
- pair ID;
- stable item ID;
- phase block index;
- phase mate stable ID;
- matched divergence anchor;
- swapped divergence anchor;
- validity flags for turning/coherence;
- matched:
  - `A_corr`;
  - `A_ctrl`;
  - `T_M`;
  - `C_M`;
- swapped:
  - `A_corr`;
  - `A_ctrl`;
  - `T_S`;
  - `C_S`;
- `X_turn`;
- `X_coh`;
- count of recurrence coordinates validated;
- optional explicitly-labeled secondary diagnostics.

Undefined scientific scalars must be encoded as JSON `null`, never NaN.

## 35. block_metrics.jsonl

Exactly 168 rows in ascending block-index order.

Each row must contain:

- block index;
- phase class;
- item A stable ID;
- item B stable ID;
- turning validity;
- `B_turn` or null;
- coherence validity;
- `B_coh` or null;
- sign classification for each valid endpoint.

No alternative block ordering is permitted.

## 36. endpoint_summary.json

Must contain for each primary endpoint:

- valid block count;
- invalid block count;
- positive count;
- negative count;
- zero count;
- effective n;
- raw exact sign-test p or null;
- Holm-adjusted p or null;
- sign effect or null;
- endpoint verdict.

Must also contain:

- overall raw-vector verdict;
- alpha `0.05`;
- multiplicity method `HOLM_M2`;
- primary endpoint count `2`.

## 37. recurrence_audit.json

Must contain at least:

- layer: `23`;
- tensor shape;
- dtype;
- device;
- total layer-23 recurrence records validated;
- exact recurrence pass count;
- maximum velocity absolute residual;
- maximum relative Frobenius residual;
- maximum scaled-tolerance residual;
- exact incoming common-state comparison count;
- incoming common-state comparison failures, required `0`.

## 38. state_hash_audit.jsonl

Must record layer-23 hashes sufficient to bind every recurrence role used for primary metrics.

Each row must identify:

- local template index;
- branch role:
  - `matched_corr`;
  - `matched_ctrl`;
  - `swapped_corr`;
  - `swapped_ctrl`;
- token index;
- SHA256 of:
  - `S_prev`;
  - `G`;
  - `W`;
  - `S_post`.

Rows must be written in deterministic item / branch-role / token-index order.

## 39. execution_manifest.json

Must bind:

- runtime Git HEAD;
- P1 implementation authority commit;
- later scientific execution-authority path;
- later scientific execution-authority Git blob;
- P1 runner SHA256;
- P1 test SHA256;
- validated observer SHA256/blob/commit;
- K2S helper SHA256/blob;
- all six P0 archive file SHA256 values;
- handoff ZIP/checkpoint/encoder identities;
- HF model/revision;
- Transformers version;
- Mamba source SHA/blob;
- primary layer 23;
- W=8;
- item count 336;
- block count 168;
- result artifact SHA256 values;
- explicit flags:
  - `scientific_model_forward_executed = true`;
  - `scientific_recurrent_state_read = true`;
  - `logits_read = false`;
  - `causal_intervention_executed = false`;
  - `learned_geometry_used = false`;
  - `outcome_selected_subspace_used = false`.

These `true` scientific flags are permitted only in a future run authorized by a separate frozen execution-authority artifact.

## 40. Scientific execution authority gate

The P1 implementation must contain a scientific mode that is **fail-closed** without a later tracked execution-authority artifact.

Scientific CLI must require an explicit argument:

`--execution-authority <repo-relative-path>`

Before any scientific model construction, the runner must verify:

1. the authority path is a tracked file in runtime HEAD;
2. working-tree bytes equal `HEAD:<path>`;
3. the authority file declares the exact P1 implementation commit/hash identities;
4. the authority file declares the exact P0 artifact hashes;
5. the authority file declares:
   `SCIENTIFIC_EXECUTION_AUTHORIZED = YES`;
6. the authority file declares:
   `SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`;
7. the authority file declares:
   `SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`;
8. the authority file keeps:
   `LOGITS_READ_AUTHORIZED = NO`;
9. the authority file keeps:
   `CAUSAL_INTERVENTION_AUTHORIZED = NO`;
10. the authority file's containing commit is an ancestor of runtime HEAD.

Without all gates, the scientific runner must stop before loading the scientific model.

## 41. Scientific CLI shape

The implementation may expose:

`--synthetic-preflight`

and:

`--execute-scientific`

Scientific mode additionally requires:

- `--execution-authority`;
- exact seed180 handoff path;
- fresh output directory.

No exploratory flags may alter:

- layer;
- W;
- endpoint formulas;
- population slice;
- pair mapping;
- metric;
- alpha;
- multiplicity;
- zero policy.

No CLI override for these frozen quantities is allowed.

## 42. Fabricated synthetic integration validation

P1 implementation validation before freeze may execute the authentic frozen model/checkpoint only on fabricated non-study text.

Synthetic integration must:

- authenticate the frozen observer/runtime;
- construct at least two fabricated phase-mate-like items;
- construct matched and swapped correction/control branches;
- exercise divergence-anchor handling;
- capture 9 target coordinates;
- exercise layer-23 extraction;
- validate exact recurrence;
- validate incoming common-state equality;
- compute turning/coherence functions;
- exercise zero-vector invalidity on fabricated tensor fixtures;
- exercise exact sign-test/Holm code on fabricated scalar fixtures;
- verify deterministic repeated synthetic metrics/state hashes.

Synthetic validation must not use any P0 candidate branch text in a model forward.

## 43. State-blind P0 access during implementation validation

P1 implementation validation may read and hash the archived P0 artifacts to prove provenance.

It may parse metadata/token contracts for static validation.

It may not feed any of the 336 P0 scientific branch texts to the model.

It may not read recurrent states for any P0 item.

## 44. Required focused tests

The P1 test suite must cover at least:

1. authority and exact two-file scope constants;
2. P0 six-file SHA binding;
3. observer SHA/blob binding;
4. K2S helper SHA/blob binding;
5. handoff/checkpoint/encoder constants;
6. frozen layer 23 and W=8 constants;
7. exact branch reconstruction;
8. token-contract revalidation;
9. 9-coordinate target construction;
10. exact incoming common-state check;
11. float64 Frobenius dot/norm/cosine;
12. zero-vector cosine undefined behavior;
13. turning endpoint formula;
14. response-coherence endpoint formula;
15. phase-block aggregation;
16. support threshold 160;
17. sign classification with exact zero;
18. exact two-sided sign test;
19. Holm m=2 adjustment;
20. endpoint verdict mapping;
21. overall verdict mapping;
22. carry/write algebraic diagnostic construction;
23. state-hash deterministic serialization;
24. JSON null instead of NaN;
25. scientific authority gate rejection when absent;
26. scientific authority gate rejection for untracked/mismatched authority;
27. no CLI override for layer/window/metric/population;
28. synthetic mode cannot call scientific population execution;
29. result artifact schema/order validation;
30. no import-time model execution.

## 45. Repository/provenance contract

Expected branch:

`longterm-k-series-native-state-kinematics`

The frozen P1 specification commit must be an ancestor of any P1 implementation/runtime HEAD.

During P1 implementation validation, allowed repository dirt is limited to:

- the two authorized new P1 implementation files;
- the historical K1 untracked pair.

No other tracked or untracked file may change.

After the P1 implementation is committed, synthetic validation must be rerun at the new implementation HEAD before push.

## 46. No scientific execution during P1 implementation

Even though the P1 runner contains a future scientific mode:

`SCIENTIFIC_EXECUTION_DURING_P1_IMPLEMENTATION = FORBIDDEN`

Implementation validation may not create or spoof an execution-authority artifact that sets scientific execution to YES for the frozen population.

The actual scientific execution-authority document must be produced only after:

1. exact P1 implementation freeze;
2. post-commit synthetic validation PASS;
3. independent P1 implementation review;
4. P1 implementation validation/readiness report freeze.

## 47. Interpretation boundary

P1 implementation success establishes only:

`SCIENTIFIC_RUNNER_IMPLEMENTATION_READY_FOR_EXECUTION_AUTHORITY_REVIEW`

It does not establish:

- turning signal;
- response-coherence signal;
- raw-vector organization;
- Branch A;
- Branch B;
- carry/write causal specialization;
- K4.

## 48. Next stage after implementation freeze

After the exact two-file P1 implementation is frozen and its synthetic validation/readiness report is frozen, the next stage may be:

`K0-RVG-P1E — Scientific Raw-Vector Execution Authority`

P1E must bind the exact committed runner/test hashes and all P0 artifact hashes.

Only P1E may turn scientific model forward and recurrent-state read from `NO` to `YES`.

P1E must keep:

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`.

## 49. Authority markers

Before this specification is frozen:

`K0_RVG_P1_SPEC_FROZEN = NO`

After this exact specification is frozen as the immediate one-file child of:

`265d060e96f98f73e4a40a9c54a88c3de482a312`

the bounded implementation authority becomes active:

`K0_RVG_P1_SPEC_FROZEN = YES`

`P1_IMPLEMENTATION_AUTHORIZED = YES`

`P1_IMPLEMENTATION_SCOPE = TWO_NEW_FILES_ONLY`

`P1_SYNTHETIC_MODEL_FORWARD_AUTHORIZED = YES`

`P1_SYNTHETIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`P1_P0_ARTIFACT_STATIC_READ_AUTHORIZED = YES`

`P1_P0_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`P1_P0_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`P1_P0_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`SCIENTIFIC_EXECUTION_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K4_EXECUTION_AUTHORIZED = NO`

Implementation must stop after the exact two-file P1 delta, fabricated synthetic validation, and independent review.
