# ContraMamba Gen5 Phase 1B — Downstream Q22 Causal-Order Correction

## 0. Status

PHASE =
`GEN5_PHASE1B_DOWNSTREAM_Q22_CAUSAL_ORDER_CORRECTION`

STATUS =
`FROZEN_ON_COMMIT`

SCIENTIFIC_EXECUTION_ALLOWED =
`NO`

MODEL_FORWARD_ALLOWED =
`NO`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This correction repairs one causal-order defect in the frozen Phase 1B bridge
design. It does not authorize scientific execution.

After freeze, the already-frozen necessity implementation authority at
`eed071f4dc93f49973da8b2313d7d1b390e0e5d1` remains active except where its
broad-Q endpoint clause is explicitly superseded below.

---

## 1. Parent authorities and frozen artifacts

PHASE1B_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

NECESSITY_IMPLEMENTATION_AUTHORITY_COMMIT =
`eed071f4dc93f49973da8b2313d7d1b390e0e5d1`

R22_C22_ARTIFACT_FREEZE_COMMIT =
`1d3542013934870aa9181d1bbaf565ff4724112c`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

All construction, cohort, checkpoint, rank, control, and intervention identities
remain unchanged.

---

## 2. Defect

The frozen Phase 1B bridge specification states that the historical broad
susceptibility endpoint

`Q = E_XG2 - E_XG4`

should be kept unchanged while the bridge intervention is applied at layer 22.

The historical implementation of that endpoint is upstream of layer 22.

The frozen historical response path is:

`reason_router_gen4_xg2_xg4_local_jacobian_fast_cuda._run_signed_probe`
->
`reason_router_gen4_k_directional_alignment_transport_runner.capture_branch`
->
`reason_router_gen4_k_directional_alignment_transport_runner.path_efficiency`
->
`reason_router_gen4_k_directional_alignment_transport_core.post4_path_efficiency`

Historical `capture_branch` captures recurrent states from the layer-17 mixer.
Historical `path_efficiency` then computes post4 path efficiency from those
layer-17 states.

Therefore a layer-22 write intervention cannot causally modify the historical
layer-17 Q readout within the same forward.

Using the historical Q implementation unchanged for Phase 1B local necessity
would make the primary endpoint causally upstream of the intervention and would
structurally prevent the intended layer-22 necessity test.

This is a design-ordering defect, not an observed negative scientific result.

---

## 3. Frozen source evidence

The correction is bound to the following frozen source identities:

`reason_router_gen4_k_directional_alignment_transport_core.py`
Git blob:
`d98b2dcd3436433c04bb56ecc57dec4240abe820`

`reason_router_gen4_k_directional_alignment_transport_runner.py`
Git blob:
`3677dd83950789e41417c3a1ffaf70b82d7003ad`

`reason_router_gen4_xg2_xg4_local_jacobian_fast_cuda.py`
Git blob:
`1749e4614f0e50d9c9bf551f672497321a5a6253`

`reason_router_gen4_family_subspace_sensitivity_fast_cuda.py`
Git blob:
`03f3bf1482913bce30bfdb665ab223a67e6e4159`

`reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda.py`
Git blob:
`2d35e5ed936fd37f4ecfc063e060290304c6bf10`

`reason_router_gen4_native_mamba_state_measurement.py`
Git blob:
`8c8d63cce182dfab66a632299e93cd5b25fc36af`

No historical source file is modified by this correction.

---

## 4. Minimal correction

The signed XG2/XG4 probe program remains unchanged.

The only scientific change is the state-trajectory readout layer.

Historical:

`PE17 = post4_path_efficiency(POST_STATE17, anchor)`

Corrected Phase 1B bridge readout:

`PE22 = post4_path_efficiency(POST_STATE22, anchor)`

The exact same frozen function
`reason_router_gen4_k_directional_alignment_transport_core.post4_path_efficiency`
must be reused.

The functional form is unchanged.

The state trajectory is moved from layer 17 to layer 22 so that the readout is
causally downstream of the layer-22 native-write intervention.

---

## 5. Frozen signed-probe semantics

The following remain unchanged from the historical broad susceptibility program:

SUBSPACE_DIM =
`5`

EPSILON =
`0.025`

BASIS_FAMILIES =
`XG2 / XG4`

ORIENTATIONS =
`+1 / -1`

BRANCHES =
`TP / TM`

SIGNED_PROBE_LOCATION =
`LAYER17_FROZEN_TARGET_TOKEN_STRONG_CHANNEL_COORDINATE`

For each basis direction `d` and orientation `o`, the exact historical signed
probe is applied at layer 17.

No PP3 neutralization is applied during necessity confirmation because the
frozen upstream necessity condition is `NATIVE`.

No direction search, epsilon search, basis refit, or response-guided selection is
allowed.

---

## 6. Corrected downstream directional response

For each layer-22 condition `c`, basis direction `d`, and orientation `o`:

1. execute the frozen signed layer-17 probe;
2. apply the frozen layer-22 write condition at the exact target token;
3. capture layer-22 recurrent `POST_STATE22` after the modified write has entered
   the recurrent update;
4. retain the layer-22 post-state trajectory over the same five-token coordinate
   `anchor..anchor+4`;
5. compute:

`PE22_tp(c,d,o) = post4_path_efficiency(S22_tp, anchor)`

`PE22_tm(c,d,o) = post4_path_efficiency(S22_tm, anchor)`

and:

`F22(c,d,o) = PE22_tp(c,d,o) - PE22_tm(c,d,o)`

The layer-22 write intervention target remains:

`target_token = anchor + 2`

for each TP/TM branch.

---

## 7. Corrected downstream broad Q22 endpoint

For each direction `d`:

`J22(c,d) = [F22(c,d,+1) - F22(c,d,-1)] / (2 * 0.025)`

Then:

`E_XG2_22(c) = mean_d_in_XG2 J22(c,d)^2`

`E_XG4_22(c) = mean_d_in_XG4 J22(c,d)^2`

Corrected Phase 1B broad endpoint:

`Q22(c) = E_XG2_22(c) - E_XG4_22(c)`

This retains:

- the same XG2/XG4 basis families;
- the same five directions per family;
- the same signed finite-difference semantics;
- the same epsilon;
- the same TP/TM branch contrast;
- the same squared directional-response aggregation;
- the same post4 path-efficiency functional.

Only the recurrent-state readout layer changes from 17 to 22.

---

## 8. Corrected necessity endpoint

Necessity population remains:

`xg1_fact_8101..xg1_fact_8400`

Layer-22 conditions remain:

1. `NATIVE22`
2. `R22_NEUTRALIZED`
3. `C22_COEFFICIENT_CONTROL`

For every signed-probe branch, let the native layer-22 write immediately before
the recurrent update be `w`.

Compute:

`a = R22^T w`

Then:

`N_R(w) = w - R22 a`

`N_C(w) = w - C22 a`

The same exact coefficient vector `a` must be used for R22 and C22 conditions.

The coefficient is derived from the native write of that exact item, branch,
basis direction, and signed orientation before modification.

Define:

`Q0_i = Q22_i(NATIVE22)`

`QR_i = Q22_i(R22_NEUTRALIZED)`

`QC_i = Q22_i(C22_COEFFICIENT_CONTROL)`

Then:

`A_R_i = Q0_i - QR_i`

`A_C_i = Q0_i - QC_i`

`D_NEC22_i = QC_i - QR_i`

The frozen necessity decision rule remains unchanged:

1. all provenance/manipulation gates pass;
2. `mean(Q0) > 0`;
3. `mean(A_R) > 0`;
4. `mean(D_NEC22) > 0`;
5. exactly one one-sided one-sample Student t-test on `D_NEC22` has `p < 0.05`.

No additional confirmatory p-value is authorized.

---

## 9. Corrected restoration endpoint

This correction also prospectively repairs the same causal-order defect for the
future restoration stage.

The restoration population remains:

`xg1_fact_8401..xg1_fact_8700`

The upstream background remains:

`PP3_NEUTRALIZED_AT_LAYER17`

The frozen donor/replacement semantics remain unchanged.

Only the broad readout becomes the same downstream `Q22` defined above.

Future restoration quantities must therefore use:

`QB_i = Q22_i(BACKGROUND)`

`QRR_i = Q22_i(R22_RESTORED)`

`QRC_i = Q22_i(C22_REPLACEMENT)`

with the already-frozen restoration contrasts and decision rule otherwise
unchanged.

No restoration implementation or execution is authorized by this correction.

---

## 10. Native-write and post-state ordering

The implementation must prove for every modified branch that:

1. native `WRITE22` is formed;
2. R22/C22 coefficients are derived from that native write;
3. the authorized replacement is applied to `WRITE22`;
4. the modified write enters the layer-22 recurrent update;
5. `POST_STATE22` is captured only after that update;
6. `PE22` is computed only from the resulting layer-22 post-state trajectory.

Any readout captured before step 4 is invalid for Q22.

---

## 11. Forward-budget identity

The corrected endpoint does not change the historical probe-count arithmetic.

Per layer-22 condition:

- 5 XG2 directions;
- 5 XG4 directions;
- 2 orientations;
- 2 TP/TM branches.

Therefore:

`FORWARDS_PER_CONDITION = 40`

`CONDITIONS_PER_PAIR = 3`

`FORWARDS_PER_PAIR = 120`

`PAIR_COUNT = 300`

`FULL_NECESSITY_FORWARD_BUDGET = 36000`

No extra baseline/model forward is permitted in the future full confirmation
run unless separately authorized as a preflight/equivalence run with a distinct
run identity.

---

## 12. Implementation-authority rebind

After this correction is frozen:

`eed071f4dc93f49973da8b2313d7d1b390e0e5d1`

remains the necessity implementation authority with the following substitution:

Old clause:

`Q = historical layer17 broad endpoint`

is superseded by:

`Q = Q22 downstream layer22 post-state broad endpoint defined in this correction`

All other `eed071f` constraints remain unchanged, including:

- exactly three new implementation files;
- no existing production-file modifications;
- frozen R22/C22 bytes;
- frozen necessity cohort;
- frozen checkpoint;
- no restoration implementation;
- no ownership implementation;
- no model forward during implementation verification;
- mandatory independent verification;
- no README update.

No second implementation-authority document is required after this correction.

---

## 13. Implementation requirements added by this correction

The bounded implementation must additionally provide:

- a layer-22 native-write intervention primitive;
- a layer-22 post-state trajectory collector;
- exact ordering checks from native write -> modified write -> recurrent update ->
  post-state capture;
- a synthetic/static test showing that the Q22 readout consumes layer-22 state
  trajectories rather than layer-17 state trajectories;
- a static verifier check that historical
  `transport_runner.path_efficiency()` is not used as the Phase 1B primary
  readout;
- a static verifier check that
  `core.post4_path_efficiency()` is reused unchanged;
- zero real model forward during implementation verification.

If exact write intervention cannot be implemented without changing non-target
computation, implementation must block.

---

## 14. Backend boundary

This correction does not select the full-run backend.

A future execution authority may select CPU slow path, CUDA slow path, or an
independently validated accelerated implementation only if it preserves the
exact Q22/write semantics above.

Any accelerated implementation requires a separate Phase 1B-specific
equivalence gate against the exact slow-path semantic reference before the full
necessity run.

Historical CPU/CUDA equivalence evidence remains background continuity evidence
only and does not by itself validate a new layer-22 write-replacement backend.

---

## 15. Interpretation boundary

The corrected endpoint is a new Phase 1B downstream bridge endpoint.

It must not be described as byte-for-byte or coordinate-identical to the
historical layer-17 Q endpoint.

The justified continuity claim is narrower:

`SAME_FROZEN_XG2_XG4_PROBE_AND_POST4_FUNCTIONAL_WITH_DOWNSTREAM_LAYER22_STATE_READOUT`

No local-necessity result exists until a separately authorized confirmation run
is executed, collected, imported, and validated.

---

## 16. Success boundary

Freezing this correction establishes only:

`PASS_PHASE1B_CAUSAL_ORDER_DEFECT_CORRECTED_READY_TO_RESUME_NECESSITY_IMPLEMENTATION`

It does not establish:

- R22 necessity;
- restoration sufficiency;
- bridge support;
- ownership benefit;
- behavioral improvement.

STATUS =
`FROZEN_ON_COMMIT`
