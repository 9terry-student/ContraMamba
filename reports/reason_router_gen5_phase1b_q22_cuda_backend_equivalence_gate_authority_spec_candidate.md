# ContraMamba Gen5 Phase 1B — Q22 CUDA Backend Equivalence Gate Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE_GATE`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED_AFTER_FREEZE =
`YES_BOUNDED`

EQUIVALENCE_EXECUTION_ALLOWED_AFTER_VERIFICATION =
`YES_ONE_PAIR_ONLY`

FULL_NECESSITY_EXECUTION_ALLOWED =
`NO`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This authority exists only to qualify a CUDA backend for the already-frozen
Gen5 Phase 1B Q22 necessity program.

It does not authorize the 300-pair necessity confirmation run.

---

## 1. Frozen parents

NECESSITY_IMPLEMENTATION_FREEZE_COMMIT =
`130a3474cf83d1ba6080561afbf886614dde4786`

Q22_CAUSAL_ORDER_CORRECTION_COMMIT =
`d31b9df1fddfb719d95a6d486ebb8f18b397f6ad`

NECESSITY_IMPLEMENTATION_AUTHORITY_COMMIT =
`eed071f4dc93f49973da8b2313d7d1b390e0e5d1`

R22_C22_ARTIFACT_FREEZE_COMMIT =
`1d3542013934870aa9181d1bbaf565ff4724112c`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

The frozen CPU semantic-reference runner must not be modified.

---

## 2. Why this gate is required

The Q22 correction explicitly requires any accelerated implementation to pass a
Phase 1B-specific equivalence gate against the exact slow-path semantics before
the full necessity run.

Historical Gen4 CPU/CUDA equivalence is supporting precedent only. It is not
sufficient for the new layer-22 write-replacement/Q22 path.

---

## 3. Gate population firewall

EQUIVALENCE_PAIR =
`xg1_fact_7801`

EQUIVALENCE_POPULATION_ROLE =
`PRIOR_CONSTRUCTION_COHORT_ONLY`

The equivalence gate must use the already-frozen construction population input
for `xg1_fact_7801`.

The necessity-confirmation population

`xg1_fact_8101..xg1_fact_8400`

must not be loaded, tokenized, executed, or inspected for scientific responses
during this gate.

The restoration-confirmation population must also remain unread.

This gate produces no confirmatory scientific evidence.

---

## 4. Exact scientific coordinates preserved

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

UPSTREAM_SIGNED_PROBE_SITE =
`FROZEN_LAYER17_TARGET_TOKEN_STRONG_CHANNEL_COORDINATE`

DOWNSTREAM_INTERVENTION_SITE =
`LAYER22_WRITE22`

DOWNSTREAM_READOUT =
`LAYER22_POST_STATE22`

READOUT_FUNCTION =
`core.post4_path_efficiency`

No scientific coordinate may differ between the reference and accelerated
backends.

---

## 5. CUDA runtime

Both backends in the equivalence run must execute on CUDA.

EXPECTED_RUNTIME =

- Python `3.12.13`
- NumPy `2.0.2`
- Torch `2.10.0+cu128`
- Transformers `5.0.0`
- CUDA runtime `12.8`
- device `Tesla T4`
- compute capability `7.5`
- kernels package `0.10.2`

Frozen accelerated-kernel identities:

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

CPU scientific model forwards are forbidden for this gate.

---

## 6. CUDA slow semantic reference

REFERENCE_BACKEND =
`TRANSFORMERS_5_0_0_MAMBA_SLOW_FORWARD_ON_CUDA`

The reference backend must preserve the exact frozen slow-path roles:

`WRITE22 = deltaB_u`

`POST_STATE22 = ssm_state after recurrent update`

and the exact frozen Q22 endpoint.

The reference must run under `torch.inference_mode()`.

No backward, training, task-head evaluation, or logits are authorized.

The model parameter payload must be unchanged before and after the reference
pass.

---

## 7. Accelerated CUDA backend

ACCELERATED_BACKEND =
`FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY`

The accelerated full forward must preserve the frozen layer-17 signed probe.

During layer 22, the backend may capture the exact frozen
`selective_scan_fn` inputs for the layer-22 mixer.

Q22 state replay must use only the exact frozen kernel bundle.

### Prefix

Use the frozen `selective_scan_fn` to obtain the recurrent state immediately
before the five-token Q22 window.

### Native per-token update

For ordinary tokens in the Q22 window, use the frozen
`selective_state_update`.

### Target WRITE22 recovery

At the frozen target token, start from the same exact pre-update state and
compute:

1. a native kernel update using the captured native input;
2. a decay-only kernel update using the same delta/A/B/C/D/gate/bias inputs but
   zero recurrent input.

Define:

`WRITE22_fast = STATE_NATIVE_UPDATE - STATE_DECAY_ONLY`

This recovery is permitted only because the frozen native recurrence has the
form:

`POST_STATE = DECAYED_PRE_STATE + WRITE`

No hand-reimplementation of the CUDA kernel recurrence equation is authorized.

### R22/C22 write replacement

Use the exact frozen write rule:

`a = R22^T WRITE22_fast`

`WRITE_R = WRITE22_fast - R22 a`

`WRITE_C = WRITE22_fast - C22 a`

Then:

`STATE_R = STATE_DECAY_ONLY + WRITE_R`

`STATE_C = STATE_DECAY_ONLY + WRITE_C`

Continue the remaining window tokens with the exact frozen
`selective_state_update`.

The same R22-derived coefficient vector must be transferred to C22.

---

## 8. Forward budget

For one pair:

- 3 layer-22 conditions
- 10 frozen directions
- 2 orientations
- 2 TP/TM branches

Therefore:

`FORWARDS_PER_BACKEND = 120`

`REFERENCE_CUDA_FORWARDS = 120`

`ACCELERATED_CUDA_FORWARDS = 120`

`TOTAL_EQUIVALENCE_MODEL_FORWARDS = 240`

No other scientific model forward is authorized by this gate.

---

## 9. Prospective tolerances

The following tolerances are frozen before equivalence outputs are observed.

These inherit the validated historical Gen4 CUDA state-equivalence scale.

WRITE22 tensor equivalence:

`ATOL = 1e-4`
`RTOL = 1e-4`

POST_STATE22 tensor equivalence at each of the five Q22 coordinates:

`ATOL = 1e-4`
`RTOL = 1e-4`

PE22 absolute tolerance:

`PE22_ATOL = 1e-4`

Derived F22 tolerance:

`F22_ATOL = 2e-4`

Because

`J22 = (F_plus - F_minus) / 0.05`,

the prospective derived J22 tolerance is:

`J22_ATOL = 0.008`

No tolerance may be relaxed after gate outputs are observed.

---

## 10. Derived E/Q tolerance

Do not invent an independent post-hoc Q tolerance.

For each reference directional response `J_ref`, define:

`B_J2 = 2 * abs(J_ref) * 0.008 + 0.008^2`

For each family:

`B_E = mean(B_J2 over its five directions)`

Then:

`B_Q = B_E_XG2 + B_E_XG4`

For every condition require:

`abs(Q22_accelerated - Q22_reference) <= B_Q`

This bound is fixed analytically from the prospective J22 tolerance.

---

## 11. Mandatory equivalence checks

The gate must compare reference vs accelerated for every exact coordinate:

- source pair identity;
- condition;
- XG2/XG4 family;
- basis index;
- orientation;
- TP/TM branch;
- target token;
- native WRITE22;
- applied R22/C22 write change;
- coefficient vector;
- correction norm;
- five POST_STATE22 tensors;
- PE22;
- F22;
- J22;
- E_XG2_22;
- E_XG4_22;
- Q22.

It must also verify:

- R22/C22 exact hashes;
- R22/C22 geometry;
- same layer-17 signed probe norm and signs;
- same target token;
- zero parameter mutation;
- zero training;
- zero backward;
- zero task-head optimization;
- zero confirmation-population scientific access.

---

## 12. Success rule

The gate passes only if every required coordinate passes all prospective
tolerances and all provenance/manipulation checks pass.

Positive gate label:

`PASS_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE`

Otherwise:

`BLOCK_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE`

There is no scientific p-value in this gate.

There is no necessity conclusion in this gate.

---

## 13. Allowed implementation files

Exactly these new files may be created:

`scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py`

`scripts/verify_reason_router_gen5_phase1b_q22_cuda_equivalence.py`

`tests/test_reason_router_gen5_phase1b_q22_cuda_equivalence.py`

The frozen necessity runner must not be modified.

Existing Gen4 production files must not be modified.

README must not be modified.

---

## 14. Implementation verification boundary

Before any CUDA equivalence execution:

- narrow tests must pass;
- independent static verifier must pass;
- model loaded = false;
- checkpoint loaded = false;
- model forward count = 0;
- CUDA scientific execution = false.

Only after that verification may the one-pair 240-forward CUDA equivalence run
be authorized by this same frozen gate authority.

---

## 15. Equivalence artifacts

The future gate run may persist only:

`q22_cuda_equivalence_items.jsonl`

`q22_cuda_equivalence_summary.json`

`artifact_manifest.json`

`SHA256SUMS.txt`

No raw full write/state tensors may be persisted.

Hashes, scalar norms, max residuals, and equivalence diagnostics are allowed.

---

## 16. Advancement

A passing equivalence gate authorizes preparation of a separate full necessity
CUDA execution authority bound to the exact qualified accelerated backend.

It does not itself authorize the 36,000-forward run.

A failed gate blocks full CUDA necessity execution. No tolerance relaxation or
backend substitution is allowed without a new correction.

---

## 17. Success boundary

Freezing this document authorizes only:

1. bounded implementation of the three CUDA-equivalence files;
2. zero-forward independent verification;
3. after verification, one-pair 240-forward CUDA equivalence execution.

It does not establish local necessity, restoration, bridge support, or ownership
benefit.

STATUS =
`FROZEN_ON_COMMIT`
