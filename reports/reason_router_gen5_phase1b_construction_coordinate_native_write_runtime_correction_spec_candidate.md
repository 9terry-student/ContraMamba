# ContraMamba Gen5 Phase 1B — Construction Coordinate and Native-Write Runtime Correction Specification

## 0. Status

PHASE = `GEN5_PHASE1B_CONSTRUCTION_COORDINATE_CORRECTION`

STATUS = `FROZEN_ON_COMMIT`

PARENT_PHASE1B_DESIGN =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

STATIC_PREPARATION_FREEZE =
`5c0d91959af1f502b667ed6ab815c949c1043cbf`

IMPLEMENTATION = `NOT_AUTHORIZED_BY_THIS_DOCUMENT`

SCIENTIFIC_EXECUTION = `NOT_AUTHORIZED`

README_UPDATE_REQUIRED = `NO`

This document corrects an underspecified construction coordinate in the frozen
Phase 1B bridge design. It does not change the Phase 1B scientific question,
population allocation, rank, confirmation endpoints, or advancement rule.

---

## 1. Defect being corrected

The frozen Phase 1B design defined:

`dW_i = w22_i(PP5_CONTROL) - w22_i(PP3_NEUTRALIZED)`

but did not define a unique `w22_i(condition)` in the presence of the historical
PP3 susceptibility protocol's:

- XG2/XG4 direction families;
- positive/negative finite-difference orientations;
- target-plus / target-minus branches.

Allowing implementation to choose an aggregation after this freeze would change
the discovered R22 subspace.

Therefore construction coordinates are fixed here before any scientific model
execution.

---

## 2. Construction does not execute the XG2/XG4 probe family

The R22/C22 construction run is a native propagation experiment, not a
susceptibility inference run.

For construction only:

`DIRECTIONAL_PROBE = NONE`

`PROBE_EPSILON = 0`

`XG2_XG4_BASIS_FORWARDS = 0`

No XG2 or XG4 directional basis is loaded or applied during R22/C22 construction.

The existing layer-17 PP3 condition transform is reused exactly, but the small
signed susceptibility probe is omitted.

This avoids introducing an arbitrary vector aggregation over ten directions and
two finite-difference orientations.

The frozen broad Q endpoint remains unchanged for later confirmation.

---

## 3. Frozen construction conditions

For every construction pair, exactly three layer-17 conditions are executed:

1. `NATIVE`
2. `PP3_NEUTRALIZED`
3. `PP5_COEFFICIENT_CONTROL`

The PP3-neutralized and PP5-control conditions must use the same native
PP3-derived coefficients for the same branch, exactly as in the frozen PP3
necessity design.

For branch-local strong-channel state `h`:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

`delta_PP3 = -a*pp3_plus - b*pp3_minus`

`delta_PP5CTRL = -a*pp5_plus - b*pp5_minus`

Mandatory gate:

`||delta_PP3||_2 = ||delta_PP5CTRL||_2`

within the inherited runtime tolerance.

No directional probe is added to either correction.

---

## 4. Frozen branches

Construction uses only the existing target branches:

- `tp = TARGET_PLUS_CELL`
- `tm = TARGET_MINUS_CELL`

with their existing tokenizer-derived target anchors and inherited
`TARGET_OFFSET`.

Reference-plus and reference-minus branches are not used to construct R22/C22.

For each condition `c`, exactly two full model forwards are required:

- target-plus forward;
- target-minus forward.

Therefore:

`FORWARDS_PER_CONDITION = 2`

`CONDITIONS_PER_PAIR = 3`

`FORWARDS_PER_PAIR = 6`

`CONSTRUCTION_PAIR_COUNT = 300`

`CONSTRUCTION_FULL_FORWARD_BUDGET = 1800`

No extra scientific forward may be added to select a coordinate, rank, channel,
layer, or sign.

---

## 5. Native layer-22 objects

At the frozen layer-22 target token, capture:

`W22 = deltaB_u[:, :, token_index, :]`

immediately before the recurrent update, and:

`S22 = ssm_state`

immediately after:

`ssm_state = discrete_A[:, :, token_index, :] * ssm_state + W22`

and before recurrent readout.

Per forward:

`shape(W22) = (1,1536,16)`

`shape(S22) = (1,1536,16)`

Flattening order is contiguous row-major over `(1536,16)` and is frozen as:

`vec : (1,1536,16) -> (24576,)`

No channel selection, masking, averaging, norm reduction, or top-k operation is
allowed before vector construction.

---

## 6. Runtime/source contract

The primary runtime contract is inherited from:

`scripts/reason_router_gen4_native_mamba_state_measurement.py`

That frozen measurement code already statically validates, for the Gen4
Transformers 5.0.0 slow path:

- `deltaB_u` as the native recurrent write term;
- `ssm_state` recurrence update;
- post-update `ssm_state` before recurrent readout;
- exact source-role ordering.

The Phase 1B construction observer may extend capture from `ssm_state` to
`deltaB_u`, but it must not redefine the recurrent equations.

The observer must bind to the same validated source identities and line roles as
the inherited Gen4 measurement contract.

K-series Transformers 5.12.1 observer code is scientific precedent only and must
not be substituted as the Gen5 runtime authority.

---

## 7. Backend boundary

Construction is authorized for design around the validated Gen4 CPU slow path,
because the exact native write variable exists at that boundary.

Historical backend evidence already includes:

`PASS_XG1_FAST_CUDA_ONE_PAIR_EQUIVALENCE`

on the same representative checkpoint, with the frozen state tolerance.

That historical result supports backend continuity but does not by itself
authorize Phase 1B confirmation on a changed backend.

Before later necessity/restoration scientific confirmation, if the Q endpoint is
computed on the slow path rather than the historically frozen fast-CUDA path, a
Phase 1B-specific outcome-blind fast/slow equivalence gate must be frozen and
passed first.

Construction itself computes no Q and no scientific p-value.

---

## 8. Branch-contrast native write coordinate

For item `i`, condition `c`, define:

`W_tp_i(c) = vec(W22 from target-plus branch)`

`W_tm_i(c) = vec(W22 from target-minus branch)`

The condition-level native write coordinate is:

`B_W_i(c) = W_tp_i(c) - W_tm_i(c)`

The condition-level post-state diagnostic is:

`B_S_i(c) = S_tp_i(c) - S_tm_i(c)`

where `S_tp_i(c)` and `S_tm_i(c)` are the corresponding flattened layer-22
post-update states.

This target-plus minus target-minus ordering is frozen.

It must not be sign-flipped based on construction outcomes.

---

## 9. Propagated role contrast

The discovery vector used to construct R22 is:

`dW_i = B_W_i(PP5_COEFFICIENT_CONTROL) - B_W_i(PP3_NEUTRALIZED)`

The matching post-state diagnostic is:

`dS_i = B_S_i(PP5_COEFFICIENT_CONTROL) - B_S_i(PP3_NEUTRALIZED)`

The PP5-minus-PP3 ordering is frozen because the historical necessity endpoint is
also oriented as matched-control minus PP3-neutralized.

Only `dW_i` may construct R22.

`dS_i` is diagnostic and must not affect basis selection.

---

## 10. R22 construction matrix

On construction population:

`xg1_fact_7801..xg1_fact_8100`

form the `300 x 24576` float64 matrix:

`D = stack_i(dW_i)`

Compute the row mean:

`mu_D = mean_i(dW_i)`

and center:

`D_c[i] = dW_i - mu_D`

R22 construction uses only `D_c`.

No normalization of individual rows is allowed.

No response weighting is allowed.

No pair dropping is allowed.

---

## 11. R22 fixed-rank SVD

Compute deterministic CPU float64 SVD of `D_c`:

`D_c = U Sigma V^T`

The Phase 1B rank remains:

`rank(R22) = 2`

Candidate basis:

`R22 = [v1, v2]`

where `v1` and `v2` are the first two right-singular vectors.

No variance threshold, elbow rule, parallel analysis, or rank sweep is allowed.

---

## 12. Rank-2 uniqueness gate

Let:

`sigma1 >= sigma2 >= sigma3`

be the first three singular values.

Define:

`gap_tol = max(1e-12, 1e-10 * max(sigma1, 1.0))`

Mandatory gate:

`sigma2 - sigma3 > gap_tol`

If this gate fails:

`BLOCKED_R22_RANK2_SUBSPACE_NOT_NUMERICALLY_IDENTIFIABLE`

No higher-rank rescue is authorized.

---

## 13. Deterministic sign canonicalization

For every saved basis vector `v`:

1. find `j* = argmax_j |v_j|`;
2. ties are resolved by the lowest coordinate index;
3. if `v[j*] < 0`, replace `v <- -v`;
4. otherwise keep `v`.

This sign rule affects serialization only.

The scientific object is the rank-2 subspace/projector.

The ordered vector pair remains sorted by descending singular value.

---

## 14. Response-blind C22 construction coordinate

The matched control is constructed from the same 300 construction items using
only the `NATIVE` condition.

Define:

`N_i = B_W_i(NATIVE)`

Compute:

`mu_N = mean_i(N_i)`

`N_c[i] = N_i - mu_N`

After R22 is frozen in memory, residualize each row:

`N_perp_i = N_c[i] - R22 * (R22^T N_c[i])`

No PP3-neutralized response, PP5-control response, dW, dS, Q, task logit, or
confirmation response may enter the C22 covariance calculation except that the
already fixed R22 span is removed geometrically.

---

## 15. C22 fixed-rank SVD

Stack:

`N_perp = stack_i(N_perp_i)`

and compute CPU float64 SVD.

Candidate control basis:

`C22 = [c1, c2]`

from the first two right-singular vectors.

Apply the same deterministic sign canonicalization.

Apply the same second-versus-third singular-value uniqueness gate.

If it fails:

`BLOCKED_C22_RANK2_CONTROL_NOT_NUMERICALLY_IDENTIFIABLE`

No alternative control search is authorized.

---

## 16. Geometry gates

Mandatory construction gates:

`R22^T R22 = I_2`

`C22^T C22 = I_2`

`R22^T C22 = 0`

within absolute tolerance:

`BASIS_ORTHOGONALITY_ATOL = 1e-10`

Also require finite values throughout and exact dimension:

`R22.shape = C22.shape = (24576,2)`

---

## 17. Construction artifacts

The construction run may persist:

1. `r22_basis.f64le`
2. `c22_basis.f64le`
3. `construction_item_audit.jsonl`
4. `construction_summary.json`
5. `artifact_manifest.json`
6. `SHA256SUMS.txt`

The basis binary layout is column-major semantic ordering serialized explicitly
as:

- first all 24576 float64 coordinates of basis vector 1;
- then all 24576 float64 coordinates of basis vector 2.

Raw per-item W22, S22, dW, or dS vectors must not be persisted.

Per-item audit may persist scalar quantities and hashes only, including:

- condition correction norms;
- PP3 coefficient matching residuals;
- W/S vector SHA256 hashes;
- dW/dS L2 norms;
- target token indices;
- manipulation residuals.

---

## 18. Construction conclusion boundary

A successful construction run may conclude only:

`PASS_GEN5_PHASE1B_R22_C22_CONSTRUCTION`

It does not establish:

- R22 necessity;
- R22 restoration sufficiency;
- bridge success;
- ownership benefit;
- task benefit.

`scientific_conclusion = null`

must be persisted.

---

## 19. Confirmation-data firewall

The construction runner must refuse to load files from:

`data/reason_router_gen5_phase1b_xg1_necessity_confirmation_v1`

and:

`data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1`

except for static existence/hash checks that do not read claims, evidence,
anchors, or any future model response.

Construction scientific inputs are restricted to:

`data/reason_router_gen5_phase1b_xg1_construction_v1`

plus already frozen historical dependencies.

---

## 20. Current next step

After this correction is frozen:

`NEXT = GEN5_PHASE1B_R22_C22_CONSTRUCTION_IMPLEMENTATION`

Implementation must include independent verification because it introduces a
native recurrent-write observer and therefore touches high-risk hidden-state
semantics.

No scientific execution is authorized by this correction.

---

## 21. Summary

CONSTRUCTION_PROBE =
`NONE`

CONSTRUCTION_CONDITIONS =
`NATIVE / PP3_NEUTRALIZED / PP5_COEFFICIENT_CONTROL`

CONSTRUCTION_BRANCH_COORDINATE =
`TARGET_PLUS_MINUS_TARGET_MINUS`

PROPAGATED_WRITE_CONTRAST =
`PP5_CONTROL_MINUS_PP3_NEUTRALIZED`

R22_SOURCE =
`CENTERED_dW`

C22_SOURCE =
`CENTERED_NATIVE_BRANCH_CONTRAST_ORTHOGONAL_TO_R22`

R22_RANK =
`2`

C22_RANK =
`2`

FULL_FORWARD_BUDGET =
`1800`

NATIVE_RUNTIME =
`GEN4_VALIDATED_TRANSFORMERS_5_0_0_SLOW_PATH`

RAW_NATIVE_VECTORS_PERSISTED =
`NO`

README_UPDATE_REQUIRED =
`NO`

STATUS =
`FROZEN_ON_COMMIT`
