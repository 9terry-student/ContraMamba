# ContraMamba Gen4 — Mamba-1.4B P5 Cross-Block JVP Transport Feasibility Audit

## Status

`STATIC_FEASIBILITY_AUDIT_CANDIDATE`

This document creates no new model output, causal response, p-value, or scientific
conclusion. It does not authorize a full scientific execution.

Current repository parent:

`5eb138560f9a638ae7c30adab58771ec68d6af54`

The bounded next step supported by this audit is implementation and local/static
validation of a one-row response-free JVP feasibility gate only.

## 1. Scientific motivation

Experiment 5 established one-shot adjacent-site specificity between the frozen
Mamba-1.4B canonical triplet

`(33,34,35)`

and the architecture-predeclared downstream adjacent triplet

`(34,35,36)`.

The open mechanistic question is not another layer search. It is:

> When the frozen canonical P5 causal plane is propagated through the exact
> block-35-to-block-36 dynamics, does its transported two-dimensional span align
> with the independently reconstructed adjacent P5 plane, or is the causal
> coordinate substantially reorganized before the adjacent intervention site?

This audit asks only whether that question is technically and geometrically
well-defined under the already frozen repository objects.

## 2. Frozen parent objects

Experiment-5 design:

`reports/reason_router_gen4_one_shot_adjacent_site_specificity_design.md`

git blob:

`9accf85f49e6278ee3d3b2b58482ffdb6b4085fc`

Canonical geometry implementation:

`scripts/reason_router_gen4_mamba14b_geometry_prepare_fast_cuda.py`

git blob:

`d221c28657cf9bb157d517b635bae42671563903`

Experiment-5 paired raw runner:

`scripts/reason_router_gen4_mamba14b_one_shot_adjacent_site_specificity_raw_fast_cuda.py`

git blob:

`99dc4175530153aff61357582efa869d7f0bf8de`

Canonical geometry summary blob:

`bad410a376d0154ced8ba0aa5b2c5397e0397ba8`

Canonical strong-index blob:

`7d86541ff368c550860dcf666ce418c5a1217ec0`

Canonical P5 plus/minus blobs:

- plus: `007c15f802bc2a07e1770586b995f1fe9d9c0b90`
- minus: `0a2a3278c01b520f0916cd48051a9a9cef77041a`

Adjacent geometry summary blob:

`d6bd05aa8d47bdf80e088fbcfb5cd1e5e77a3866`

Adjacent strong-index blob:

`dc5f8f0af4d0375a88fbee93c8769bf4b64492cd`

Adjacent P5 plus/minus blobs:

- plus: `6d83d1fc64bcbb7b1c9a779399dd405d82a063a6`
- minus: `0ff81176cd5be099eb0da344293621d74b3ce39b`

Existing analytic local-VJP precedent:

`scripts/reason_router_gen4_precursor_v4_analytic_vjp_one_row_equivalence.py`

git blob:

`92ede8e49fa5a329a5d292db714c7c854138e369`

## 3. Exact coordinate identity

Both Experiment-5 sites intervene at the same semantic tensor location inside
their respective Mamba mixers:

- output of `mixer.in_proj`;
- content/x half only;
- first `4096` channels of the `8192`-wide in-projection output;
- target token `A_IDENTITY + 2`;
- gate half held unchanged;
- all non-target tokens held unchanged by the intervention hook.

The ambient content coordinate is therefore:

`R^4096`

at both block 35 and block 36.

The canonical and adjacent frozen P5 plane files are not stored in this full
ambient coordinate. They are stored in site-local strong-coordinate systems.

Canonical site:

- intervention block: `35`;
- strong dimension: `829`;
- XG2/XG4 basis shape: `[829,5]`.

Adjacent site:

- intervention block: `36`;
- strong dimension: `1205`;
- XG2/XG4 basis shape: `[1205,5]`.

Therefore a direct dot product between the stored canonical and adjacent P5
vectors is invalid. Each vector must first be scattered through its frozen
strong-index list into the common `4096`-channel ambient content space.

## 4. Response-free structural mask comparison

This audit inspected only the already frozen strong-index files. It did not read
XG1 causal-response values or compute any P5-to-P5 plane angle.

Observed strong-mask relationship:

- canonical strong channels: `829`;
- adjacent strong channels: `1205`;
- intersection: `223`;
- union: `1811`;
- Jaccard overlap: `0.12313638873550524`;
- fraction of canonical strong channels also strong at adjacent site:
  `0.2689987937273824`;
- fraction of adjacent strong channels also strong at canonical site:
  `0.18506224066390042`.

This structural fact does not establish P5 rotation or explain the Experiment-5
causal sign reversal. It only establishes that the two compressed coordinate
systems differ enough that an explicit ambient-space transport construction is
required.

No principal angle, projector overlap, Procrustes score, or causal-response
statistic is computed in this audit.

## 5. State-conditioned transport map

A single input-independent matrix `T_35_to_36` is not justified by the current
Mamba computation.

The block dynamics are state- and input-conditioned. The appropriate local object
is a row-specific Jacobian.

For a frozen input row `q` and target token `tau`, let

`x35_q,tau in R^4096`

denote the content half of the block-35 `in_proj` output at the target token.

Define the local map

`Phi_q : R^4096 -> R^4096`

as follows:

1. run the frozen input to the block-35 `in_proj`;
2. detach the complete block-35 `in_proj` output and reintroduce it as the local
   differentiation boundary;
3. vary only the target-token content half;
4. keep the block-35 gate half fixed;
5. keep every other token in the block-35 `in_proj` output fixed;
6. propagate through the remainder of block 35 and the ordinary block-36
   normalization/in-projection computation;
7. read the block-36 target-token content half before its depthwise convolution.

Then

`J_q = d Phi_q / d x35_q,tau`

is the exact local cross-block transport Jacobian for this question.

This definition matches the already frozen intervention-site semantics. It is not
a residual-stream approximation and does not compare unrelated compressed
coordinates.

## 6. Frozen source and target planes for a future gate

Scatter the canonical P5 vectors through the canonical strong indices:

`U35 = [u35_plus, u35_minus] in R^(4096 x 2)`.

Scatter the adjacent P5 vectors through the adjacent strong indices:

`U36 = [u36_plus, u36_minus] in R^(4096 x 2)`.

Both columns are already frozen unit vectors in their respective strong spaces.
Zero-filling outside the frozen strong masks preserves their ambient L2 norms and
orthogonality.

For a row-specific Jacobian, the desired push-forward is:

`W_q = J_q U35`.

If `W_q` has numerical rank 2, orthonormalize it to:

`Q_q = orth(W_q)`.

Only after an execution design is prospectively frozen may scientific transport
metrics such as the following be evaluated over a population:

- principal singular values of `Q_q^T U36`;
- the corresponding two principal angles;
- normalized projector overlap
  `0.5 * ||Q_q^T U36||_F^2`;
- orthogonal-Procrustes residual;
- transported-vector norms and conditioning.

This audit does not calculate those quantities.

## 7. Existing backend evidence and remaining gap

The repository already establishes two useful facts.

First, the Experiment-5 fast-CUDA runner can intervene exactly at the required
`in_proj` content-half site while preserving the gate half, non-strong channels,
and other tokens.

Second, Precursor-v4 established an analytic local leaf plus reverse-mode VJP
through the exact project CUDA kernels for a related Mamba-370M workload.

Those facts make the proposed transport technically plausible.

They do not establish forward JVP support for this Mamba-1.4B cross-block map.

The repository currently contains no validated evidence that:

- `torch.func.jvp`,
- `torch.autograd.functional.jvp`,
- or an equivalent exact analytic push-forward

works through the exact Mamba-1.4B fast-CUDA kernel path for this internal-to-
internal map.

Therefore full transport execution is not yet technically authorized.

## 8. Bounded next feasibility gate

The next implementation should be a one-row response-free feasibility gate only.

Frozen gate row:

- family: `XG2`;
- source pair: `xg2_fact_301`;
- cell: `C2_NAME`;
- anchor: `A_IDENTITY`;
- target offset: `+2`;
- source site: block-35 `in_proj` content half;
- target site: block-36 `in_proj` content half;
- source directions: canonical `P5_plus` and `P5_minus`.

Why this row is admissible:

- it is the first row of the already frozen response-free XG2 geometry population;
- the choice does not use Experiment-5 XG1 response;
- it does not select a row from a transport outcome;
- it is used only to establish operator feasibility.

The one-row gate may report:

- whether both analytic JVPs are obtainable;
- finite/nonfinite status;
- source and target tensor shapes;
- transported-vector L2 norms;
- numerical rank of the two transported vectors;
- exact runtime/backend provenance;
- model forward/JVP accounting.

It must not report or use:

- Experiment-5 `D_CAN`, `D_ADJ`, or `S`;
- XG1 response values;
- a population transport conclusion;
- a p-value;
- a threshold chosen from JVP outcomes;
- a layer, token, plane, or row search.

A later scientific design must be frozen separately if the one-row JVP gate passes.

## 9. Pass/fail interpretation

Technical feasibility PASS requires:

1. exact frozen model/checkpoint/site identities;
2. successful block-35 local differentiation boundary;
3. successful analytic push-forward of both canonical P5 basis vectors to the
   block-36 target content coordinate;
4. finite transported vectors;
5. transported span numerical rank `2`;
6. no parameter gradient or update;
7. no training;
8. no response-dependent selection.

A failure of the exact fast-CUDA forward-JVP path is not itself a scientific
failure of the transport hypothesis.

It means only that the requested estimator/backend combination has not yet been
established. Any fallback estimator or backend would require a separate bounded
equivalence argument before scientific use.

## 10. Feasibility conclusion

`PASS_FOR_BOUNDED_ONE_ROW_JVP_GATE_IMPLEMENTATION`

The cross-block transport question is geometrically well-defined with the frozen
Experiment-5 objects.

The decisive implementation requirement is to establish an exact analytic
block-35-content to block-36-content JVP on one frozen response-free row before
any population transport measurement is designed or executed.

`SCIENTIFIC_EXECUTION_AUTHORIZED = FALSE`

`STATISTICAL_TESTING_AUTHORIZED = FALSE`

`TRAINING_AUTHORIZED = FALSE`

`KAGGLE_EXECUTION_AUTHORIZED_BY_THIS_AUDIT = FALSE`
