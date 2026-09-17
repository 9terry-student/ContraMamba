# Gen4-K PP3 XG1 Independent-Generator External Transport — Prospective Scope

Base HEAD:

`83b8cf7e73eb68343cc4c97e3f0698df4b587408`

Branch:

`gen4-k-xg2-basis-holdout`

## Purpose

The completed Gen4-K XG2-basis fresh-index holdout established that the frozen
XG2 top-5 subspace has greater local squared directional sensitivity than the
frozen XG4 top-5 subspace on both XG2 and XG4 source families.

The subsequent static projector-contrast localization identified the third
principal pair of the frozen XG2/XG4 subspaces as the strongest shared
mechanistic candidate:

- the same third positive contrast mode was dominant in XG2 and XG4;
- the same third principal plane was the largest positive-net plane in both;
- secondary plane structure remained distributed and family-specific.

This document prospectively freezes the next question before any XG1
model-derived scientific response is inspected:

> Does the already frozen third principal projector-contrast plane transport
> as a positive local susceptibility contrast to a structurally independent
> XG1 generator family?

To avoid ambiguity with historical ContraMamba P3/P3-W7 terminology, this
document names the current object:

`PP3 = principal-pair-3`

PP3 does not refer to the historical P3 research phase.

## External target population

External generator family:

`xg1_independent_structured_records_v1`

Frozen rows:

`data/reason_router_gen4_xg1_cross_generator_v1/synthetic_reason_router_six_cell.jsonl`

Rows SHA256:

`6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f`

Frozen source facts:

`data/reason_router_gen4_xg1_cross_generator_v1/structured_source_facts.jsonl`

Source-facts SHA256:

`fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0`

Frozen structural manifest:

`data/reason_router_gen4_xg1_cross_generator_v1/structural_manifest.json`

Structural-manifest SHA256:

`f7c881dd1a4a400e600eea4b03b46e05a48bf0da07d29905462fc3543d8af822`

Structural freeze commit:

`d9029801fd47636c155b1c846c433fc561424c8f`

XG1 design freeze commit:

`a31d2bc5ab4b939f52e969c89f3783feb9c3b233`

XG1 builder Git blob:

`c830026935a6c9f4990c6a3315c75fd5580e7264`

The structural manifest records:

- exactly 300 source pairs;
- exactly 1800 six-cell rows;
- pair IDs `xg1_fact_001` through `xg1_fact_300`;
- deterministic byte regeneration;
- no historical-generator production dependency;
- zero prior-holdout claim overlap;
- zero prior-holdout evidence overlap;
- no model geometry;
- no response fields;
- no endpoint values;
- no scientific model execution.

The exact primary population is all 300 source pairs in ascending pair-ID order.

No pair may be removed, substituted, reordered, or replaced based on a model
response.

Any structural, tokenizer, provenance, finite-value, or execution-contract
failure blocks the affected experiment. It does not authorize replacement by
another pair.

## Frozen tokenizer and anchor eligibility

Existing XG1 tokenizer/anchor eligibility artifact:

`reports/reason_router_gen4_xg1_tokenizer_anchor_eligibility_4a926a1/`

Eligibility summary Git blob:

`f31ec132ca42458300f97292dcf31110e09ef856`

Anchor-manifest SHA256:

`ebce31f5f93f33da8eb238cdc4d66510ea7ade03bdbedac88ec8d6b82f9e739e`

Eligibility result:

`PASS_300_OF_300`

Frozen active tokenizer reference:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Tokenizer file SHA256:

`b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf`

Tokenizer-config SHA256:

`9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb`

Special-tokens-map SHA256:

`57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8`

Encoding contract:

`claim[:63] + EOS(0) + evidence[:64]`

Maximum sequence length:

`128`

The existing eligibility analysis used zero model forwards and did not inspect
scientific outcomes.

## Frozen source subspaces

Do not fit, rotate, optimize, or re-rank a basis on XG1.

The two source subspaces remain exactly the frozen XG2/XG4 top-5 subspaces
already used for the completed fresh-index holdout and static localization.

Frozen Phase-1 artifact root:

`reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3/`

XG2 `alignment_delta_h.pt` SHA256:

`b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`

XG4 `alignment_delta_h.pt` SHA256:

`792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

For each family `f`, reconstruct its frozen top-5 basis exactly as previously
defined:

1. normalize each of the original 300 direction rows to unit norm;
2. compute the float64 CPU uncentered second moment;
3. use `torch.linalg.eigh`;
4. select the five largest-eigenvalue eigenvectors;
5. preserve the frozen sign-canonicalization and orthonormality checks.

No XG1 response may enter this reconstruction.

## Frozen PP3 construction

Let the two frozen orthonormal basis matrices be:

`B2` for XG2

and

`B4` for XG4.

Compute:

`B2^T B4 = U diag(c_1,...,c_5) V^T`

using float64 CPU `torch.linalg.svd`.

Principal-pair numbering is frozen by the descending singular-value order
returned for the five principal pairs.

PP3 is exactly the third principal pair.

The already localized frozen PP3 values are:

`c_3 = 0.16115855024319592`

`theta_3 = 80.7258509922 degrees`

`s_3 = sin(theta_3) = 0.98692852916688512`

Let:

`u_3 = B2 U[:,3]`

`v_3 = B4 V[:,3]`

where the notation `[:,3]` denotes the third principal vector in one-based
scientific numbering.

Construct the in-plane orthogonal coordinate:

`q_3 = (v_3 - c_3 u_3) / s_3`

In the orthonormal coordinate system `[u_3, q_3]`, construct:

`D_3 = [[s_3^2, -c_3 s_3], [-c_3 s_3, -s_3^2]]`

The two eigenvalues of `D_3` are:

`-s_3`

and

`+s_3`.

Use float64 CPU `torch.linalg.eigh(D_3)`.

The eigenvector associated with `+s_3` defines `PP3+`.

The eigenvector associated with `-s_3` defines `PP3-`.

Map both two-dimensional eigenvectors back to ambient hidden-state space via
`[u_3, q_3]`.

For each resulting ambient vector independently, canonicalize sign
deterministically:

- locate the ambient coordinate with maximum absolute magnitude;
- require that coordinate to be positive;
- if negative, multiply the whole vector by `-1`.

Both final PP3 vectors must be unit-norm and mutually orthogonal.

No response-guided sign selection is permitted.

Before any XG1 scientific execution, preparation must materialize and hash the
exact frozen PP3+ and PP3- vectors and verify that the reconstructed
`c_3`, `s_3`, eigenvalues, norms, orthogonality, and deterministic signs match
this scope.

## Fixed local probe

Use exactly:

`epsilon = 0.025`

For each unit PP3 direction `w`, preserve the already frozen intervention
semantics:

forward probe input delta:

`delta_h = +2 * epsilon * w`

reverse probe input delta:

`delta_h = -2 * epsilon * w`

with the runtime applying the corresponding half-delta to the plus/minus
branches.

For external XG1 pair `i`:

`F_i(+epsilon; w) = PE(tp, +epsilon*w) - PE(tm, -epsilon*w)`

`F_i(-epsilon; w) = PE(tp, -epsilon*w) - PE(tm, +epsilon*w)`

Define:

`J_i(w) = [F_i(+epsilon; w) - F_i(-epsilon; w)] / (2*epsilon)`

All observed quantities must be finite.

No new baseline model forward is part of the scientific endpoint.

The checkpoint, hidden-state intervention location, anchor semantics, runtime
half-delta semantics, path-efficiency observation semantics, and scientific
forward accounting must remain identical to the completed frozen Gen4-K
holdout execution unless a provenance-only compatibility correction is
explicitly frozen before scientific execution.

No model selection or checkpoint substitution is allowed.

## Primary external-transport endpoint

For every one of the 300 frozen XG1 source pairs, observe:

`J_i(PP3+)`

and

`J_i(PP3-)`

directly by finite difference.

Define the single frozen primary item-level endpoint:

`C_PP3_i = (s_3 / 5) * [J_i(PP3+)^2 - J_i(PP3-)^2]`

where:

`s_3 = 0.98692852916688512`

The factor `1/5` is retained so that `C_PP3` is on the exact contribution scale
used by the previously frozen five-dimensional projector-contrast
decomposition.

The scientific hypothesis is:

`mean(C_PP3) > 0`

over the exact 300-pair XG1 population.

This is an unsigned local-susceptibility contrast.

It does not require a common sign of `J_i(PP3+)`.

## Primary confirmatory rule

Use exactly one confirmatory test:

one-sample Student t-test on the 300 `C_PP3_i` values with one-sided
alternative:

`H1: mean(C_PP3) > 0`

Familywise alpha:

`0.05`

There is exactly one primary test, therefore no multiplicity correction is
required.

Assign:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

only if:

1. all required artifacts and provenance checks pass;
2. all 300 frozen source pairs are present and valid;
3. `mean(C_PP3) > 0`; and
4. the frozen one-sided test rejects at `alpha = 0.05`.

Otherwise assign:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_NOT_ESTABLISHED_ON_XG1_EXTERNAL_GENERATOR`

There is no rescue rule.

## Signed PP3+ diagnostic

Because PP3+ is now frozen before XG1 response observation, the direct
finite-difference values `J_i(PP3+)` may be recorded prospectively.

Allowed descriptive diagnostics are:

- mean `J(PP3+)`;
- mean absolute `J(PP3+)`;
- fraction of pairs with `J(PP3+) > 0`;
- SD of `J(PP3+)`.

These signed quantities are diagnostic only.

They receive no confirmatory p-value and do not affect the primary scientific
label.

A matching signed orientation must not be described by this experiment alone
as a universal signed causal direction.

## Allowed descriptive primary outputs

Without additional hypothesis tests, report:

- mean, median, and SD of `C_PP3`;
- mean and SD of `J(PP3+)^2`;
- mean and SD of `J(PP3-)^2`;
- the signed PP3+ diagnostics listed above;
- finite-value audit;
- exact pair-count audit;
- exact scientific-forward-count audit.

## Scientific forward budget

Each PP3 direction requires exactly four scientific forwards per source pair.

There are exactly two frozen PP3 directions:

- PP3+
- PP3-

Therefore the frozen budget is:

- `8` scientific forwards per source pair;
- `2400` scientific forwards total over 300 XG1 pairs;
- `0` new baseline scientific forwards.

No training, backward pass, task-head evaluation, logits analysis, basis
fitting, or scientific baseline run is part of this experiment.

## Outcome-blindness boundary

Before this scope freeze:

- no XG1 scientific finite-difference response for PP3+ or PP3- has been
  observed;
- no XG1 `C_PP3` value has been observed;
- no XG1 PP3 scientific p-value has been computed.

Existing XG1 structural and tokenizer eligibility work is allowed evidence
because it contains no model forward or scientific endpoint.

After this scope is committed, outcome-blind preparation may perform only:

- exact XG1 dataset/hash/schema/order validation;
- reuse and validation of the existing XG1 tokenizer/anchor eligibility
  artifact;
- exact reconstruction and materialization of frozen PP3+ and PP3-;
- PP3 vector/hash/norm/orthogonality validation;
- one dedicated scientific runner implementation;
- static/unit tests;
- fail-closed runtime/provenance/forward-budget checks.

Scientific response must remain unobserved until preparation and execution
authority are frozen.

## Prohibited post-hoc operations

The following are prohibited for this confirmatory transport experiment:

- re-ranking PP1 through PP5 on XG1;
- testing PP1, PP2, PP4, or PP5 as rescue hypotheses;
- rotating PP3 using XG1 response;
- searching neighboring PP3 directions;
- fitting a new XG1 basis;
- changing subspace dimension;
- epsilon sweep;
- checkpoint sweep;
- layer or intervention-location sweep;
- subgroup search;
- tail search;
- response-guided row exclusion;
- response-guided replacement of failed rows or pairs;
- alternative sign convention chosen after response observation;
- combining PP3 with another plane after seeing XG1 response;
- opposite-direction confirmatory testing;
- alternate primary thresholds;
- new multiplicity families;
- promotion of the signed PP3+ diagnostic into the confirmatory endpoint.

Failure of the frozen PP3 primary test must remain a negative transport result.

## Interpretation boundary

A positive result may establish that a specific projector-contrast mechanism,
discovered from the frozen XG2/XG4 geometry and frozen before XG1 scientific
observation, transports to the structurally independent XG1 generator family.

It would not establish:

- that PP3 is the sole cause of the full five-dimensional XG2-basis effect;
- universal transport to arbitrary generators or natural data;
- universal signed causal orientation;
- robustness to other checkpoints, layers, epsilon values, or model families.

A negative result would establish only that PP3 external transport was not
confirmed under this frozen XG1 experiment.

It would not erase the already established same-generator fresh-index
five-dimensional result.

## Next boundary

This scope freeze authorizes outcome-blind preparation only.

It does not itself authorize scientific XG1 PP3 model execution.

Scientific execution requires a later explicit execution freeze after the
dedicated runner, tests, exact PP3 vector identities, and runtime provenance
checks are ready.
