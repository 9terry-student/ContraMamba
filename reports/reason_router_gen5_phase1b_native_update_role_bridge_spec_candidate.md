# ContraMamba Gen5 — Phase 1B Native-Update Role Bridge Specification Candidate

## 0. Status

PHASE = `GEN5_PHASE1B_NATIVE_UPDATE_ROLE_BRIDGE_SPECIFICATION`

STATUS = `FROZEN_ON_COMMIT`

IMPLEMENTATION = `NOT_AUTHORIZED`

SCIENTIFIC_EXECUTION = `NOT_AUTHORIZED`

TRAINING = `NOT_AUTHORIZED`

TASK_EVALUATION = `NOT_AUTHORIZED`

KAGGLE = `NOT_AUTHORIZED`

This document prospectively defines the first bridge required by Gen5 Phase 1.

It does not implement the bridge and does not authorize model execution.

---

## 1. Immediate scientific parent

GEN5_PHASE1_FREEZE_COMMIT =
`e8b5642cab0762ce9ee0ffbe8450820fc7f22710`

GEN5_PHASE1_OBJECT =
`STATE_UPDATE_AUTHORITY`

GEN5_PHASE1_REQUIREMENT =
`ROLE_TO_NATIVE_UPDATE_BRIDGE_REQUIRED`

The Phase 1 rule remains binding:

`DIRECT_PP3_TO_OWNER_PROMOTION = FORBIDDEN`

`DIRECT_K_ALIGNMENT_TO_OWNER_PROMOTION = FORBIDDEN`

---

## 2. Scientific question

Can the already validated 130M layer-17 operational causal role be linked
prospectively to a locally causal native recurrent-update realization at a
pre-specified downstream layer-22 site?

The bridge is successful only if a downstream layer-22 realization derived from
the upstream causal-role perturbation:

1. transports to a new population;
2. is locally necessary relative to a matched response-blind control; and
3. restores the frozen broad susceptibility endpoint from the upstream
   role-neutralized background more strongly than a matched replacement control.

This is a causal mediation-style bridge.

It is not a claim of coordinate identity between layer 17 and layer 22.

---

## 3. Frozen model identity for the bridge

All Phase 1B discovery and confirmation work must use the exact Gen4
representative checkpoint on which the PP3 causal-role program is grounded.

CHECKPOINT_PATH =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

HF_MODEL_FAMILY =
`state-spaces/mamba-130m-hf`

No other checkpoint may be substituted.

No checkpoint sweep is allowed.

---

## 4. Relation to K-series evidence

The strongest K-series causal falsification used a different frozen whole-model
checkpoint:

K_CHECKPOINT_SHA256 =
`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

K_ENCODER_CANONICAL_DIGEST =
`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Those identifiers use a different provenance/digest surface from the Gen4
native-backbone signature and must not be treated as directly equal or unequal
without an explicit same-algorithm audit.

Therefore the K-series result is used here only as independent prior motivation
for:

- examining native recurrent write/post-state rather than a generic hidden-state
  scalar;
- using layer 22 as the pre-specified downstream bridge site;
- preserving write and post-state as separate mechanistic endpoints.

Phase 1B does NOT assume that the K-series strong-side alignment direction is the
same causal role as PP3.

A later static identity audit may compare the two checkpoints with one canonical
native-backbone hashing rule. That audit is informative but is not a prerequisite
for this bridge because all bridge execution is confined to the Gen4 checkpoint
above.

---

## 5. Frozen upstream causal-role semantics

The upstream intervention must reuse the already validated layer-17 PP3
necessity/restoration semantics without modification.

UPSTREAM_LAYER =
`17`

UPSTREAM_TOKEN =
`FROZEN_TARGET_TOKEN`

UPSTREAM_HOOK =
`EXISTING_LAYER17_MIXER_IN_PROJ_INTERVENTION_SEMANTICS`

UPSTREAM_ROLE_PLANE =
`PP3`

UPSTREAM_MATCHED_CONTROL_PLANE =
`PP5`

The exact frozen PP3/PP5 vectors and hashes must be inherited from the validated
Gen4 PP3 artifacts.

The bridge implementation may not:

- recompute PP3;
- re-fit PP3;
- choose a different plane;
- change layer;
- change target-token definition;
- change strong-channel masking semantics;
- choose a new matched control after observing bridge responses.

---

## 6. Frozen upstream broad endpoint

The upstream causal-role program uses the broad susceptibility endpoint:

`Q_i(c) = E_XG2_i(c) - E_XG4_i(c)`

with the already frozen XG2/XG4 signed intervention semantics.

Phase 1B keeps this endpoint unchanged.

The bridge does not introduce task accuracy, task logits, or behavioral labels
as primary outcomes.

---

## 7. Pre-specified downstream native-update site

DOWNSTREAM_LAYER =
`22`

DOWNSTREAM_TOKEN =
`SAME_FROZEN_TARGET_TOKEN_INDEX_AS_THE_UPSTREAM_XG1_FORWARD`

DOWNSTREAM_NATIVE_OBJECTS =

1. `WRITE22`
2. `POST_STATE22`

The layer-22 site is fixed before bridge outcomes are observed.

It is motivated by the independent K-series native recurrence program, but the
Phase 1B result will be new evidence on the Gen4 representative checkpoint and
fresh XG1 populations.

No layer sweep is authorized.

The K-series `k=2` coordinate is not copied into the XG1 bridge. The bridge uses
the existing XG1 target-token coordinate inherited from the PP3 program.

---

## 8. Native vector representation

For each branch and condition, capture the full native layer-22 recurrent write
and post-update state at the target token using the existing validated native
Mamba capture semantics.

Define:

`w22_i(c) = vec(WRITE22_i(c))`

`s22_i(c) = vec(POST_STATE22_i(c))`

The vectorization order must be frozen by implementation and proven identical
across all conditions.

No response-dependent channel filtering is allowed.

No top-k channel selection is allowed.

---

## 9. Fresh population allocation

The deterministic XG1 lineage already covers frozen pair identities through:

`xg1_fact_001..xg1_fact_7800`

Phase 1B reserves the next three non-overlapping 300-pair populations:

### Bridge construction population

`xg1_fact_7801..xg1_fact_8100`

Purpose:

- construct the layer-22 candidate role realization only;
- no confirmatory scientific conclusion.

### Bridge necessity confirmation population

`xg1_fact_8101..xg1_fact_8400`

Purpose:

- prospective local necessity test.

### Bridge restoration confirmation population

`xg1_fact_8401..xg1_fact_8700`

Purpose:

- prospective restoration test.

Before any model execution, static preparation must establish for all three
populations:

- deterministic XG1 generation;
- exact 300-pair count;
- six-cell structure;
- no pair-ID overlap with `001..7800`;
- no claim/evidence-row overlap with prior frozen XG1 populations;
- no overlap among the three Phase 1B populations;
- tokenizer/anchor eligibility;
- zero response fields present during cohort construction.

If deterministic generation of these exact ranges cannot be established from the
frozen XG1 generator semantics, Phase 1B is blocked rather than reassigned to a
different cohort.

---

## 10. Discovery-only propagated-role construction

The construction population is used only to define a candidate layer-22 role
realization.

For each item, obtain the existing upstream matched pair:

- `PP3_NEUTRALIZED`
- `PP5_COEFFICIENT_CONTROL`

under identical model/input semantics.

Capture `w22` and `s22`.

Define the PP3-specific propagated write contrast:

`dW_i = w22_i(PP5_CONTROL) - w22_i(PP3_NEUTRALIZED)`

and the corresponding post-state contrast:

`dS_i = s22_i(PP5_CONTROL) - s22_i(PP3_NEUTRALIZED)`

The ownership realization is constructed from `dW`, because Phase 1 selected
write authority as the first ownership semantic.

`dS` is a downstream bridge diagnostic and must not be used to select the write
subspace.

---

## 11. Fixed minimal rank

The first bridge tests a minimal two-dimensional downstream realization:

`rank(R22) = 2`

This rank is fixed prospectively.

It is chosen to test whether the already validated two-dimensional upstream role
admits a minimal two-dimensional downstream write realization.

It is NOT a claim that causal-role dimensionality is invariant across depth.

If rank-2 fails, Phase 1B fails as specified.

A rank sweep, variance-threshold rank selection, or post-hoc higher-rank rescue is
not authorized.

Any later higher-rank hypothesis requires a new scientific design.

---

## 12. Construction of R22

On `xg1_fact_7801..8100` only:

1. stack centered discovery vectors `dW_i`;
2. compute the deterministic SVD;
3. freeze the first two right-singular directions;
4. use deterministic sign canonicalization;
5. store them as an orthonormal basis of `R22`.

The construction must not inspect:

- necessity confirmation outcomes;
- restoration confirmation outcomes;
- task labels beyond already frozen cohort semantics;
- downstream task logits;
- task accuracy;
- behavioral outcomes.

The resulting basis vectors and all construction identities must be frozen before
either confirmation population is executed.

---

## 13. Response-blind matched control C22

The primary control must be rank matched and response blind.

On the same construction population, use only the NATIVE condition layer-22
`WRITE22` vectors and no PP3-vs-control response labels.

Construct a deterministic native write covariance basis.

After removing the span of `R22`, define `C22` as the first two orthonormal
native-geometry directions in the residual covariance basis.

Therefore:

`rank(C22) = rank(R22) = 2`

and:

`R22^T C22 = 0`

within frozen numerical tolerance.

C22 is a geometry-matched control, not a causal-role candidate.

Its construction must not use `dW`, `dS`, Q values, task logits, or confirmation
responses except for the orthogonal exclusion of the already frozen `R22` span.

---

## 14. Layer-22 coefficient-transfer control semantics

For a native write vector `w`, define R22 coefficients:

`a = R22^T w`

R22 neutralization:

`N_R(w) = w - R22 a`

Matched C22 coefficient-transfer control:

`N_C(w) = w - C22 a`

The same R22-derived coefficient vector `a` is transferred to C22.

Because both bases are orthonormal and rank matched:

`||w - N_R(w)||_2 = ||w - N_C(w)||_2`

must hold within frozen numerical tolerance.

This is the layer-22 analogue of the existing PP3/PP5 matched-control logic.

---

## 15. Confirmation A — local necessity

Population:

`xg1_fact_8101..xg1_fact_8400`

Upstream layer-17 condition:

`NATIVE`

At layer 22 compare:

1. `NATIVE22`
2. `R22_NEUTRALIZED`
3. `C22_COEFFICIENT_CONTROL`

The remainder of the model forward is unchanged.

For each item compute the unchanged broad endpoint:

`Q0_i`
`QR_i`
`QC_i`

Define:

`A_R_i = Q0_i - QR_i`

`A_C_i = Q0_i - QC_i`

Primary necessity contrast:

`D_NEC22_i = QC_i - QR_i`

Positive necessity requires:

1. all provenance/manipulation gates pass;
2. `mean(Q0) > 0`;
3. `mean(A_R) > 0`;
4. `mean(D_NEC22) > 0`;
5. one-sided one-sample Student t-test on `D_NEC22` gives `p < 0.05`.

Exactly one confirmatory p-value is authorized for the necessity stage.

Positive label:

`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

Otherwise:

`GEN5_R22_LOCAL_NECESSITY_NOT_ESTABLISHED`

---

## 16. Confirmation B — restoration from upstream role neutralization

Population:

`xg1_fact_8401..xg1_fact_8700`

Start from the frozen upstream condition:

`PP3_NEUTRALIZED_AT_LAYER17`

At layer 22, the implementation must have a paired native donor capture for the
same item/branch under the identical unmodified forward.

Let the native donor R22 coefficient be:

`a_native = R22^T w22_native`

Let `w22_B` be the layer-22 write under upstream PP3 neutralization.

Compare:

### B

Upstream PP3-neutralized background with no layer-22 restoration.

### RR

Restore the exact native R22 component:

`w22_RR = w22_B + R22 a_native`

### RC

Matched C22 replacement:

`w22_RC = w22_B + C22 a_native`

The same coefficient vector is used in RR and RC.

Thus the addition norms must match exactly within frozen tolerance.

No coefficient fitting is allowed on the restoration population.

---

## 17. Restoration endpoint

For each item compute:

`Q_B_i`
`Q_RR_i`
`Q_RC_i`

Define:

`S_R_i = Q_RR_i - Q_B_i`

`S_C_i = Q_RC_i - Q_B_i`

Primary restoration contrast:

`D_SUF22_i = Q_RR_i - Q_RC_i`

Positive restoration requires:

1. all provenance/manipulation gates pass;
2. `mean(Q_RR) > 0`;
3. `mean(S_R) > 0`;
4. `mean(D_SUF22) > 0`;
5. one-sided one-sample Student t-test on `D_SUF22` gives `p < 0.05`.

Exactly one confirmatory p-value is authorized for the restoration stage.

Positive label:

`GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED`

Otherwise:

`GEN5_R22_RESTORATION_NOT_ESTABLISHED`

---

## 18. Bridge success rule

The complete Phase 1B bridge is supported only if BOTH independent confirmation
stages are positive:

1. `GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`
2. `GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED`

Then the bounded bridge conclusion is:

`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

This supports R22 as a prospectively constructed, locally necessary, and
restoration-capable native write realization associated with the already validated
upstream causal role.

It does NOT establish exact coordinate transport.

---

## 19. Bridge failure classes

### B0 — static/provenance block

Examples:

- cohort overlap;
- tokenizer/anchor failure;
- checkpoint mismatch;
- plane hash mismatch;
- capture identity mismatch;
- construction leak;
- invalid matched-control algebra.

Conclusion:

`PHASE1B_BLOCKED_NO_SCIENTIFIC_INTERPRETATION`

### B1 — propagated rank-2 realization not confirmably necessary

Necessity stage fails.

Conclusion:

`GEN5_RANK2_NATIVE_WRITE_BRIDGE_NOT_ESTABLISHED`

No restoration execution is scientifically interpreted as rescue if necessity has
already failed.

### B2 — necessity positive, restoration negative

Conclusion:

`GEN5_R22_NECESSITY_SUPPORTED_BUT_MEDIATION_RESTORATION_NOT_ESTABLISHED`

This is insufficient to authorize state-update ownership.

### B3 — necessity and restoration positive

Conclusion:

`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

Only B3 advances the ownership program.

---

## 20. Mandatory manipulation checks

Every scientific run must verify:

- exact checkpoint SHA256;
- exact PP3/PP5 basis hashes;
- exact R22/C22 basis hashes;
- exact population identities;
- target-token equality across conditions;
- R22/C22 orthonormality;
- R22/C22 cross-orthogonality;
- coefficient-transfer norm equality;
- no unintended layer/token modifications;
- layer-22 write intervention applied only at the frozen site;
- post-state captured after the modified write enters the recurrent update;
- all non-target model parameters identical;
- zero training;
- zero backward passes;
- zero task-head optimization;
- zero response-guided row dropping.

Any failed mandatory manipulation check blocks scientific interpretation.

---

## 21. Native write/post-state bridge diagnostics

Although the primary confirmatory decision uses the frozen broad Q endpoint,
Phase 1B must also report descriptive native bridge quantities:

- R22-projected `WRITE22` magnitude;
- C22-projected `WRITE22` magnitude;
- full `WRITE22` change norm;
- full `POST_STATE22` change norm;
- itemwise sign of write reduction/recovery;
- itemwise sign of post-state reduction/recovery;
- carry change if capture is available under the existing native observer.

These are mechanistic diagnostics.

They must not introduce additional confirmatory p-values.

A bridge can be statistically positive on Q while showing unexpected native-state
diagnostics; such discordance must be reported rather than repaired post hoc.

---

## 22. Native-backbone identity audit between parent programs

A CPU-only static audit may compute the same canonical native-backbone signature
for the K-series A0 checkpoint and the Gen4 representative checkpoint.

This audit asks only whether the `mamba.*` parameter payload is byte-identical
under one common hashing algorithm.

Possible outcomes:

`PARENT_NATIVE_BACKBONE_IDENTICAL`

or

`PARENT_NATIVE_BACKBONE_DIFFERENT`

Neither outcome changes the Phase 1B checkpoint.

If identical, the K-series layer-22 evidence has stronger direct backbone
continuity with Phase 1B.

If different, K-series remains independent site/endpoint motivation only.

No scientific result may be inferred from the identity audit itself.

---

## 23. Forbidden actions

Phase 1B forbids:

- layer search;
- target-token search;
- checkpoint search;
- rank search;
- channel search;
- top-k search;
- alternative plane search;
- alternative control search after outcomes;
- cohort replacement after outcomes;
- row filtering based on response;
- task-score optimization;
- tuning intervention magnitude;
- using confirmation data to construct R22 or C22;
- using post-state response to construct the write owner;
- promoting a failed rank-2 bridge by silently increasing rank.

---

## 24. Execution separation

This specification does not authorize implementation.

The future sequence is:

`PHASE1B_SPEC_FREEZE`
->
`STATIC_COHORT_AND_IDENTITY_PREPARATION`
->
`IMPLEMENTATION`
->
`INDEPENDENT_VERIFICATION`
->
`EXECUTION_AUTHORITY`
->
`CONSTRUCTION_RUN`
->
`FREEZE_R22_C22`
->
`NECESSITY_CONFIRMATION`
->
`RESTORATION_CONFIRMATION`
->
`VALIDATED_BRIDGE_INTERPRETATION`

Because hidden-state intervention semantics are high risk, implementation must
receive independent verification before scientific execution.

---

## 25. Advancement rule

State-update ownership implementation is authorized for design only after a
validated B3 bridge result.

If B3 is established, the next scientific stage is:

`GEN5_PHASE2_STATE_UPDATE_OWNERSHIP_IMPLEMENTATION_DESIGN`

using:

- `R22` as the protected causal-role native-write realization;
- `C22` as the matched ownership-null control;
- frozen backbone;
- shared read access;
- unchanged gradient authority;
- G5-C0 / G5-C1 / G5-M1 comparison semantics from Phase 1.

If B3 is not established, Gen5 does not proceed to that implementation under the
current state-update ownership hypothesis.

A new hypothesis would require a new design.

---

## 26. Phase 1B summary

MODEL =
`state-spaces/mamba-130m-hf`

BRIDGE_CHECKPOINT =
`G3-GROUP-D-HALF_SEED180_1ff3fcf2`

UPSTREAM_CAUSAL_ROLE =
`LAYER17_PP3`

DOWNSTREAM_CANDIDATE_SITE =
`LAYER22_NATIVE_WRITE`

CONSTRUCTION_POPULATION =
`XG1_7801_8100`

NECESSITY_CONFIRMATION_POPULATION =
`XG1_8101_8400`

RESTORATION_CONFIRMATION_POPULATION =
`XG1_8401_8700`

R22_RANK =
`2`

MATCHED_CONTROL =
`RESPONSE_BLIND_RANK2_C22`

PRIMARY_NECESSITY_ENDPOINT =
`D_NEC22 = Q_C22_CONTROL - Q_R22_NEUTRALIZED`

PRIMARY_RESTORATION_ENDPOINT =
`D_SUF22 = Q_R22_RESTORED - Q_C22_REPLACEMENT`

BRIDGE_SUCCESS_REQUIRES =
`NECESSITY_AND_RESTORATION`

TASK_PERFORMANCE_PRIMARY =
`NO`

README_UPDATE_REQUIRED =
`NO`

STATUS =
`FROZEN_ON_COMMIT`
