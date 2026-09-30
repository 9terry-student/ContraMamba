# ContraMamba Gen5 — Phase 2 State-Update Ownership Implementation Design Candidate

## 0. Status

PHASE =
`GEN5_PHASE2_STATE_UPDATE_OWNERSHIP_IMPLEMENTATION_DESIGN`

STATUS =
`CANDIDATE_FOR_FREEZE`

IMPLEMENTATION =
`NOT_AUTHORIZED`

TRAINING =
`NOT_AUTHORIZED`

SCIENTIFIC_EXECUTION =
`NOT_AUTHORIZED`

TASK_EVALUATION =
`NOT_AUTHORIZED`

KAGGLE =
`NOT_AUTHORIZED`

This document defines the first implementable Gen5 state-update ownership test
after the successful Phase 1B role-to-native-update bridge.

It does not authorize source modification, training, evaluation, causal
execution, or GPU use.

---

## 1. Immediate scientific parents

GEN5_PHASE1_MINIMAL_DESIGN_COMMIT =
`e8b5642cab0762ce9ee0ffbe8450820fc7f22710`

GEN5_PHASE1_MINIMAL_DESIGN =
`reports/reason_router_gen5_state_update_ownership_phase1_minimal_causal_design_candidate.md`

GEN5_PHASE1B_BRIDGE_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

GEN5_PHASE1B_BRIDGE_DESIGN =
`reports/reason_router_gen5_phase1b_native_update_role_bridge_spec_candidate.md`

GEN5_PHASE1B_FINAL_EVIDENCE_FREEZE_COMMIT =
`47154c02b6e691bbd2e577d934b0ef0145a04607`

NECESSITY_EVIDENCE_FREEZE_COMMIT =
`4f7dd3a9e0ca2606a7eae8e204e416f32adc884e`

RESTORATION_EVIDENCE_FREEZE_COMMIT =
`47154c02b6e691bbd2e577d934b0ef0145a04607`

FROZEN_NECESSITY_CONCLUSION =
`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

FROZEN_RESTORATION_CONCLUSION =
`GEN5_R22_RESTORATION_OVER_MATCHED_C22_REPLACEMENT_SUPPORTED`

PHASE1B_BRIDGE_CONCLUSION =
`GEN5_LAYER17_CAUSAL_ROLE_TO_LAYER22_NATIVE_WRITE_REALIZATION_BRIDGE_SUPPORTED`

Phase 2 is authorized for design because the Phase 1B B3 condition is satisfied.

---

## 2. Scientific question

Does protecting the prospectively authenticated layer-22 causal-role native-write
realization `R22` from a learned additive correction preserve causal-role
integrity more strongly than protecting the response-blind, rank-matched control
realization `C22`?

The primary question is mechanistic ownership.

It is not whether the correction improves task accuracy.

---

## 3. Frozen parent model

MODEL =
`state-spaces/mamba-130m-hf`

REPRESENTATIVE_CHECKPOINT =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

PARENT_ARM =
`G3-GROUP-D-HALF`

All parameters of the historical parent model are frozen in Phase 2:

- Mamba backbone;
- normalization parameters;
- reason-router heads;
- comparator parameters;
- final decision head;
- every historical trainable scalar or projection.

Only the new Phase 2 correction module may receive optimizer updates.

No parent parameter may be added to the optimizer.

---

## 4. Frozen owner and matched control

OWNER_SUBSPACE =
`R22`

OWNER_SUBSPACE_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

MATCHED_CONTROL_SUBSPACE =
`C22`

MATCHED_CONTROL_SUBSPACE_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

RANK_R22 =
`2`

RANK_C22 =
`2`

The bases are immutable.

Phase 2 may not:

- reconstruct either basis;
- rotate either basis;
- re-fit either basis;
- select a different rank;
- choose a different control after training;
- use Phase 2 responses to alter either basis.

---

## 5. Ownership comparison family

### G5-C0 — unrestricted learned correction

`DeltaW_eff = DeltaW_theta`

This is a contextual baseline.

It is not the primary capacity-matched ownership control.

### G5-C1 — matched ownership-null correction

`DeltaW_eff = (I - P_C22) DeltaW_theta`

The learned correction is denied write authority in the frozen rank-2,
response-blind `C22` subspace.

### G5-M1 — causal-role-owned correction

`DeltaW_eff = (I - P_R22) DeltaW_theta`

The learned correction is denied write authority in the frozen rank-2
causal-role realization `R22`.

### Primary comparison

`G5-M1 vs G5-C1`

Only the protected rank-2 subspace identity differs.

---

## 6. Native update semantics

For every active recurrent step `t` at layer 22:

`S22_t = Carry22_t + Write22_native_t + DeltaW_eff_t`

The original native write is never replaced or projected.

The ownership operator acts only on the newly introduced correction.

Therefore:

`NATIVE_WRITE_AUTHORITY = UNCHANGED`

`CORRECTION_WRITE_AUTHORITY = EXPERIMENTAL_VARIABLE`

`READ_ACCESS = SHARED`

`GRADIENT_AUTHORITY = OTHERWISE_UNCHANGED`

No information-flow masking or new gradient detach may be added as part of the
Phase 2 ownership variable.

The operator is applied at layer 22 only.

It is applied at every active non-padding recurrent step with the same semantics
in C0, C1, and M1.

This global application does not assert that R22 has separately validated
semantic meaning at every token. The scientific interpretation remains bounded
to preservation of the previously validated target-site causal role.

---

## 7. Minimal correction module

The first correction module is a rank-2, bias-free linear residual map.

Let `x22_t` be the exact frozen tensor entering the layer-22 Mamba mixer before
its input projection.

Let:

`A_theta : R^(d_model) -> R^2`

`B_theta : R^2 -> R^(STATE_WIDTH)`

Define:

`DeltaW_theta_t = B_theta A_theta x22_t`

No nonlinearity is used.

No bias is used.

No normalization is added.

No gate is learned.

No token-dependent owner is learned.

No rank search is allowed.

STATE_WIDTH =
`24576`

CORRECTION_BOTTLENECK_RANK =
`2`

The implementation must verify the exact `d_model` from the frozen model config
and fail closed if it differs from the expected model identity.

The correction parameter count is therefore:

`2 * d_model + 2 * 24576`

with no additional trainable ownership parameters.

---

## 8. Initialization

For each Phase 2 training seed:

- `A_theta` uses one deterministic seeded initialization;
- `B_theta` is initialized exactly to zero;
- all three arms receive byte-identical initial `A_theta` and `B_theta`;
- initial `DeltaW_theta` is therefore exactly zero;
- C0, C1, and M1 must reproduce the frozen parent forward before any optimizer
  step within the frozen numerical tolerance.

The initial zero-output condition is a mandatory implementation gate.

A failure of step-0 parent equivalence blocks training.

---

## 9. Differentiable projector semantics

The implementation must use the frozen basis matrices as non-trainable tensors.

For any correction vector `v`:

`P_R22(v) = R22 (R22^T v)`

`P_C22(v) = C22 (C22^T v)`

C1:

`v_eff = v - P_C22(v)`

M1:

`v_eff = v - P_R22(v)`

The projector must remain in the autograd graph with respect to `v`.

The basis tensors themselves must have:

`requires_grad = False`

Mandatory numerical checks:

- `R22^T R22 = I`;
- `C22^T C22 = I`;
- `R22^T C22 = 0`;
- for M1, `||R22^T DeltaW_eff||` is within frozen tolerance;
- for C1, `||C22^T DeltaW_eff||` is within frozen tolerance;
- parameter count is identical across C0/C1/M1;
- input tensor identity is identical across C0/C1/M1.

---

## 10. Existing Phase 1B replay is not a training backend

The Phase 1B qualified CUDA replay backend is a forward causal-assay backend.

It uses detached captures and inference-only replay surfaces.

It is not authorized as a backward/training implementation merely because its
forward outputs were validated.

Phase 2 therefore requires a new autograd-preserving layer-22 native-write
correction path.

No scientific training may begin until the new path passes independent
forward-and-backward validation.

---

## 11. Trainable-parameter boundary

During Phase 2 training:

TRAINABLE =
`A_theta, B_theta ONLY`

FROZEN =
`ALL HISTORICAL PARENT PARAMETERS + R22 + C22`

The optimizer must be constructed from an explicit allow-list of the correction
parameters.

Mandatory audit:

- every optimizer parameter ID belongs to `A_theta` or `B_theta`;
- every correction parameter is present exactly once;
- no historical parent parameter is present;
- no frozen basis tensor is present;
- after backward, parent parameter gradients are absent;
- correction gradients are finite;
- after optimizer step, parent parameter hashes remain unchanged.

---

## 12. Adaptation pressure objective

The correction is trained under task-driven pressure that is separate from the
primary causal-role assay.

The first Phase 2 training objective is:

`FINAL_3WAY_CROSS_ENTROPY_ONLY`

using the historical parent's frozen final three-way logits:

`REFUTE / NOT_ENTITLED / SUPPORT`

No causal-role endpoint is part of the training loss.

Forbidden training losses:

- `Q`-based loss;
- `D_NEC22` loss;
- `D_SUF22` loss;
- R22 coefficient loss;
- C22 coefficient loss;
- native-state reconstruction loss;
- PP3/PP5 causal loss;
- auxiliary frame/predicate/sufficiency/polarity loss;
- primary-reason loss;
- task-margin sweep;
- ownership regularizer beyond the hard projector.

This prevents direct optimization of the primary ownership endpoint.

Task loss is adaptation pressure, not the primary scientific result.

---

## 13. Training-data inheritance and static provenance gate

Phase 2 does not invent a new supervised training dataset.

The adaptation training rows must come from the exact controlled-data lineage
used by the frozen representative `G3-GROUP-D-HALF` checkpoint.

Before implementation training is authorized, a CPU-only static provenance audit
must freeze:

- exact dataset path;
- exact physical SHA256;
- exact semantic SHA256 if available;
- exact pair-level train split;
- exact split seed;
- exact ordered train-row identity;
- exact label mapping;
- confirmation that Phase 2 primary assay rows are absent.

The exact optimizer/training envelope must also be recovered from the
representative checkpoint provenance or its authoritative execution record:

- optimizer family;
- learning rate;
- weight decay;
- batch size;
- epoch count / exact update-step count;
- data order / sampler rule;
- scheduler rule, if any.

These values must be inherited, not guessed and not tuned.

If the representative training envelope cannot be uniquely recovered, Phase 2
training is blocked until a separate static reconciliation resolves it.

No dev metric may be used to tune Phase 2 hyperparameters.

---

## 14. Checkpoint selection

Phase 2 uses no performance-based checkpoint selection.

For every arm and seed:

`SELECTED_STATE = FINAL_FIXED_TRAINING_STEP`

No early stopping.

No best-dev selection.

No task-score selection.

No causal-role endpoint selection.

No averaging or SWA.

This prevents task or mechanistic response from choosing the evaluated state.

---

## 15. Phase 2 training seeds

The first experiment uses exactly three pre-specified correction-training seeds:

`5201`

`5202`

`5203`

The parent checkpoint is identical across all seeds.

The training dataset and split are identical across all seeds.

Only deterministic correction initialization and permitted stochastic training
order may depend on the Phase 2 seed according to the frozen training envelope.

No additional seed may be added after observing results.

---

## 16. Fresh primary ownership-assay population

Reserve the next non-overlapping XG1 range:

PRIMARY_OWNERSHIP_ASSAY_POPULATION =
`xg1_fact_8701..xg1_fact_9000`

PAIR_COUNT =
`300`

This population is used only for the primary causal-role integrity assay.

It may not be used for:

- correction training;
- dev selection;
- optimizer selection;
- hyperparameter selection;
- architecture selection;
- rank selection;
- owner/control selection.

Before training, static preparation must establish:

- exact deterministic generation;
- exact 300-pair count;
- six-cell structure;
- no overlap with `001..8700`;
- tokenizer/anchor eligibility;
- no response fields;
- no training labels or task metrics used for Phase 2 model selection.

If the exact range cannot be generated under the frozen XG1 lineage, Phase 2
blocks rather than substituting another cohort.

---

## 17. Primary causal-role integrity assay

For every trained arm `A` in `{C0, C1, M1}`, every Phase 2 seed, and every fresh
ownership-assay item, keep the learned correction active.

Reuse the frozen Phase 1B restoration semantics.

Start from:

`PP3_NEUTRALIZED_AT_LAYER17`

Let the native donor write be captured before the Phase 2 correction is added.

Because the historical backbone is frozen, the native donor definition remains:

`a_native = R22^T w22_native`

For each arm, define the corrected PP3-neutralized background:

`w22_B^A = w22_B_native + DeltaW_eff^A`

Then:

`w22_RR^A = w22_B^A + R22 a_native`

`w22_RC^A = w22_B^A + C22 a_native`

The same `a_native` is used for RR and RC.

The Phase 2 correction remains active identically across B, RR, and RC.

For each item:

`I_A = D_SUF22^A = Q_RR^A - Q_RC^A`

This is the arm-specific restoration-integrity score.

---

## 18. Seed aggregation

The three Phase 2 training seeds are not treated as independent item
replications.

For each fresh item `i`, first average the arm-specific restoration-integrity
score across the three frozen training seeds:

`Ibar_M1_i = mean_seed(I_M1_i)`

`Ibar_C1_i = mean_seed(I_C1_i)`

Then define the primary ownership contrast:

`D_OWN_i = Ibar_M1_i - Ibar_C1_i`

The confirmatory sample size is therefore the 300 fresh items, not
`300 x 3`.

This prevents pseudo-replication across training seeds.

---

## 19. Primary statistical decision

Exactly one confirmatory p-value is authorized for the first Phase 2 ownership
experiment:

`one-sided paired / one-sample Student t-test on D_OWN`

with alternative:

`mean(D_OWN) > 0`

No confirmatory p-value is assigned to C0.

No task-performance p-value is authorized.

No seed-specific confirmatory p-value is authorized.

---

## 20. Positive ownership rule

Phase 2 state-update ownership is supported only if all provenance,
implementation, optimization, and manipulation gates pass and all of the
following hold:

1. `mean(Ibar_M1) > 0`;
2. seed-averaged `mean(Q_RR_M1) > 0`;
3. seed-averaged `mean(S_R_M1) > 0`;
4. `mean(D_OWN) > 0`;
5. the single one-sided Student t-test gives `p < 0.05`.

Positive label:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_PRESERVES_CAUSAL_ROLE_INTEGRITY_OVER_MATCHED_C22_CONTROL`

Otherwise:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

A valid negative result is terminal for this exact Phase 2 hypothesis.

It may not be rescued by changing rank, layer, projector, correction width,
training objective, seed count, or assay cohort.

---

## 21. G5-C0 role

G5-C0 is descriptive context only.

Report:

- correction norm;
- R22 overlap before ownership projection;
- task loss trajectory;
- fresh ownership-assay `I_C0`;
- native-state diagnostics;
- tertiary task metrics.

C0 does not enter the primary confirmatory test because it has unrestricted
write access and therefore is not capacity-matched to the rank-2 protected arms.

---

## 22. Mandatory training diagnostics

For every arm and seed report:

- exact parent checkpoint SHA256;
- exact correction initialization SHA256;
- exact initial parent-equivalence residual;
- trainable parameter names/count/numel;
- parent parameter hash before training;
- parent parameter hash after training;
- optimizer parameter ownership;
- total optimizer steps;
- final correction parameter SHA256;
- correction output norm distribution;
- pre-projection R22 overlap;
- pre-projection C22 overlap;
- post-projection forbidden-subspace residual;
- finite-gradient checks;
- finite-parameter checks.

No raw full recurrent state needs to be persisted unless separately authorized.

---

## 23. Primary mechanistic diagnostics

On the fresh ownership-assay population report descriptively:

- `I_C0`, `I_C1`, `I_M1`;
- `D_OWN`;
- `Q_B`, `Q_RR`, `Q_RC`;
- `S_R`, `S_C`;
- R22 projection of the proposed correction before projection;
- R22 projection after M1 projection;
- C22 projection after C1 projection;
- full correction L2 norm;
- target POST_STATE22 change norm;
- parameter immutability checks.

These diagnostics do not introduce additional confirmatory tests.

---

## 24. Tertiary task-functional readout

Task-functional metrics are tertiary.

They may be reported on a frozen task dev split after training, but:

- they do not select checkpoints;
- they do not select hyperparameters;
- they do not alter the primary causal-role conclusion;
- task improvement cannot rescue a failed primary ownership result;
- task degradation does not by itself falsify the ownership mechanism if the
  primary mechanistic endpoint is positive.

No task-performance p-value is authorized in the first Phase 2 experiment.

---

## 25. Implementation verification gates

Because the new code changes native recurrent-update semantics and backward
flow, implementation verification is mandatory before any scientific training.

At minimum the implementation must prove:

### Forward gate

- correction disabled / zero-output reproduces the frozen parent;
- C0/C1/M1 are identical at step 0;
- native WRITE22 is unchanged before correction addition;
- only layer 22 receives the correction;
- no non-target parameter changes occur.

### Projector gate

- exact R22/C22 identity;
- exact rank 2;
- orthonormality and cross-orthogonality;
- M1 post-projection R22 residual within tolerance;
- C1 post-projection C22 residual within tolerance;
- equal correction parameter count across arms.

### Backward gate

- loss backward reaches `A_theta` and `B_theta`;
- gradients are finite;
- no parent parameter gradient is accumulated;
- no basis tensor gradient is accumulated;
- optimizer contains only correction parameters;
- one optimizer step changes correction parameters;
- the same step leaves every historical parent parameter byte-identical.

### CUDA / reference gate

A narrow reference implementation and the intended CUDA training path must agree
on correction application and gradient semantics within prospectively frozen
tolerances before full training authority is opened.

No Phase 1B inference-only replay equivalence result substitutes for this new
backward gate.

---

## 26. Forbidden actions

Phase 2 forbids:

- owner-rank sweep;
- correction-rank sweep;
- correction-width sweep;
- layer search;
- token search;
- owner/control replacement;
- R22/C22 reconstruction;
- using Phase 2 assay responses in training;
- causal-role endpoint loss;
- task-based checkpoint selection;
- causal-based checkpoint selection;
- parent backbone unfreezing;
- parent head training;
- information-flow gating;
- new gradient detach ownership manipulation;
- adding a second correction module;
- seed expansion after results;
- post-hoc cohort replacement;
- row dropping based on response;
- task-score rescue of a mechanistic failure.

---

## 27. Required staging sequence

The authorized sequence after this design is frozen is:

`PHASE2_DESIGN_FREEZE`

->

`PHASE2_STATIC_TRAINING_PROVENANCE_AND_XG1_8701_9000_PREPARATION`

->

`PHASE2_DIFFERENTIABLE_WRITE_CORRECTION_IMPLEMENTATION_AUTHORITY`

->

`PHASE2_IMPLEMENTATION`

->

`PHASE2_FORWARD_BACKWARD_INDEPENDENT_VERIFICATION`

->

`PHASE2_TRAINING_EXECUTION_AUTHORITY`

->

`G5_C0_C1_M1_TRAINING`

->

`PHASE2_PRIMARY_CAUSAL_ROLE_INTEGRITY_ASSAY`

->

`COLLECT_IMPORT_VALIDATE`

->

`PHASE2_SCIENTIFIC_INTERPRETATION`

Training or causal execution before the corresponding execution authority is
forbidden.

---

## 28. Immediate next stage after design freeze

NEXT_STAGE =
`GEN5_PHASE2_STATIC_TRAINING_PROVENANCE_AND_XG1_8701_9000_PREPARATION`

This next stage is CPU/static only.

It must not:

- load the model for scientific execution;
- train;
- evaluate;
- use CUDA;
- compute Phase 2 outcomes.

Its purpose is to freeze the exact inherited training envelope and prepare the
fresh primary ownership-assay population before implementation can observe any
Phase 2 response.

---

## 29. Phase 2 summary

OWNERSHIP_DIMENSION =
`STATE_UPDATE_AUTHORITY`

OWNER =
`R22`

MATCHED_CONTROL =
`C22`

OWNER_RANK =
`2`

CORRECTION_RANK =
`2`

CORRECTION_TYPE =
`BIAS_FREE_LINEAR_LOW_RANK_NATIVE_WRITE_RESIDUAL`

CORRECTION_SITE =
`LAYER22_NATIVE_WRITE`

NATIVE_WRITE_MODIFIED =
`NO`

PARENT_MODEL_TRAINABLE =
`NO`

CORRECTION_ONLY_TRAINABLE =
`YES`

READ_ACCESS =
`SHARED`

GRADIENT_AUTHORITY =
`UNCHANGED_EXCEPT_NORMAL_BACKPROP_TO_NEW_CORRECTION`

TRAINING_OBJECTIVE =
`FINAL_3WAY_CROSS_ENTROPY_ONLY`

CHECKPOINT_SELECTION =
`FINAL_FIXED_STEP_ONLY`

TRAINING_SEEDS =
`5201,5202,5203`

PRIMARY_ASSAY =
`FRESH_XG1_8701_9000_RESTORATION_INTEGRITY`

PRIMARY_COMPARISON =
`G5_M1_VS_G5_C1`

CONFIRMATORY_P_VALUE_COUNT =
`1`

TASK_PERFORMANCE_PRIMARY =
`NO`

STATUS =
`CANDIDATE_FOR_FREEZE`
