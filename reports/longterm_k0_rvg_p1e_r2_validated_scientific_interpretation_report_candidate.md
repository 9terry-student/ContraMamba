# K0-RVG-P1E-R2 Validated Scientific Interpretation Report Candidate

**Status:** validated scientific interpretation candidate.

**Date:** 2026-09-12

**Scientific execution authority commit:**

`db8f72bc8121a06a5dc9120611fa56b36eaec26f`

**Corrected implementation commit:**

`50a1daa781e47d1c0f1ba158beb445878e049a65`

**R2 numerical-guard correction authority commit:**

`d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`

**Validated scientific execution output directory:**

`C:\Users\Home1\Desktop\ContraMamba-K0-RVG-P1-Runs\p1-r2-scientific-db8f72b-v1`

This report interprets only the already validated corrected P1 scientific artifacts.

It does not authorize a new model forward, rerun, training step, causal intervention, K4 execution, successor-branch implementation, or any modification of the frozen P1 estimand.

## 1. Evidence state

The evidence chain is complete through artifact/provenance validation.

`P1_R2_CODE_CORRECTNESS = PASS`

`P1_R2_SCIENTIFIC_EXECUTION = COMPLETE`

`P1_R2_ARTIFACT_PROVENANCE_VALIDATION = PASS`

`P1_R2_SCIENTIFIC_INTERPRETATION_READY = YES`

Validated artifact status:

`PASS_VALIDATED_CORRECTED_P1_SCIENTIFIC_ARTIFACTS`

Exact artifact set:

1. `item_metrics.jsonl`
2. `block_metrics.jsonl`
3. `endpoint_summary.json`
4. `recurrence_audit.json`
5. `state_hash_audit.jsonl`
6. `execution_manifest.json`

Validated SHA256:

`item_metrics.jsonl = 7a8ac4cb347a1a64c2dd69653a4bc575641679667534a89f42d44e257f9550e9`

`block_metrics.jsonl = c00197ab3c93ee2bb0d3951dbb47b452a0f0aa2ba920219c28707ab544e50530`

`endpoint_summary.json = 5138fb7456e8626093753e188a3e334d79c91e5d9bc2a465073e4ae43823ec0d`

`recurrence_audit.json = f97dd5a3d934eccc91fde8430ebfea8c67018b7ef2af6a940e5adf968c4cdd5f`

`state_hash_audit.jsonl = 8b9d68a81f245b2dce46238dd915a9cf6eb0a8032556fb44b92a9c8cfce2a274`

`execution_manifest.json = 20bd491dbebaa3a882e7fce02d1ed039fec036bec08d7bd01287f109e6d74b26`

Canonical JSON/JSONL serialization:

`PASS`

Item count:

`336`

Phase-block count:

`168`

State-hash audit row count:

`12096`

## 2. Primary preregistered result

Frozen overall verdict:

`RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED`

Turning endpoint:

- valid blocks:
  `168`
- invalid blocks:
  `0`
- positive blocks:
  `1`
- negative blocks:
  `0`
- exact-zero blocks:
  `167`
- effective nonzero sign-test support:
  `1`
- raw exact two-sided sign-test p:
  `1.0`
- Holm-adjusted p:
  `1.0`
- frozen endpoint verdict:
  `TURNING_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED`

Response-coherence endpoint:

- valid blocks:
  `168`
- invalid blocks:
  `0`
- positive blocks:
  `1`
- negative blocks:
  `0`
- exact-zero blocks:
  `167`
- effective nonzero sign-test support:
  `1`
- raw exact two-sided sign-test p:
  `1.0`
- Holm-adjusted p:
  `1.0`
- frozen endpoint verdict:
  `RESPONSE_COHERENCE_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED`

Therefore the preregistered positive-direction claim is not established for either primary endpoint.

## 3. Support interpretation

This result is not caused by vector-norm support failure.

Both primary endpoints are valid on:

`168 / 168`

phase blocks.

The failure to establish direction arises because exact-zero block values dominate the frozen sign-test family.

For both endpoints:

`effective_n = positive_count + negative_count = 1`

The reported:

`sign_effect = 1.0`

must not be interpreted as a strong positive effect.

It means only that the single nonzero block was positive.

With one effective observation, the exact two-sided sign-test p-value is necessarily:

`1.0`

and no directional signal is established.

## 4. Item-level degeneracy precedes block aggregation

The read-only post-validation diagnostic establishes that the block-level degeneracy is already present at the item level.

Turning:

- valid items:
  `336`
- exact-zero `X_turn` items:
  `335`
- positive `X_turn` items:
  `1`
- negative `X_turn` items:
  `0`
- items with exact:
  `T_M == T_S`
  count:
  `335`

Response coherence:

- valid items:
  `336`
- exact-zero `X_coh` items:
  `335`
- positive `X_coh` items:
  `1`
- negative `X_coh` items:
  `0`
- items with exact:
  `C_M == C_S`
  count:
  `335`

Therefore:

`P1_ITEM_LEVEL_MATCHED_SWAPPED_ENDPOINT_IDENTITY = 335_OF_336`

for both primary endpoint constructions.

This is the central scientific observation of the validated P1 result.

## 5. Zero-block mechanism

For both primary endpoints, all `167` exact-zero blocks arise by:

`both_item_values_zero`

Count:

`167`

Exact cancellation of two nonzero item values:

`0`

Other exact-zero averaging mechanism:

`0`

Therefore the preregistered phase-pair aggregation did not create the endpoint degeneracy.

The degeneracy is already present before block averaging.

`P1_ZERO_BLOCKS_DUE_TO_PHASE_PAIR_CANCELLATION = NO`

`P1_ZERO_BLOCKS_DUE_TO_BOTH_ITEMS_ZERO = 167_OF_167`

## 6. The single nonzero block

Both endpoints have exactly the same single nonzero phase block:

`block_index = 163`

Phase class:

`25`

Item A:

`local_template_index = 163`

Stable ID:

`k0-rvg-p0-v1:770595a19a44b178faa6d32dae44b7bb0833a744dfecd4e27e2239e7c4c152a2`

Item B:

`local_template_index = 331`

Stable ID:

`k0-rvg-p0-v1:4b00403f3bd53c3426accbfa5202a2d3375501f837bcfe82e962840442d2c6be`

Turning:

- item A:
  `X_turn = 1.181793825300037e-08`
- item B:
  `X_turn = 0.0`
- block:
  `B_turn = 5.908969126500185e-09`

Response coherence:

- item A:
  `X_coh = 4.066333458840887e-08`
- item B:
  `X_coh = 0.0`
- block:
  `B_coh = 2.0331667294204436e-08`

These magnitudes are recorded descriptively only.

They do not define a new threshold.

They do not justify tolerance retuning.

They do not create an exploratory exception to the frozen sign-test contract.

## 7. Corrected recurrence result

The numerical-guard correction performed its intended integrity role during the scientific run.

Exact native recurrence accepted:

`12096 / 12096`

Hard integrity gate:

`EXACT_NATIVE_RECURRENCE`

Incoming common-state comparisons:

`2688`

Incoming common-state failures:

`0`

Rearrangement diagnostic pass count:

`12078`

Rearrangement diagnostic exceedance count:

`18`

Rearrangement blocking:

`false`

Frozen descriptive tolerances:

`velocity_atol = 1e-6`

`velocity_rtol = 1e-5`

Maximum absolute rearrangement residual:

`3.814697265625e-06`

Maximum relative Frobenius residual:

`2.0347988090282514e-07`

Maximum scaled tolerance residual:

`1.5940370559692383`

The `18` rearrangement diagnostic exceedances were retained because exact native recurrence passed.

This confirms that the corrected R2 guard avoided the previously identified false hard failure without changing the scientific population or frozen endpoint definitions.

`R2_NUMERICAL_GUARD_OPERATION = VALIDATED_IN_SCIENTIFIC_EXECUTION`

The recurrence diagnostic exceedances are not scientific endpoint evidence.

## 8. What the P1 result supports

The validated P1 evidence supports the following bounded conclusion:

Under the frozen K0-RVG-P1 observational raw-native-state geometry, layer-23 measurement, `W=8` temporal window, frozen matched-versus-swapped construction, raw Frobenius endpoint definitions, frozen P0 population, and preregistered block/sign-test inference, the study does not establish matched-condition raw native vector organization on either turning or response coherence.

More specifically, the frozen primary contrasts are exactly zero for `335 / 336` items on both endpoints.

Thus the failure is not mainly statistical noise around a broad nonzero effect and is not caused by phase-block cancellation.

The primary matched-versus-swapped endpoint contrast is almost everywhere degenerate under this exact measurement construction.

## 9. What the P1 result does not support

This result does not establish any of the following stronger claims:

- that Mamba recurrent states contain no task-relevant information;
- that native recurrent-state dynamics contain no temporal structure;
- that no lower-dimensional, conditional, local, nonlinear, or alternative native-state geometry could separate matched from swapped conditions;
- that all layers behave like layer 23;
- that all windows behave like `W=8`;
- that the matched and swapped recurrent tensors are themselves identical;
- that carry and write components are mechanistically irrelevant;
- that causal intervention would fail;
- that Branch A or Branch B is automatically selected;
- that a different scientific estimand should now be tuned on this same result.

The validated evidence concerns only the frozen P1 estimand.

## 10. Scientific interpretation of the degeneracy

The strongest justified interpretation is:

`P1_PRIMARY_CONTRAST_DEGENERATE_AT_ITEM_LEVEL = YES`

The exact primary contrast definitions satisfy:

`X_turn = 0`

and:

`X_coh = 0`

for `335 / 336` items.

This means the frozen matched-versus-swapped P1 endpoint construction sees essentially no differential organization across the population.

Because the equality occurs before phase-block aggregation, the appropriate next scientific question is not to retune the sign test or block aggregation.

The next question must instead examine why the frozen observational geometry maps matched and swapped conditions to equal endpoint values for almost every item.

That question must be addressed under a separately frozen successor-stage authority.

## 11. Falsification status

The narrow P1 hypothesis tested by the frozen primary endpoints is not supported.

`P1_RAW_NATIVE_VECTOR_ORGANIZATION_POSITIVE_DIRECTIONAL_HYPOTHESIS = NOT_ESTABLISHED`

This is a valid negative result for the frozen P1 measurement construction.

It is not a falsification of the broader native-state-kinematics research program.

The broader program remains open because P1 tested one specific observational geometry at one layer and one temporal construction.

## 12. Provenance-valid scientific conclusion

The following conclusion is now admissible because code correctness, execution completion, artifact provenance, and endpoint recomputation have all passed independently:

`P1_VALIDATED_SCIENTIFIC_CONCLUSION = RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED`

The associated mechanistic characterization is:

`P1_ENDPOINT_DEGENERACY_LOCATION = ITEM_LEVEL_BEFORE_BLOCK_AGGREGATION`

`P1_EXACT_ZERO_ITEM_COUNT_TURNING = 335_OF_336`

`P1_EXACT_ZERO_ITEM_COUNT_RESPONSE_COHERENCE = 335_OF_336`

`P1_SINGLE_NONZERO_BLOCK = 163`

`P1_SINGLE_NONZERO_BLOCK_SHARED_BY_BOTH_ENDPOINTS = YES`

## 13. Branch-selection boundary

This interpretation report does not itself activate a successor branch.

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED_UNTIL_INTERPRETATION_REPORT_FREEZE`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

After this exact interpretation report is frozen, the next authorized controller action is to inspect the frozen successor-branch definitions and choose the branch that directly addresses the validated item-level degeneracy without converting the program into post-hoc hyperparameter tuning.

No training, evaluation, model forward, or new scientific execution is authorized by this report.

## 14. Final markers

`P1_R2_SCIENTIFIC_EXECUTION_COMPLETE = YES`

`P1_R2_ARTIFACT_PROVENANCE_VALID = YES`

`P1_R2_ENDPOINT_RECOMPUTATION_MATCH = YES`

`P1_VALIDATED_SCIENTIFIC_CONCLUSION = RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED`

`P1_PRIMARY_CONTRAST_DEGENERATE_AT_ITEM_LEVEL = YES`

`P1_EXACT_ZERO_ITEM_COUNT_TURNING = 335_OF_336`

`P1_EXACT_ZERO_ITEM_COUNT_RESPONSE_COHERENCE = 335_OF_336`

`P1_ZERO_BLOCKS_DUE_TO_PHASE_PAIR_CANCELLATION = NO`

`P1_ZERO_BLOCKS_DUE_TO_BOTH_ITEMS_ZERO = 167_OF_167`

`P1_SINGLE_NONZERO_BLOCK = 163`

`P1_SINGLE_NONZERO_BLOCK_SHARED_BY_BOTH_ENDPOINTS = YES`

`R2_EXACT_RECURRENCE_ACCEPTED = 12096_OF_12096`

`R2_REARRANGEMENT_DIAGNOSTIC_EXCEEDANCE_COUNT = 18`

`R2_INCOMING_COMMON_STATE_FAILURES = 0`

`SCIENTIFIC_RERUN_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED_UNTIL_INTERPRETATION_REPORT_FREEZE`

This report becomes the frozen P1 scientific interpretation only after this exact document is committed and pushed as the immediate one-file child of:

`db8f72bc8121a06a5dc9120611fa56b36eaec26f`
