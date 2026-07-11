# Stage167-A V6B Native Scalar Schema Report

Stage166-A reproduced a non-collapsed Stage63-like external distribution from a checkpoint-backed `v6b_minimal` run. The supplied Stage166-A result has 1000 external rows, accuracy 0.295, macro-F1 0.28791024061267173, and prediction counts REFUTE 178, NOT_ENTITLED 536, SUPPORT 286.

## Schema Diagnosis

The original 10-field scalar validator incorrectly treated vnext-only fields as required for `v6b_minimal`. For `v6b_minimal`, the complete architecture-native scalar schema is the seven common/native fields:

- `frame_prob`
- `predicate_coverage_prob`
- `sufficiency_prob`
- `entitlement_prob`
- `polarity_margin`
- `positive_energy`
- `negative_energy`

The vnext-only fields remain unsupported for `v6b_minimal` and must not be synthesized or aliased from `entitlement_prob`:

- `compositional_entitlement_prob`
- `learned_entitlement_prob`
- `learned_entitlement_logit`

## Stage166 Reinterpretation

Stage166-A is not a scalar-export failure. It provides complete coverage of the seven architecture-common/native scalar fields for the 1000 external rows. The native coverage is expected to be 7/7, with `native_required_scalar_pass = true` and `all_requested_scalar_pass = false` because vnext-only fields are unsupported for this architecture.

Expected decision: `STAGE167A_V6B_NATIVE_SCALAR_EXPORT_READY_FOR_COMMON_SCALAR_ANALYSIS`.

## Parser Correction

The clean prediction artifact parser now inspects the whole JSON document before falling back to JSONL. It supports JSON arrays, JSON objects containing prediction lists under common keys, ordinary JSONL, and separate reporting for valid scalar or non-dictionary JSON values.

## Next Stage

The next analytical stage may compare recovered and regressed examples using the common seven-scalar subset.

No model decision behavior, training losses, checkpoint selection, training data, external-label tuning, or missing scalar synthesis is changed by Stage167-A.