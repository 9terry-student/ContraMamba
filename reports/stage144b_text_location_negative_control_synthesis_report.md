# Stage144-B Text Location Negative-Control Synthesis

## 1. Summary decision

Decision: `STAGE144B_TEXT_LOC_DISJOINT_NEGATIVE_CONTROLS_EXPOSE_FALSE_POSITIVE_RISK`

Stage144-A exposes serious false-positive risk for raw `text_loc_disjoint`. The failure is concentrated in alias/surface-form SUPPORT and organization-like SUPPORT cases.

Raw `text_loc_disjoint` remains useful as an optional shadow diagnostic, but it is not final-integration ready. It must not be routed into final logits or final predictions.

## 2. Why Stage144-A was needed

Stage142-B showed robust broader stress behavior across existing prediction JSONLs, including a false SUPPORT reduction of 53, a false NOT_ENTITLED increase of 3, and `feature_correct_support_fp = 0`.

Stage144-A was needed because broad existing exports did not directly stress negative controls that could reveal whether the raw set-based location policy overfires on SUPPORT rows with aliases, surface-form variation, organization-like names, or overlapping multi-location evidence.

## 3. Aggregate result

Stage144-A used 97 targeted rows and the Stage142 analyzer produced `STAGE142_TEXT_LOCATION_GUARD_SHADOW_MIXED`.

- Changed rows: 26
- False SUPPORT: 22 -> 10
- Delta false SUPPORT: -12
- False NOT_ENTITLED: 0 -> 14
- Delta false NOT_ENTITLED: +14
- Feature false SUPPORT true positives: 12
- Feature correct SUPPORT false positives: 14
- Feature false SUPPORT false negatives: 10
- Feature precision for false SUPPORT among SUPPORT predictions: 0.46153846153846156
- Feature recall for false SUPPORT among SUPPORT predictions: 0.5454545454545454

The aggregate result confirms a real diagnostic signal, but the false-positive burden blocks final integration.

## 4. Positive-control success

The policy still succeeds on simple explicit location mismatch.

For `simple_location_mismatch_ne`, all 12 rows changed, false SUPPORT dropped from 12 to 0, and false NOT_ENTITLED did not increase. This is the desired positive-control behavior for explicit disjoint location mismatch.

## 5. Negative-control passes

The policy is safe on exact same-location, overlap/multi-location support, no-location support, and non-SUPPORT prediction scope.

- `same_location_exact_support`: 12 rows, 0 changed.
- `same_location_with_extra_distractor_support`: 12 rows, 0 changed.
- `multi_location_overlap_support`: 12 rows, 0 changed.
- `no_location_support`: 15 rows, 0 changed.
- `non_support_prediction_safety`: 4 rows, 0 changed.

These passes show that the policy scope and overlap behavior work in several important safe cases.

## 6. Failure modes

The most serious failures are concentrated in alias/surface-form SUPPORT and organization-like SUPPORT.

`alias_surface_support` produced 8 changed rows out of 8 and introduced 8 false NOT_ENTITLED errors. Examples include New York City vs New York, NYC vs New York City, Los Angeles vs LA, and San Francisco vs SF.

`organization_alias_surface_support` produced 6 changed rows out of 6 and introduced 6 false NOT_ENTITLED errors. Examples include Beacon Harbor Labs vs Beacon Harbor Laboratory, Oxford Research Center vs Oxford Research Centre, and Falcon Ridge Institute vs Falcon Ridge Initiative.

The policy also misses some intended NOT_ENTITLED cases. `route_reversal_ne` left all 6 false SUPPORT errors unchanged, reflecting a known set-based policy limitation on order or role mismatches. `lowercase_location_mismatch_ne` left all 4 false SUPPORT errors unchanged, reflecting a known case-sensitive extractor miss.

## 7. Interpretation

Raw `text_loc_disjoint` should remain an optional shadow-only diagnostic. It identifies explicit disjoint location mismatch and can reduce false SUPPORT in targeted positive controls.

It is not final-integration ready because alias and organization-like support controls create 14 false NOT_ENTITLED errors. The next action is to refine the extractor and policy before any further final integration discussion.

## 8. Safety policy

The workflow remains shadow-only and diagnostic-only.

It does not modify final logits, final predictions, training, checkpoint selection, Stage128 guard behavior, Stage15 behavior, external-data training usage, or thresholds used for model selection.

## 9. Stage145 recommendation

Stage145 should refine the extractor/policy and rerun Stage144.

Recommended requirements:

- Keep the analyzer shadow-only.
- Do not modify source predictions.
- Do not integrate with final logits.
- Add alias/canonicalization handling for common location aliases.
- Add organization-like suffix awareness.
- Re-run Stage144 negative controls after refinement.

Avoid final prediction override, training loss integration, using gold labels or `intervention_type` as policy inputs, and claiming final integration readiness.
