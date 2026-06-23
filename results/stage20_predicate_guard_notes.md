# Stage20-A Predicate Guard Notes

## Purpose

Stage20-A tests whether the remaining `predicate_mismatch` false-entitlement can be corrected if a predicate mismatch signal is available. It is a diagnostic upper-bound and detector analysis, not a trained model or architecture change.

## Detector modes

- `oracle_probe_type`: flags `stage15_probe_type == predicate_mismatch`. This is an explicit upper bound and is not deployable.
- `metadata_predicate`: uses available probe metadata such as `source_intervention_type == predicate_swap` or `stage15_original_probe_type == predicate_swap`.
- `lexical_predicate`: uses a conservative predicate-family conflict map and only flags when claim/evidence have high content overlap and contain known conflicting predicate families.

## Guard modes

- `hard_override`: flagged examples are set to `NOT_ENTITLED`.
- `pseudo_logit_penalty`: flagged examples receive a pseudo-logit shift toward `NOT_ENTITLED`.

## Lexical safety principle

The lexical detector must not punish generic lexical difference. `surface_control` is supposed to remain SUPPORT. The detector therefore requires both high content overlap and a known predicate-family conflict.

## Interpretation

If oracle mode fixes predicate mismatch while preserving controls, predicate failure is correctable given a reliable predicate signal. If lexical mode preserves controls but has low recall, Stage20 points toward a learned predicate comparator rather than a broader lexical-distance heuristic.

## Caveat

Stage20-A is post-processing over existing predictions. It should not be described as a trained predicate guard or end-to-end model result.

