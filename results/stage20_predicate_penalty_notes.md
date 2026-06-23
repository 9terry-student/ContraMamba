# Stage20-B Soft Predicate Penalty Notes

## Purpose

Stage20-B tests whether a finite soft predicate penalty can reproduce most of the Stage20-A predicate-guard correction without using a hard override. It is a post-processing diagnostic, not a trained model.

## Detector reuse

The sweep reuses the Stage20-A predicate detectors:

- `oracle_probe_type`
- `lexical_predicate`

The lexical detector remains conservative: it requires high content overlap plus a known predicate-family conflict. This is meant to avoid punishing `surface_control` or generic lexical variation.

## Penalty rule

For flagged predicate mismatch rows, the pseudo-logit penalty applies:

```text
REFUTE       -= penalty
NOT_ENTITLED += penalty
SUPPORT      -= penalty
```

If true logits are unavailable, probabilities are converted to pseudo-logits with `log(max(prob, eps))`.

## Selection rule

The analyzer reports the smallest valid predicate penalty that:

1. reduces predicate_mismatch false-entitled mean by at least 90% relative to base;
2. keeps `surface_control` changed_count at zero across all seeds;
3. keeps `temporal_erased` changed_count at zero across all seeds;
4. keeps `sufficiency_control` changed_count at zero across all seeds;
5. keeps `temporal_mismatch` changed_count at zero across all seeds;
6. keeps `frame_location_mismatch` and `frame_role_mismatch` changed_count at zero across all seeds.

## Kaggle multi-seed result

Local validation only had seed1 and produced a degenerate smallest valid penalty of 0.0 (seed1's predicate_mismatch group had 0 base false-entitled cases after Stage19 temporal calibration, so the 90% reduction threshold was trivially met).

The real Kaggle run used seeds 1, 2, and 3 and reported:

- **Smallest valid predicate penalty: 1.25** (lexical_predicate detector, pseudo_logit_penalty mode)
- At penalty 1.25: predicate_mismatch false-entitled mean reduced by ≥ 90% relative to base
- At penalty 1.25: changed_count = 0 for all six control groups across all seeds

Do not re-derive or override this result from local seed1-only runs.

## Caveat

Stage20-B is not an architecture or training result. A positive result means the existing predictions are correctable by a finite predicate decision bias when a predicate mismatch signal is available.

