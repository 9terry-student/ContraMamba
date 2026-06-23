# Stage18-A Temporal Penalty Notes

## Purpose

Stage18-A tests whether a finite soft temporal penalty can reproduce the Stage17 hard-override correction for temporal mismatch examples. It is a diagnostic post-processing experiment, not a trained model and not an architecture change.

## Comparator source

The sweep reuses the Stage17 conservative temporal comparator. Explicit weekday/month anchors dominate weaker eventive cue words, so predicate rows such as `launched ... during January` are not flagged as mismatches when both sides share the same explicit date.

## Penalty modes

- `pseudo_logit_penalty`: uses real logits if present; otherwise converts `final_probs` to pseudo-logits via `log(max(prob, 1e-8))`.
- `prob_penalty`: shifts probability mass from REFUTE/SUPPORT toward NOT_ENTITLED. This is less principled and should be used only as a fallback.

For `pseudo_logit_penalty`, when the temporal comparator flags a mismatch:

- add `+penalty` to NOT_ENTITLED;
- add `-penalty` to REFUTE;
- add `-penalty` to SUPPORT;
- re-softmax and take the adjusted argmax.

Penalty `0.0` should reproduce the original predictions exactly.

## Selection rule

The analyzer reports the smallest penalty that:

1. reduces `temporal_mismatch` false-entitled count by at least 90%;
2. changes zero `temporal_erased` predictions;
3. changes zero `surface_control` predictions;
4. changes zero `sufficiency_control` predictions.

If no penalty satisfies these constraints, the report states that no valid soft penalty was found.

## Caveat

Stage18-A should be framed as a soft temporal-comparator diagnostic. A positive result would suggest that explicit temporal comparison can be approximated by a finite decision bias; it would not be evidence of a trained temporal reasoning model.

