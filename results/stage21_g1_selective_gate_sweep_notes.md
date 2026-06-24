# Stage21-G1 Selective Preservation Gate Sweep Notes

## Purpose

Stage21-G0 showed that applying a global NOT_ENTITLED logit shift to all unflagged OOD
records reduced SUPPORT over-rejection on `surface_control` and `temporal_erased`, but
caused severe false-entitled regressions on `frame_location_mismatch`,
`frame_role_mismatch`, and `sufficiency_control`.

Stage21-G1 tests whether the shift can be made safe by restricting it to unflagged records
that look preservation-like according to model-internal auxiliary scores (`frame_prob`,
`sufficiency_prob`, `predicate_coverage_prob`). The shift is eval-only and post-hoc; it
does not affect training, losses, or checkpoint selection.

## Sweep Design

- **Gates**: `high_sufficiency`, `high_frame_sufficiency`, `high_frame_suff_predicate`
  (and optionally `high_frame`)
- **Thresholds**: applied as a lower bound on each auxiliary probability required by the gate
- **Shifts**: values subtracted from the NOT_ENTITLED final logit for gate-selected records
- **Unflagged definition**: `temporal_flag == 0` AND `predicate_flag == 0`
- All (gate x threshold x shift) triples were evaluated on a single best-dev checkpoint
  forward pass per seed; no re-training was performed.

## Pass/Fail Criteria (passes_g1_gate)

A triple is considered safe if all five conditions hold:

1. `keeps_temporal_guard`: `temporal_mismatch_fe_mean == 0.0`
2. `keeps_predicate_guard`: `predicate_mismatch_fe_mean == 0.0`
3. `safe_sufficiency`: `sufficiency_control_fe_mean <= 0.15`
4. `safe_frame_location`: `frame_location_mismatch_fe_mean <= 0.40`
5. `safe_frame_role`: `frame_role_mismatch_fe_mean <= 0.40`

## Main Findings

### Temporal and predicate guards were preserved

For all tested configurations, `temporal_mismatch` and `predicate_mismatch`
false-entitled rates remained near zero. Because the selective gate applies only to
unflagged records (both flags == 0), the comparator-driven guard on flagged records is
structurally unaffected.

### No configuration passed the full G1 safety criterion

Zero rows passed all five conditions.

The configurations that most effectively reduced `surface_control` and `temporal_erased`
false-not-entitled rates also caused large false-entitled regressions on
`frame_location_mismatch` and/or `frame_role_mismatch`, exceeding the 0.40 safety
threshold. The auxiliary scores (`sufficiency_prob`, `frame_prob`,
`predicate_coverage_prob`) do not provide a clean separating boundary between
preservation-like records and frame-mismatch records when used as a selective gate for
NE logit depression.

### G1 is rejected as a safe positive calibration method

Because no (gate, threshold, shift) triple satisfied the full safety criterion,
Stage21-G1 is rejected as a viable standalone calibration approach.

The fundamental problem is that `surface_control` and `temporal_erased` records (which
should be preserved as SUPPORT) share auxiliary score profiles with `frame_location_mismatch`
and `frame_role_mismatch` records (which should remain NOT_ENTITLED). Auxiliary-score-only
gating cannot separate these two populations at any tested threshold or gate combination.

## Paper Framing

Stage21-G1 exposes the **preservation-vs-frame-mismatch boundary** as the core challenge.
The auxiliary scores produced by the v6B model (frame sufficiency, predicate coverage,
sufficiency probability) are not sufficient to distinguish:

- True SUPPORT records whose temporal/predicate surface forms were erased or swapped
  (`surface_control`, `temporal_erased`) — which should be preserved as SUPPORT
- True NOT_ENTITLED records with wrong frame or role assignments
  (`frame_location_mismatch`, `frame_role_mismatch`) — which should remain NOT_ENTITLED

Auxiliary-score-only selective NE shifting therefore exposes this boundary rather than
solving it. Selective recalibration would require a richer internal signal, such as an
explicit training-time preservation loss or a learned gate trained to distinguish the two
populations.

## Top-10 Rows by Best Preservation

Sorted by `surface_control_fne_mean` then `temporal_erased_fne_mean` (lower is better).

| gate | threshold | shift | selected_rate_mean | surface_control_fne_mean | temporal_erased_fne_mean | sufficiency_control_fe_mean | frame_location_mismatch_fe_mean | frame_role_mismatch_fe_mean | temporal_mismatch_fe_mean | predicate_mismatch_fe_mean | passes_g1_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_sufficiency | 0.60 | 0.4 | 0.613725 | 0.283333 | 0.573333 | 0.050000 | 0.733333 | 0.766667 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.70 | 0.4 | 0.395098 | 0.336667 | 0.580000 | 0.050000 | 0.666667 | 0.683333 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.80 | 0.4 | 0.272549 | 0.346667 | 0.630000 | 0.050000 | 0.650000 | 0.650000 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.60 | 0.3 | 0.613725 | 0.350000 | 0.626667 | 0.050000 | 0.650000 | 0.683333 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.70 | 0.3 | 0.395098 | 0.356667 | 0.630000 | 0.050000 | 0.633333 | 0.683333 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.60 | 0.25 | 0.613725 | 0.360000 | 0.650000 | 0.050000 | 0.633333 | 0.650000 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.70 | 0.25 | 0.395098 | 0.360000 | 0.653333 | 0.050000 | 0.633333 | 0.650000 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.80 | 0.3 | 0.272549 | 0.363333 | 0.663333 | 0.050000 | 0.633333 | 0.650000 | 0.000000 | 0.000000 | False |
| high_sufficiency | 0.80 | 0.25 | 0.272549 | 0.363333 | 0.676667 | 0.050000 | 0.633333 | 0.650000 | 0.000000 | 0.000000 | False |
| high_frame | 0.60 | 0.4 | 0.503922 | 0.366667 | 0.566667 | 0.176667 | 0.650000 | 0.650000 | 0.000000 | 0.000000 | False |

## Top-10 Rows with passes_g1_gate == True

_No rows._

## Conclusion

Stage21-G1 confirms that post-hoc selective NE logit shifting using model-internal
auxiliary scores does not constitute a safe improvement over Stage21-E3 or Stage21-G0.
The temporal/predicate comparator guard is preserved by construction, but the
frame-mismatch boundary is violated whenever the NE logit is depressed sufficiently to
reduce false-not-entitled rate on preservation controls.

Recommended next steps:

- Training-time preservation signal (e.g., SUPPORT reconstruction loss on erased/surface
  control records added to the training mix)
- A learned boundary gate trained discriminatively to separate surface/temporal-erased
  from frame-mismatch probe types
- Contrastive calibration that conditions explicitly on the expected intervention type
  at inference time
