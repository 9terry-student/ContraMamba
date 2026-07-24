# Stage196-B2-B6P9-P3-R1 Runtime Metric Closure Repair Spec

## Authority

Stage196-B2-B6P9-P3-R1 repairs runtime metric emission for the teacher-state
observer sidecars created by the P9-P3 separate observational runs. It does not
change teacher lifecycle timing, student training, loss computation, optimizer
behavior, checkpoint state semantics, run manifests, orchestration, or analysis
logic.

## Runtime Defects

Two previous-step runs completed successfully and satisfied lifecycle closure,
but emitted invalid target-total metrics. Direction rows reported all evaluated
teacher targets as exact ties while `direction_teacher_total_targets` was zero.
Candidate-order rows reported all evaluated teacher pairs as exact ties while
`order_teacher_total_pairs` was zero. Both violate the v1 closure contract:
`total = exact_ties + positive + negative`.

The epoch CSV also recorded epochs as `2, 4, 6, ..., 40` while the batch JSONL
recorded `1, 2, 3, ..., 20`. This identifies epoch identity as having been
numerically accumulated rather than preserved as an identity field.

## Structural All-Tie Interpretation

The previous-step teacher candidate may be structurally degenerate in the frozen
training configuration. With one successful optimizer step per epoch, the
observer reads before the step and updates the teacher to the current student
after the successful step. At the next epoch observation, the student has not
changed again, so the previous-step teacher can equal the student. All-tie
direction and candidate-order targets are therefore valid observational findings,
not lifecycle failures.

The repair must not delay teacher updates, move observation after optimizer
steps, skip successful updates, introduce a two-step lag, add epsilon tie
breaking, alter exact-tie semantics, perturb parameters, or change student
quantities.

## Total Semantics

For direction metrics:

`direction_teacher_total_targets = direction_teacher_exact_tie_targets + direction_teacher_positive_sign_targets + direction_teacher_negative_sign_targets`

For candidate-order metrics:

`order_teacher_total_pairs = order_teacher_exact_tie_pairs + order_teacher_positive_pair_targets + order_teacher_negative_pair_targets`

`total` means every evaluated target or pair, including exact ties. Active count
remains `positive + negative` and must not be overloaded into the total fields.
Each batch row is validated before being accumulated or written; a closure
violation fails explicitly.

## Agreement Rates

Agreement and disagreement counts are computed only over comparable active
teacher/student targets or pairs, excluding exact ties. When the comparable
active denominator is zero, agreement and disagreement counts are zero and both
agreement and flip rates are `null`. The observer must not fabricate zero rates
for zero denominators.

## Epoch Aggregation

Epoch CSV aggregation classifies fields explicitly:

- Identity fields such as `epoch`, `mode`, `target_family`, `run_name`, and
  `schema_version` are preserved and must not be summed.
- Additive interval counts such as target/pair totals, exact ties,
  positive/negative counts, and agreement/disagreement counts are summed across
  rows in the epoch.
- Cumulative lifecycle counters such as teacher read/update counts, teacher and
  student forward counts, successful optimizer steps, and skipped optimizer
  steps use the last or maximum observed cumulative value for the epoch.
- Current-state measurements such as parameter distances, exact match rate,
  state bytes, parameter count, and buffer count use last-observation semantics.
- Rates are recomputed from aggregated numerators and active comparable
  denominators; per-batch rates are not averaged. Zero denominators emit `null`.
- `rows` equals the number of batch-metric records aggregated into the epoch.

The accumulator must not initialize from the first row and then add that same row
again. In particular, the epoch identity emitted to CSV remains the actual
1-based epoch sequence `1, 2, 3, ..., E`.

## Lifecycle Invariance

The repair preserves previous-step observation before optimizer step, update only
after successful optimizer step, AMP-skipped step non-advancement, previous-epoch
start-of-current-epoch boundary semantics, EMA update after successful optimizer
step, teacher eval mode, teacher no-grad execution, teacher exclusion from the
optimizer, RNG preservation, student mode restoration, checkpoint namespace, and
five-sidecar output closure.

## Rerun Requirements

The affected P9-P3 previous-step direction and candidate-order runs must be
rerun after this repair. The expected structural finding may remain all ties, but
it must be emitted correctly as `total > 0`, `ties = total`, `positive = 0`,
`negative = 0`, active count zero, agreement/disagreement counts zero, and null
agreement/flip rates.
