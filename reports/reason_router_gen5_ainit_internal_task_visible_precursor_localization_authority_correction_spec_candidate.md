# Gen5 A-init Internal Task-Visible Precursor Localization Authority Correction

SOURCE_INTERNAL_PRECURSOR_AUTHORITY_COMMIT=e7eba19b102016131e4990f724c825cbec49ec5c
SOURCE_FORWARD_JACOBIAN_RECOVERY_EVIDENCE_FREEZE_COMMIT=a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e
SOURCE_CONFIRMATORY_EVIDENCE_FREEZE_COMMIT=1468938af9753fa9f4a511d4e7f740dea0110bba

STATUS=READY_FOR_INTERNAL_PRECURSOR_AUTHORITY_CORRECTION_FREEZE

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Defect

The committed internal precursor authority defines exactly eight new
stage-specific signed-permutation controls whose identities are derived only
from:

`SHA256("GEN5_INTERNAL_PRECURSOR_V1|<stage>|<control_index>")`

The same authority also requires the `layer22_out_proj` stage to authenticate
against historical actual/control ratios from the frozen forward-Jacobian
recovery:

- directional-gain ratio: approximately `9.8220`
- row-space-energy ratio: approximately `57.1938`

Those historical ratios were produced by a different frozen control family.
The source nullness audit records exactly:

`control_seeds = [310001,310002,310003,310004,310005,310006,310007,310008]`

Therefore the historical actual/control ratios are not invariant identity
quantities and are not required to be reproduced by a newly defined control
family.

Requiring both the new SHA-derived controls and the old control-family ratios
would make the execution contract internally inconsistent.

This defect was identified before any internal-precursor scientific execution
or metrics were produced.

## Correction

This correction supersedes only the conflicting `layer22_out_proj`
control-ratio authentication clause of the source authority.

The new SHA-derived eight-control family remains unchanged and remains the only
orientation-control family authorized for the internal precursor audit.

At `layer22_out_proj`, hard recovery authentication MUST use only quantities
that are independent of signed-permutation-control identity.

Required recovery targets are:

- same-training-RNG / different-A grouped true-forward directional gain:
  `0.0003051091215898591`
- same-training-RNG / different-A grouped local task-row-space squared-energy
  fraction:
  `0.004158599422848653`

The execution must compare the newly observed values against these frozen
targets under predeclared numerical tolerances recorded before scientific
interpretation.

The historical ratios:

- `9.822032145892573`
- `57.19382760440825`

may be recorded as provenance/context for the old
`310001..310008` control family only.

They MUST NOT be used as pass/fail criteria for the new SHA-derived control
family and MUST NOT be used to tune the new controls.

## Unchanged scientific controls and criteria

All other clauses of the source authority remain unchanged, including:

- frozen Phase3A P0 dev rows only;
- exact 3x3 checkpoint grid;
- five ordered internal stages;
- true-forward `joint` analysis-gradient semantics;
- exact local two-row task projector;
- finite visible/complement interventions;
- exactly eight SHA-derived signed-permutation controls per stage;
- primary pair class and natural pair controls;
- precursor criterion:
  - `0.60 <= R_visible <= 1.40`
  - `R_complement <= 0.05`
  - `R_interaction <= 0.05`
  - `E_task_actual / E_task_control_mean >= 5`
- deterministic two-GPU pair-orientation sharding;
- no training, optimizer, parameter gradient, checkpoint mutation, or
  confirmatory-population access.

No stage definition, threshold, pair subset, k, population, or scientific
outcome rule is changed by this correction.

## Execution stop condition amendment

Replace the source stop condition:

`layer22_out_proj fails recovered-Jacobian authentication`

with the precise condition:

> Stop if the `layer22_out_proj` grouped actual true-forward directional gain
> or grouped actual local task-row-space squared-energy fraction fails to match
> the frozen recovery targets above within the predeclared implementation
> tolerances.

Do not stop merely because the newly defined SHA-derived controls yield
different actual/control ratios from the historical `310001..310008` control
family.

## Boundary

This is an execution-contract consistency correction made before scientific
execution.

It does not authorize any new data, training, model change, threshold change,
post-hoc rescue, or result-dependent control selection.
