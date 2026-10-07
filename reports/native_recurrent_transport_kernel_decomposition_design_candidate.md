# Native Recurrent Transport Kernel Decomposition — Scientific Design Candidate

Status: `SCIENTIFIC_DESIGN_READY_FOR_IMPLEMENTATION`

Base validated evidence commit:

`6d36e49931389c21e9299b84dfd880151f71b8d3`

Source validated run:

`gen5-native-reconvergence-raw-write-transport-3f73ec3-r1`

## 1. Scientific target

The previous validated transport audit localized the strongest A-init-specific
functional reconvergence to:

`raw_write -> recurrent_state`

The next question is therefore narrower:

> Why does the native recurrence retain the fixed raw-write task-visible
> component much more strongly than its fixed complementary residual?

No new training, checkpoint fitting, basis search, downstream projector refit,
or threshold tuning is allowed.

## 2. Exact recurrence identity

For a fixed source checkpoint and example, the intervention difference obeys

`delta_s_t = A_t * delta_s_(t-1) + delta_w_t`

with elementwise

`A_t = exp(a_continuous * discrete_time_step_t)`.

The recurrence coefficients are fixed by the source context and do not depend
on whether the injected raw-write perturbation is FULL, VISIBLE, or COMPLEMENT.

For component `c` in `{VISIBLE, COMPLEMENT}`, expand exactly:

`delta_s_t^c = sum_{tau <= t} K_(t,tau) * delta_w_tau^c`

where

`K_(t,tau) = product_{u=tau+1..t} A_u`

and `K_(tau,tau) = 1`.

This makes the recurrence mechanism analyzable without fitting any new model.

## 3. Exact energy decomposition

For each example and ordered orientation, define raw component energy

`R_c = sum_tau ||delta_w_tau^c||^2`.

Define propagated self-energy

`S_c = sum_t sum_{tau <= t} ||K_(t,tau) * delta_w_tau^c||^2`.

Define exact recurrent energy

`E_c = sum_t ||sum_{tau <= t} K_(t,tau) * delta_w_tau^c||^2`.

Define cross-token interference

`C_c = E_c - S_c`.

Then exactly:

`E_c = S_c + C_c`.

Define self-propagation gain

`G_c = S_c / R_c`.

Define interference factor

`Q_c = E_c / S_c`.

Therefore the already-validated recurrent retention is

`RET_c(recurrent_state) = G_c * Q_c`

and the validated selective retention decomposes exactly as

`SELECTIVE_RETENTION = (G_complement / G_visible) * (Q_complement / Q_visible)`.

In log space:

`log SELECTIVE_RETENTION = L_kernel + L_interference`

where

`L_kernel = log(G_complement / G_visible)`

and

`L_interference = log(Q_complement / Q_visible)`.

This is the primary scientific decomposition.

## 4. Mechanism interpretations

### KERNEL_WEIGHTING_DOMINANT

`L_kernel` accounts for most of the negative selective-retention log effect,
while `L_interference` is comparatively small.

Interpretation:

The fixed native recurrence kernel preferentially preserves the task-visible
raw-write component because that component is aligned with longer-lived
state/time coordinates.

### TEMPORAL_INTERFERENCE_DOMINANT

`L_interference` accounts for most of the negative selective-retention log
effect.

Interpretation:

The complement is not merely placed in faster-decaying coordinates; its
propagated token contributions cancel one another more strongly across time.

### MIXED_KERNEL_AND_INTERFERENCE

Both terms are materially negative.

Interpretation:

Selective reconvergence arises jointly from recurrence-kernel weighting and
cross-token cancellation.

No threshold-based categorical promotion is required. The primary output is
the continuous exact decomposition.

## 5. Lag-resolved diagnostic

To localize kernel weighting without introducing a fitted model, additionally
report

`S_c(lag) = sum_{t-tau=lag} ||K_(t,tau) * delta_w_tau^c||^2`.

For each component report:

- lag-0 fraction;
- lag-1 fraction;
- lag-2..4 fraction;
- lag-5..8 fraction;
- lag-9+ fraction;
- energy-weighted mean lag;
- cumulative lag at 50% and 90% of propagated self-energy.

These are descriptive diagnostics only.

## 6. Immediate one-step survival diagnostic

Define

`J1_c = sum_tau ||A_(tau+1) * delta_w_tau^c||^2 / sum_tau ||delta_w_tau^c||^2`

over positions with a valid next token.

This isolates immediate native decay weighting before long-horizon accumulation.

Report PRIMARY_A and CONTROL_R distributions of:

`J1_complement / J1_visible`.

This is not a replacement for the exact `L_kernel` decomposition.

## 7. Population and controls

Use exactly the same frozen population and orientation contract as the validated
transport run:

- 840 dev examples;
- 60,094 valid tokens;
- 18 ordered PRIMARY_A orientations;
- 18 ordered CONTROL_R orientations;
- both directions for every unordered pair;
- same source-local two-margin raw-write projector;
- same checkpoint grid;
- same dev order and encoding;
- same valid-token mask.

Source-matched control comparison remains mandatory.

## 8. Aggregation

Primary quantities are computed per example before aggregation.

For each ordered orientation report sums sufficient to reconstruct:

- `R_visible`, `R_complement`;
- `S_visible`, `S_complement`;
- `E_visible`, `E_complement`;
- `C_visible`, `C_complement`;
- lag-resolved self-energy;
- one-step survival numerator/denominator.

For orientation-level ratios, use ratio-of-summed energies.

For unordered-pair summaries, preserve the existing symmetric log aggregation.

For PRIMARY_A versus CONTROL_R comparison, report:

- median and IQR over nine unordered pairs;
- source-matched PRIMARY-minus-CONTROL log differences for all nine source
  cells.

## 9. Mandatory authentication

The new audit must authenticate against the frozen validated transport result.

Required replay checks:

1. same 36 ordered orientations;
2. same valid-token count;
3. raw component reconstruction:
   `VISIBLE + COMPLEMENT = FULL`;
4. recurrent `E_visible` and `E_complement` reproduce the validated
   `recurrent_state` stage energies within a frozen numerical tolerance;
5. reconstructed selective retention reproduces the validated pair-stage
   metrics;
6. exact checkpoint/dev/runtime/kernel identities;
7. no training, optimizer step, checkpoint mutation, confirmatory population,
   or VitaminC access.

The historical projector outcome scalar remains context-only under the frozen
corrected policy and must not be reintroduced as a gate.

## 10. Execution-efficiency boundary

The new execution should stop at `recurrent_state`.

It does not need to run:

- C readout;
- gate;
- out projection;
- layer 23;
- final head logits.

The only model-gradient computation needed is the already-frozen source-local
two-margin projector construction.

The recurrence decomposition itself is deterministic arithmetic on:

- fixed raw-write components;
- fixed source `A_t` coefficients.

Use two independent single-GPU workers with the existing disjoint example
shards unless a strictly more efficient implementation is proven equivalent
before execution.

## 11. Falsification / decision logic

The current hypothesis that early recurrence is the operative reconvergence
mechanism is strengthened only if the exact decomposition explains the
validated selective-retention collapse without relying on downstream stages.

Possible outcomes:

- large negative `L_kernel`, small `L_interference`:
  recurrence-kernel alignment / decay weighting;
- small `L_kernel`, large negative `L_interference`:
  temporal cancellation;
- both materially negative:
  mixed recurrence mechanism;
- neither explains the validated effect:
  implementation or metric inconsistency; stop and audit.

## 12. Claim boundary

This audit can explain how the fixed recurrence maps raw-write visible and
complement components differently.

It does not establish:

- semantic meaning of individual native state dimensions;
- universal behavior outside the frozen population;
- a unique microscopic causal variable;
- a new architecture;
- a training intervention.

## 13. Next action

Implement one bounded recurrence-only audit with unit tests for the exact
energy identity and validated-transport replay authentication.

Training/evaluation allowed: `NO`.

Kaggle scientific execution should occur only after local static validation and
a frozen implementation commit.
