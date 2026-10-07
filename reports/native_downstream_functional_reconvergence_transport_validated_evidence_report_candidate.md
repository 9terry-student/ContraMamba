# Native Downstream Functional Reconvergence — Validated Evidence Report

Status: **VALIDATED**

Run:
`gen5-native-reconvergence-raw-write-transport-3f73ec3-r1`

Execution commit:
`3f73ec30e194cc3670ed8294c798fb5f5ebe42bd`

Authentication policy:
`CURRENT_SEMANTIC_AUTH_WITH_HISTORICAL_PROJECTOR_CONTEXT_NON_GATING`

## 1. Question

This run tests how a source-local two-margin task-visible raw-write component and
its complementary residual are transported through the frozen native Mamba
downstream map.

The scientific object is fixed to:

- 18 ordered PRIMARY_A orientations;
- 18 ordered CONTROL_R orientations;
- 840 dev examples and 60,094 valid tokens;
- source-local raw-write projector fitted once at the source;
- no downstream projector refit, rotation, or basis search;
- stages:
  `raw_write`,
  `recurrent_state`,
  `c_readout_pre_gate`,
  `gated_scan`,
  `layer22_out_proj`;
- source-map FULL / VISIBLE / COMPLEMENT counterfactual transport;
- no training, no new checkpoint, no confirmatory 9601/9900 loading, and no
  VitaminC execution.

## 2. Execution and provenance status

The fresh corrected run completed successfully with:

- worker 0 rows `0:416`, 36 orientations;
- worker 1 rows `416:840`, 36 orientations;
- 486 fused transport calls total;
- 243 source-gradient forwards total;
- 36 merged ordered orientations;
- 180 pair-stage metric rows;
- `PASS_VALIDATED_TRANSPORT_ARTIFACT`;
- overall semantic authentication `PASS`;
- exit code `0`.

The imported handoff ZIP SHA256 was:

`4a9bc1a6a69f3f8e1a3eec917fd9ee00e16cb5ca8a2dce859c6fd5d05fcc2c60`

Final artifact SHA256 values:

- `transport_summary.json`:
  `6c5d4438c8cb0dfbabaf00e7893d8c41897da90d354b5826faec5ba82f3cc464`
- `pair_stage_transport_metrics.jsonl`:
  `fb7abbd92fb82033fc310d5f9fae98e42345e51c5159716e432e5a50d46045de`
- `shard_manifest.json`:
  `cebd15304cf2c47df5bbec5ba58fc6c9426c4a3ca4d8626bbe269d6096ee59c4`
- `run_provenance.json`:
  `40a40d22cd5b9b04b8b965af488fb9bf340084fd14a2278eb9a5162fd4af16a4`

## 3. Authentication

All invariant semantic checks used by the corrected authentication policy passed.

The historical raw-write projector scalar remained:

- observed: `0.00042701348720395292`;
- historical target: `0.000323695863574`;
- historical comparison: `FAIL`;
- `gate_required = false`;
- role: `HISTORICAL_OUTCOME_CONTEXT_ONLY`.

This scalar is retained as historical scientific context rather than treated as
an implementation-invariant fingerprint. The correction did not alter the
historical target or relax its tolerance.

Frozen two-margin functional authentication passed:

- `R_visible`
  - observed: `0.89495988377002311`
  - target: `0.89495800000199999`
  - absolute error: `1.8837680231253984e-06`
- `R_complement`
  - observed: `0.0044058372665501536`
  - target: `0.0044066019000500002`
  - absolute error: `7.6463349984665085e-07`
- `R_interaction`
  - observed: `0.0084390489993514464`
  - target: `0.0084395184338499993`
  - absolute error: `4.6943449855292585e-07`

Thus the prior raw-write functional decomposition is reproduced to substantially
tighter error than the frozen `0.002` functional tolerance.

## 4. Canonical symmetric-pair selective retention

Selective retention is the downstream retention of the complementary component
relative to the task-visible component. Raw-write is normalized to `1` by
construction.

| Stage | PRIMARY_A median | CONTROL_R median | PRIMARY / CONTROL |
|---|---:|---:|---:|
| raw_write | 1 | 1 | 1 |
| recurrent_state | 0.0123531320 | 0.0466631449 | 0.26473 |
| c_readout_pre_gate | 0.00170490341 | 0.00281944456 | 0.60469 |
| gated_scan | 0.000507319691 | 0.000547217580 | 0.92709 |
| layer22_out_proj | 0.000434972766 | 0.000481617842 | 0.90315 |

The largest PRIMARY-versus-CONTROL separation occurs immediately at
`recurrent_state`: the PRIMARY median is approximately `3.78x` lower than the
CONTROL median.

At `c_readout_pre_gate`, PRIMARY remains approximately `1.65x` lower.

By `gated_scan` and `layer22_out_proj`, the grouped medians are much closer.

## 5. Source-matched control comparison

For each of the nine source cells, the run compares the mean log selective
retention of its two PRIMARY_A targets against its two CONTROL_R targets.

Number of sources with lower PRIMARY than CONTROL selective retention:

- `raw_write`: `0/9` — equal by construction;
- `recurrent_state`: `9/9`;
- `c_readout_pre_gate`: `9/9`;
- `gated_scan`: `6/9`;
- `layer22_out_proj`: `6/9`.

At `recurrent_state`, the source-matched PRIMARY-minus-CONTROL log differences
range from:

`-1.9493957319` to `-1.0395849953`

with median:

`-1.3736048447`.

At `c_readout_pre_gate`, all nine remain negative, with median:

`-0.6239623072`.

At the final two stages, exactly the three `A6202-*` source cells become positive
while the six `A6201-*` and `A6203-*` sources remain negative.

This shows that the A-init-specific differential filtering is globally consistent
across the factorial grid at the recurrence and pre-gate readout stages, but
becomes source-dependent after gating.

## 6. Scientific interpretation

The validated evidence localizes the strongest A-init-specific downstream
functional reconvergence effect to the transition from `raw_write` into the
native Mamba recurrent state.

The raw-write decomposition starts with equal normalized retention
(`SELECTIVE_RETENTION = 1`). After the first recurrent transport, the
complementary residual is retained far less strongly relative to the task-visible
component, and this suppression is substantially stronger for PRIMARY_A
differences than for matched CONTROL_R differences.

This differential is not created by the final task head. It is already present
at `recurrent_state` and persists through `c_readout_pre_gate`.

The later `gated_scan` and `layer22_out_proj` stages do not strengthen the
factor-wide A-init differential. Instead, the PRIMARY-versus-CONTROL separation
narrows and becomes source-dependent, with the `A6202-*` sources reversing sign.

Therefore the supported bounded conclusion is:

> **Native functional reconvergence is driven primarily by selective transport
> at the raw-write-to-recurrent-state transition: task-visible raw-write
> components survive downstream far more strongly than their complementary
> residuals, and this relative filtering is specifically stronger for A-init
> differences than for matched RNG controls at the recurrence and pre-gate
> readout stages. The later gate and output projection mainly preserve or
> partially reconverge this separation rather than originating it.**

## 7. Relation to prior Gen5 evidence

This result is consistent with the prior precursor evidence that a very small
source-local task-visible raw-write component carries most of the endpoint
functional effect, while the much larger complementary representational
residual contributes little task effect.

It also narrows the previously unresolved
`native downstream functional reconvergence mechanism`:

- the mechanism is not adequately described as generic representational
  alignment;
- it does not require a downstream basis refit to become visible;
- the strongest factor-specific selection appears at the native recurrent-state
  transport step;
- later stages reduce the factor-wide differential and introduce
  source-dependent heterogeneity.

The evidence is descriptive with respect to the frozen transport statistic; no
new threshold-based mechanism class is introduced.

## 8. Limitations

This run supports localization of differential transport, not a claim that the
recurrent transition is the unique microscopic cause of reconvergence.

The source-local task-visible subspace is defined by the frozen two-margin
projector, so conclusions are conditional on that task-coordinate definition.

The historical projector-energy scalar is retained as non-gating context and
does not numerically reproduce its old target. That discrepancy does not affect
the validated finite functional decomposition or the current transport
semantics, but it should remain documented rather than silently discarded.

No new training, checkpoint optimization, hyperparameter tuning, confirmatory
seed population, or VitaminC evidence was introduced.

## 9. Result

**Validated result:** the previously unresolved native downstream functional
reconvergence mechanism is substantially localized to **early native recurrent
transport**, specifically the `raw_write -> recurrent_state` transition, with
continued but weaker factor-wide separation at `c_readout_pre_gate` and
source-dependent partial reconvergence after gating.

No further execution is required to establish this bounded result.
