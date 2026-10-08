# Native Recurrent Transport Kernel Decomposition — Validated Evidence Report

Status: **VALIDATED**

Run:
`gen5-recurrent-kernel-decomposition-09577cf-r1`

Execution commit:
`09577cf4b3a4f84d1988647e153eda42c65a8664`

## 1. Question

The prior validated native downstream transport result localized the strongest
A-init-specific differential filtering to the transition from `raw_write` into
`recurrent_state`.

This run asks a narrower recurrence-only question:

> Within the frozen `raw_write -> recurrent_state` transition, how much of the
> visible-versus-complement selective-retention difference is attributable to
> self-propagation through the recurrent kernel, and how much is attributable
> to cross-token interference?

The scientific object is fixed to:

- the same frozen Phase3A P0 dev population;
- 840 dev examples and 60,094 valid tokens;
- 18 ordered PRIMARY_A orientations;
- 18 ordered CONTROL_R orientations;
- the frozen source-local two-margin raw-write projector;
- no downstream projector refit;
- no `c_readout_pre_gate`, `gated_scan`, `layer22_out_proj`, or final-head
  transport in this audit;
- no training, optimizer, checkpoint mutation, confirmatory 9601/9900 loading,
  or VitaminC execution.

## 2. Execution and provenance status

The corrected run completed and imported successfully.

Execution:

- worker 0 rows: `0:416`;
- worker 1 rows: `416:840`;
- merged ordered orientations: `36`;
- symmetric unordered pairs: `18`;
- validated transport replay: `PASS`;
- run exit code: `0`.

Imported handoff ZIP SHA256:

`92d45a11e420082b8a925be69d02ce2b0c3a9607d6a9f43ce1b64d14c0fc6f05`

Validated imported artifact hashes:

- `recurrent_kernel_decomposition_summary.json`:
  `4072aec0f92a66ccb1435c4deb600bd60e58fbc078d646c2fd91a9cea295ba96`
- `orientation_kernel_decomposition_metrics.jsonl`:
  `939a7411c9c30ad8172a01bf1a5d568e637e45866f45865721dc0c600a5107d5`
- `shard_manifest.json`:
  `b1efe53d34c401d2a4ab6de0560973f91b77f16eab4dcee5376effef3246e78c`

Static imported-artifact validation passed:

- rows: `36`;
- dev rows: `840`;
- valid tokens: `60094`;
- validated transport replay max energy relative error:
  `2.5587146447890154e-06`;
- validated transport replay max log absolute error:
  `2.5959654101903595e-06`;
- maximum decomposition log-identity absolute error:
  `8.8817841970012523e-16`;
- maximum raw reconstruction absolute error:
  `7.4505805969238281e-09`.

The numerical implementation encountered active-prefix dynamic ranges far beyond
a naively centered float64 exponential representation. The corrected
overflow-safe GPU FFT implementation was exercised in the scientific run; the
maximum observed FFT half-log span was:

`6252619.5183000565`.

The run nevertheless replayed the frozen transport evidence and exact
decomposition identity within the stated tolerances.

## 3. Exact decomposition

For each ordered orientation, define:

- `R`: raw-write component energy;
- `S`: recurrent self-propagated energy with cross-token cross terms removed;
- `E`: full recurrent-state energy;
- `G = S / R`;
- `Q = E / S`.

The decomposition is:

`L_kernel = log(G_complement / G_visible)`

`L_interference = log(Q_complement / Q_visible)`

and exactly:

`LOG_SELECTIVE_RETENTION = L_kernel + L_interference`.

Here:

`SELECTIVE_RETENTION = (E_complement / R_complement) / (E_visible / R_visible)`.

Therefore a more negative log selective-retention value means that the
complement is retained less strongly relative to the task-visible component;
equivalently, recurrent transport is more selectively favorable to the visible
component.

`L_kernel < 0` means the complement has a lower self-propagation gain than the
visible component.

`L_interference < 0` means the complement-to-visible ratio of the full
interference multiplier `Q = E/S` is below one. This is a differential
cross-token-interference statement; it does not by itself classify the absolute
interference in either component as constructive or destructive.

## 4. Grouped symmetric-pair result

Across the nine symmetric unordered pairs per group:

| Metric | PRIMARY_A median | CONTROL_R median |
|---|---:|---:|
| `L_kernel` | -2.66474358553 | -2.31308081907 |
| `L_interference` | -1.74646194109 | -0.751720203131 |
| `LOG_SELECTIVE_RETENTION` | -4.39384565937 | -3.06480102220 |
| `LOG_J1_RATIO` | -2.17696487176 | -1.53130919605 |
| `SELECTIVE_RETENTION` | 0.01235313182 | 0.04666312587 |

Thus the canonical symmetric-pair recurrent selective-retention ratio is
approximately `3.78x` lower in PRIMARY_A than in CONTROL_R
(`0.0123531` versus `0.0466631`).

The kernel term is more negative in PRIMARY_A, but the larger grouped
PRIMARY-versus-CONTROL separation appears in the interference term.

## 5. Source-matched PRIMARY minus CONTROL

For every one of the nine source cells:

- `PRIMARY_A - CONTROL_R` is negative for `L_kernel`;
- `PRIMARY_A - CONTROL_R` is negative for `L_interference`;
- `PRIMARY_A - CONTROL_R` is negative for total log selective retention.

Sign counts:

| Metric | Negative | Positive | Zero |
|---|---:|---:|---:|
| `L_kernel` | 9 | 0 | 0 |
| `L_interference` | 9 | 0 | 0 |
| `LOG_SELECTIVE_RETENTION` | 9 | 0 | 0 |

Source-matched median differences:

- `ΔL_kernel = -0.344368505841`;
- `ΔL_interference = -1.03744402952`;
- `ΔLOG_SELECTIVE_RETENTION = -1.37360369906`.

The median absolute interference differential is about `3.01x` the median
absolute kernel differential.

More directly, source by source, the interference term contributes approximately
`69.2%` to `75.8%` of the total absolute log differential, with a median of
approximately `72.9%`.

This is descriptive decomposition, not a threshold-based mechanism
classification.

## 6. Orientation-level consistency

All 18 PRIMARY_A ordered orientations have:

- `L_kernel < 0`;
- `L_interference < 0`;
- `LOG_SELECTIVE_RETENTION < 0`.

All 18 CONTROL_R ordered orientations also have the same three negative signs.

Therefore recurrent transport is visible-selective in both factorial groups.
The scientific distinction is not the existence of visible-selective filtering,
but its substantially stronger magnitude for PRIMARY_A.

The PRIMARY-specific strengthening is globally consistent across the nine
source-matched cells for both decomposition terms, with the interference
difference consistently larger than the kernel difference.

## 7. Scientific interpretation

The validated recurrence-only evidence resolves the immediate mechanism question
as follows.

The `raw_write -> recurrent_state` transition does not obtain its
visible-selective filtering from a single effect.

First, the native recurrent kernel itself preferentially preserves the
task-visible component relative to its complement. This contribution is present
for both PRIMARY_A and CONTROL_R and is systematically stronger for PRIMARY_A.

Second, the full recurrent-state energy contains an additional
cross-token-interference contribution. The complement-to-visible differential in
this interference term is also systematically more negative for PRIMARY_A than
for matched CONTROL_R.

Crucially, the A-init-specific increment in selective filtering is numerically
dominated by this second term. Across all nine source-matched cells, the
interference term accounts for roughly seven-tenths to three-quarters of the
absolute log PRIMARY-minus-CONTROL separation.

Therefore the supported bounded conclusion is:

> **The previously localized A-init-specific selective filtering at
> `raw_write -> recurrent_state` is jointly produced by recurrent
> self-propagation and cross-token interference, but the larger share of the
> PRIMARY_A-versus-CONTROL_R differential is carried by the cross-token
> interference term. The recurrent kernel contributes consistently and
> nontrivially, but kernel gain alone does not explain the magnitude of the
> factor-specific recurrent-state separation.**

This narrows the native-Mamba mechanism from “early recurrent transport” to a
more specific statement: the dominant factor-specific differential is created
inside recurrence through how tokenwise propagated writes combine, rather than
through recurrent self-propagation gain alone.

## 8. Relation to prior validated transport evidence

The prior validated downstream transport run showed:

- raw-write selective retention is normalized to equality;
- the strongest PRIMARY-versus-CONTROL separation appears immediately at
  `recurrent_state`;
- the separation remains at `c_readout_pre_gate`;
- later gating/output stages partially reconverge and become source-dependent.

The present run explains the first of those transitions without executing the
later stages.

The recurrence-only decomposition shows that:

- native kernel propagation already favors visible over complement;
- A-init strengthens that kernel differential;
- an even larger A-init-specific separation appears in the interference term;
- the exact sum of these two terms reproduces the previously validated
  recurrent-state selective-retention statistic.

Thus the earlier localization result and the present decomposition are mutually
consistent: later readout/gating is not required to generate the primary
factor-specific separation.

## 9. Boundaries and limitations

This run does **not** establish that a single microscopic pairwise interaction,
token lag, state coordinate, or sign of interference is the unique causal
mechanism.

In particular:

- `L_interference` is an exact aggregate differential term derived from
  `Q = E/S`; it does not alone identify which token-token cross terms dominate;
- no new mechanism-classification threshold was defined;
- no scientific p-values were introduced;
- no confirmatory 9601/9900 population was loaded;
- no VitaminC evidence was used;
- no downstream transport stages were executed in this audit;
- conclusions remain conditional on the frozen source-local task-visible
  projector and frozen Phase3A P0 dev population.

The next mechanistic question, if pursued, is therefore not another kernel-gain
sweep. It is whether the dominant `L_interference` differential can be
localized to a bounded temporal/token-interaction structure using the frozen
evidence and a separately authorized analysis.

## 10. Result

**Validated result:** the recurrent-state A-init-specific selective-retention
effect is not explained by recurrent self-propagation gain alone.

Both `L_kernel` and `L_interference` contribute with the same factor-specific
direction in all nine source-matched cells, but the differential
cross-token-interference term is consistently larger and accounts for roughly
`69%–76%` of the absolute source-matched log separation.

No additional GPU execution is required to establish this bounded result.
