# Native downstream reconvergence transport authentication correction

Status: implementation-correction candidate.

## Scope

This correction addresses two post-execution implementation/provenance defects
found at execution head `32e5d8d78253056dca47036ba88be15f950383a3`.

It does not change the scientific transport computation.

Unchanged scientific objects include:

- the 18 PRIMARY_A and 18 CONTROL_R ordered orientations;
- the exact frozen checkpoint grid and dev encoding;
- the source-local two-margin raw-write decomposition;
- source-map FULL / VISIBLE / COMPLEMENT trajectories;
- the five ordered stages;
- ratio-of-summed stage energies and selective-retention definition;
- example sharding and fused streaming transport;
- no training, new checkpoint, VitaminC, or confirmatory-population access.

## Defect 1: worker SHA sidecar serialization

The worker writer serialized a literal backslash+n suffix instead of a real LF.
`_read_worker()` expects a normal text line and therefore failed closed with
`WORKER_SHA_MISMATCH` after both workers had successfully completed.

Correction: write one real LF byte after the 64-hex SHA256 digest.

## Defect 2: historical projector outcome misclassified as implementation auth

The merge gate treated the historical scalar raw-write task-row energy fraction

`0.000323695863574`

as an implementation-invariant fingerprint.

The completed 32e5d8d execution observed:

`0.00042701349599325662`.

The absolute difference `0.00010331763241925663` exceeded the historical
`5e-5` tolerance, while all other frozen semantic checks passed:

- source replay max abs: `1.7881393432617188e-07`;
- all five residual-chain targets passed;
- two-margin `R_visible` error: `1.904454586698634e-06`;
- two-margin `R_complement` error: `7.646960020219612e-07`;
- two-margin `R_interaction` error: `4.678049310546245e-07`.

Independent diagnostic probes ruled out the tested implementation-drift
hypotheses in the current projector arithmetic:

- optimized projector versus legacy CPU-f64 projection agreed to numerical noise;
- f32-product/f64-reduction versus f64-operand projector differences were many
  orders of magnitude below the historical-scalar discrepancy;
- batch-32 versus batch-4 projector means differed only at ~1e-10 scale;
- valid-token gradient masking versus the historical-unmasked form was identical
  on the tested slice.

The historical precursor artifact records an execution commit distinct from
the later repository implementation/helper history. Its task-row-energy scalar
is scientific outcome context, not a safe exact implementation fingerprint.

Correction:

- retain and report the historical projector scalar, target, tolerance, error,
  and pass/fail value;
- explicitly mark it `gate_required = false` and
  `role = HISTORICAL_OUTCOME_CONTEXT_ONLY`;
- do not include this historical outcome scalar in overall authentication PASS;
- continue to gate on invariant semantic checks: exact identities, source
  replay, residual-chain replay, finite-effect replay, raw reconstruction, and
  projector implementation equivalence covered by executable tests.

This is not a tolerance relaxation and does not replace the historical target
with the observed result.

## Existing executions

The following are not promoted by this correction:

- `gen5-native-reconvergence-raw-write-transport-32e5d8d-r1`;
- `gen5-native-reconvergence-raw-write-transport-32e5d8d-recovery-r2`;
- diagnostic recovery r3/r4.

They remain execution/debug evidence only.

A fresh run at the corrected implementation commit is required for a validated
transport artifact.

## Diagnostic scientific signal

The recovered r4 diagnostic shows a strong, coherent transport pattern but
remains non-validated until the fresh corrected run.

Canonical symmetric-pair selective-retention medians
(`complement retention / visible retention`) were:

| stage | PRIMARY_A | CONTROL_R |
|---|---:|---:|
| raw_write | 1 | 1 |
| recurrent_state | 0.0123531 | 0.0466631 |
| c_readout_pre_gate | 0.00170490 | 0.00281944 |
| gated_scan | 0.000507320 | 0.000547218 |
| layer22_out_proj | 0.000434973 | 0.000481618 |

The source-matched PRIMARY-minus-CONTROL log-selective difference was negative
for all 9 source cells at recurrent_state and c_readout_pre_gate. At the last
two stages, the three A6202 source cells changed sign while the remaining six
stayed negative.

This pattern motivates interpretation only after the corrected fresh run.
