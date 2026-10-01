# Gen5 Optimization-Path Bypass Stage A Static Geometry Interpretation

## Status

PASS_STATIC_GEOMETRY_AUDIT_V2

Source repository HEAD:

`ae61d962443cb374086420c62b33dd86441b4d8f`

Source Phase3A execution commit:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

Execution mode:

`READ_ONLY_CPU_NO_FORWARD_NO_BACKWARD_NO_TRAINING`

## Geometry objects

The audit keeps distinct:

1. output write geometry `col(B)`,
2. input read geometry `row(A)`,
3. full correction operator `M = BA`,
4. existing authenticated operator-energy fractions `F_R` and `F_C`.

All nine A, B, and BA objects retained effective rank 2.

## Primary result

Within the same seed across P0/PR/PC:

- mean A affinity: `0.9999642374840724`
- mean B affinity: `0.999281353911041`
- mean BA signed Frobenius cosine: `0.9997925663338935`

Within the same pressure across seeds:

- mean A affinity: `0.1442479041531371`
- mean B affinity: `0.47797477201701416`
- mean BA signed Frobenius cosine: `0.4510457788826586`

The pressure manipulation therefore changed the learned correction geometry far less than changing the training seed.

This holds for the input read subspace, output write subspace, and complete learned operator.

## Interpretation

Stage A supports the following bounded conclusions:

- P0/PR/PC pressure did not materially redirect the learned correction geometry within a fixed seed.
- Learned solution geometry is substantially seed-dependent.
- Cross-seed B planes are not simply identical, yet retain structured non-random overlap.
- The current evidence is consistent with a structured family of alternative optimization solutions.

Stage A does not establish:

- downstream functional substitutability,
- step-0 gradient misalignment,
- loss shortcut behavior,
- downstream Jacobian degeneracy,
- a non-substitutable R22 optimization bottleneck.

## Null calibration

The descriptive rank-2 random-subspace affinity expectations are:

- B in 24576 dimensions: `8.138020833333333e-05`
- A in 768 dimensions: `0.0026041666666666665`

These are descriptive references only and are not scientific p-values.

## Next stage

The next scientific analysis is the existing-checkpoint forward-only functional decomposition:

`FULL / R22_ONLY / R22_REMOVED / ZERO`

A is held fixed and only the B output-write component is decomposed.

The first diagnostic population is the frozen Phase3A dev domain under each checkpoint's matched pressure.

The unused 9601..9900 confirmatory population remains untouched.
