# ContraMamba Gen4 — Forced-Decisive Stage A Static Analysis

## Status

This report applies exactly the preregistered confirmatory Stage A family to the frozen forced-decisive raw artifact.

Result: `FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

Scope: forced-decisive two-class finite grammar only.

## Frozen input

- Raw evidence freeze commit: `880eab834c442054642773935a88fa60a31287c3`
- Rows SHA256: `5cc15d2626b350108c2c532ab8502d87aacb0d5253f82905ab83dc438aff5fcd`
- Summary SHA256: `0dc283213dda376b9684ee34e093de3ad6e09a425ea4fa4884390fa8b14555c5`
- Partition SHA256: `871fb5c1e2c62f247c284ceae409ef9ec19acc75f63ffc832829c2664db46311`
- Forced grammar SHA256: `62c9c53871f68fcc0f57d38c96b75d2391ee6ddc473a4325a522f91c9249bd00`
- Confirmatory unsupported: 169
- Confirmatory supported: 62

## Preregistered primary family

- Signal: `p3_component_l2`
- Comparison orientation: unsupported minus supported
- Offsets: `t*-4`, `t*-3`, `t*-2`, `t*-1`
- Test: two-sided Welch t-test
- Multiplicity: Holm across exactly 4 p-values
- Familywise alpha: 0.05

## Results

| Offset | N unsupported | N supported | Mean unsupported | Mean supported | Difference U-S | Welch t | df | Raw p | Holm p | Significant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| -4 | 169 | 62 | 2.02937105314 | 2.18400868575 | -0.154637632608 | -0.779785063974 | 92.1042239209 | 0.437516626842 | 1 | NO |
| -3 | 169 | 62 | 1.92714511304 | 2.17469169841 | -0.247546585379 | -1.52590299003 | 95.5629401672 | 0.13033696675 | 0.521347867 | NO |
| -2 | 169 | 62 | 5.83810587666 | 6.05930278804 | -0.221196911374 | -0.967253890087 | 114.805652193 | 0.335450342324 | 1 | NO |
| -1 | 169 | 62 | 2.56156893071 | 2.60770756027 | -0.046138629555 | -0.286402663676 | 110.411436884 | 0.775106445427 | 1 | NO |

## Interpretation boundary

No preregistered pre-emission offset passed Holm-adjusted `p < 0.05`.

The bounded Stage A label is:

`FORCED_DECISIVE_PRE_EMISSION_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`

This does not support temporal separation under the prospectively frozen forced-decisive two-class grammar.

This report does not establish spontaneous hallucination prediction, behavior with abstention available, free-form generation generalization, prospective Stage B prediction, or causal Stage C prevention.

No training, backward pass, selection reopening, layer scan, rescue, Stage B, or Stage C was executed.
