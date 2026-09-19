# Gen4 AVeriTeC 370M Fixed Mirror Steering — Frozen Analysis

## Result

`AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED`

## Primary endpoint

- N: `2799`
- native-correct: `331`
- native-incorrect: `2468`
- C (native-wrong → steer-correct): `0`
- D (native-correct → steer-wrong): `0`
- discordant C+D: `0`
- one-sided exact p-value: `NOT_ESTIMABLE` (`C + D = 0`)
- alpha: `0.05`

## Preservation and success gates

- damage rate on native-correct: `0.0`
- fixed damage threshold: `0.05`
- C + D > 0: `False`
- primary p < alpha: `False`
- C > D: `False`
- damage rate <= threshold: `True`
- all four gates pass: `False`

## Required descriptive diagnostics

- native accuracy: `0.11825652018578063`
- mirror-steer accuracy: `0.11825652018578063`
- matched-control accuracy: `0.11825652018578063`
- correction rate on native errors: `0.0`
- damage rate on native-correct: `0.0`
- net accuracy change: `0.0`
- mean correct-class margin change, all rows (steer-native): `3.804646130231427e-05`
- mean margin change, target/native-incorrect: `2.3042106180699965e-05`
- mean margin change, preservation/native-correct: `0.0001499218342332632`
- mean matched-control contrast (native-control): `4.031383967802844e-05`
- fraction with steer margin > native: `0.5151839942836728`
- fraction with M_steer > M_native > M_control: `0.49803501250446586`

Mapped-label-specific correction/damage counts and full prediction transition tables are recorded in `steering_analysis.json`.

## Interpretation

The specific fixed mirror extrapolation did not establish useful steering under the preregistered utility/harm rule.

This analysis adds exactly one primary p-value when the endpoint is estimable and adds no subgroup/control p-values. No rescue, tuning, new model forward, CUDA execution, training, or backward pass occurs.

## Claim boundary

This result does not establish:

- general benchmark superiority
- four-class AVeriTeC performance
- retrieval competence
- free-form hallucination prevention
- a pre-emission precursor
- cross-scale steering
- universal P3 rank identity
- optimal steering magnitude
