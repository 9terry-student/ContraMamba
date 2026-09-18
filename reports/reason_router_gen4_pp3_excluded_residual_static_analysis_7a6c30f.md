# Gen4 PP3-Excluded Residual Static Characterization

## Status

`STATIC_EXPLORATORY_RESIDUAL_CHARACTERIZATION_NO_INFERENCE`

This analysis reuses only already validated directional-Jacobian artifacts and frozen XG2/XG4 basis geometry.

- new model forwards: `0`
- checkpoint loads: `0`
- GPU use: `false`
- p-values: `0`
- training/backward/task-head/logit analysis: `none`

The purpose is hypothesis generation for the PP3-excluded distributed residual,
not confirmatory promotion of any secondary plane.

## Frozen decomposition

For every item:

`Q = (1/5) g^T (P2 - P4) g`

The same frozen five principal contrast planes P1..P5 are used.
P3 is excluded only after exact reconstruction; no secondary plane is selected.

Frozen positive contrast eigenvalues:

- P1: `0.87061814189182751`
- P2: `0.94755022112376253`
- P3: `0.98692852916688445`
- P4: `0.99848952673382496`
- P5: `0.9998679284285461`

## Cohort summaries

### XG2_601_900

- N: `300`
- mean Q: `1.3987206143704023e-07`
- mean PP3 positive-net share: `0.5271789779054183`
- PP3-excluded residual mean Q: `6.5672801781134341e-08`
- residual effective |net| plane count: `2.88052225688`
- residual effective positive-gain plane count: `2.9479854366`
- residual net sign pattern [P1,P2,P4,P5]: `[1, 1, -1, 1]`
- dominant residual |net| plane (descriptive only): `P5`
- dominant residual positive-gain plane (descriptive only): `P5`
- max |Q reconstruction residual|: `3.4410713482205951e-22`

| Plane | positive gain | negative penalty | net |
|---|---:|---:|---:|
| P1 | 2.00005655765291e-08 | -4.0799781233712202e-09 | 1.5920587453157877e-08 |
| P2 | 2.9655242304908248e-08 | -8.9439540574987843e-09 | 2.0711288247409458e-08 |
| P3 | 8.5835715132312529e-08 | -1.1636455476406546e-08 | 7.4199259655905986e-08 |
| P4 | 1.8015989907783953e-09 | -2.6772964092099815e-09 | -8.7569741843158632e-10 |
| P5 | 3.7181168134320547e-08 | -7.264544635321946e-09 | 2.9916623498998595e-08 |

### XG4_601_900

- N: `300`
- mean Q: `3.7773950385492771e-07`
- mean PP3 positive-net share: `0.5554900659499483`
- PP3-excluded residual mean Q: `1.5581314777397315e-07`
- residual effective |net| plane count: `2.46425864287`
- residual effective positive-gain plane count: `2.67284910374`
- residual net sign pattern [P1,P2,P4,P5]: `[1, -1, 1, 1]`
- dominant residual |net| plane (descriptive only): `P4`
- dominant residual positive-gain plane (descriptive only): `P4`
- max |Q reconstruction residual|: `6.3527471044072525e-22`

| Plane | positive gain | negative penalty | net |
|---|---:|---:|---:|
| P1 | 4.4500064363405042e-08 | -2.583004145232737e-08 | 1.8670022911077672e-08 |
| P2 | 1.671928843445605e-09 | -2.3446961223178364e-08 | -2.177503237973276e-08 |
| P3 | 2.3888399758288331e-07 | -1.6957641501928785e-08 | 2.2192635608095456e-07 |
| P4 | 1.2047741121265835e-07 | -4.4215359621736195e-09 | 1.1605587525048473e-07 |
| P5 | 1.1249673650401968e-07 | -6.963445451187619e-08 | 4.2862281992143499e-08 |

### XG1_601_900_NATIVE

- N: `300`
- mean Q: `1.8678970016861958e-07`
- mean PP3 positive-net share: `0.389716431288496`
- PP3-excluded residual mean Q: `1.1399468481745706e-07`
- residual effective |net| plane count: `2.99637613719`
- residual effective positive-gain plane count: `3.27884775218`
- residual net sign pattern [P1,P2,P4,P5]: `[1, 1, 1, 1]`
- dominant residual |net| plane (descriptive only): `P5`
- dominant residual positive-gain plane (descriptive only): `P5`
- max |Q reconstruction residual|: `7.411538288475128e-22`

| Plane | positive gain | negative penalty | net |
|---|---:|---:|---:|
| P1 | 3.3413722352810883e-08 | -4.8264768560862166e-09 | 2.8587245496724666e-08 |
| P2 | 4.1260185759663282e-08 | -7.4600410186026122e-09 | 3.3800144741060673e-08 |
| P3 | 1.1264468103100171e-07 | -3.9849665679839038e-08 | 7.2795015351162654e-08 |
| P4 | 1.0624075332193178e-08 | -7.6841322159987355e-09 | 2.9399431161944443e-09 |
| P5 | 5.6808185524647937e-08 | -8.1408340611706564e-09 | 4.8667351463477282e-08 |

### XG1_901_1200_NATIVE

- N: `300`
- mean Q: `1.8577153849600195e-07`
- mean PP3 positive-net share: `0.3875785659534635`
- PP3-excluded residual mean Q: `1.1377047201075295e-07`
- residual effective |net| plane count: `3.00945411181`
- residual effective positive-gain plane count: `3.28187872443`
- residual net sign pattern [P1,P2,P4,P5]: `[1, 1, 1, 1]`
- dominant residual |net| plane (descriptive only): `P5`
- dominant residual positive-gain plane (descriptive only): `P5`
- max |Q reconstruction residual|: `5.8233515123733148e-22`

| Plane | positive gain | negative penalty | net |
|---|---:|---:|---:|
| P1 | 3.3256555431813544e-08 | -4.7130264343648967e-09 | 2.854352899744865e-08 |
| P2 | 4.1142160928562876e-08 | -7.2959329047067071e-09 | 3.3846228023856171e-08 |
| P3 | 1.1197644404971483e-07 | -3.9975377564465684e-08 | 7.2001066485249133e-08 |
| P4 | 1.0674417460361738e-08 | -7.5753214674600315e-09 | 3.0990959929017052e-09 |
| P5 | 5.6585463070156892e-08 | -8.3038440736104662e-09 | 4.8281618996546424e-08 |

## PP3-excluded pairwise profile agreement

| left | right | residual net cosine | residual positive-gain cosine | net-sign agreement |
|---|---|---:|---:|---:|
| XG2_601_900 | XG4_601_900 | 0.203540997541 | 0.605744633033 | 0.5 |
| XG2_601_900 | XG1_601_900_NATIVE | 0.997089170572 | 0.993025059465 | 0.75 |
| XG2_601_900 | XG1_901_1200_NATIVE | 0.996848470736 | 0.992948233651 | 0.75 |
| XG4_601_900 | XG1_601_900_NATIVE | 0.266026829592 | 0.688564569141 | 0.75 |
| XG4_601_900 | XG1_901_1200_NATIVE | 0.267146281362 | 0.689098308545 | 0.75 |
| XG1_601_900_NATIVE | XG1_901_1200_NATIVE | 0.99998797118 | 0.999999143852 | 1 |

## Interpretation boundary

These quantities are descriptive and exploratory. They may motivate a future prospectively frozen residual object on a fresh population, but they do not establish any P1/P2/P4/P5 plane as causal, transportable, necessary, or sufficient.

No secondary plane should be promoted from this analysis without a new prospectively frozen question and fresh evidence.
