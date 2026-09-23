# ICLR Pair-Resampling Robustness Analysis

## Status

`CPU_ONLY_FROZEN_ARTIFACT_RESAMPLING_RESULT`

Source HEAD: `ef1806bef9e4a7f55e1f7e77f7a0f08592fb4abe`

No model execution, tokenizer execution, training, evaluation, forward pass, backward pass, representation collection, row filtering, or response-guided selection occurred.

Bootstrap protocol: B=10000, percentile 95% pair-resampling intervals, fixed seed=20260924.

Split-half protocol: 1000 repeated balanced 150/150 splits, fixed seed=20260925.

These intervals quantify pair-resampling stability on the frozen cohorts. They are not a noise ceiling and not an independent-population replication.

## Objective mean pair-resampling

| Scale | Objective | Point mean | 95% interval | Contains 0 | P(boot > 0) |
|---|---|---:|---:|:---:|---:|
| 130M | task | 0.0022407321 | [0.00083299857, 0.0036227012] | no | 0.9994 |
| 130M | lm | -0.056416846 | [-0.068825987, -0.044195009] | no | 0.0000 |
| 370M | task | 0.0010179127 | [0.00066497744, 0.0013934152] | no | 1.0000 |
| 370M | lm | -0.0019753191 | [-0.0065514743, 0.0025359293] | yes | 0.2043 |
| 790M | task | 0.0045534229 | [0.0034547763, 0.0057362699] | no | 1.0000 |
| 790M | lm | -0.015964364 | [-0.022894954, -0.0091581402] | no | 0.0000 |
| 1.4B | task | -0.002390895 | [-0.0029845848, -0.0018318703] | no | 0.0000 |
| 1.4B | lm | 0.0078331828 | [0.0046518585, 0.011012203] | no | 1.0000 |
| 2.8B | task | -0.00014834991 | [-0.00048601639, 0.00018914839] | yes | 0.1964 |
| 2.8B | lm | 0.016448402 | [0.010704811, 0.022231041] | no | 1.0000 |

## Geometry joint pair-resampling

The same bootstrap multiplicities are applied jointly to all five scales within a generator family. XG2 and XG4 remain separate.

| Family | Scale pair | Point CKA | 95% interval |
|---|---|---:|---:|
| XG2 | 130M--370M | 0.742546 | [0.686168, 0.804516] |
| XG2 | 130M--790M | 0.435747 | [0.373114, 0.533184] |
| XG2 | 130M--1.4B | 0.342510 | [0.288400, 0.434264] |
| XG2 | 130M--2.8B | 0.555414 | [0.498702, 0.633222] |
| XG2 | 370M--790M | 0.564059 | [0.496527, 0.655139] |
| XG2 | 370M--1.4B | 0.504483 | [0.436225, 0.597320] |
| XG2 | 370M--2.8B | 0.649344 | [0.592576, 0.718996] |
| XG2 | 790M--1.4B | 0.663678 | [0.592873, 0.746388] |
| XG2 | 790M--2.8B | 0.639963 | [0.575241, 0.721765] |
| XG2 | 1.4B--2.8B | 0.532185 | [0.469506, 0.617164] |
| XG4 | 130M--370M | 0.542295 | [0.471883, 0.630262] |
| XG4 | 130M--790M | 0.451862 | [0.385099, 0.544526] |
| XG4 | 130M--1.4B | 0.481304 | [0.407584, 0.581593] |
| XG4 | 130M--2.8B | 0.544277 | [0.488900, 0.621725] |
| XG4 | 370M--790M | 0.558569 | [0.487361, 0.649232] |
| XG4 | 370M--1.4B | 0.436834 | [0.381434, 0.521894] |
| XG4 | 370M--2.8B | 0.590983 | [0.537345, 0.665149] |
| XG4 | 790M--1.4B | 0.444332 | [0.392263, 0.526154] |
| XG4 | 790M--2.8B | 0.392717 | [0.347751, 0.472452] |
| XG4 | 1.4B--2.8B | 0.455698 | [0.410225, 0.532717] |

## Repeated balanced split-half stability

Each half produces the fixed 10-dimensional off-diagonal cross-scale CKA vector. Metrics compare the two vectors within each split.

| Family | Metric | Median | 2.5% | 97.5% |
|---|---|---:|---:|---:|
| XG2 | pearson | 0.875577 | 0.670669 | 0.973706 |
| XG2 | spearman | 0.878788 | 0.636364 | 0.963636 |
| XG2 | mad | 0.057331 | 0.027105 | 0.099749 |
| XG2 | rmse | 0.068640 | 0.033487 | 0.117836 |
| XG4 | pearson | 0.652928 | 0.231557 | 0.895784 |
| XG4 | spearman | 0.672727 | 0.115152 | 0.878788 |
| XG4 | mad | 0.056955 | 0.028731 | 0.098158 |
| XG4 | rmse | 0.068166 | 0.035012 | 0.115344 |

## Interpretation boundary

- Geometry intervals are sampling-robustness summaries for the existing 300 controlled pairs, not model-remeasurement uncertainty.
- Objective intervals are descriptive pair-resampling uncertainty. They do not retroactively create a prospective five-scale inferential family.
- Point-estimate sign vectors remain point estimates; a cell whose resampling interval contains zero is not individually sign-stable under this resampling analysis.
- No permutation control is included in this frozen stage.
