# Gen4 seed181 independent checkpoint replication — result report

## Status

The frozen seed181 independent checkpoint replication supports the prespecified homolog-restoration replication claim.

This report records an executed and imported result. It does not authorize additional training, evaluation, checkpoint mutation, or outcome-dependent rescue.

## Execution and provenance

- run: `g4k-seed181-checkpoint-replication-8e96fd1-retry1`
- execution commit: `8e96fd164ac0321add7d6dff8d7020872824b8ab`
- checkpoint arm: `G3-GROUP-D-HALF`
- checkpoint seed: `181`
- selected epoch: `19`
- checkpoint SHA256: `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`
- registered command SHA256: `04930df45c03be9b41b3762ad1eb80ba633ca5497a92c5fdee8cc550fe2613f7`
- imported handoff ZIP SHA256: `d979af65831bc49f8601329e2dff4d9d7e5702c44e119207f185e800e0a10507`
- run exit code: `0`
- imported files validated: `9`
- scientific model forwards: `50,400`
- XG1 evaluation population: `xg1_fact_901..1200`, `N=300`
- analysis SHA256: `9c7eb59266455669fb5b8fe309414824e691f0cf3816ca6ff90059162185e645`

The raw execution explicitly recorded:

- `PRIMARY_INFERENCE_EXECUTED=False`
- `SCIENTIFIC_CONCLUSION=None`

Primary inference was performed only after successful local import by the frozen analysis script.

## Response-blind geometry transfer

The seed181 homolog/control planes were selected without using seed181 causal-response magnitude, p-values, restoration outcomes, or finite-epsilon reconstruction outcomes.

The frozen selection produced:

- seed180 PP3-matched seed181 homolog: `P3`
- matched control: `P5`
- homolog projector overlap: `1.9999999999999951`

Seed180-PP3 projector overlaps across the five seed181 planes were:

`[1.0801441853300552e-30, 1.695403841287025e-30, 1.9999999999999951, 2.619986948095786e-31, 6.100585878968926e-31]`

Thus the response-blind homolog selection was numerically unambiguous. The overlap itself is descriptive geometric evidence and is not treated as an independent causal result.

## Prespecified confirmatory restoration result

The inherited restoration contrast was:

`D_SUF = (Q_R3 - Q_B) - (Q_R5 - Q_B)`

with the one-sided one-sample test

`H0: E[D_SUF] <= 0`

against

`H1: E[D_SUF] > 0`.

All four frozen positive-replication gates passed:

- mean `Q_restored = 1.8577153849600195e-07 > 0`
- mean `S_homolog = 8.069442497094753e-08 > 0`
- mean `D_SUF = 4.078872356598754e-08 > 0`
- one-sided `t(299) = 20.105176219299196`, `p = 9.010901911776754e-58 < 0.05`

Additional means:

- mean `Q_neutralized = 1.0507711352505441e-07`
- mean `Q_control = 1.4498281493001444e-07`
- mean `S_control = 3.9905701404959994e-08`

Frozen analyzer label:

`SEED181_HOMOLOG_RESTORATION_REPLICATION_SUPPORTED`

## Finite-epsilon principal reconstruction

This analysis remains descriptive only and has no binary promotion threshold.

- mean native Q: `1.8577153849600195e-07`
- mean principal reconstruction: `1.8607038369640398e-07`
- mean residual: `-2.988452004020009e-10`
- RMSE: `8.592936258069666e-09`
- normalized RMSE over RMS native Q: `0.0392067058468371`
- MAE: `3.164292668625018e-09`
- Pearson correlation: `0.9972791521198436`
- sign agreement: `1.0`

## Scientific interpretation

Under the frozen seed181 design, the response-blind seed180-PP3-matched seed181 P3 component restores the internal recurrent-state endpoint more strongly than the matched P5 control and satisfies every prespecified confirmatory gate.

This supports replication of the internal homolog-restoration effect across the seed180 to seed181 checkpoint/training-seed change within the fixed arm and experimental construction.

It does not yet establish that this recurrent-state geometry changes downstream task decisions. The demonstrated causality remains causality with respect to the measured internal recurrent-state endpoint.

## Next research direction

The next high-value test is a prospective fresh-holdout behavioral bridge, rather than another geometry-localization sweep.

The bridge should reuse frozen geometry/intervention semantics and carry the same intervention through the downstream task head on an untouched population. Before observing behavioral responses, freeze the population, `N`, intervention site, homolog/control identities, primary metric, contrast, statistical test, and success criterion.

The intended primary behavioral metric is the correct-class logit margin

`M = z_y - max_{c != y} z_c`

with primary matched causal contrast

`D_BEH = M_R - M_C`.

The primary hypothesis should remain one-sided:

`H0: E[D_BEH] <= 0`

versus

`H1: E[D_BEH] > 0`.

Behavioral necessity (`M_0 - M_N`), accuracy, prediction-flip rate, and item-wise association between internal-effect magnitude and behavioral-effect magnitude should remain secondary/descriptive rather than becoming additional primary gates.

If practical under the frozen design, applying the identical prospective behavioral bridge to both seed180 and seed181 checkpoints would test whether the downstream behavioral consequence replicates as well as the internal restoration effect.

No behavioral result has yet been executed or established.
