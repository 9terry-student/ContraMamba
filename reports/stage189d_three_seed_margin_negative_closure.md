# Stage189-D authoritative negative closure

**Decision:** `STAGE189D_THREE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE`

The Stage189 artifacts remain valid internal diagnostic evidence. The compatible-positive absolute margin objective at weight `0.05` is closed as a model-advancement candidate.

## Frozen identity and topology

- Training commit: `bee2f5ad452d1d9f57b30f444d18835dbffdbecf`
- Trainer SHA256: `24b01c5799c762772fe1700204afae59f8566898f65e7f3eefa4ac57ac6f126f`
- Dataset SHA256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- Stage185 sidecar semantic SHA256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`
- Training seeds: `174, 175, 176`; fixed split seed: `174`
- Train/dev: `2880/720`; train-compatible: `1440`
- ELIGIBLE/INELIGIBLE/UNRESOLVED: `605/716/119`

## Direct observations

| Seed | Macro-F1 delta | SUPPORT recall delta | False-entitlement delta | Polarity delta | Hard guardrails |
|---:|---:|---:|---:|---:|---|
| 174 | +0.0089975 | +0.0224719 | -3 | 0 | pass |
| 175 | -0.00364033 | -0.0898876 | -13 | 0 | fail |
| 176 | -0.00944096 | -0.247191 | -40 | +1 | fail |

Aggregate means were `-0.00136126` macro-F1, `-0.104869` SUPPORT recall, and `-18.6667` false entitlement.

Eligible posthoc frame-logit mean/median/positive-fraction deltas were `+0.251300/+0.279331/0.960331` for seed 174, `+0.131983/+0.101600/0.970248` for seed 175, and `-0.392888/-0.395615/0.0` for seed 176. Stage182-B compatible-FN newly harmed counts were `0/2/7`.

## Closure interpretation

Seed 174's single-seed benefit did not replicate robustly. False-entitlement reduction came with severe SUPPORT-recall loss. Seed 176 additionally showed polarity regression and opposite-direction frame-logit movement. Relative selectivity statistics do not override these absolute mechanism and clean failures.

No external evaluation, additional simple weight sweep, or seed extension is authorized at this stage.
