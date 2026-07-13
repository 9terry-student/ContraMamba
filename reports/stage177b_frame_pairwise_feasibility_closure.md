# Stage177-B frame-pairwise feasibility closure

## Decision

`STAGE177B_FRAME_PAIRWISE_OBJECTIVE_FEASIBLE`

Targets come only from the actual clean-data `frame_compatible_label`. Family names never infer targets.

## Pair topology

The deterministic clean split contains 300 pairs: 240 train pairs (2,880 rows) and 60 dev pairs (720 rows), with zero pair overlap. Every pair contains six compatible and six incompatible rows, producing 36 comparisons per pair. This gives 8,640 train comparisons and 2,160 dev comparisons. No train or dev pair is malformed, and the top 10% of pairs contribute exactly 10% of comparisons.

## Baseline gap

| Split | Comparison ranking accuracy | Pair-normalized ranking accuracy | Mean compatible − incompatible gap |
|---|---:|---:|---:|
| Train | 0.950347 | 0.950347 | 3.080331 |
| Dev | 0.939352 | 0.939352 | 3.099228 |

The dev pair-bootstrap 95% confidence interval for ranking accuracy is `[0.9263888888888889, 0.9518634259259259]`.

## Objective contract

For every clean train pair, form every compatible × incompatible comparison:

```python
gap = compatible_frame_logit - incompatible_frame_logit
comparison_loss = torch.nn.functional.softplus(-gap)
pair_loss = comparison_loss.mean()
```

The unweighted Stage177-C loss is the mean of `pair_loss` over eligible pairs. Every pair therefore has equal weight. The objective has no margin, threshold, teacher, detached or no-grad reference, cross-pair negative, final-label target, final-classifier-logit target, external data, or `time_swap` data.

The existing direct frame BCE remains unchanged. Stage177-C adds within-pair ordering on the same native `frame_logit`; parameter sharing may propagate gradients into upstream representations, but final classifier logits are not directly targeted.

## Implementation topology

The trainer already produces a full clean-train differentiable `output["frame_logit"]` aligned with stable train-row ordering. Stage177-C can build a validated train-only pair index before training and consume that native tensor without an additional counterpart forward, dev lookup, checkpoint teacher, detached reference, or cross-pair lookup.

## Closure and safety

Default-off Stage177-C implementation and one-epoch wiring smoke are authorized. A 20-epoch pilot, weight sweep, margin or threshold addition, multi-seed run, external evaluation or tuning, `time_swap`, and final-logit pairwise loss are not authorized. No smoke or training is executed by this patch.
