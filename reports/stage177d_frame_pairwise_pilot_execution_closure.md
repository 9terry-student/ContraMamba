# Stage177-D frame-pairwise pilot execution closure

**Execution decision:** `STAGE177D_SINGLE_SEED_FRAME_PAIRWISE_PILOT_COMPLETED`

## Configuration

The single authorized pilot used seed 174 for 20 epochs with `v6b_minimal`, the Mamba backbone, `state-spaces/mamba-130m-hf`, CUDA, and `data/controlled_v5_v3_without_time_swap.jsonl`. Stage174-C and Stage175-B were off with weight 0. Stage177-C used `pair_softplus` with weight 0.05. Epoch 20 was selected and its checkpoint was persisted. No external data or `time_swap` data was used.

## Selected clean-dev result

| Metric | Value |
|---|---:|
| Final accuracy | 0.8680555820465088 |
| Final macro-F1 | 0.8286042092431251 |
| Frame accuracy | 0.8402777910232544 |
| Predicate accuracy | 0.8583333492279053 |
| Sufficiency accuracy | 1.0 |
| Polarity accuracy on entitled rows | 1.0 |

The selected prediction counts were 493 `NOT_ENTITLED`, 91 `REFUTE`, and 136 `SUPPORT`. These aggregate metrics and counts exactly match the seed-174 baseline selected checkpoint.

## Stage177-C trajectory

Pair-ranking accuracy rose from 0.586574 at epoch 1 to 0.944676 at epoch 20; the run mean was 0.864433. Every epoch had 240 eligible pairs, zero malformed pairs, and 8,640 raw comparisons. The loss stayed finite and differentiable, required no extra counterpart forward, and did not directly target final-classifier logits.

## Limitation and next stage

Aggregate equality does not establish row-level prediction equality or checkpoint equivalence. Stage177-E must compare the two persisted checkpoints directly. This closure does not authorize a weight sweep, multi-seed execution, or external evaluation.
