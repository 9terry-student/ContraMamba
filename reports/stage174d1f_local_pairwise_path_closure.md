# Stage174-D1F clean local-pairwise path closure

**Decision:** `STAGE174D1F_CLEAN_LOCAL_PAIRWISE_OBJECTIVE_DIRECTION_CONFLICT_PATH_CLOSED`

## Experimental scope

All runs used `v6b_minimal` with a Mamba backbone, seed 174, 20 epochs, and `data/controlled_v5_v3_without_time_swap.jsonl`. `time_swap` was excluded. Checkpoints were selected only on internal clean-dev; final classifier CE used `output["logits"]`, never `loss_logits`. No external evaluation ran. The runtime validated 240 train pair groups with zero malformed or skipped groups.

## Results

| Arm | Epoch | Accuracy | Macro-F1 | Δ Macro-F1 | SUPPORT recall | Δ recall | NOT_ENTITLED | SUPPORT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (`off`) | 20 | 0.868056 | 0.828604 | — | 0.730337 | — | 493 | 136 |
| local-only 0.01 | 20 | 0.872222 | 0.824706 | -0.003898 | 0.662921 | -0.067416 | 508 (+15) | 121 (-15) |
| local-only 0.02 | 19 | 0.866667 | 0.822442 | -0.006162 | 0.685393 | -0.044944 | 500 (+7) | 129 (-7) |
| local-only 0.05 | 19 | 0.868056 | 0.822352 | -0.006252 | 0.674157 | -0.056180 | 503 (+10) | 126 (-10) |
| local-only 0.10 | 20 | 0.883333 | 0.823343 | -0.005261 | 0.573034 | -0.157303 | 532 (+39) | 97 (-39) |

Baseline REFUTE predictions were 91, as were all listed arms. At weight 0.10, SUPPORT precision was 0.525773, accuracy delta was +0.015277, frame-accuracy delta was -0.015278, and predicate-accuracy delta was -0.015277. Disabling polarity preservation at weight 0.05 (`polarity_preservation_weight=0.0`) produced identical clean-dev metrics, so polarity preservation was not the primary cause.

## Closure interpretation

Stage174-C’s runtime and provenance paths worked and its entitlement loss/counts were active. Nevertheless, every tested weight (0.01, 0.02, 0.05, 0.10) failed to improve macro-F1 and moved SUPPORT predictions toward NOT_ENTITLED; the effect strengthened at 0.10. The apparent high-weight accuracy gain is therefore a conservative majority-class shift, not balanced clean improvement. This is an objective-direction conflict, not a simple weight-tuning problem.

Stage174-C remains available as a reproducible negative path and stays default `off`. This path is closed without three-seed expansion, longer training, another weight sweep, or external evaluation. Stage175-A proceeds only as a static clean-data feasibility audit of SUPPORT-anchor preservation.
