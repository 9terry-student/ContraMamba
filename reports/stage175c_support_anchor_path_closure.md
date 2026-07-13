# Stage175-C — relative SUPPORT-anchor preservation closure

**Decision:** `STAGE175C_RELATIVE_SUPPORT_ANCHOR_PRESERVATION_NO_CLEAN_BENEFIT_PATH_CLOSED`

## Experimental scope

This closure covers `v6b_minimal` with the Mamba backbone `state-spaces/mamba-130m-hf`, seed 174, 20 epochs, train/eval batch sizes 2/4, and gradient accumulation 2. The only main dataset was `data/controlled_v5_v3_without_time_swap.jsonl`. Stage174-C was off (`weight=0.0`); treatment enabled Stage175-B `paraphrase_margin` with `weight=0.05` and `tolerance=0.10`. Selection used internal clean dev only. Final classifier CE consumed `output["logits"]`. No external evaluation or time-swap data was used.

## Clean-dev result

| Metric | Baseline | Treatment | Delta |
|---|---:|---:|---:|
| selected epoch | 20 | 20 | 0 |
| final accuracy | 0.868056 | 0.883333 | +0.015277 |
| macro-F1 | 0.828604 | 0.823343 | -0.005261 |
| frame accuracy | 0.840278 | 0.826389 | -0.013889 |
| predicate accuracy | 0.856944 | 0.841667 | -0.015277 |
| SUPPORT precision | 0.477941 | 0.525773 | +0.047832 |
| SUPPORT recall | 0.730337 | 0.573034 | -0.157303 |
| SUPPORT F1 | 0.577778 | 0.548387 | -0.029391 |
| NOT_ENTITLED predictions | 493 | 532 | +39 |
| REFUTE predictions | 91 | 91 | 0 |
| SUPPORT predictions | 136 | 97 | -39 |

The treatment improved accuracy and SUPPORT precision, but degraded macro-F1, SUPPORT recall/F1, frame accuracy, and predicate accuracy. The aggregate distribution shifted exactly 39 predictions from SUPPORT mass toward NOT_ENTITLED mass; row-level transition attribution is deferred to Stage176-A.

## Runtime activity

The eligible mapping contained the expected 121 train SUPPORT-anchor pairs. Across 20 epochs it processed 2,420 current rows, 2,420 detached reference rows, and 620 reference batches, with 222 active and 2,198 zero hinge violations. No row was malformed or skipped. Aggregate current/reference SUPPORT margins were -0.514620/-0.784678; the raw `reference - current` gap was -0.270058. Mean unweighted/weighted losses were 0.011612/0.000581.

At epoch 20 there were 11 active violations. Current/reference mean margins were 0.172791/-0.269488, their raw gap was -0.442279, and weighted loss was 0.000804.

## Established findings and interpretation

1. Eligible mapping and reference forward operated correctly.
2. Each current paraphrase mapped to the exact detached same-pair canonical `none` row.
3. No malformed, missing, ambiguous, or same-row reference error occurred.
4. The objective protected paraphrase margin relative to canonical margin.
5. Clean-dev SUPPORT recall and macro-F1 nevertheless deteriorated.
6. SUPPORT predictions decreased by 39 while NOT_ENTITLED predictions increased by 39.
7. Relative margin preservation does not guarantee an absolute SUPPORT decision.
8. A detached reference is not a frozen teacher; canonical reference margins can move between optimizer steps.
9. The objective directly protects only current paraphrases, not canonical `none` rows or other SUPPORT rows.
10. This path will not be expanded through weight/tolerance sweeps, three seeds, longer training, or external evaluation.

## Closure

The Stage175-B implementation remains available but default-off. Its clean-improvement path is closed: no weight sweep, tolerance sweep, three-seed expansion, long-training expansion, or external evaluation is authorized. Stage176-A is the next step and is strictly a clean-dev diagnostic attribution audit.
