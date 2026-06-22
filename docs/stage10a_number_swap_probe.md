# Stage 10A Number-Swap Single-Axis Probe

## Question

`time_swap` is a catastrophic confidently inverted case in the current v3 predictions, but entity, event, and predicate swaps are mostly rejected. The sharper question is whether the failure is temporal-specific or reflects a broader inability to compare values within the same semantic slot type.

Stage 10A changes exactly one numeric quantity while preserving the subject, predicate, object type, time expression, and sentence form. Example:

```text
Claim:    Aster Company sold 100 units in April.
Evidence: Aster Company sold 300 units in April.
```

## Label ontology

The probe assigns `number_swap` the label `NOT_ENTITLED`. This follows the existing `time_swap` ontology: a same-frame slot-value mismatch is treated as evidence that does not license the claim, rather than as an explicit polarity refutation. The auxiliary labels likewise match the time-swap convention: frame compatibility is 0, predicate coverage is 1, and sufficiency is 1.

If a future benchmark instead defines numeric mismatch as `REFUTE`, it must be evaluated separately as a value contradiction. The probe should not silently force `NOT_ENTITLED` under a different ontology.

## Decision rule

- If `number_swap` and `time_swap` both have high false-entitled rates and high frame, predicate, sufficiency, and entitlement pass probabilities, the result supports a same-type low-surface-change or slot-value comparison failure.
- If `number_swap` is correctly rejected while `time_swap` fails, the result supports a temporal-specific failure.
- Otherwise, the result is inconclusive or mixed.

## Reproducible inference pipeline

The archived v3 artifacts do not include model checkpoints, so new number-swap predictions require retraining. `scripts/train_and_export_stage10a_number_swap.py` reproduces the frozen-Mamba `v3_no_polarity_flip` balanced-auditor configuration, selects the best development epoch by macro-F1, restores its trainable state, and exports predictions for all 60 `none` and 60 `number_swap` probe records. The export uses the same schema as the existing v3 prediction files.

The script fixes the polarity-flip intervention losses to zero while retaining the other intervention-aware losses, weighted label loss, balanced sampling, a frozen `state-spaces/mamba-130m-hf` encoder, and the established 0.003 head learning rate. The selected preset is written into export metadata.

Once the new predictions exist, `scripts/evaluate_number_swap_probe.py` compares them with the corresponding balanced-auditor v3 predictions and extracts `time_swap` rows from the latter.

### Kaggle commands

Generate the fixed probe only if it is absent:

```bash
python scripts/create_number_swap_probe.py --num-pairs 60 --output data/stage10a_number_swap_probe.jsonl
```

Train and export each balanced-auditor seed:

```bash
python scripts/train_and_export_stage10a_number_swap.py --seed 1 --device cuda --output-predictions-json /kaggle/working/stage10a_number_swap_seed1_preds.json
python scripts/train_and_export_stage10a_number_swap.py --seed 2 --device cuda --output-predictions-json /kaggle/working/stage10a_number_swap_seed2_preds.json
python scripts/train_and_export_stage10a_number_swap.py --seed 3 --device cuda --output-predictions-json /kaggle/working/stage10a_number_swap_seed3_preds.json
```

Compare number and time swaps for each seed:

```bash
python scripts/evaluate_number_swap_probe.py --number-preds /kaggle/working/stage10a_number_swap_seed1_preds.json --time-preds /kaggle/working/v3_no_polarity_flip_seed1_preds.json --output-csv /kaggle/working/stage10a_number_swap_seed1.csv --output-md /kaggle/working/stage10a_number_swap_seed1.md
python scripts/evaluate_number_swap_probe.py --number-preds /kaggle/working/stage10a_number_swap_seed2_preds.json --time-preds /kaggle/working/v3_no_polarity_flip_seed2_preds.json --output-csv /kaggle/working/stage10a_number_swap_seed2.csv --output-md /kaggle/working/stage10a_number_swap_seed2.md
python scripts/evaluate_number_swap_probe.py --number-preds /kaggle/working/stage10a_number_swap_seed3_preds.json --time-preds /kaggle/working/v3_no_polarity_flip_seed3_preds.json --output-csv /kaggle/working/stage10a_number_swap_seed3.csv --output-md /kaggle/working/stage10a_number_swap_seed3.md
```

## Claim constraints

- Do not generalize the time-swap result into broad evidence-presence versus claim-match blindness; entity, event, and predicate swaps already contradict that broad framing.
- Do not call temporal mismatch the root cause until the number-swap contrast is evaluated.
- Do not infer a shared gate mechanism from global correlation alone; use Stage 9D.
