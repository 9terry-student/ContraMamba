"""Train the v3 balanced auditor and export Stage 10A probe predictions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_controlled_v5 import load_jsonl, split_by_pair_id  # noqa: E402
from scripts.create_number_swap_probe import REQUIRED_FIELDS  # noqa: E402
from scripts.train_controlled_v5 import (  # noqa: E402
    build_mamba_model,
    build_model,
    build_vocab,
    cache_frozen_encoder_states,
    encode_mamba_records,
    encode_records,
    evaluate,
    move_inputs,
    prediction_records,
    restore_trainable_state,
    run_training,
    write_predictions_json,
)


DEFAULT_DATA = REPO_ROOT / "data" / "controlled_v5_v3.jsonl"
DEFAULT_PROBE = REPO_ROOT / "data" / "stage10a_number_swap_probe.jsonl"
BALANCED_NO_POLARITY_FLIP_LOSS = {
    "lambda_frame_preserve": 1.0,
    "lambda_frame_anchor": 1.0,
    "lambda_predicate_contrast": 1.0,
    "lambda_predicate_anchor": 1.0,
    "lambda_sufficiency_contrast": 1.0,
    "lambda_polarity_flip": 0.0,
    "lambda_polarity_margin_anchor": 0.0,
    "lambda_paraphrase_preserve": 1.0,
    "lambda_entitlement_preserve": 1.0,
    "lambda_logit_preserve": 1.0,
    "ranking_margin": 0.5,
    "polarity_margin_min": 1.0,
}


def load_number_swap_probe(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("number-swap probe is empty")
    for record in records:
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"{record.get('id', '<unknown>')} is missing {sorted(missing)}")
        if record["intervention_type"] not in {"none", "number_swap"}:
            raise ValueError("probe may contain only none and number_swap interventions")
        if record["intervention_type"] == "number_swap" and not (
            record["final_label"] == "NOT_ENTITLED"
            and record["frame_compatible_label"] == 0
            and record["predicate_covered_label"] == 1
            and record["sufficiency_label"] == 1
            and record["polarity_label"] == "NONE"
            and record["primary_failure_type"] == "frame"
        ):
            raise ValueError("number_swap ontology must remain NOT_ENTITLED via frame failure")
    interventions = {record["intervention_type"] for record in records}
    if interventions != {"none", "number_swap"}:
        raise ValueError("probe must contain both none and number_swap interventions")
    return records


def _pad_to_shared_length(input_sets: Sequence[dict[str, torch.Tensor]]) -> int:
    max_length = max(inputs["input_ids"].shape[1] for inputs in input_sets)
    for inputs in input_sets:
        difference = max_length - inputs["input_ids"].shape[1]
        if difference:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                inputs[key] = F.pad(inputs[key], (0, difference), value=0)
    return max_length


def export_probe_predictions(
    path: Path,
    records: list[dict],
    output: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    predictions = prediction_records(records, output)
    write_predictions_json(path, metadata, predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--probe-data", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output-predictions-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--backbone", choices=("mamba", "dummy"), default="mamba")
    parser.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--head-lr", type=float, default=0.003)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    records = load_jsonl(args.data)
    probe_records = load_number_swap_probe(args.probe_data)
    train_records, dev_records = split_by_pair_id(records, args.dev_ratio, args.seed)

    if args.backbone == "mamba":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        bundles = [
            encode_mamba_records(rows, tokenizer, args.max_length)
            for rows in (train_records, dev_records, probe_records)
        ]
        model = build_mamba_model(args.model_name, freeze_encoder=True, freeze_a_log=True)
    else:
        vocab = build_vocab(records + probe_records)
        bundles = [encode_records(rows, vocab) for rows in (train_records, dev_records, probe_records)]
        model = None

    train_bundle, dev_bundle, probe_bundle = bundles
    input_sets = [move_inputs(bundle["model_inputs"], device) for bundle in bundles]
    train_inputs, dev_inputs, probe_inputs = input_sets
    shared_length = _pad_to_shared_length(input_sets)
    if model is None:
        model = build_model(len(vocab), shared_length)
    model = model.to(device)
    if args.backbone == "mamba":
        print("Caching frozen Mamba token states for train/dev/probe...")
        for inputs in input_sets:
            cache_frozen_encoder_states(model, inputs)

    report = run_training(
        model,
        train_inputs,
        dev_inputs,
        train_records,
        dev_records,
        train_bundle,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        encoder_lr=None,
        weighted_label_loss=True,
        balanced_sampler=True,
        use_intervention_loss=True,
        ranking_weight=2.0,
        loss_config=BALANCED_NO_POLARITY_FLIP_LOSS,
        seed=args.seed,
        run_name="v3_no_polarity_flip",
        select_metric="final_macro_f1",
        capture_best_trainable_state=True,
    )
    best_state = report.pop("_best_trainable_state")
    if best_state is None:
        raise RuntimeError("training did not capture a best-epoch state")
    restore_trainable_state(model, best_state)
    probe_output, _ = evaluate(model, probe_inputs, probe_records)
    metadata = {
        "data_path": str(args.data),
        "probe_data_path": str(args.probe_data),
        "seed": args.seed,
        "best_epoch": report["best_epoch"],
        "backbone": args.backbone,
        "model_name": args.model_name if args.backbone == "mamba" else None,
        "freeze_encoder": True,
        "freeze_a_log": True,
        "use_intervention_loss": True,
        "weighted_label_loss": True,
        "balanced_sampler": True,
        "configuration": "v3_no_polarity_flip",
        "loss_config": BALANCED_NO_POLARITY_FLIP_LOSS,
    }
    export_probe_predictions(
        args.output_predictions_json, probe_records, probe_output, metadata
    )
    print(
        json.dumps(
            {
                "output": str(args.output_predictions_json),
                "predictions": len(probe_records),
                "best_epoch": report["best_epoch"],
                "seed": args.seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
