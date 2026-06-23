"""Train ContraMamba-v6A on controlled intervention data.

This is intentionally a minimal v5-compatible training entrypoint. It reuses
the established controlled-v5 data, metrics, intervention-loss, reporting, and
prediction-export helpers while swapping in ContraMambaV6A and excluding the
known-corrupted time_swap intervention by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba import ContraMambaV6A  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


def build_model(
    vocab_size: int, max_length: int, hidden_size: int = 48
) -> ContraMambaV6A:
    backbone = v5.ControlledDummyBackbone(vocab_size, hidden_size, max_length)
    return ContraMambaV6A(
        backbone=backbone,
        frame_size=32,
        predicate_size=32,
        sufficiency_size=32,
        energy_size=24,
        composer_hidden_size=32,
        dropout=0.0,
        decision_mode="explicit_product",
        product_loss_weight=0.25,
    )


def build_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
) -> ContraMambaV6A:
    model = ContraMambaV6A(
        model_name=model_name,
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        energy_size=64,
        composer_hidden_size=64,
        dropout=0.1,
        freeze_a_log=freeze_a_log,
        decision_mode="explicit_product",
        product_loss_weight=0.25,
    )
    for parameter in model.mamba.parameters():
        parameter.requires_grad = not freeze_encoder
    if freeze_a_log:
        for name, parameter in model.mamba.named_parameters():
            if "A_log" in name:
                parameter.requires_grad = False
    return model


def prediction_records_v6a(records: list[dict], output: dict[str, Any]) -> list[dict]:
    probabilities = torch.softmax(output["logits"], dim=-1).detach().cpu()
    predictions = output["predictions"].detach().cpu()
    product_logits = output.get("product_logits")
    product_predictions = (
        product_logits.argmax(dim=-1).detach().cpu()
        if product_logits is not None
        else None
    )
    product_logits_cpu = (
        product_logits.detach().cpu() if product_logits is not None else None
    )
    composer_correction_logits = output.get("composer_correction_logits")
    composer_correction_logits_cpu = (
        composer_correction_logits.detach().cpu()
        if composer_correction_logits is not None
        else None
    )
    composer_logits = output.get("composer_logits")
    composer_logits_cpu = (
        composer_logits.detach().cpu() if composer_logits is not None else None
    )
    scalar_keys = (
        "frame_prob",
        "predicate_coverage_prob",
        "sufficiency_prob",
        "entitlement_prob",
        "polarity_margin",
        "product_entitlement_prob",
    )
    scalars = {
        key: output[key].detach().cpu()
        for key in scalar_keys
        if key in output
    }
    exported: list[dict] = []
    for index, record in enumerate(records):
        item = {
            "id": record["id"],
            "pair_id": record["pair_id"],
            "intervention_type": record["intervention_type"],
            "claim": record["claim"],
            "evidence": record["evidence"],
            "gold_final_label": record["final_label"],
            "pred_final_label": v5.ID_TO_FINAL_LABEL[int(predictions[index])],
            "final_probs": probabilities[index].tolist(),
            **{key: float(values[index]) for key, values in scalars.items()},
        }
        if product_predictions is not None:
            item["product_pred_final_label"] = v5.ID_TO_FINAL_LABEL[
                int(product_predictions[index])
            ]
        if product_logits_cpu is not None:
            item["product_logits"] = product_logits_cpu[index].tolist()
        if composer_correction_logits_cpu is not None:
            item["composer_correction_logits"] = (
                composer_correction_logits_cpu[index].tolist()
            )
        if composer_logits_cpu is not None:
            item["composer_logits"] = composer_logits_cpu[index].tolist()
        exported.append(item)
    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = v5.build_parser()
    parser.add_argument(
        "--exclude-interventions",
        nargs="*",
        default=["time_swap"],
        help="Intervention types to exclude from train/dev. Default excludes corrupted time_swap.",
    )
    return parser


def _filter_excluded_interventions(
    records: list[dict[str, Any]], excluded_interventions: list[str] | None
) -> list[dict[str, Any]]:
    if not excluded_interventions:
        return records
    excluded = set(excluded_interventions)
    before = len(records)
    filtered = [
        record
        for record in records
        if record.get("intervention_type") not in excluded
    ]
    after = len(filtered)
    print(
        f"Excluded interventions={sorted(excluded)}: "
        f"{before} -> {after} records"
    )
    return filtered


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    v5.prediction_records = prediction_records_v6a

    records = v5.load_jsonl(args.data)
    records = _filter_excluded_interventions(records, args.exclude_interventions)
    train_records, dev_records = v5.split_by_pair_id(
        records, dev_ratio=args.dev_ratio, seed=args.seed
    )
    if args.backbone == "dummy":
        vocab = v5.build_vocab(records)
        train_bundle = v5.encode_records(train_records, vocab)
        dev_bundle = v5.encode_records(dev_records, vocab)
        model = None
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        train_bundle = v5.encode_mamba_records(
            train_records, tokenizer, args.max_length
        )
        dev_bundle = v5.encode_mamba_records(dev_records, tokenizer, args.max_length)
        model = build_mamba_model(
            args.model_name,
            freeze_encoder=args.freeze_encoder,
            freeze_a_log=args.freeze_a_log,
        )
    train_inputs = v5.move_inputs(train_bundle["model_inputs"], device)
    dev_inputs = v5.move_inputs(dev_bundle["model_inputs"], device)
    max_length = max(
        train_inputs["input_ids"].shape[1], dev_inputs["input_ids"].shape[1]
    )
    for inputs in (train_inputs, dev_inputs):
        difference = max_length - inputs["input_ids"].shape[1]
        if difference:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                inputs[key] = F.pad(inputs[key], (0, difference), value=0)

    if model is None:
        model = build_model(len(vocab), max_length)
    model = model.to(device)
    if args.backbone == "mamba" and args.freeze_encoder:
        print("Caching frozen Mamba token states for train/dev...")
        v5.cache_frozen_encoder_states(model, train_inputs)
        v5.cache_frozen_encoder_states(model, dev_inputs)
    print(
        f"controlled v6A | backbone={args.backbone} "
        f"train={len(train_records)} ({len(set(train_bundle['pair_ids']))} pairs) "
        f"dev={len(dev_records)} ({len(set(dev_bundle['pair_ids']))} pairs) "
        f"freeze_encoder={args.freeze_encoder} device={device}"
    )

    requested_loss_config = {
        "lambda_frame_preserve": args.lambda_frame_preserve,
        "lambda_frame_anchor": args.lambda_frame_anchor,
        "lambda_predicate_contrast": args.lambda_predicate_contrast,
        "lambda_predicate_anchor": args.lambda_predicate_anchor,
        "lambda_sufficiency_contrast": args.lambda_sufficiency_contrast,
        "lambda_polarity_flip": args.lambda_polarity_flip,
        "lambda_polarity_margin_anchor": args.lambda_polarity_margin_anchor,
        "lambda_paraphrase_preserve": args.lambda_paraphrase_preserve,
        "lambda_entitlement_preserve": args.lambda_entitlement_preserve,
        "lambda_logit_preserve": args.lambda_logit_preserve,
        "ranking_margin": args.ranking_margin,
        "polarity_margin_min": args.polarity_margin_min,
    }
    configurations = (
        v5.sweep_presets(args.ranking_margin)
        if args.loss_sweep
        else {"single": requested_loss_config}
    )
    initial_head_state = v5.capture_head_state(model)
    reports: dict[str, dict[str, Any]] = {}
    for run_name, loss_config in configurations.items():
        v5.restore_head_state(model, initial_head_state)
        torch.manual_seed(args.seed)
        reports[run_name] = v5.run_training(
            model,
            train_inputs,
            dev_inputs,
            train_records,
            dev_records,
            train_bundle,
            epochs=args.epochs,
            lr=args.lr,
            head_lr=args.head_lr,
            encoder_lr=args.encoder_lr,
            weighted_label_loss=args.weighted_label_loss,
            balanced_sampler=args.balanced_sampler,
            use_intervention_loss=args.use_intervention_loss or args.loss_sweep,
            ranking_weight=args.ranking_weight,
            loss_config=loss_config,
            seed=args.seed,
            run_name=run_name,
            select_metric=args.select_metric,
        )

    report = {
        "configuration": {
            "backbone": args.backbone,
            "model_name": args.model_name if args.backbone == "mamba" else None,
            "freeze_encoder": args.freeze_encoder,
            "freeze_a_log": args.freeze_a_log,
            "weighted_label_loss": args.weighted_label_loss,
            "balanced_sampler": args.balanced_sampler,
            "use_intervention_loss": args.use_intervention_loss,
            "loss_sweep": args.loss_sweep,
            "exclude_interventions": args.exclude_interventions,
            "model_version": "v6A",
        },
        "runs": reports,
    }
    prediction_exports = {
        name: run_report.pop("_best_dev_predictions")
        for name, run_report in reports.items()
    }
    if args.output_predictions_json is not None:
        if len(reports) != 1:
            parser.error("--output-predictions-json requires a single non-sweep run")
        run_name, run_report = next(iter(reports.items()))
        metadata = {
            "data_path": str(args.data),
            "seed": args.seed,
            "best_epoch": run_report["best_epoch"],
            "backbone": args.backbone,
            "model_name": args.model_name if args.backbone == "mamba" else None,
            "freeze_encoder": args.freeze_encoder,
            "weighted_label_loss": args.weighted_label_loss,
            "balanced_sampler": args.balanced_sampler,
            "use_intervention_loss": args.use_intervention_loss,
            "loss_config": run_report["loss_config"],
            "exclude_interventions": args.exclude_interventions,
            "model_version": "v6A",
        }
        v5.write_predictions_json(
            args.output_predictions_json,
            metadata,
            prediction_exports[run_name],
        )
    if len(reports) == 1:
        single = next(iter(reports.values()))
        for key in (
            "final_epoch",
            "best_epoch",
            "best_dev_metrics",
            "best_dev_interventions",
            "best_dev_pairwise_checks",
        ):
            report[key] = single[key]
    for run_name, run_report in reports.items():
        distribution = run_report["dev_prediction_distribution"]
        if len(distribution) == 1:
            collapsed_label = next(iter(distribution))
            print(
                f"WARNING: run {run_name} dev predictions collapsed to "
                f"the single label {collapsed_label}",
                file=sys.stderr,
            )
    print("\nFINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        v5.write_report_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
