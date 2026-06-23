"""Train ContraMamba-v6B-minimal on controlled intervention data.

Minimal v6B wrapper: reuses v5 training infrastructure, adds temporal/predicate
comparator alphas with learnable scaling. No composer, no product_final_loss.
All CE/pairwise/intervention losses consume final calibrated logits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba.comparator_flags import (  # noqa: E402
    predicate_mismatch_flags_from_intervention_type,
    predicate_mismatch_flags_from_probe,
    temporal_mismatch_flags_from_probe,
    temporal_mismatch_flags_none,
)
from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


def build_model(
    vocab_size: int, max_length: int, hidden_size: int = 48
) -> ContraMambaV6BMinimal:
    backbone = v5.ControlledDummyBackbone(vocab_size, hidden_size, max_length)
    return ContraMambaV6BMinimal(
        backbone=backbone,
        frame_size=32,
        predicate_size=32,
        sufficiency_size=32,
        energy_size=24,
        dropout=0.0,
        decision_mode="explicit_product",
        use_temporal_comparator=True,
        use_predicate_comparator=True,
        alpha_temporal_init=1.25,
        alpha_predicate_init=1.25,
    )


def build_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
) -> ContraMambaV6BMinimal:
    model = ContraMambaV6BMinimal(
        model_name=model_name,
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        energy_size=64,
        dropout=0.1,
        freeze_a_log=freeze_a_log,
        decision_mode="explicit_product",
        use_temporal_comparator=True,
        use_predicate_comparator=True,
        alpha_temporal_init=1.25,
        alpha_predicate_init=1.25,
    )
    for parameter in model.mamba.parameters():
        parameter.requires_grad = not freeze_encoder
    if freeze_a_log:
        for name, parameter in model.mamba.named_parameters():
            if "A_log" in name:
                parameter.requires_grad = False
    return model


def extract_flags(
    records: list[dict],
    flag_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract temporal and predicate flags based on source."""
    if flag_source == "stage15_probe_type":
        temporal_flags = temporal_mismatch_flags_from_probe(records, device)
        predicate_flags = predicate_mismatch_flags_from_probe(records, device)
    elif flag_source == "controlled_heuristic":
        temporal_flags = temporal_mismatch_flags_none(records, device)
        predicate_flags = predicate_mismatch_flags_from_intervention_type(records, device)
    elif flag_source == "none":
        temporal_flags = temporal_mismatch_flags_none(records, device)
        predicate_flags = temporal_mismatch_flags_none(records, device)
    else:
        raise ValueError(f"unknown flag_source: {flag_source}")
    return temporal_flags, predicate_flags


def compute_class_weights_v6b(
    records: list[dict],
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    n_classes = len(v5.FinalLabel)
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for record in records:
        counts[v5.FINAL_LABEL_TO_ID[record["final_label"]]] += 1.0
    counts = counts.clamp_min(1.0)
    if mode == "inverse_freq":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse_freq":
        weights = 1.0 / counts.sqrt()
    else:
        raise ValueError(f"unknown class_weighting mode: {mode!r}")
    weights = weights / weights.mean()
    return weights.to(device)


def evaluate_ood_v6b(
    model: ContraMambaV6BMinimal,
    records: list[dict],
    inputs: dict[str, torch.Tensor],
    temporal_flags: torch.Tensor,
    predicate_flags: torch.Tensor,
) -> tuple[dict[str, Any], list[dict]]:
    """Evaluate trained model on OOD records using final output logits/predictions."""
    from collections import defaultdict

    model.eval()
    with torch.no_grad():
        output = model(
            **v5.model_feature_inputs(inputs),
            temporal_mismatch_flags=temporal_flags,
            predicate_mismatch_flags=predicate_flags,
        )

    overall_metrics = v5.compute_metrics(output, inputs)
    predictions_export = prediction_records_v6b(records, output)

    predictions_cpu = output["predictions"].detach().cpu()
    labels_cpu = inputs["final_labels"].detach().cpu()
    not_entitled_id = v5.FINAL_LABEL_TO_ID.get("NOT_ENTITLED")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[record.get("stage15_probe_type", "unknown")].append(idx)

    group_metrics: dict[str, dict[str, Any]] = {}
    for group_name, indices in sorted(groups.items()):
        g_preds = predictions_cpu[indices]
        g_labels = labels_cpu[indices]
        n = len(indices)
        accuracy = (g_preds == g_labels).float().mean().item()
        f1_values = []
        for label in v5.FinalLabel:
            label_id = int(label)
            predicted = g_preds == label_id
            actual = g_labels == label_id
            tp = (predicted & actual).sum().item()
            prec_denom = predicted.sum().item()
            rec_denom = actual.sum().item()
            prec = tp / prec_denom if prec_denom else 0.0
            rec = tp / rec_denom if rec_denom else 0.0
            f1_values.append(2.0 * prec * rec / (prec + rec) if prec + rec else 0.0)
        macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        pred_dist: dict[str, int] = {}
        for pred_id in g_preds.tolist():
            label_name = v5.ID_TO_FINAL_LABEL[pred_id]
            pred_dist[label_name] = pred_dist.get(label_name, 0) + 1
        false_entitled_count = 0
        if not_entitled_id is not None:
            for pred_id, gold_id in zip(g_preds.tolist(), g_labels.tolist()):
                if gold_id == not_entitled_id and pred_id != not_entitled_id:
                    false_entitled_count += 1
        false_entitled_rate = false_entitled_count / n if n > 0 else 0.0
        group_metrics[group_name] = {
            "n": n,
            "final_accuracy": accuracy,
            "final_macro_f1": macro_f1,
            "prediction_distribution": dict(sorted(pred_dist.items())),
            "false_entitled_count": false_entitled_count,
            "false_entitled_rate": false_entitled_rate,
        }

    return {
        "n_records": len(records),
        "overall_metrics": overall_metrics,
        "group_metrics": group_metrics,
    }, predictions_export


def prediction_records_v6b(records: list[dict], output: dict[str, Any]) -> list[dict]:
    """Export predictions with v6b metadata."""
    probabilities = torch.softmax(output["logits"], dim=-1).detach().cpu()
    predictions = output["predictions"].detach().cpu()
    scalar_keys = (
        "frame_prob",
        "predicate_coverage_prob",
        "sufficiency_prob",
        "entitlement_prob",
        "polarity_margin",
    )
    scalars = {key: output[key].detach().cpu() for key in scalar_keys if key in output}
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
            **{key: float(scalars[key][index]) for key in scalar_keys if key in scalars},
        }
        exported.append(item)
    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = v5.build_parser()
    parser.add_argument(
        "--use-temporal-comparator",
        action="store_true",
        default=True,
        help="Use learnable temporal comparator alpha",
    )
    parser.add_argument(
        "--use-predicate-comparator",
        action="store_true",
        default=True,
        help="Use learnable predicate comparator alpha",
    )
    parser.add_argument(
        "--flag-source",
        choices=("stage15_probe_type", "controlled_heuristic", "none"),
        default="controlled_heuristic",
        help="Source for temporal/predicate flags",
    )
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=None,
        help="Max train records (for smoke testing)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: tiny settings, small data",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse_freq", "sqrt_inverse_freq"),
        default="none",
        help="Class weighting mode for CE classification loss (none preserves existing behavior)",
    )
    parser.add_argument(
        "--ood-data",
        type=Path,
        default=None,
        help="Optional OOD probe data for post-training evaluation",
    )
    parser.add_argument(
        "--output-ood-json",
        type=Path,
        default=None,
        help="Path to write OOD evaluation summary JSON",
    )
    parser.add_argument(
        "--output-ood-predictions-json",
        type=Path,
        default=None,
        help="Path to write OOD per-example predictions JSON",
    )
    parser.add_argument(
        "--ood-flag-source",
        choices=("stage15_probe_type", "controlled_heuristic", "none"),
        default=None,
        help="Flag source for OOD evaluation; defaults to --flag-source if not set",
    )
    return parser


def load_ood_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prediction_distribution_from_records(records: list[dict]) -> dict[str, int]:
    """Compute prediction distribution from exported prediction records."""
    from collections import Counter
    predictions = [record.get("pred_final_label") for record in records]
    return dict(sorted(Counter(predictions).items()))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    # Smoke mode overrides
    if args.smoke:
        args.epochs = 2
        args.max_train_records = 16
        print("[SMOKE MODE] epochs=2, max_train_records=16")

    records = v5.load_jsonl(args.data)
    if args.max_train_records is not None:
        records = records[: args.max_train_records]

    train_records, dev_records = v5.split_by_pair_id(
        records, dev_ratio=args.dev_ratio, seed=args.seed
    )

    ce_class_weights = compute_class_weights_v6b(train_records, args.class_weighting, device)
    label_counts: dict[str, int] = {name: 0 for name in v5.ID_TO_FINAL_LABEL.values()}
    for _record in train_records:
        label_counts[_record["final_label"]] += 1

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
        train_bundle = v5.encode_mamba_records(train_records, tokenizer, args.max_length)
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
        f"controlled v6b_minimal | backbone={args.backbone} "
        f"train={len(train_records)} dev={len(dev_records)} "
        f"flag_source={args.flag_source} freeze_encoder={args.freeze_encoder}"
    )

    # Extract flags aligned to train/dev records
    train_temporal_flags, train_predicate_flags = extract_flags(
        train_records, args.flag_source, device
    )
    dev_temporal_flags, dev_predicate_flags = extract_flags(
        dev_records, args.flag_source, device
    )

    # Wrap v5 training to accept flags
    original_run_training = v5.run_training

    def run_training_v6b(
        model,
        train_inputs,
        dev_inputs,
        train_records,
        dev_records,
        train_bundle,
        *,
        epochs,
        lr,
        head_lr,
        encoder_lr,
        weighted_label_loss,
        balanced_sampler,
        use_intervention_loss,
        ranking_weight,
        loss_config,
        seed,
        run_name,
        select_metric="final_macro_f1",
        capture_best_trainable_state=False,
        smoke_mode=False,
        ce_class_weights=None,
    ):
        """Modified run_training that passes flags to v6b model."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        optimizer = v5.build_optimizer(model, lr, head_lr, encoder_lr)
        sampling_generator = torch.Generator().manual_seed(seed)
        best_epoch = 0
        best_score = float("-inf")
        best_dev_metrics = None
        best_dev_interventions = None
        best_dev_pairwise_checks = None
        best_dev_predictions = None
        best_trainable_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # CRITICAL: Pass flags to v6b model forward
            output = model(
                **v5.model_feature_inputs(train_inputs),
                temporal_mismatch_flags=train_temporal_flags,
                predicate_mismatch_flags=train_predicate_flags,
            )

            indices = v5.sample_indices(
                train_inputs["final_labels"], balanced_sampler, sampling_generator
            )
            losses = v5.controlled_losses(
                output, train_inputs, indices,
                False if ce_class_weights is not None else weighted_label_loss,
            )
            if ce_class_weights is not None:
                selected_labels = train_inputs["final_labels"].index_select(0, indices)
                new_label_loss = F.cross_entropy(
                    output["logits"].index_select(0, indices),
                    selected_labels,
                    weight=ce_class_weights,
                )
                non_label_total = losses["total"] - losses["label"]
                losses = dict(losses)
                losses["label"] = new_label_loss
                losses["total"] = non_label_total + new_label_loss

            if use_intervention_loss:
                from contramamba import intervention_pairwise_losses

                # Pairwise losses consume output["logits"] (final logits from v6b)
                pairwise_losses = intervention_pairwise_losses(
                    output,
                    train_bundle["pair_ids"],
                    train_bundle["intervention_types"],
                    train_inputs["final_labels"],
                    **loss_config,
                )
                active_intervention_loss = pairwise_losses["total"]
            else:
                active_intervention_loss = (
                    ranking_weight * v5.intervention_objective(output, train_records)
                )
            total_loss = losses["total"] + active_intervention_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Evaluate with flags
            model.eval()
            with torch.no_grad():
                train_output = model(
                    **train_inputs,
                    temporal_mismatch_flags=train_temporal_flags,
                    predicate_mismatch_flags=train_predicate_flags,
                )
                dev_output = model(
                    **dev_inputs,
                    temporal_mismatch_flags=dev_temporal_flags,
                    predicate_mismatch_flags=dev_predicate_flags,
                )
            train_metrics = v5.compute_metrics(train_output, train_inputs)
            dev_metrics = v5.compute_metrics(dev_output, dev_inputs)

            if select_metric not in dev_metrics or not isinstance(
                dev_metrics[select_metric], (int, float)
            ):
                raise ValueError(f"unsupported select_metric: {select_metric!r}")
            score = float(dev_metrics[select_metric])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_dev_metrics = dev_metrics
                best_dev_interventions = v5.intervention_diagnostics(
                    dev_records, dev_output
                )
                # Skip pairwise checks in smoke mode (may have incomplete variants)
                if not smoke_mode:
                    best_dev_pairwise_checks = v5.pairwise_checks(dev_records, dev_output)
                best_dev_predictions = prediction_records_v6b(dev_records, dev_output)
                if capture_best_trainable_state:
                    best_trainable_state = v5.capture_trainable_state(model)

            print(
                f"run={run_name} "
                + v5.format_epoch(
                    epoch,
                    total_loss.item(),
                    losses,
                    train_metrics,
                    dev_metrics,
                    active_intervention_loss,
                )
            )

        report = {
            "run_name": run_name,
            "final_epoch": epochs,
            "best_epoch": best_epoch,
            "select_metric": select_metric,
            "best_dev_metrics": best_dev_metrics,
            "best_dev_interventions": best_dev_interventions,
            "best_dev_pairwise_checks": best_dev_pairwise_checks,
            "_best_dev_predictions": best_dev_predictions,
            "loss_config": loss_config,
        }
        if best_trainable_state is not None:
            report["best_trainable_state"] = best_trainable_state
        return report

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
        reports[run_name] = run_training_v6b(
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
            smoke_mode=args.smoke,
            ce_class_weights=ce_class_weights,
        )

    # Capture learned alphas
    alpha_temporal = float(model.alpha_temporal().detach()) if model.alpha_temporal_raw else 0.0
    alpha_predicate = float(model.alpha_predicate().detach()) if model.alpha_predicate_raw else 0.0
    temporal_flag_count = int(train_temporal_flags.sum().item())
    predicate_flag_count = int(train_predicate_flags.sum().item())

    report = {
        "configuration": {
            "seed": args.seed,
            "random_seed": args.seed,
            "numpy_seed": args.seed,
            "torch_seed": args.seed,
            "cuda_seed": args.seed if torch.cuda.is_available() else None,
            "data_seed": args.seed,
            "backbone": args.backbone,
            "model_name": args.model_name if args.backbone == "mamba" else None,
            "freeze_encoder": args.freeze_encoder,
            "freeze_a_log": args.freeze_a_log,
            "weighted_label_loss": args.weighted_label_loss,
            "balanced_sampler": args.balanced_sampler,
            "use_intervention_loss": args.use_intervention_loss,
            "loss_sweep": args.loss_sweep,
            "model_version": "v6b_minimal",
            "use_temporal_comparator": args.use_temporal_comparator,
            "use_predicate_comparator": args.use_predicate_comparator,
            "flag_source": args.flag_source,
            "alpha_temporal": alpha_temporal,
            "alpha_predicate": alpha_predicate,
            "temporal_flag_count": temporal_flag_count,
            "predicate_flag_count": predicate_flag_count,
            "final_logits_used": True,
            "time_swap_used": False,
            "pairwise_checks_skipped": args.smoke,
            "pairwise_checks_skip_reason": "incomplete smoke subset" if args.smoke else None,
            "class_weighting": args.class_weighting,
            "class_weights": ce_class_weights.tolist() if ce_class_weights is not None else None,
            "class_counts": label_counts,
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
            "model_version": "v6b_minimal",
            "use_temporal_comparator": args.use_temporal_comparator,
            "use_predicate_comparator": args.use_predicate_comparator,
            "flag_source": args.flag_source,
            "alpha_temporal": alpha_temporal,
            "alpha_predicate": alpha_predicate,
            "final_logits_used": True,
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
        distribution = prediction_distribution_from_records(prediction_exports[run_name])
        if len(distribution) == 1:
            collapsed_label = next(iter(distribution))
            print(
                f"WARNING: run {run_name} dev predictions collapsed to "
                f"the single label {collapsed_label}",
                file=sys.stderr,
            )

    if args.ood_data is not None:
        ood_flag_source = args.ood_flag_source if args.ood_flag_source is not None else args.flag_source
        print(f"[OOD EVAL] loading {args.ood_data} flag_source={ood_flag_source}")
        ood_records = load_ood_jsonl(args.ood_data)
        if args.backbone == "dummy":
            ood_bundle = v5.encode_records(ood_records, vocab)
        else:
            ood_bundle = v5.encode_mamba_records(ood_records, tokenizer, args.max_length)
        ood_inputs = v5.move_inputs(ood_bundle["model_inputs"], device)
        ood_seq_len = ood_inputs["input_ids"].shape[1]
        if ood_seq_len < max_length:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                ood_inputs[key] = F.pad(ood_inputs[key], (0, max_length - ood_seq_len), value=0)
        elif ood_seq_len > max_length:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                ood_inputs[key] = ood_inputs[key][:, :max_length]
        ood_temporal_flags, ood_predicate_flags = extract_flags(
            ood_records, ood_flag_source, device
        )
        ood_summary, ood_predictions = evaluate_ood_v6b(
            model, ood_records, ood_inputs, ood_temporal_flags, ood_predicate_flags
        )
        ood_summary["train_flag_source"] = args.flag_source
        ood_summary["ood_flag_source"] = ood_flag_source
        report["ood_evaluation"] = ood_summary
        if args.output_ood_json is not None:
            v5.write_report_json(ood_summary, args.output_ood_json)
        if args.output_ood_predictions_json is not None:
            ood_metadata = {
                "ood_data_path": str(args.ood_data),
                "seed": args.seed,
                "backbone": args.backbone,
                "model_version": "v6b_minimal",
                "train_flag_source": args.flag_source,
                "ood_flag_source": ood_flag_source,
                "final_logits_used": True,
            }
            v5.write_predictions_json(
                args.output_ood_predictions_json, ood_metadata, ood_predictions
            )

    print("\nFINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        v5.write_report_json(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
