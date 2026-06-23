"""Stage 13 OOD validation: v5 clean retrain vs v6A residual.

This runner trains both models on controlled_v5_v3 with the corrupted
``time_swap`` intervention excluded before pair_id splitting, restores each
best dev checkpoint, evaluates an external OOD probe, and writes comparable
prediction/metric reports.

The default OOD probe is the locally available Stage10A number-swap probe. If a
Stage10C surface/temporality probe exists in another environment, pass it with
``--ood-data``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import train_controlled_v5 as v5  # noqa: E402
from scripts import train_controlled_v6a as v6a  # noqa: E402


LABELS = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]
ENTITLED = {"REFUTE", "SUPPORT"}
DEFAULT_LOSS_CONFIG = {
    "lambda_frame_preserve": 1.0,
    "lambda_frame_anchor": 1.0,
    "lambda_predicate_contrast": 1.0,
    "lambda_predicate_anchor": 1.0,
    "lambda_sufficiency_contrast": 1.0,
    "lambda_polarity_flip": 1.0,
    "lambda_polarity_margin_anchor": 1.0,
    "lambda_paraphrase_preserve": 1.0,
    "lambda_entitlement_preserve": 1.0,
    "lambda_logit_preserve": 1.0,
    "ranking_margin": 0.5,
    "polarity_margin_min": 1.0,
}
OOD_GROUP_FIELDS = (
    "ood_group",
    "probe_group",
    "surface_temporality_group",
    "temporality_group",
    "group",
    "intervention_type",
    "primary_failure_type",
)


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def filter_time_swap(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before = len(records)
    filtered = [
        record
        for record in records
        if record.get("intervention_type") != "time_swap"
    ]
    print(f"Excluded interventions=['time_swap']: {before} -> {len(filtered)} records")
    return filtered


def load_ood_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load OOD probes without enforcing controlled-v5 intervention vocabulary."""

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"OOD data is empty: {path}")
    required = {
        "id",
        "pair_id",
        "claim",
        "evidence",
        "final_label",
        "frame_compatible_label",
        "predicate_covered_label",
        "sufficiency_label",
        "polarity_label",
        "intervention_type",
    }
    for index, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"OOD row {index} missing fields: {sorted(missing)}")
    return records


def pad_to_shared_length(input_sets: Sequence[dict[str, torch.Tensor]]) -> int:
    max_length = max(inputs["input_ids"].shape[1] for inputs in input_sets)
    for inputs in input_sets:
        difference = max_length - inputs["input_ids"].shape[1]
        if difference:
            for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
                inputs[key] = F.pad(inputs[key], (0, difference), value=0)
    return max_length


def per_label_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for label in LABELS:
        tp = sum(1 for row in records if row["gold_final_label"] == label and row["pred_final_label"] == label)
        fp = sum(1 for row in records if row["gold_final_label"] != label and row["pred_final_label"] == label)
        fn = sum(1 for row in records if row["gold_final_label"] == label and row["pred_final_label"] != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        result[label] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def prediction_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    per_label = per_label_metrics(predictions)
    correct = sum(row["gold_final_label"] == row["pred_final_label"] for row in predictions)
    false_entitled = [
        row
        for row in predictions
        if row["gold_final_label"] == "NOT_ENTITLED" and row["pred_final_label"] in ENTITLED
    ]
    gold_not_entitled = sum(row["gold_final_label"] == "NOT_ENTITLED" for row in predictions)
    return {
        "n": len(predictions),
        "accuracy": safe_div(correct, len(predictions)),
        "macro_f1": mean(per_label[label]["f1"] for label in LABELS),
        "false_entitled_count": len(false_entitled),
        "false_entitled_rate": safe_div(len(false_entitled), gold_not_entitled),
        "prediction_distribution": dict(Counter(row["pred_final_label"] for row in predictions)),
        "per_label": per_label,
    }


def choose_group_field(records: list[dict[str, Any]]) -> str | None:
    for field in OOD_GROUP_FIELDS:
        if any(field in record for record in records):
            return field
    return None


def by_group_metrics(predictions: list[dict[str, Any]], group_field: str | None) -> dict[str, Any]:
    if group_field is None:
        return {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[str(row.get(group_field, "UNKNOWN"))].append(row)
    return {group: prediction_metrics(rows) for group, rows in sorted(groups.items())}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, metadata: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    write_json(path, {"metadata": metadata, "predictions": predictions})


def prepare_bundles(
    *,
    records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    ood_records: list[dict[str, Any]],
    backbone: str,
    model_name: str,
    max_length: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]], Any]:
    if backbone == "mamba":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        bundles = [
            v5.encode_mamba_records(rows, tokenizer, max_length)
            for rows in (train_records, dev_records, ood_records)
        ]
        vocab = None
    else:
        vocab = v5.build_vocab(records + ood_records)
        bundles = [
            v5.encode_records(rows, vocab)
            for rows in (train_records, dev_records, ood_records)
        ]
    inputs = [v5.move_inputs(bundle["model_inputs"], device) for bundle in bundles]
    pad_to_shared_length(inputs)
    return bundles, inputs, vocab


def build_stage13_model(
    system: str,
    *,
    backbone: str,
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
    vocab: Any,
    train_inputs: dict[str, torch.Tensor],
    dev_inputs: dict[str, torch.Tensor],
    ood_inputs: dict[str, torch.Tensor],
) -> torch.nn.Module:
    if system == "v5":
        if backbone == "mamba":
            model = v5.build_mamba_model(model_name, freeze_encoder, freeze_a_log)
        else:
            max_length = max(train_inputs["input_ids"].shape[1], dev_inputs["input_ids"].shape[1], ood_inputs["input_ids"].shape[1])
            model = v5.build_model(len(vocab), max_length)
    elif system == "v6a":
        if backbone == "mamba":
            model = v6a.build_mamba_model(model_name, freeze_encoder, freeze_a_log)
        else:
            max_length = max(train_inputs["input_ids"].shape[1], dev_inputs["input_ids"].shape[1], ood_inputs["input_ids"].shape[1])
            model = v6a.build_model(len(vocab), max_length)
    else:
        raise ValueError(f"unknown system: {system}")
    return model


def cache_if_needed(model: torch.nn.Module, inputs: Iterable[dict[str, torch.Tensor]], backbone: str) -> None:
    if backbone == "mamba":
        print("Caching frozen Mamba token states...")
        for item in inputs:
            v5.cache_frozen_encoder_states(model, item)


def train_and_eval_system(
    *,
    system: str,
    seed: int,
    records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    ood_records: list[dict[str, Any]],
    backbone: str,
    model_name: str,
    max_length: int,
    device: torch.device,
    epochs: int,
    lr: float,
    head_lr: float | None,
    encoder_lr: float | None,
    freeze_encoder: bool,
    freeze_a_log: bool,
    output_prefix: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    random.seed(seed)
    torch.manual_seed(seed)

    original_prediction_records = v5.prediction_records
    if system == "v6a":
        v5.prediction_records = v6a.prediction_records_v6a
    else:
        v5.prediction_records = original_prediction_records

    bundles, inputs, vocab = prepare_bundles(
        records=records,
        train_records=train_records,
        dev_records=dev_records,
        ood_records=ood_records,
        backbone=backbone,
        model_name=model_name,
        max_length=max_length,
        device=device,
    )
    train_bundle, _dev_bundle, _ood_bundle = bundles
    train_inputs, dev_inputs, ood_inputs = inputs
    model = build_stage13_model(
        system,
        backbone=backbone,
        model_name=model_name,
        freeze_encoder=freeze_encoder,
        freeze_a_log=freeze_a_log,
        vocab=vocab,
        train_inputs=train_inputs,
        dev_inputs=dev_inputs,
        ood_inputs=ood_inputs,
    ).to(device)
    cache_if_needed(model, inputs, backbone)

    report = v5.run_training(
        model,
        train_inputs,
        dev_inputs,
        train_records,
        dev_records,
        train_bundle,
        epochs=epochs,
        lr=lr,
        head_lr=head_lr,
        encoder_lr=encoder_lr,
        weighted_label_loss=True,
        balanced_sampler=True,
        use_intervention_loss=True,
        ranking_weight=2.0,
        loss_config=DEFAULT_LOSS_CONFIG,
        seed=seed,
        run_name=f"stage13_{system}",
        select_metric="final_macro_f1",
        capture_best_trainable_state=True,
    )
    best_state = report.pop("_best_trainable_state")
    if best_state is None:
        raise RuntimeError(f"{system} seed {seed} did not capture best trainable state")
    v5.restore_trainable_state(model, best_state)
    ood_output, _ = v5.evaluate(model, ood_inputs, ood_records)
    prediction_fn = v6a.prediction_records_v6a if system == "v6a" else original_prediction_records
    ood_predictions = prediction_fn(ood_records, ood_output)
    group_field = choose_group_field(ood_records)
    ood_metrics = prediction_metrics(ood_predictions)
    ood_group_metrics = by_group_metrics(ood_predictions, group_field)

    metadata = {
        "system": system,
        "seed": seed,
        "backbone": backbone,
        "model_name": model_name if backbone == "mamba" else None,
        "epochs": epochs,
        "best_epoch": report["best_epoch"],
        "train_data": "data/controlled_v5_v3.jsonl with time_swap excluded before split",
        "ood_group_field": group_field,
        "accepted_v6a": system == "v6a",
    }
    metrics_payload = {
        "metadata": metadata,
        "train_dev_report": report,
        "ood_metrics": ood_metrics,
        "ood_group_metrics": ood_group_metrics,
    }
    metrics_path = output_prefix.parent / f"{output_prefix.name}_{system}_seed{seed}.json"
    preds_path = output_prefix.parent / f"{output_prefix.name}_{system}_seed{seed}_preds.json"
    write_json(metrics_path, metrics_payload)
    write_predictions(preds_path, metadata, ood_predictions)

    v5.prediction_records = original_prediction_records
    return metrics_payload, ood_predictions


def transition_rows(seed: int, v5_predictions: list[dict[str, Any]], v6_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v5_by_id = {row["id"]: row for row in v5_predictions}
    v6_by_id = {row["id"]: row for row in v6_predictions}
    if set(v5_by_id) != set(v6_by_id):
        raise ValueError(f"seed {seed} v5/v6 prediction ids do not match")
    rows = []
    for item_id in sorted(v5_by_id):
        left = v5_by_id[item_id]
        right = v6_by_id[item_id]
        gold = left["gold_final_label"]
        if gold != right["gold_final_label"]:
            raise ValueError(f"gold label mismatch for {item_id}")
        v5_pred = left["pred_final_label"]
        v6_pred = right["pred_final_label"]
        rows.append(
            {
                "seed": seed,
                "id": item_id,
                "pair_id": left["pair_id"],
                "group": left.get(choose_group_field([left]) or "intervention_type", "UNKNOWN"),
                "intervention_type": left.get("intervention_type", "UNKNOWN"),
                "gold_final_label": gold,
                "v5_pred": v5_pred,
                "v6a_pred": v6_pred,
                "v5_wrong_v6a_correct": int(v5_pred != gold and v6_pred == gold),
                "v5_correct_v6a_wrong": int(v5_pred == gold and v6_pred != gold),
                "v5_false_entitled_fixed_by_v6a": int(
                    gold == "NOT_ENTITLED" and v5_pred in ENTITLED and v6_pred == "NOT_ENTITLED"
                ),
                "new_v6a_false_entitled": int(
                    gold == "NOT_ENTITLED" and v6_pred in ENTITLED and v5_pred == "NOT_ENTITLED"
                ),
            }
        )
    return rows


def flatten_metric_rows(all_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in all_metrics:
        metadata = payload["metadata"]
        system = metadata["system"]
        seed = metadata["seed"]
        ood = payload["ood_metrics"]
        rows.append(
            {
                "seed": seed,
                "system": system,
                "group": "__all__",
                "n": ood["n"],
                "accuracy": ood["accuracy"],
                "macro_f1": ood["macro_f1"],
                "false_entitled_count": ood["false_entitled_count"],
                "false_entitled_rate": ood["false_entitled_rate"],
                "prediction_distribution": json.dumps(ood["prediction_distribution"], sort_keys=True),
            }
        )
        for group, metrics in payload["ood_group_metrics"].items():
            rows.append(
                {
                    "seed": seed,
                    "system": system,
                    "group": group,
                    "n": metrics["n"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "false_entitled_count": metrics["false_entitled_count"],
                    "false_entitled_rate": metrics["false_entitled_rate"],
                    "prediction_distribution": json.dumps(metrics["prediction_distribution"], sort_keys=True),
                }
            )
    return rows


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["group"])].append(row)
    output = []
    for (system, group), items in sorted(grouped.items()):
        for metric in ("accuracy", "macro_f1", "false_entitled_count", "false_entitled_rate"):
            values = [float(row[metric]) for row in items]
            output.append(
                {
                    "system": system,
                    "group": group,
                    "metric": metric,
                    "mean": mean(values),
                    "std": sample_std(values),
                    "n": len(values),
                    "formatted": f"{mean(values):.3f} +/- {sample_std(values):.3f}",
                }
            )
    return output


def summarize_transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    output = []
    for group, items in sorted(grouped.items()):
        output.append(
            {
                "group": group,
                "n": len(items),
                "v5_wrong_v6a_correct": sum(int(row["v5_wrong_v6a_correct"]) for row in items),
                "v5_correct_v6a_wrong": sum(int(row["v5_correct_v6a_wrong"]) for row in items),
                "v5_false_entitled_fixed_by_v6a": sum(int(row["v5_false_entitled_fixed_by_v6a"]) for row in items),
                "new_v6a_false_entitled": sum(int(row["new_v6a_false_entitled"]) for row in items),
            }
        )
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(output_prefix: Path, metrics_payloads: list[dict[str, Any]], transition_items: list[dict[str, Any]]) -> None:
    metric_rows = flatten_metric_rows(metrics_payloads)
    aggregate_rows = aggregate_metric_rows(metric_rows)
    transition_summary = summarize_transitions(transition_items)
    write_csv(output_prefix.parent / f"{output_prefix.name}_metrics.csv", metric_rows)
    write_csv(output_prefix.parent / f"{output_prefix.name}_aggregate.csv", aggregate_rows)
    write_csv(output_prefix.parent / f"{output_prefix.name}_transitions.csv", transition_items)
    write_csv(output_prefix.parent / f"{output_prefix.name}_transition_summary.csv", transition_summary)

    md = "\n\n".join(
        [
            "# Stage 13 OOD v5 Clean vs v6A Residual",
            "## Aggregate metrics",
            markdown_table(aggregate_rows, ["system", "group", "metric", "mean", "std", "formatted"]),
            "## Per-seed/group metrics",
            markdown_table(metric_rows, ["seed", "system", "group", "n", "accuracy", "macro_f1", "false_entitled_count", "false_entitled_rate", "prediction_distribution"]),
            "## Transition summary",
            markdown_table(transition_summary, ["group", "n", "v5_wrong_v6a_correct", "v5_correct_v6a_wrong", "v5_false_entitled_fixed_by_v6a", "new_v6a_false_entitled"]),
            "## Notes",
            "This is an OOD-probe validation report. The default local probe is Stage10A number_swap because no Stage10C surface/temporality file is present in this checkout. Use --ood-data to evaluate a Stage10C file when available.",
        ]
    )
    summary_path = output_prefix.parent / f"{output_prefix.name}_v5_vs_v6a_summary.md"
    summary_path.write_text(md + "\n", encoding="utf-8")
    print(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "controlled_v5_v3.jsonl")
    parser.add_argument("--ood-data", type=Path, default=REPO_ROOT / "data" / "stage10a_number_swap_probe.jsonl")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--backbone", choices=("dummy", "mamba"), default="dummy")
    parser.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--head-lr", type=float, default=0.003)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-a-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results" / "stage13_ood")
    return parser.parse_args()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    output_prefix = args.output_prefix
    if args.smoke:
        output_prefix = output_prefix.parent / "stage13_smoke"

    records = filter_time_swap(v5.load_jsonl(args.data))
    ood_records = load_ood_jsonl(args.ood_data)
    print(f"Loaded OOD records: {len(ood_records)} from {args.ood_data}")
    print(f"OOD group field: {choose_group_field(ood_records)}")

    all_metrics: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"\n=== Stage13 seed {seed} ===")
        train_records, dev_records = v5.split_by_pair_id(records, dev_ratio=args.dev_ratio, seed=seed)
        print(f"split seed={seed}: train={len(train_records)} dev={len(dev_records)} ood={len(ood_records)}")
        seed_predictions: dict[str, list[dict[str, Any]]] = {}
        for system in ("v5", "v6a"):
            print(f"\n--- training/evaluating {system} seed={seed} ---")
            payload, predictions = train_and_eval_system(
                system=system,
                seed=seed,
                records=records,
                train_records=train_records,
                dev_records=dev_records,
                ood_records=ood_records,
                backbone=args.backbone,
                model_name=args.model_name,
                max_length=args.max_length,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                head_lr=args.head_lr,
                encoder_lr=args.encoder_lr,
                freeze_encoder=args.freeze_encoder,
                freeze_a_log=args.freeze_a_log,
                output_prefix=output_prefix,
            )
            all_metrics.append(payload)
            seed_predictions[system] = predictions
        all_transitions.extend(transition_rows(seed, seed_predictions["v5"], seed_predictions["v6a"]))

    write_summary(output_prefix, all_metrics, all_transitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
