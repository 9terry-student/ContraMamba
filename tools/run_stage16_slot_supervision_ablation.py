"""Stage16 slot-violation supervision ablation.

This runner tests whether v5 slot failures improve when controlled hard
negatives and optional slot-violation supervision are added to training. It does
not introduce a new architecture; the default v5 path is unchanged unless
Stage16 flags are enabled.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import train_controlled_v5 as v5  # noqa: E402
from tools import run_stage13_ood_v5_vs_v6a as stage13  # noqa: E402


SLOT_TARGETS = {"temporal", "frame", "predicate"}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def choose_group_field(records: list[dict[str, Any]]) -> str | None:
    return stage13.choose_group_field(records)


def slot_violation_kind(record: dict[str, Any]) -> str | None:
    stage14_type = record.get("stage14_probe_type")
    stage15_type = record.get("stage15_probe_type")
    if stage14_type == "temporality_shift" or stage15_type == "temporal_mismatch":
        return "temporal"
    if stage14_type == "frame_swap" or stage15_type in {
        "frame_location_mismatch",
        "frame_role_mismatch",
    }:
        return "frame"
    if stage14_type == "predicate_swap" or stage15_type == "predicate_mismatch":
        return "predicate"
    return None


def normalize_hard_negative(record: dict[str, Any], *, repeat_index: int) -> dict[str, Any]:
    """Make Stage14 probe rows safe for controlled training bookkeeping."""

    row = dict(record)
    original_id = row["id"]
    suffix = f"__stage16_hardneg_r{repeat_index}"
    source_pair_id = row.get("source_pair_id") or row.get("pair_id")
    source_type = row.get("source_intervention_type") or row.get("stage14_probe_type")
    unique_intervention = (
        f"stage16_{row.get('stage14_probe_type', 'probe')}_"
        f"{source_type}_{original_id.replace('__', '_')}_r{repeat_index}"
    )
    row["id"] = f"{original_id}{suffix}"
    row["pair_id"] = source_pair_id
    row["intervention_type"] = unique_intervention
    row["stage16_hard_negative"] = True
    row["stage16_source_id"] = original_id
    row["stage16_slot_violation_kind"] = slot_violation_kind(record)
    return row


def select_hard_negatives(
    *,
    path: Path,
    selected_types: Sequence[str],
    max_per_type: int,
    repeat: int,
    seed: int,
    allowed_source_pair_ids: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if max_per_type < 0:
        raise ValueError("--stage16-hard-negative-max-per-type must be non-negative")
    if repeat < 1:
        raise ValueError("--stage16-hard-negative-train-repeat must be positive")
    if max_per_type == 0 or not selected_types:
        return [], Counter()

    rows = load_jsonl(path)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for probe_type in selected_types:
        candidates = [
            row
            for row in rows
            if row.get("stage14_probe_type") == probe_type
            and (row.get("final_label") or row.get("label")) == "NOT_ENTITLED"
            and (row.get("source_pair_id") or row.get("pair_id")) in allowed_source_pair_ids
        ]
        rng.shuffle(candidates)
        chosen = candidates[:max_per_type]
        counts[probe_type] = len(chosen)
        for repeat_index in range(repeat):
            selected.extend(
                normalize_hard_negative(row, repeat_index=repeat_index)
                for row in chosen
            )
    return selected, counts


def adjusted_loss_config(intervention_weight: float | None) -> dict[str, float]:
    config = dict(stage13.DEFAULT_LOSS_CONFIG)
    if intervention_weight is None:
        return config
    for key in list(config):
        if key.startswith("lambda_"):
            config[key] = config[key] * intervention_weight
    return config


def build_slot_targets(
    train_records: Sequence[dict[str, Any]],
    targets: set[str],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    frame_values = []
    predicate_values = []
    active_values = []
    for record in train_records:
        kind = slot_violation_kind(record)
        frame_active = kind in {"temporal", "frame"} and kind in targets
        predicate_active = kind == "predicate" and kind in targets
        frame_values.append(1.0 if frame_active else 0.0)
        predicate_values.append(1.0 if predicate_active else 0.0)
        active_values.append(1.0 if frame_active or predicate_active else 0.0)
    return {
        "frame": torch.tensor(frame_values, dtype=torch.float32, device=device),
        "predicate": torch.tensor(predicate_values, dtype=torch.float32, device=device),
        "active": torch.tensor(active_values, dtype=torch.float32, device=device),
    }


def slot_violation_loss(
    output: dict[str, Any],
    indices: torch.Tensor,
    slot_targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    selected_frame = slot_targets["frame"].index_select(0, indices)
    selected_predicate = slot_targets["predicate"].index_select(0, indices)
    selected_active = slot_targets["active"].index_select(0, indices)
    if not torch.any(selected_active > 0):
        return output["logits"].sum() * 0.0

    components = []
    if torch.any(selected_frame > 0):
        # Violation probability is modeled as sigmoid(-frame_logit), using the
        # existing frame head without adding a new architecture component.
        components.append(
            F.binary_cross_entropy_with_logits(
                -output["frame_logit"].index_select(0, indices),
                selected_frame,
                reduction="none",
            )
        )
    if torch.any(selected_predicate > 0):
        components.append(
            F.binary_cross_entropy_with_logits(
                -output["predicate_coverage_logit"].index_select(0, indices),
                selected_predicate,
                reduction="none",
            )
        )
    if not components:
        return output["logits"].sum() * 0.0
    stacked = torch.stack(components, dim=0).mean(dim=0)
    return (stacked * selected_active).sum() / selected_active.sum().clamp_min(1.0)


@contextmanager
def stage16_training_patches(
    *,
    train_records: Sequence[dict[str, Any]],
    frame_loss_weight: float | None,
    slot_violation_loss_weight: float,
    slot_targets: set[str],
    device: torch.device,
):
    original_controlled_losses = v5.controlled_losses
    if (
        frame_loss_weight is None
        and slot_violation_loss_weight <= 0
    ):
        yield
        return

    target_tensors = build_slot_targets(train_records, slot_targets, device=device)

    def controlled_losses_with_stage16(
        output: dict[str, Any],
        inputs: dict[str, torch.Tensor],
        indices: torch.Tensor,
        weighted_label_loss: bool,
    ) -> dict[str, torch.Tensor]:
        losses = original_controlled_losses(output, inputs, indices, weighted_label_loss)
        if frame_loss_weight is not None:
            extra_frame = (frame_loss_weight - 1.0) * losses["frame"]
            losses["total"] = losses["total"] + extra_frame
            losses["stage16_frame_weight_extra"] = extra_frame
        if slot_violation_loss_weight > 0:
            slot_loss = slot_violation_loss(output, indices, target_tensors)
            losses["stage16_slot_violation"] = slot_loss
            losses["total"] = losses["total"] + slot_violation_loss_weight * slot_loss
        return losses

    v5.controlled_losses = controlled_losses_with_stage16
    try:
        yield
    finally:
        v5.controlled_losses = original_controlled_losses


@contextmanager
def stage16_intervention_config(intervention_loss_weight: float | None):
    original_config = stage13.DEFAULT_LOSS_CONFIG
    if intervention_loss_weight is None:
        yield
        return
    stage13.DEFAULT_LOSS_CONFIG = adjusted_loss_config(intervention_loss_weight)
    try:
        yield
    finally:
        stage13.DEFAULT_LOSS_CONFIG = original_config


def metadata_update(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    hard_negative_counts: Counter[str],
    injected_count: int,
) -> None:
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "stage16_hard_negative_data": str(args.stage16_hard_negative_data),
            "stage16_hard_negative_enabled": args.stage16_hard_negative_enabled,
            "stage16_hard_negative_types": args.stage16_hard_negative_types,
            "stage16_hard_negative_max_per_type": args.stage16_hard_negative_max_per_type,
            "stage16_hard_negative_train_repeat": args.stage16_hard_negative_train_repeat,
            "stage16_frame_loss_weight": args.stage16_frame_loss_weight,
            "stage16_intervention_loss_weight": args.stage16_intervention_loss_weight,
            "stage16_slot_violation_loss_weight": args.stage16_slot_violation_loss_weight,
            "stage16_slot_violation_targets": args.stage16_slot_violation_targets,
            "stage16_injected_hard_negative_count": injected_count,
            "stage16_injected_hard_negative_counts_by_type": dict(hard_negative_counts),
            "stage16_note": (
                "Hard negatives are injected into train only; clean dev remains unchanged. "
                "Default OOD evaluation uses Stage15 diagnostics derived from Stage14, so "
                "this is a supervision ablation, not final OOD validation."
            ),
        }
    )


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


def write_stage16_summary(
    output_prefix: Path,
    payloads: list[dict[str, Any]],
    *,
    hard_negative_counts: Counter[str],
    injected_count: int,
) -> None:
    rows = []
    group_rows = []
    for payload in payloads:
        meta = payload["metadata"]
        clean = payload["train_dev_report"]["best_dev_metrics"]
        ood = payload["ood_metrics"]
        rows.append(
            {
                "seed": meta["seed"],
                "system": meta["system"],
                "clean_macro_f1": clean["final_macro_f1"],
                "clean_accuracy": clean["final_accuracy"],
                "ood_macro_f1": ood["macro_f1"],
                "ood_accuracy": ood["accuracy"],
                "ood_false_entitled_count": ood["false_entitled_count"],
                "ood_false_entitled_rate": ood["false_entitled_rate"],
            }
        )
        for group, metrics in payload.get("ood_group_metrics", {}).items():
            group_rows.append(
                {
                    "seed": meta["seed"],
                    "system": meta["system"],
                    "group": group,
                    "n": metrics["n"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "false_entitled_count": metrics["false_entitled_count"],
                    "false_entitled_rate": metrics["false_entitled_rate"],
                    "prediction_distribution": json.dumps(
                        metrics["prediction_distribution"],
                        sort_keys=True,
                    ),
                }
            )
    stage13.write_csv(output_prefix.parent / f"{output_prefix.name}_metrics.csv", rows)
    stage13.write_csv(output_prefix.parent / f"{output_prefix.name}_ood_groups.csv", group_rows)
    md = "\n\n".join(
        [
            "# Stage16 Slot Violation Supervision Ablation",
            "## Run summary",
            markdown_table(
                rows,
                [
                    "seed",
                    "system",
                    "clean_macro_f1",
                    "clean_accuracy",
                    "ood_macro_f1",
                    "ood_accuracy",
                    "ood_false_entitled_count",
                    "ood_false_entitled_rate",
                ],
            ),
            "## OOD group metrics",
            markdown_table(
                group_rows,
                [
                    "seed",
                    "system",
                    "group",
                    "n",
                    "accuracy",
                    "macro_f1",
                    "false_entitled_count",
                    "false_entitled_rate",
                    "prediction_distribution",
                ],
            ),
            "## Stage16 configuration",
            f"Injected hard negatives: {injected_count}",
            f"Injected hard negatives by type: {dict(hard_negative_counts)}",
            "Clean dev is unchanged. Hard negatives are train-only. Stage15 OOD evaluation is diagnostic and derived from Stage14, so this should be interpreted as a supervision ablation rather than final OOD validation.",
        ]
    )
    path = output_prefix.parent / f"{output_prefix.name}_summary.md"
    path.write_text(md + "\n", encoding="utf-8")
    print(md)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "controlled_v5_v3.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--backbone", choices=("dummy", "mamba"), default="mamba")
    parser.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--system", choices=("v5",), default="v5")
    parser.add_argument("--ood-data", type=Path, default=REPO_ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl")
    parser.add_argument("--output-prefix", type=Path, default=REPO_ROOT / "results" / "stage16_slot_supervision")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-a-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage16-hard-negative-data", type=Path, default=REPO_ROOT / "data" / "stage14_ood_probe_v2.jsonl")
    parser.add_argument("--stage16-hard-negative-types", default="temporality_shift,frame_swap")
    parser.add_argument("--stage16-hard-negative-max-per-type", type=int, default=100)
    parser.add_argument("--stage16-hard-negative-train-repeat", type=int, default=1)
    parser.add_argument("--stage16-frame-loss-weight", type=float, default=None)
    parser.add_argument("--stage16-intervention-loss-weight", type=float, default=None)
    parser.add_argument("--stage16-slot-violation-loss-weight", type=float, default=0.0)
    parser.add_argument("--stage16-slot-violation-targets", default="temporal,frame")
    args = parser.parse_args(argv)
    args.stage16_hard_negative_enabled = any(
        item.startswith("--stage16-hard-negative") for item in raw_argv
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage16_slot_violation_loss_weight < 0:
        raise ValueError("--stage16-slot-violation-loss-weight must be non-negative")
    slot_targets = set(parse_csv(args.stage16_slot_violation_targets))
    unknown_targets = slot_targets - SLOT_TARGETS
    if unknown_targets:
        raise ValueError(f"unknown slot violation targets: {sorted(unknown_targets)}")

    torch.set_num_threads(1)
    device = stage13.resolve_device(args.device)
    clean_records = stage13.filter_time_swap(v5.load_jsonl(args.data))
    ood_records = stage13.load_ood_jsonl(args.ood_data)
    hard_negative_types = (
        parse_csv(args.stage16_hard_negative_types)
        if args.stage16_hard_negative_enabled
        else []
    )
    all_payloads: list[dict[str, Any]] = []
    aggregate_hard_counts: Counter[str] = Counter()
    aggregate_injected = 0

    print(
        "Stage16 config: "
        f"system={args.system} backbone={args.backbone} epochs={args.epochs} "
        f"hard_negative_enabled={args.stage16_hard_negative_enabled} "
        f"hard_negative_types={hard_negative_types} "
        f"slot_loss_weight={args.stage16_slot_violation_loss_weight} "
        f"slot_targets={sorted(slot_targets)}"
    )
    print(f"Loaded OOD records: {len(ood_records)} from {args.ood_data}")
    print(f"OOD group field: {choose_group_field(ood_records)}")

    for seed in args.seeds:
        print(f"\n=== Stage16 seed {seed} ===")
        train_records, dev_records = v5.split_by_pair_id(
            clean_records,
            dev_ratio=args.dev_ratio,
            seed=seed,
        )
        train_pair_ids = {record["pair_id"] for record in train_records}
        hard_negatives, hard_counts = select_hard_negatives(
            path=args.stage16_hard_negative_data,
            selected_types=hard_negative_types,
            max_per_type=(
                args.stage16_hard_negative_max_per_type
                if args.stage16_hard_negative_enabled
                else 0
            ),
            repeat=args.stage16_hard_negative_train_repeat,
            seed=seed,
            allowed_source_pair_ids=train_pair_ids,
        )
        train_records_augmented = train_records + hard_negatives
        all_training_records_for_vocab = clean_records + hard_negatives
        aggregate_hard_counts.update(hard_counts)
        aggregate_injected += len(hard_negatives)
        print(f"Injected hard negatives seed={seed}: {len(hard_negatives)} {dict(hard_counts)}")
        print(
            f"split seed={seed}: train_clean={len(train_records)} "
            f"train_augmented={len(train_records_augmented)} dev_clean={len(dev_records)} "
            f"ood={len(ood_records)}"
        )

        output_prefix = args.output_prefix
        with stage16_training_patches(
            train_records=train_records_augmented,
            frame_loss_weight=args.stage16_frame_loss_weight,
            slot_violation_loss_weight=args.stage16_slot_violation_loss_weight,
            slot_targets=slot_targets,
            device=device,
        ), stage16_intervention_config(args.stage16_intervention_loss_weight):
            payload, _predictions = stage13.train_and_eval_system(
                system=args.system,
                seed=seed,
                records=all_training_records_for_vocab,
                train_records=train_records_augmented,
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
                ranking_weight=args.ranking_weight,
                freeze_encoder=args.freeze_encoder,
                freeze_a_log=args.freeze_a_log,
                output_prefix=output_prefix,
                log_v6a_diagnostics=False,
                v6a_correction_scale=1.0,
                v6a_correction_l2=0.0,
                v6a_learnable_correction_scale=False,
                v6a_learnable_correction_init=0.1,
                v6a_product_gated_correction=False,
                v6a_gate_hidden_dim=16,
                v6a_gate_detach_features=True,
                v6a_gate_trust_loss_weight=0.0,
                v6a_gate_trust_margin=0.2,
                v6a_gate_trust_mode="none",
            )
        metadata_update(
            payload,
            args=args,
            hard_negative_counts=hard_counts,
            injected_count=len(hard_negatives),
        )
        metrics_path = output_prefix.parent / f"{output_prefix.name}_{args.system}_seed{seed}.json"
        stage13.write_json(metrics_path, payload)
        preds_path = output_prefix.parent / f"{output_prefix.name}_{args.system}_seed{seed}_preds.json"
        if preds_path.exists():
            preds_payload = json.loads(preds_path.read_text(encoding="utf-8"))
            preds_payload["metadata"] = dict(payload["metadata"])
            stage13.write_json(preds_path, preds_payload)
        all_payloads.append(payload)

    write_stage16_summary(
        args.output_prefix,
        all_payloads,
        hard_negative_counts=aggregate_hard_counts,
        injected_count=aggregate_injected,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
