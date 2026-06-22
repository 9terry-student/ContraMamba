"""Stage 4A: train ContraMamba-v5 on the controlled seed with a dummy backbone."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba import (  # noqa: E402
    ContraMambaV5,
    FinalLabel,
    PolarityLabel,
    intervention_pairwise_losses,
)
from scripts.build_controlled_v5 import load_jsonl, split_by_pair_id  # noqa: E402


FINAL_LABEL_TO_ID = {label.name: int(label) for label in FinalLabel}
POLARITY_LABEL_TO_ID = {label.name: int(label) for label in PolarityLabel}
ID_TO_FINAL_LABEL = {value: key for key, value in FINAL_LABEL_TO_ID.items()}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "claim_mask",
    "evidence_mask",
    "final_labels",
    "frame_compatible_labels",
    "predicate_covered_labels",
    "sufficiency_labels",
    "polarity_labels",
}
MODEL_FEATURE_KEYS = {"input_ids", "attention_mask", "claim_mask", "evidence_mask"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


class ControlledDummyBackbone(nn.Module):
    """Learnable token/position encoder used only for controlled Stage 4A."""

    def __init__(self, vocab_size: int, hidden_size: int, max_length: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # Keep train and held-out vocabulary items in the same deterministic
        # random-feature space. Only the shared encoder and v5 heads learn.
        self.token_embedding.weight.requires_grad_(False)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        states = self.token_embedding(input_ids) + self.position_embedding(positions)
        return SimpleNamespace(last_hidden_state=self.encoder(states))


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(records: list[dict]) -> dict[str, int]:
    tokens = sorted(
        {
            token
            for record in records
            for field in ("claim", "evidence")
            for token in tokenize(record[field])
        }
    )
    return {
        "<pad>": 0,
        "<unk>": 1,
        "<sep>": 2,
        **{token: index + 3 for index, token in enumerate(tokens)},
    }


def encode_records(records: list[dict], vocab: dict[str, int]) -> dict[str, Any]:
    encoded: list[list[int]] = []
    claim_lengths: list[int] = []
    evidence_lengths: list[int] = []
    for record in records:
        claim_tokens = tokenize(record["claim"])
        evidence_tokens = tokenize(record["evidence"])
        claim_ids = [vocab.get(token, vocab["<unk>"]) for token in claim_tokens]
        evidence_ids = [vocab.get(token, vocab["<unk>"]) for token in evidence_tokens]
        encoded.append(claim_ids + [vocab["<sep>"]] + evidence_ids)
        claim_lengths.append(len(claim_ids))
        evidence_lengths.append(len(evidence_ids))

    max_length = max(map(len, encoded))
    size = len(records)
    input_ids = torch.zeros(size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(size, max_length, dtype=torch.bool)
    claim_mask = torch.zeros_like(attention_mask)
    evidence_mask = torch.zeros_like(attention_mask)
    for index, ids in enumerate(encoded):
        length = len(ids)
        claim_length = claim_lengths[index]
        evidence_start = claim_length + 1
        input_ids[index, :length] = torch.tensor(ids)
        attention_mask[index, :length] = True
        claim_mask[index, :claim_length] = True
        evidence_mask[index, evidence_start : evidence_start + evidence_lengths[index]] = True

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
        "final_labels": torch.tensor(
            [FINAL_LABEL_TO_ID[record["final_label"]] for record in records]
        ),
        "frame_compatible_labels": torch.tensor(
            [record["frame_compatible_label"] for record in records],
            dtype=torch.float32,
        ),
        "predicate_covered_labels": torch.tensor(
            [record["predicate_covered_label"] for record in records],
            dtype=torch.float32,
        ),
        "sufficiency_labels": torch.tensor(
            [record["sufficiency_label"] for record in records],
            dtype=torch.float32,
        ),
        "polarity_labels": torch.tensor(
            [POLARITY_LABEL_TO_ID[record["polarity_label"]] for record in records]
        ),
    }
    assert set(model_inputs) == MODEL_INPUT_KEYS
    return {
        "model_inputs": model_inputs,
        "pair_ids": [record["pair_id"] for record in records],
        "intervention_types": [record["intervention_type"] for record in records],
    }


def encode_mamba_records(
    records: list[dict], tokenizer: Any, max_length: int = 128
) -> dict[str, Any]:
    """Tokenize claim/evidence separately so pair masks remain exact."""

    separator_id = tokenizer.eos_token_id
    if separator_id is None:
        separator_id = tokenizer.pad_token_id
    if separator_id is None:
        raise ValueError("Mamba tokenizer requires an eos or pad token id")

    encoded: list[list[int]] = []
    claim_lengths: list[int] = []
    evidence_lengths: list[int] = []
    claim_budget = max(1, (max_length - 1) // 2)
    evidence_budget = max_length - claim_budget - 1
    for record in records:
        claim_ids = tokenizer.encode(
            record["claim"], add_special_tokens=False, truncation=True, max_length=claim_budget
        )
        evidence_ids = tokenizer.encode(
            record["evidence"],
            add_special_tokens=False,
            truncation=True,
            max_length=evidence_budget,
        )
        if not claim_ids or not evidence_ids:
            raise ValueError(f"tokenization produced an empty span for {record['id']}")
        encoded.append(claim_ids + [separator_id] + evidence_ids)
        claim_lengths.append(len(claim_ids))
        evidence_lengths.append(len(evidence_ids))

    input_ids = torch.full(
        (len(records), max_length),
        fill_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else separator_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(len(records), max_length, dtype=torch.bool)
    claim_mask = torch.zeros_like(attention_mask)
    evidence_mask = torch.zeros_like(attention_mask)
    for index, ids in enumerate(encoded):
        length = len(ids)
        claim_length = claim_lengths[index]
        evidence_start = claim_length + 1
        input_ids[index, :length] = torch.tensor(ids)
        attention_mask[index, :length] = True
        claim_mask[index, :claim_length] = True
        evidence_mask[index, evidence_start : evidence_start + evidence_lengths[index]] = True

    labels = encode_label_tensors(records)
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
        **labels,
    }
    assert set(model_inputs) == MODEL_INPUT_KEYS
    return {
        "model_inputs": model_inputs,
        "pair_ids": [record["pair_id"] for record in records],
        "intervention_types": [record["intervention_type"] for record in records],
    }


def encode_label_tensors(records: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "final_labels": torch.tensor(
            [FINAL_LABEL_TO_ID[record["final_label"]] for record in records]
        ),
        "frame_compatible_labels": torch.tensor(
            [record["frame_compatible_label"] for record in records], dtype=torch.float32
        ),
        "predicate_covered_labels": torch.tensor(
            [record["predicate_covered_label"] for record in records], dtype=torch.float32
        ),
        "sufficiency_labels": torch.tensor(
            [record["sufficiency_label"] for record in records], dtype=torch.float32
        ),
        "polarity_labels": torch.tensor(
            [POLARITY_LABEL_TO_ID[record["polarity_label"]] for record in records]
        ),
    }


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def model_feature_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {key: inputs[key] for key in MODEL_FEATURE_KEYS}
    if "encoder_hidden_states" in inputs:
        result["encoder_hidden_states"] = inputs["encoder_hidden_states"]
    return result


def cache_frozen_encoder_states(
    model: ContraMambaV5,
    inputs: dict[str, torch.Tensor],
    batch_size: int = 8,
) -> None:
    if any(parameter.requires_grad for parameter in model.mamba.parameters()):
        raise ValueError("encoder caching requires a fully frozen encoder")
    model.mamba.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs["input_ids"].shape[0], batch_size):
            ids = inputs["input_ids"][start : start + batch_size]
            chunks.append(model.mamba(input_ids=ids).last_hidden_state)
    inputs["encoder_hidden_states"] = torch.cat(chunks, dim=0)


def pair_index(records: list[dict]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for index, record in enumerate(records):
        result[record["pair_id"]][record["intervention_type"]] = index
    return dict(result)


def intervention_objective(output: dict[str, Any], records: list[dict]) -> torch.Tensor:
    """Output-only ranking/preservation objective; metadata is never a model input."""

    terms: list[torch.Tensor] = []
    margin = 0.5
    for variants in pair_index(records).values():
        original = variants["none"]
        paraphrase = variants["paraphrase"]
        entity = variants["entity_swap"]
        event = variants["event_swap"]
        predicate = variants["predicate_swap"]
        deletion = variants["evidence_deletion"]
        truncation = variants["evidence_truncation"]
        flip = variants["polarity_flip"]

        for changed in (entity, event):
            terms.append(
                F.relu(
                    margin
                    - output["frame_logit"][original]
                    + output["frame_logit"][changed]
                )
            )
        terms.append(
            F.relu(
                margin
                - output["predicate_coverage_logit"][original]
                + output["predicate_coverage_logit"][predicate]
            )
        )
        for changed in (deletion, truncation):
            terms.append(
                F.relu(
                    margin
                    - output["sufficiency_logit"][original]
                    + output["sufficiency_logit"][changed]
                )
            )

        for key in (
            "frame_prob",
            "predicate_coverage_prob",
            "sufficiency_prob",
            "entitlement_prob",
        ):
            terms.append((output[key][original] - output[key][paraphrase]).square())
        terms.append(
            (output["logits"][original] - output["logits"][paraphrase])
            .square()
            .mean()
        )
        terms.append(
            (output["frame_prob"][original] - output["frame_prob"][predicate]).square()
        )
        terms.append(
            (output["entitlement_prob"][original] - output["entitlement_prob"][flip]).square()
        )
    return torch.stack(terms).mean()


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=len(FinalLabel)).float().clamp_min(1.0)
    weights = labels.numel() / (len(FinalLabel) * counts)
    return weights / weights.mean()


def sample_indices(
    labels: torch.Tensor,
    balanced: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    if not balanced:
        return torch.arange(labels.shape[0], device=labels.device)
    counts = torch.bincount(labels.detach().cpu(), minlength=len(FinalLabel)).float()
    example_weights = counts.clamp_min(1.0).reciprocal()[labels.detach().cpu()]
    sampled = torch.multinomial(
        example_weights,
        num_samples=labels.shape[0],
        replacement=True,
        generator=generator,
    )
    return sampled.to(labels.device)


def controlled_losses(
    output: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    indices: torch.Tensor,
    weighted_label_loss: bool,
) -> dict[str, torch.Tensor]:
    selected_labels = inputs["final_labels"].index_select(0, indices)
    weights = class_weights(inputs["final_labels"]).to(output["logits"].device)
    label_loss = F.cross_entropy(
        output["logits"].index_select(0, indices),
        selected_labels,
        weight=weights if weighted_label_loss else None,
    )
    frame_loss = F.binary_cross_entropy_with_logits(
        output["frame_logit"],
        inputs["frame_compatible_labels"],
    )
    predicate_loss = F.binary_cross_entropy_with_logits(
        output["predicate_coverage_logit"],
        inputs["predicate_covered_labels"],
    )
    sufficiency_loss = F.binary_cross_entropy_with_logits(
        output["sufficiency_logit"],
        inputs["sufficiency_labels"],
    )
    entitled = inputs["final_labels"] != int(FinalLabel.NOT_ENTITLED)
    if torch.any(entitled):
        polarity_logits = torch.stack(
            [
                torch.zeros_like(output["negative_energy"]),
                output["negative_energy"],
                output["positive_energy"],
            ],
            dim=-1,
        )
        polarity_loss = F.cross_entropy(
            polarity_logits[entitled],
            inputs["polarity_labels"][entitled],
        )
    else:
        polarity_loss = output["logits"].sum() * 0.0
    total = label_loss + frame_loss + predicate_loss + sufficiency_loss + polarity_loss
    return {
        "total": total,
        "label": label_loss,
        "frame": frame_loss,
        "predicate": predicate_loss,
        "sufficiency": sufficiency_loss,
        "polarity": polarity_loss,
    }


def compute_metrics(output: dict[str, Any], inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    predictions = output["predictions"]
    entitled = inputs["final_labels"] != int(FinalLabel.NOT_ENTITLED)
    polarity_predictions = torch.where(
        output["polarity_margin"] >= 0,
        torch.full_like(predictions, int(PolarityLabel.SUPPORT)),
        torch.full_like(predictions, int(PolarityLabel.REFUTE)),
    )

    def binary_accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
        return ((probabilities >= 0.5) == labels.bool()).float().mean().item()

    polarity_accuracy = (
        (polarity_predictions[entitled] == inputs["polarity_labels"][entitled])
        .float()
        .mean()
        .item()
        if torch.any(entitled)
        else float("nan")
    )
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in FinalLabel:
        label_id = int(label)
        predicted = predictions == label_id
        actual = inputs["final_labels"] == label_id
        true_positive = (predicted & actual).sum().item()
        precision_denominator = predicted.sum().item()
        recall_denominator = actual.sum().item()
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label.name] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)

    return {
        "final_accuracy": (predictions == inputs["final_labels"]).float().mean().item(),
        "final_macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
        "frame_accuracy": binary_accuracy(
            output["frame_prob"], inputs["frame_compatible_labels"]
        ),
        "predicate_accuracy": binary_accuracy(
            output["predicate_coverage_prob"], inputs["predicate_covered_labels"]
        ),
        "sufficiency_accuracy": binary_accuracy(
            output["sufficiency_prob"], inputs["sufficiency_labels"]
        ),
        "polarity_accuracy_entitled": polarity_accuracy,
        "prediction_distribution": final_prediction_distribution(output),
    }


def intervention_diagnostics(
    records: list[dict], output: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["intervention_type"]].append(index)
    result: dict[str, dict[str, Any]] = {}
    for intervention, indices in sorted(grouped.items()):
        index_tensor = torch.tensor(indices, device=output["logits"].device)
        predictions = output["predictions"].index_select(0, index_tensor).detach().cpu()
        distribution = Counter(ID_TO_FINAL_LABEL[int(value)] for value in predictions)
        result[intervention] = {
            key: output[key].index_select(0, index_tensor).mean().item()
            for key in (
                "frame_prob",
                "predicate_coverage_prob",
                "sufficiency_prob",
                "entitlement_prob",
                "polarity_margin",
            )
        }
        result[intervention]["prediction_distribution"] = dict(sorted(distribution.items()))
    return result


def final_prediction_distribution(output: dict[str, Any]) -> dict[str, int]:
    predictions = output["predictions"].detach().cpu()
    return dict(
        sorted(Counter(ID_TO_FINAL_LABEL[int(value)] for value in predictions).items())
    )


def pairwise_checks(records: list[dict], output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = {
        key: output[key].detach().cpu()
        for key in (
            "frame_prob",
            "predicate_coverage_prob",
            "sufficiency_prob",
            "entitlement_prob",
            "polarity_margin",
            "predictions",
        )
    }
    accumulators: dict[str, list[float]] = defaultdict(list)
    booleans: dict[str, list[bool]] = defaultdict(list)
    for variants in pair_index(records).values():
        original = variants["none"]
        paraphrase = variants["paraphrase"]
        entity = variants["entity_swap"]
        event = variants["event_swap"]
        predicate = variants["predicate_swap"]
        deletion = variants["evidence_deletion"]
        truncation = variants["evidence_truncation"]
        flip = variants["polarity_flip"]

        entity_drop = (values["frame_prob"][original] - values["frame_prob"][entity]).item()
        event_drop = (values["frame_prob"][original] - values["frame_prob"][event]).item()
        predicate_frame_delta = torch.abs(
            values["frame_prob"][original] - values["frame_prob"][predicate]
        ).item()
        predicate_drop = (
            values["predicate_coverage_prob"][original]
            - values["predicate_coverage_prob"][predicate]
        ).item()
        deletion_drop = (
            values["sufficiency_prob"][original]
            - values["sufficiency_prob"][deletion]
        ).item()
        truncation_drop = (
            values["sufficiency_prob"][original]
            - values["sufficiency_prob"][truncation]
        ).item()
        paraphrase_gate_delta = max(
            abs((values[key][original] - values[key][paraphrase]).item())
            for key in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob")
        )
        flip_entitlement_delta = abs(
            (values["entitlement_prob"][original] - values["entitlement_prob"][flip]).item()
        )
        margin_reversed = bool(
            values["polarity_margin"][original] * values["polarity_margin"][flip] < 0
        )

        accumulators["entity_frame_drop"].append(entity_drop)
        accumulators["event_frame_drop"].append(event_drop)
        accumulators["predicate_frame_delta"].append(predicate_frame_delta)
        accumulators["predicate_coverage_drop"].append(predicate_drop)
        accumulators["deletion_sufficiency_drop"].append(deletion_drop)
        accumulators["truncation_sufficiency_drop"].append(truncation_drop)
        accumulators["paraphrase_gate_delta"].append(paraphrase_gate_delta)
        accumulators["flip_entitlement_delta"].append(flip_entitlement_delta)
        booleans["entity_frame_lower"].append(entity_drop > 0)
        booleans["event_frame_lower"].append(event_drop > 0)
        booleans["predicate_disentangled"].append(
            predicate_frame_delta < 0.5 * entity_drop and predicate_drop > 0
        )
        booleans["deletion_sufficiency_lower"].append(deletion_drop > 0)
        booleans["truncation_sufficiency_lower"].append(truncation_drop > 0)
        booleans["paraphrase_preserved"].append(
            paraphrase_gate_delta < 0.1
            and bool(
                values["predictions"][original]
                == values["predictions"][paraphrase]
            )
        )
        booleans["polarity_flip_preserved_and_reversed"].append(
            flip_entitlement_delta < 0.15 and margin_reversed
        )

    return {
        name: {
            "pass_rate": sum(results) / len(results),
            "passed": all(results),
        }
        for name, results in booleans.items()
    } | {
        name: {"mean": sum(results) / len(results)}
        for name, results in accumulators.items()
    }


def evaluate(
    model: ContraMambaV5,
    inputs: dict[str, torch.Tensor],
    records: list[dict],
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    return output, compute_metrics(output, inputs)


def build_model(vocab_size: int, max_length: int, hidden_size: int = 48) -> ContraMambaV5:
    backbone = ControlledDummyBackbone(vocab_size, hidden_size, max_length)
    return ContraMambaV5(
        backbone=backbone,
        frame_size=32,
        predicate_size=32,
        sufficiency_size=32,
        energy_size=24,
        dropout=0.0,
        decision_mode="explicit_product",
    )


def build_mamba_model(
    model_name: str,
    freeze_encoder: bool,
    freeze_a_log: bool,
) -> ContraMambaV5:
    model = ContraMambaV5(
        model_name=model_name,
        frame_size=128,
        predicate_size=128,
        sufficiency_size=128,
        energy_size=64,
        dropout=0.1,
        freeze_a_log=freeze_a_log,
        decision_mode="explicit_product",
    )
    for parameter in model.mamba.parameters():
        parameter.requires_grad = not freeze_encoder
    if freeze_a_log:
        for name, parameter in model.mamba.named_parameters():
            if "A_log" in name:
                parameter.requires_grad = False
    return model


def build_optimizer(
    model: ContraMambaV5,
    lr: float,
    head_lr: float | None,
    encoder_lr: float | None,
) -> torch.optim.Optimizer:
    encoder_parameters = [
        parameter for parameter in model.mamba.parameters() if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in model.mamba.parameters()}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups: list[dict[str, Any]] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_lr or lr})
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr or lr})
    if not groups:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def format_epoch(
    epoch: int,
    total_loss: float,
    losses: dict[str, torch.Tensor],
    train_metrics: dict[str, Any],
    dev_metrics: dict[str, Any],
    intervention_loss: torch.Tensor,
) -> str:
    return (
        f"epoch={epoch:03d} total={total_loss:.4f} "
        f"label={losses['label'].item():.4f} "
        f"frame={losses['frame'].item():.4f} "
        f"predicate={losses['predicate'].item():.4f} "
        f"sufficiency={losses['sufficiency'].item():.4f} "
        f"polarity={losses['polarity'].item():.4f} "
        f"intervention={intervention_loss.item():.4f} | "
        f"train final={train_metrics['final_accuracy']:.3f} "
        f"macroF1={train_metrics['final_macro_f1']:.3f} "
        f"frame={train_metrics['frame_accuracy']:.3f} "
        f"predicate={train_metrics['predicate_accuracy']:.3f} "
        f"sufficiency={train_metrics['sufficiency_accuracy']:.3f} "
        f"polarity={train_metrics['polarity_accuracy_entitled']:.3f} | "
        f"dev final={dev_metrics['final_accuracy']:.3f} "
        f"macroF1={dev_metrics['final_macro_f1']:.3f} "
        f"frame={dev_metrics['frame_accuracy']:.3f} "
        f"predicate={dev_metrics['predicate_accuracy']:.3f} "
        f"sufficiency={dev_metrics['sufficiency_accuracy']:.3f} "
        f"polarity={dev_metrics['polarity_accuracy_entitled']:.3f}"
    )


def capture_head_state(model: ContraMambaV5) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("mamba.")
    }


def restore_head_state(
    model: ContraMambaV5, head_state: dict[str, torch.Tensor]
) -> None:
    current = model.state_dict()
    with torch.no_grad():
        for name, value in head_state.items():
            current[name].copy_(value)


def run_training(
    model: ContraMambaV5,
    train_inputs: dict[str, torch.Tensor],
    dev_inputs: dict[str, torch.Tensor],
    train_records: list[dict],
    dev_records: list[dict],
    train_bundle: dict[str, Any],
    *,
    epochs: int,
    lr: float,
    head_lr: float | None,
    encoder_lr: float | None,
    weighted_label_loss: bool,
    balanced_sampler: bool,
    use_intervention_loss: bool,
    ranking_weight: float,
    loss_config: dict[str, float],
    seed: int,
    run_name: str,
) -> dict[str, Any]:
    optimizer = build_optimizer(model, lr, head_lr, encoder_lr)
    sampling_generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(**model_feature_inputs(train_inputs))
        indices = sample_indices(
            train_inputs["final_labels"], balanced_sampler, sampling_generator
        )
        losses = controlled_losses(
            output, train_inputs, indices, weighted_label_loss
        )
        if use_intervention_loss:
            pairwise_losses = intervention_pairwise_losses(
                output,
                train_bundle["pair_ids"],
                train_bundle["intervention_types"],
                **loss_config,
            )
            active_intervention_loss = pairwise_losses["total"]
        else:
            active_intervention_loss = (
                ranking_weight * intervention_objective(output, train_records)
            )
        total_loss = losses["total"] + active_intervention_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_output, train_metrics = evaluate(model, train_inputs, train_records)
        _, dev_metrics = evaluate(model, dev_inputs, dev_records)
        print(
            f"run={run_name} "
            + format_epoch(
                epoch,
                total_loss.item(),
                losses,
                train_metrics,
                dev_metrics,
                active_intervention_loss,
            )
        )

    train_output, train_metrics = evaluate(model, train_inputs, train_records)
    dev_output, dev_metrics = evaluate(model, dev_inputs, dev_records)
    return {
        "run_name": run_name,
        "loss_config": loss_config,
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "train_prediction_distribution": final_prediction_distribution(train_output),
        "dev_prediction_distribution": final_prediction_distribution(dev_output),
        "train_interventions": intervention_diagnostics(train_records, train_output),
        "dev_interventions": intervention_diagnostics(dev_records, dev_output),
        "train_pairwise_checks": pairwise_checks(train_records, train_output),
        "dev_pairwise_checks": pairwise_checks(dev_records, dev_output),
    }


def sweep_presets(ranking_margin: float) -> dict[str, dict[str, float]]:
    shared = {
        "lambda_predicate_contrast": 1.0,
        "lambda_sufficiency_contrast": 1.0,
        "lambda_polarity_flip": 1.0,
        "lambda_polarity_margin_anchor": 1.0,
        "lambda_paraphrase_preserve": 1.0,
        "lambda_entitlement_preserve": 1.0,
        "lambda_logit_preserve": 1.0,
        "ranking_margin": ranking_margin,
        "polarity_margin_min": 1.0,
    }
    return {
        "A_stage4c": {
            **shared,
            "lambda_frame_preserve": 1.0,
            "lambda_frame_anchor": 0.0,
            "lambda_predicate_anchor": 0.0,
        },
        "B_high_frame": {
            **shared,
            "lambda_frame_preserve": 3.0,
            "lambda_frame_anchor": 0.0,
            "lambda_predicate_anchor": 0.0,
        },
        "C_anchor": {
            **shared,
            "lambda_frame_preserve": 1.0,
            "lambda_frame_anchor": 1.0,
            "lambda_predicate_anchor": 1.0,
        },
        "D_anchor_reduced_frame": {
            **shared,
            "lambda_frame_preserve": 0.25,
            "lambda_frame_anchor": 1.0,
            "lambda_predicate_anchor": 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "controlled_v5_seed.jsonl"
    )
    parser.add_argument("--backbone", choices=("dummy", "mamba"), default="dummy")
    parser.add_argument("--model-name", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--weighted-label-loss", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--encoder-lr", type=float)
    parser.add_argument("--freeze-encoder", type=parse_bool, default=True)
    parser.add_argument("--freeze-a-log", type=parse_bool, default=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--use-intervention-loss", action="store_true")
    parser.add_argument("--lambda-frame-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-frame-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-predicate-contrast", type=float, default=1.0)
    parser.add_argument("--lambda-predicate-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-sufficiency-contrast", type=float, default=1.0)
    parser.add_argument("--lambda-polarity-flip", type=float, default=1.0)
    parser.add_argument("--lambda-polarity-margin-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-paraphrase-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-entitlement-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-logit-preserve", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--polarity-margin-min", type=float, default=1.0)
    parser.add_argument("--loss-sweep", action="store_true")
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    records = load_jsonl(args.data)
    train_records, dev_records = split_by_pair_id(
        records, dev_ratio=args.dev_ratio, seed=args.seed
    )
    if args.backbone == "dummy":
        vocab = build_vocab(records)
        train_bundle = encode_records(train_records, vocab)
        dev_bundle = encode_records(dev_records, vocab)
        model = None
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Mamba tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        train_bundle = encode_mamba_records(train_records, tokenizer, args.max_length)
        dev_bundle = encode_mamba_records(dev_records, tokenizer, args.max_length)
        model = build_mamba_model(
            args.model_name,
            freeze_encoder=args.freeze_encoder,
            freeze_a_log=args.freeze_a_log,
        )
    train_inputs = move_inputs(train_bundle["model_inputs"], device)
    dev_inputs = move_inputs(dev_bundle["model_inputs"], device)
    max_length = max(
        train_inputs["input_ids"].shape[1], dev_inputs["input_ids"].shape[1]
    )
    # Pad each split to the shared positional range.
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
        cache_frozen_encoder_states(model, train_inputs)
        cache_frozen_encoder_states(model, dev_inputs)
    print(
        f"controlled Stage 4D | backbone={args.backbone} "
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
        sweep_presets(args.ranking_margin)
        if args.loss_sweep
        else {"single": requested_loss_config}
    )
    initial_head_state = capture_head_state(model)
    reports: dict[str, dict[str, Any]] = {}
    for run_name, loss_config in configurations.items():
        restore_head_state(model, initial_head_state)
        torch.manual_seed(args.seed)
        reports[run_name] = run_training(
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
        },
        "runs": reports,
    }
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


if __name__ == "__main__":
    main()
