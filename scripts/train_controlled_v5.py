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

from contramamba import ContraMambaV5, FinalLabel, PolarityLabel  # noqa: E402
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


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


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


def compute_metrics(output: dict[str, Any], inputs: dict[str, torch.Tensor]) -> dict[str, float]:
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
    return {
        "final_accuracy": (predictions == inputs["final_labels"]).float().mean().item(),
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


def format_epoch(
    epoch: int,
    total_loss: float,
    output: dict[str, Any],
    train_metrics: dict[str, float],
    dev_metrics: dict[str, float],
) -> str:
    return (
        f"epoch={epoch:03d} total={total_loss:.4f} "
        f"label={output['label_loss'].item():.4f} "
        f"frame={output['frame_loss'].item():.4f} "
        f"predicate={output['predicate_loss'].item():.4f} "
        f"sufficiency={output['sufficiency_loss'].item():.4f} "
        f"polarity={output['polarity_loss'].item():.4f} | "
        f"train final={train_metrics['final_accuracy']:.3f} "
        f"frame={train_metrics['frame_accuracy']:.3f} "
        f"predicate={train_metrics['predicate_accuracy']:.3f} "
        f"sufficiency={train_metrics['sufficiency_accuracy']:.3f} "
        f"polarity={train_metrics['polarity_accuracy_entitled']:.3f} | "
        f"dev final={dev_metrics['final_accuracy']:.3f} "
        f"frame={dev_metrics['frame_accuracy']:.3f} "
        f"predicate={dev_metrics['predicate_accuracy']:.3f} "
        f"sufficiency={dev_metrics['sufficiency_accuracy']:.3f} "
        f"polarity={dev_metrics['polarity_accuracy_entitled']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "controlled_v5_seed.jsonl"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
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
    vocab = build_vocab(records)
    train_bundle = encode_records(train_records, vocab)
    dev_bundle = encode_records(dev_records, vocab)
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

    model = build_model(len(vocab), max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(
        f"controlled Stage 4A | train={len(train_records)} ({len(set(train_bundle['pair_ids']))} pairs) "
        f"dev={len(dev_records)} ({len(set(dev_bundle['pair_ids']))} pairs) device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(**train_inputs)
        ranking_loss = intervention_objective(output, train_records)
        total_loss = output["loss"] + args.ranking_weight * ranking_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_output, train_metrics = evaluate(model, train_inputs, train_records)
        _, dev_metrics = evaluate(model, dev_inputs, dev_records)
        print(format_epoch(epoch, total_loss.item(), output, train_metrics, dev_metrics))

    train_output, train_metrics = evaluate(model, train_inputs, train_records)
    dev_output, dev_metrics = evaluate(model, dev_inputs, dev_records)
    report = {
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "train_prediction_distribution": final_prediction_distribution(train_output),
        "dev_prediction_distribution": final_prediction_distribution(dev_output),
        "train_interventions": intervention_diagnostics(train_records, train_output),
        "dev_interventions": intervention_diagnostics(dev_records, dev_output),
        "train_pairwise_checks": pairwise_checks(train_records, train_output),
        "dev_pairwise_checks": pairwise_checks(dev_records, dev_output),
    }
    print("\nFINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
