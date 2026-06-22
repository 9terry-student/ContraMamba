"""Overfit ContraMamba-v5 on tiny, handcrafted intervention pairs.

This experiment deliberately uses a learnable dummy token backbone. It is a
structural smoke test, not a benchmark or a claim about generalization.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contramamba import ContraMambaV5, FinalLabel, PolarityLabel  # noqa: E402


FINAL_LABEL_TO_ID = {label.name: int(label) for label in FinalLabel}
POLARITY_LABEL_TO_ID = {label.name: int(label) for label in PolarityLabel}
ID_TO_FINAL_LABEL = {value: key for key, value in FINAL_LABEL_TO_ID.items()}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DummyTokenBackbone(nn.Module):
    """Small order-invariant token encoder used only by this toy experiment."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=self.token_mlp(self.embedding(input_ids))
        )


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not 12 <= len(records) <= 30:
        raise ValueError("toy dataset must contain 12–30 records")
    required_interventions = {
        "none",
        "paraphrase",
        "entity_swap",
        "predicate_swap",
        "evidence_deletion",
        "polarity_flip",
    }
    found = {record["intervention_type"] for record in records}
    if found != required_interventions:
        raise ValueError(f"unexpected intervention set: {sorted(found)}")
    return records


def build_vocab(records: list[dict]) -> dict[str, int]:
    words = sorted(
        {
            token
            for record in records
            for field in ("claim", "evidence")
            for token in tokenize(record[field])
        }
    )
    return {"<pad>": 0, "<sep>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def collate(records: list[dict], vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    encoded: list[list[int]] = []
    claim_lengths: list[int] = []
    evidence_lengths: list[int] = []
    for record in records:
        claim = [vocab[token] for token in tokenize(record["claim"])]
        evidence = [vocab[token] for token in tokenize(record["evidence"])]
        encoded.append(claim + [vocab["<sep>"]] + evidence)
        claim_lengths.append(len(claim))
        evidence_lengths.append(len(evidence))

    max_length = max(map(len, encoded))
    batch_size = len(records)
    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    claim_mask = torch.zeros_like(attention_mask)
    evidence_mask = torch.zeros_like(attention_mask)
    for index, token_ids in enumerate(encoded):
        length = len(token_ids)
        claim_length = claim_lengths[index]
        input_ids[index, :length] = torch.tensor(token_ids)
        attention_mask[index, :length] = True
        claim_mask[index, :claim_length] = True
        evidence_start = claim_length + 1
        evidence_mask[index, evidence_start : evidence_start + evidence_lengths[index]] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
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


def paired_indices(records: list[dict]) -> dict[str, dict[str, int]]:
    pairs: dict[str, dict[str, int]] = {}
    for index, record in enumerate(records):
        pairs.setdefault(record["pair_id"], {})[record["intervention_type"]] = index
    return pairs


def intervention_loss(output: dict, pairs: dict[str, dict[str, int]]) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    margin = 0.45
    for variants in pairs.values():
        original = variants["none"]
        paraphrase = variants["paraphrase"]
        entity = variants["entity_swap"]
        predicate = variants["predicate_swap"]
        deletion = variants["evidence_deletion"]
        flip = variants["polarity_flip"]

        terms.append(F.relu(margin - output["frame_logit"][original] + output["frame_logit"][entity]))
        terms.append(F.relu(margin - output["predicate_coverage_logit"][original] + output["predicate_coverage_logit"][predicate]))
        terms.append(F.relu(margin - output["sufficiency_logit"][original] + output["sufficiency_logit"][deletion]))

        for key in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob"):
            terms.append((output[key][original] - output[key][paraphrase]).square())
        terms.append((output["logits"][original] - output["logits"][paraphrase]).square().mean())

        terms.append((output["frame_prob"][original] - output["frame_prob"][predicate]).square())
        terms.append((output["entitlement_prob"][original] - output["entitlement_prob"][flip]).square())

    return torch.stack(terms).mean()


def evaluate_checks(records: list[dict], output: dict, pairs: dict[str, dict[str, int]]) -> dict[str, bool]:
    values = {key: output[key].detach().cpu() for key in (
        "frame_prob",
        "predicate_coverage_prob",
        "sufficiency_prob",
        "entitlement_prob",
        "polarity_margin",
        "predictions",
    )}
    checks: dict[str, list[bool]] = {
        "entity_swap lowers frame": [],
        "predicate_swap preserves frame better and lowers predicate": [],
        "evidence_deletion lowers sufficiency": [],
        "paraphrase preserves gates and label": [],
        "polarity_flip preserves entitlement and reverses margin": [],
    }
    for variants in pairs.values():
        original = variants["none"]
        paraphrase = variants["paraphrase"]
        entity = variants["entity_swap"]
        predicate = variants["predicate_swap"]
        deletion = variants["evidence_deletion"]
        flip = variants["polarity_flip"]

        original_frame = values["frame_prob"][original]
        entity_drop = original_frame - values["frame_prob"][entity]
        predicate_drop = torch.abs(original_frame - values["frame_prob"][predicate])
        checks["entity_swap lowers frame"].append(bool(entity_drop > 0.25))
        checks["predicate_swap preserves frame better and lowers predicate"].append(bool(
            predicate_drop < entity_drop
            and values["predicate_coverage_prob"][original]
            - values["predicate_coverage_prob"][predicate] > 0.25
        ))
        checks["evidence_deletion lowers sufficiency"].append(bool(
            values["sufficiency_prob"][original] - values["sufficiency_prob"][deletion] > 0.25
        ))
        gate_preserved = all(
            torch.equal(values[key][original], values[key][paraphrase])
            for key in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob")
        )
        checks["paraphrase preserves gates and label"].append(bool(
            gate_preserved
            and values["predictions"][original] == values["predictions"][paraphrase]
        ))
        checks["polarity_flip preserves entitlement and reverses margin"].append(bool(
            torch.abs(values["entitlement_prob"][original] - values["entitlement_prob"][flip]) < 0.12
            and values["polarity_margin"][original] > 0
            and values["polarity_margin"][flip] < 0
        ))
    return {name: all(results) for name, results in checks.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "toy_interventions_v5.jsonl"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    records = load_records(args.data)
    vocab = build_vocab(records)
    batch = collate(records, vocab)
    pairs = paired_indices(records)

    backbone = DummyTokenBackbone(len(vocab), hidden_size=32)
    model = ContraMambaV5(
        backbone=backbone,
        frame_size=24,
        predicate_size=24,
        sufficiency_size=24,
        energy_size=16,
        dropout=0.0,
        decision_mode="explicit_product",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Toy records: {len(records)} | vocabulary: {len(vocab)} | steps: {args.steps}")
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        output = model(**batch)
        paired_loss = intervention_loss(output, pairs)
        loss = output["loss"] + 2.0 * paired_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step={step:04d} total={loss.item():.6f} "
                f"supervised={output['loss'].item():.6f} paired={paired_loss.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        output = model(**batch)

    print("\nPer-example outputs")
    for index, record in enumerate(records):
        prediction = ID_TO_FINAL_LABEL[int(output["predictions"][index])]
        print(
            f"{record['id']:<27} "
            f"frame={output['frame_prob'][index].item():.3f} "
            f"predicate={output['predicate_coverage_prob'][index].item():.3f} "
            f"sufficiency={output['sufficiency_prob'][index].item():.3f} "
            f"entitlement={output['entitlement_prob'][index].item():.3f} "
            f"E+={output['positive_energy'][index].item():.3f} "
            f"E-={output['negative_energy'][index].item():.3f} "
            f"prediction={prediction}"
        )

    checks = evaluate_checks(records, output, pairs)
    print("\nIntervention checks")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Toy intervention checks failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
