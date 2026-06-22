"""Create the Stage 10A deterministic number-swap probe dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


NUMBER_SWAP_FINAL_LABEL = "NOT_ENTITLED"
REQUIRED_FIELDS = {
    "id",
    "pair_id",
    "claim",
    "evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "intervention_type",
}
COMPANIES = ("Aster Company", "Beacon Labs", "Cedar Works", "Delta Foods", "Elm Systems")
VERBS = ("sold", "shipped", "produced", "delivered", "ordered")
ITEMS = ("units", "devices", "packages", "components", "tickets")
MONTHS = ("April", "May", "June", "July", "August")


def _record(
    pair_id: str,
    intervention: str,
    claim: str,
    evidence: str,
    final_label: str,
    frame: int,
    polarity: str,
    failure: str,
) -> dict:
    return {
        "id": f"{pair_id}__{intervention}",
        "pair_id": pair_id,
        "claim": claim,
        "evidence": evidence,
        "final_label": final_label,
        "frame_compatible_label": frame,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": polarity,
        "primary_failure_type": failure,
        "intervention_type": intervention,
    }


def build_number_swap_probe(num_pairs: int = 30) -> list[dict]:
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    records = []
    for index in range(num_pairs):
        company = COMPANIES[index % len(COMPANIES)]
        verb = VERBS[(index // len(COMPANIES)) % len(VERBS)]
        item = ITEMS[(index * 2) % len(ITEMS)]
        month = MONTHS[(index * 3) % len(MONTHS)]
        quantity = 100 + 10 * index
        swapped_quantity = quantity + 200
        pair_id = f"number_probe_{index + 1:03d}"
        claim = f"{company} {verb} {quantity} {item} in {month}."
        records.extend(
            [
                _record(pair_id, "none", claim, claim, "SUPPORT", 1, "SUPPORT", "none"),
                _record(
                    pair_id,
                    "number_swap",
                    claim,
                    f"{company} {verb} {swapped_quantity} {item} in {month}.",
                    NUMBER_SWAP_FINAL_LABEL,
                    0,
                    "NONE",
                    "frame",
                ),
            ]
        )
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("number-swap probe generated duplicate ids")
    if any(REQUIRED_FIELDS - set(record) for record in records):
        raise RuntimeError("number-swap probe generated an incomplete record")
    return records


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_jsonl(args.output, build_number_swap_probe(args.num_pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
