"""Build the Stage34-A held-out structured coverage probe.

Diagnostic only. The generated probe is for external evaluation of Stage33
structured coverage generalization and must not be used for training,
calibration, threshold selection, loss computation, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "stage34a_heldout_coverage_probe.jsonl"

LABEL_TO_ID = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}

HELDOUT_PAIRS = [
    ("animals", "dogs", "at the shelter"),
    ("instruments", "violins", "in the orchestra"),
    ("documents", "invoices", "in the archive"),
    ("machines", "turbines", "at the plant"),
    ("rooms", "kitchens", "in the building"),
    ("courses", "laboratory courses", "in the catalog"),
    ("devices", "routers", "on the network"),
    ("roads", "bike lanes", "in the transit plan"),
    ("policies", "privacy policies", "in the handbook"),
    ("messages", "urgent messages", "in the queue"),
    ("accounts", "administrator accounts", "on the platform"),
    ("files", "encrypted files", "in the folder"),
    ("buildings", "libraries", "on the campus"),
    ("regions", "coastal districts", "in the province"),
    ("teams", "youth teams", "in the league"),
    ("products", "refurbished products", "in the warehouse"),
    ("tickets", "priority tickets", "in the support system"),
    ("materials", "recycled materials", "in the shipment"),
    ("medicines", "antibiotics", "in the clinic"),
    ("facilities", "clinics", "in the county"),
]

GROUPS: list[tuple[str, str, str, str]] = [
    ("heldout_all_to_some_support", "SUPPORT", "quantifier", "ENTAILMENT_PRESERVE"),
    ("heldout_some_to_all_not_entitled", "NOT_ENTITLED", "quantifier", "OVERCLAIM_NE"),
    ("heldout_none_to_some_refute", "REFUTE", "negation", "CONTRADICTION_REFUTE"),
    ("heldout_some_to_none_refute", "REFUTE", "negation", "CONTRADICTION_REFUTE"),
    ("heldout_only_to_base_support", "SUPPORT", "exclusive", "ENTAILMENT_PRESERVE"),
    ("heldout_also_to_only_not_entitled", "NOT_ENTITLED", "exclusive", "OVERCLAIM_NE"),
    ("heldout_specific_to_general_support", "SUPPORT", "specific_general", "ENTAILMENT_PRESERVE"),
    ("heldout_general_to_specific_not_entitled", "NOT_ENTITLED", "specific_general", "OVERCLAIM_NE"),
    ("heldout_whole_to_part_support", "SUPPORT", "whole_part", "ENTAILMENT_PRESERVE"),
    ("heldout_part_to_whole_not_entitled", "NOT_ENTITLED", "whole_part", "OVERCLAIM_NE"),
    ("heldout_collection_to_member_support", "SUPPORT", "collection_member", "ENTAILMENT_PRESERVE"),
    ("heldout_member_to_collection_not_entitled", "NOT_ENTITLED", "collection_member", "OVERCLAIM_NE"),
    ("heldout_region_to_subregion_support", "SUPPORT", "region_subregion", "ENTAILMENT_PRESERVE"),
    ("heldout_subregion_to_region_not_entitled", "NOT_ENTITLED", "region_subregion", "OVERCLAIM_NE"),
    ("heldout_category_to_subcategory_support", "SUPPORT", "category_subcategory", "ENTAILMENT_PRESERVE"),
    ("heldout_subcategory_to_category_not_entitled", "NOT_ENTITLED", "category_subcategory", "OVERCLAIM_NE"),
    ("heldout_role_to_specialized_role_support", "SUPPORT", "role_specialized", "ENTAILMENT_PRESERVE"),
    ("heldout_specialized_role_to_role_not_entitled", "NOT_ENTITLED", "role_specialized", "OVERCLAIM_NE"),
    ("heldout_material_to_variant_support", "SUPPORT", "material_variant", "ENTAILMENT_PRESERVE"),
    ("heldout_variant_to_material_not_entitled", "NOT_ENTITLED", "material_variant", "OVERCLAIM_NE"),
]


def row_text(group: str, whole: str, part: str, suffix: str, idx: int) -> tuple[str, str]:
    specific = f"{part} {suffix}"
    general = f"{whole} {suffix}"
    if group == "heldout_all_to_some_support":
        return f"Some {whole} {suffix} passed inspection.", f"All {whole} {suffix} passed inspection."
    if group == "heldout_some_to_all_not_entitled":
        return f"All {whole} {suffix} passed inspection.", f"Some {whole} {suffix} passed inspection."
    if group == "heldout_none_to_some_refute":
        return f"Some {whole} {suffix} passed inspection.", f"No {whole} {suffix} passed inspection."
    if group == "heldout_some_to_none_refute":
        return f"No {whole} {suffix} passed inspection.", f"Some {whole} {suffix} passed inspection."
    if group == "heldout_only_to_base_support":
        return f"The {part} {suffix} were reviewed.", f"Only the {part} {suffix} were reviewed."
    if group == "heldout_also_to_only_not_entitled":
        return f"Only the {part} {suffix} were reviewed.", f"The {part} {suffix} were also reviewed."
    if group == "heldout_specific_to_general_support":
        return f"The {whole} {suffix} were updated.", f"The newly certified {whole} {suffix} were updated."
    if group == "heldout_general_to_specific_not_entitled":
        return f"The newly certified {whole} {suffix} were updated.", f"The {whole} {suffix} were updated."
    if group in {
        "heldout_whole_to_part_support",
        "heldout_collection_to_member_support",
        "heldout_region_to_subregion_support",
        "heldout_category_to_subcategory_support",
        "heldout_role_to_specialized_role_support",
        "heldout_material_to_variant_support",
    }:
        return f"The {specific} passed inspection.", f"All {general} passed inspection."
    if group in {
        "heldout_part_to_whole_not_entitled",
        "heldout_member_to_collection_not_entitled",
        "heldout_subregion_to_region_not_entitled",
        "heldout_subcategory_to_category_not_entitled",
        "heldout_specialized_role_to_role_not_entitled",
        "heldout_variant_to_material_not_entitled",
    }:
        return f"All {general} passed inspection.", f"The {specific} passed inspection."
    raise ValueError(f"Unhandled group {group!r} at index {idx}")


def build_rows(examples_per_group: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_idx, (group, gold, family, route) in enumerate(GROUPS):
        for item_idx in range(examples_per_group):
            whole, part, suffix = HELDOUT_PAIRS[(group_idx + item_idx) % len(HELDOUT_PAIRS)]
            claim, evidence = row_text(group, whole, part, suffix, item_idx)
            row_id = f"stage34a_{group}_{item_idx:02d}"
            rows.append({
                "id": row_id,
                "pair_id": row_id,
                "claim": claim,
                "evidence": evidence,
                "final_label": gold,
                "label": LABEL_TO_ID[gold],
                "gold_label": gold,
                "group": group,
                "stage34_family": family,
                "stage34_relation": f"{whole}->{part}",
                "stage34_expected_route": route,
                "stage34_is_heldout": True,
            })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-group", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_rows(args.examples_per_group)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
