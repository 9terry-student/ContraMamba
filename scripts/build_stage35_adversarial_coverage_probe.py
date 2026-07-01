"""Build the Stage35-A adversarial held-out structured coverage probe.

Diagnostic only. The generated probe is for external evaluation of structured
coverage robustness and must not be used for training, calibration, threshold
selection, loss computation, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "stage35a_adversarial_coverage_probe.jsonl"

LABEL_TO_ID = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}

ADVERSARIAL_PAIRS = [
    ("workers", "night-shift workers"),
    ("vehicles", "electric vans"),
    ("suppliers", "local suppliers"),
    ("documents", "tax forms"),
    ("hospitals", "pediatric hospitals"),
    ("users", "verified users"),
    ("containers", "refrigerated containers"),
    ("flights", "red-eye flights"),
    ("rooms", "isolation rooms"),
    ("courses", "graduate seminars"),
    ("sensors", "infrared sensors"),
    ("payments", "recurring payments"),
    ("customers", "enterprise customers"),
    ("farms", "organic farms"),
    ("cases", "urgent cases"),
    ("committees", "ethics committees"),
    ("laboratories", "chemistry laboratories"),
    ("packages", "fragile packages"),
    ("licenses", "export licenses"),
    ("vessels", "cargo vessels"),
    ("permits", "temporary permits"),
    ("devices", "wearable devices"),
    ("schools", "rural schools"),
    ("reports", "audit reports"),
    ("requests", "refund requests"),
    ("stations", "charging stations"),
    ("shelters", "emergency shelters"),
    ("warehouses", "cold-storage warehouses"),
    ("teams", "reserve teams"),
    ("medicines", "prescription medicines"),
]

PREDICATES = [
    "affected",
    "covered",
    "delayed",
    "reached",
    "included",
    "excluded",
    "required",
    "approved",
    "denied",
    "suspended",
    "notified",
    "protected",
    "audited",
    "repaired",
    "recalled",
    "updated",
    "blocked",
    "assigned",
    "registered",
    "reimbursed",
]

LOCATIONS = [
    "at the north depot",
    "inside the regional office",
    "near the harbor terminal",
    "within the city program",
    "at the emergency desk",
    "inside the county registry",
]

TIMES = [
    "on Monday",
    "during the winter review",
    "before the quarterly audit",
    "after the policy update",
    "in 2024",
    "during the overnight window",
]

DISTRACTORS = [
    "visitors",
    "contractors",
    "inspectors",
    "vendors",
    "coordinators",
    "trainees",
]

GROUPS: list[dict[str, str]] = [
    {"group": "adv_whole_to_part_support_verb_diverse", "gold": "SUPPORT", "family": "subset", "perturbation": "verb_diverse", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_part_to_whole_not_entitled_verb_diverse", "gold": "NOT_ENTITLED", "family": "subset_reverse", "perturbation": "verb_diverse", "route": "OVERCLAIM_NE"},
    {"group": "adv_whole_to_part_support_fronted_modifier", "gold": "SUPPORT", "family": "whole_part", "perturbation": "fronted_modifier", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_part_to_whole_not_entitled_fronted_modifier", "gold": "NOT_ENTITLED", "family": "whole_part_reverse", "perturbation": "fronted_modifier", "route": "OVERCLAIM_NE"},
    {"group": "adv_whole_to_part_support_postnominal_modifier", "gold": "SUPPORT", "family": "whole_part", "perturbation": "postnominal_modifier", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_part_to_whole_not_entitled_postnominal_modifier", "gold": "NOT_ENTITLED", "family": "whole_part_reverse", "perturbation": "postnominal_modifier", "route": "OVERCLAIM_NE"},
    {"group": "adv_whole_to_part_support_sentence_order_flip", "gold": "SUPPORT", "family": "subset", "perturbation": "sentence_order_flip", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_part_to_whole_not_entitled_sentence_order_flip", "gold": "NOT_ENTITLED", "family": "subset_reverse", "perturbation": "sentence_order_flip", "route": "OVERCLAIM_NE"},
    {"group": "adv_all_except_subset_not_entitled", "gold": "NOT_ENTITLED", "family": "exception", "perturbation": "all_except", "route": "OVERCLAIM_NE"},
    {"group": "adv_all_except_subset_support_for_nonexcluded", "gold": "SUPPORT", "family": "exception", "perturbation": "all_except", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_no_except_subset_support", "gold": "SUPPORT", "family": "exception", "perturbation": "no_except", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_no_except_nonexcluded_refute", "gold": "REFUTE", "family": "exception", "perturbation": "no_except", "route": "CONTRADICTION_REFUTE"},
    {"group": "adv_exactly_some_to_all_not_entitled", "gold": "NOT_ENTITLED", "family": "quantifier", "perturbation": "exactly_some", "route": "OVERCLAIM_NE"},
    {"group": "adv_all_to_at_least_some_support", "gold": "SUPPORT", "family": "quantifier", "perturbation": "at_least_some", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_not_all_to_some_not_entitled", "gold": "NOT_ENTITLED", "family": "quantifier", "perturbation": "not_all", "route": "OVERCLAIM_NE"},
    {"group": "adv_none_to_any_refute", "gold": "REFUTE", "family": "negation", "perturbation": "none_any", "route": "CONTRADICTION_REFUTE"},
    {"group": "adv_passive_active_support", "gold": "SUPPORT", "family": "voice", "perturbation": "passive_active", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_passive_active_reverse_not_entitled", "gold": "NOT_ENTITLED", "family": "voice_reverse", "perturbation": "passive_active", "route": "OVERCLAIM_NE"},
    {"group": "adv_coordination_support", "gold": "SUPPORT", "family": "coordination", "perturbation": "coordination", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_coordination_distractor_not_entitled", "gold": "NOT_ENTITLED", "family": "coordination", "perturbation": "coordination_distractor", "route": "OVERCLAIM_NE"},
    {"group": "adv_numeric_subset_support", "gold": "SUPPORT", "family": "numeric", "perturbation": "numeric_subset", "route": "ENTAILMENT_PRESERVE"},
    {"group": "adv_numeric_reverse_not_entitled", "gold": "NOT_ENTITLED", "family": "numeric_reverse", "perturbation": "numeric_reverse", "route": "OVERCLAIM_NE"},
    {"group": "adv_temporal_scope_not_entitled", "gold": "NOT_ENTITLED", "family": "scope", "perturbation": "temporal_scope", "route": "OVERCLAIM_NE"},
    {"group": "adv_location_scope_not_entitled", "gold": "NOT_ENTITLED", "family": "scope", "perturbation": "location_scope", "route": "OVERCLAIM_NE"},
]


def article(noun: str) -> str:
    return "an" if noun[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def postnominal(part: str) -> str:
    words = part.split(" ", 1)
    if len(words) == 1:
        return part
    modifier, head = words
    return f"{head} that were marked as {modifier}"


def text_for_group(group: str, whole: str, part: str, predicate: str, idx: int) -> tuple[str, str]:
    loc = LOCATIONS[idx % len(LOCATIONS)]
    time = TIMES[idx % len(TIMES)]
    distractor = DISTRACTORS[idx % len(DISTRACTORS)]
    nonexcluded_part = f"standard {whole}"
    part_after_head = postnominal(part)
    count_all = 12 + (idx % 5)
    count_part = 3 + (idx % 4)

    if group == "adv_whole_to_part_support_verb_diverse":
        return (
            f"The {part} were {predicate} {time}.",
            f"All {whole} were {predicate} {time}.",
        )
    if group == "adv_part_to_whole_not_entitled_verb_diverse":
        return (
            f"All {whole} were {predicate} {time}.",
            f"The {part} were {predicate} {time}.",
        )
    if group == "adv_whole_to_part_support_fronted_modifier":
        return (
            f"{part.capitalize()} were {predicate} {loc}.",
            f"Every {whole} was {predicate} {loc}.",
        )
    if group == "adv_part_to_whole_not_entitled_fronted_modifier":
        return (
            f"Every {whole} was {predicate} {loc}.",
            f"{part.capitalize()} were {predicate} {loc}.",
        )
    if group == "adv_whole_to_part_support_postnominal_modifier":
        return (
            f"The {part_after_head} were {predicate} {time}.",
            f"All {whole} were {predicate} {time}.",
        )
    if group == "adv_part_to_whole_not_entitled_postnominal_modifier":
        return (
            f"All {whole} were {predicate} {time}.",
            f"The {part_after_head} were {predicate} {time}.",
        )
    if group == "adv_whole_to_part_support_sentence_order_flip":
        return (
            f"{time.capitalize()}, the {part} were {predicate}.",
            f"The office {predicate} every {whole} {time}.",
        )
    if group == "adv_part_to_whole_not_entitled_sentence_order_flip":
        return (
            f"The office {predicate} every {whole} {time}.",
            f"{time.capitalize()}, the {part} were {predicate}.",
        )
    if group == "adv_all_except_subset_not_entitled":
        return (
            f"The {part} were {predicate}.",
            f"All {whole} except the {part} were {predicate}.",
        )
    if group == "adv_all_except_subset_support_for_nonexcluded":
        return (
            f"The {nonexcluded_part} were {predicate}.",
            f"All {whole} except the {part} were {predicate}.",
        )
    if group == "adv_no_except_subset_support":
        return (
            f"The {part} were {predicate}.",
            f"No {whole} except the {part} were {predicate}.",
        )
    if group == "adv_no_except_nonexcluded_refute":
        return (
            f"The {nonexcluded_part} were {predicate}.",
            f"No {whole} except the {part} were {predicate}.",
        )
    if group == "adv_exactly_some_to_all_not_entitled":
        return (
            f"All {whole} were {predicate}.",
            f"Only some {whole} were {predicate}.",
        )
    if group == "adv_all_to_at_least_some_support":
        return (
            f"At least some {whole} were {predicate}.",
            f"All {whole} were {predicate}.",
        )
    if group == "adv_not_all_to_some_not_entitled":
        return (
            f"Some {whole} were {predicate}.",
            f"Not all {whole} were {predicate}.",
        )
    if group == "adv_none_to_any_refute":
        return (
            f"Some {whole} were {predicate}.",
            f"No {whole} were {predicate}.",
        )
    if group == "adv_passive_active_support":
        return (
            f"The agency {predicate} the {part}.",
            f"Every {whole} was {predicate} by the agency.",
        )
    if group == "adv_passive_active_reverse_not_entitled":
        return (
            f"Every {whole} was {predicate} by the agency.",
            f"The agency {predicate} the {part}.",
        )
    if group == "adv_coordination_support":
        return (
            f"The {part} were {predicate}.",
            f"All {whole} and all {distractor} were {predicate}.",
        )
    if group == "adv_coordination_distractor_not_entitled":
        return (
            f"All {distractor} were {predicate}.",
            f"All {whole} were {predicate}, and some {distractor} were {predicate}.",
        )
    if group == "adv_numeric_subset_support":
        return (
            f"The {part} among the {whole} were {predicate}.",
            f"All {count_all} {whole} were {predicate}.",
        )
    if group == "adv_numeric_reverse_not_entitled":
        return (
            f"All {count_all} {whole} were {predicate}.",
            f"{count_part} {part} were {predicate}.",
        )
    if group == "adv_temporal_scope_not_entitled":
        return (
            f"The {part} were {predicate} in 2025.",
            f"All {whole} were {predicate} in 2024.",
        )
    if group == "adv_location_scope_not_entitled":
        return (
            f"The {part} in the west district were {predicate}.",
            f"All {whole} in the east district were {predicate}.",
        )
    raise ValueError(f"Unhandled Stage35 group {group!r}")


def build_rows(examples_per_group: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_idx, spec in enumerate(GROUPS):
        group = spec["group"]
        gold = spec["gold"]
        for item_idx in range(examples_per_group):
            pair_idx = (group_idx * examples_per_group + item_idx) % len(ADVERSARIAL_PAIRS)
            whole, part = ADVERSARIAL_PAIRS[pair_idx]
            predicate = PREDICATES[(group_idx + item_idx) % len(PREDICATES)]
            claim, evidence = text_for_group(group, whole, part, predicate, item_idx)
            row_id = f"stage35a_{group}_{item_idx:02d}"
            relation = f"{whole}->{part}"
            rows.append({
                "id": row_id,
                "pair_id": row_id,
                "claim": claim,
                "evidence": evidence,
                "final_label": gold,
                "gold_label": gold,
                "label": LABEL_TO_ID[gold],
                "group": group,
                "intervention_type": group,
                "normalized_intervention": group,
                "primary_failure_type": relation,
                "failure_type": relation,
                "source": "stage35a_adversarial_coverage_probe",
                "split": "external_stage35_adversarial",
                "stage35_family": spec["family"],
                "stage35_relation": relation,
                "stage35_perturbation": spec["perturbation"],
                "stage35_expected_route": spec["route"],
                "stage35_is_adversarial": True,
            })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]], examples_per_group: int) -> None:
    counts = Counter(row["group"] for row in rows)
    payload = {
        "stage": "Stage35-A",
        "row_count": len(rows),
        "examples_per_group": examples_per_group,
        "group_count": len(counts),
        "group_counts": dict(sorted(counts.items())),
        "label_counts": dict(Counter(row["final_label"] for row in rows)),
        "source": "stage35a_adversarial_coverage_probe",
        "split": "external_stage35_adversarial",
        "diagnostic_only": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-group", type=int, default=25)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO_ROOT / "reports" / "stage35a_adversarial_coverage_probe_report.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_rows(args.examples_per_group)
    write_jsonl(args.output, rows)
    write_report(args.report_json, rows, args.examples_per_group)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Report -> {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
