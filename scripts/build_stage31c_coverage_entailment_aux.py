"""Stage31-C: directional Coverage/Entailment auxiliary data builder.

This auxiliary dataset is diagnostic supervision for a readout head. It is
separate from the Stage31-A/B evaluation probe and must not be used for final
evaluation, calibration, threshold selection, or checkpoint selection.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "stage31c_coverage_entailment_aux.jsonl"
REPORT_MD_PATH = ROOT / "reports" / "stage31c_coverage_entailment_aux_report.md"
REPORT_JSON_PATH = ROOT / "reports" / "stage31c_coverage_entailment_aux_report.json"

FINAL_LABEL_TO_ID = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}
COVERAGE_LABEL_TO_ID = {
    "ENTAILS_SUPPORT": 0,
    "OVERCLAIM_NOT_ENTITLED": 1,
    "CONTRADICTS_REFUTE": 2,
    "OTHER_RESIDUAL": 3,
}

CLASSES = list(COVERAGE_LABEL_TO_ID)
ROWS_PER_CLASS = 200
DEV_PER_CLASS = 40


def _compat(final_label: str) -> dict[str, Any]:
    sufficient = 0 if final_label == "NOT_ENTITLED" else 1
    polarity = {
        "SUPPORT": "SUPPORT",
        "NOT_ENTITLED": "NONE",
        "REFUTE": "REFUTE",
    }[final_label]
    return {
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": sufficient,
        "evidence_sufficient_label": sufficient,
        "polarity_label": polarity,
    }


def _row(
    *,
    idx: int,
    class_label: str,
    group: str,
    evidence: str,
    claim: str,
    final_label: str,
    coverage_relation: str,
    notes: str,
) -> dict[str, Any]:
    row_id = f"stage31c_{class_label.lower()}_{idx:03d}"
    split = "dev" if idx % 5 == 4 else "train"
    row = {
        "id": row_id,
        "pair_id": row_id,
        "claim": claim,
        "evidence": evidence,
        "final_label": final_label,
        "label": final_label,
        "gold": FINAL_LABEL_TO_ID[final_label],
        "label_id": FINAL_LABEL_TO_ID[final_label],
        "coverage_direction_label": class_label,
        "coverage_direction_id": COVERAGE_LABEL_TO_ID[class_label],
        "coverage_relation": coverage_relation,
        "expected_owner": "coverage_entailment_direction",
        "group": group,
        "split": split,
        "notes": notes,
        "intervention_type": group,
        "probe_type": group,
    }
    row.update(_compat(final_label))
    return row


ENTAIL_TEMPLATES = [
    (
        "all_to_some_aux",
        "all_entails_some",
        "All {plural} in the {place} {verb}.",
        "Some {plural} in the {place} {verb}.",
        "All-to-some weakening preserves SUPPORT.",
    ),
    (
        "specific_to_general_aux",
        "specific_entails_general",
        "{name} {specific_action} during the {event}.",
        "{name} {general_action} during the {event}.",
        "Specific evidence entails a weaker general claim.",
    ),
    (
        "whole_to_part_aux",
        "whole_entails_included_part",
        "The complete {whole} was {status} after inspection.",
        "The {part} of the {whole} was {status} after inspection.",
        "Whole-set evidence covers an included part.",
    ),
    (
        "only_to_base_aux",
        "only_entails_base_membership",
        "{name} was the only {role} recognized by the {body}.",
        "{name} was a {role} recognized by the {body}.",
        "Only membership entails base membership.",
    ),
]

OVERCLAIM_TEMPLATES = [
    (
        "some_to_all_aux",
        "some_does_not_entail_all",
        "Some {plural} in the {place} {verb}.",
        "All {plural} in the {place} {verb}.",
        "Some-to-all is an overclaim.",
    ),
    (
        "general_to_specific_aux",
        "general_does_not_entail_specific",
        "{name} {general_action} during the {event}.",
        "{name} {specific_action} during the {event}.",
        "General evidence does not entail a stronger specific claim.",
    ),
    (
        "part_to_whole_aux",
        "part_does_not_entail_whole",
        "The {part} of the {whole} was {status} after inspection.",
        "The complete {whole} was {status} after inspection.",
        "Part evidence does not cover the whole.",
    ),
    (
        "also_to_only_aux",
        "also_does_not_entail_only",
        "{name} was also a {role} recognized by the {body}.",
        "{name} was the only {role} recognized by the {body}.",
        "Also membership does not entail exclusivity.",
    ),
]

CONTRADICT_TEMPLATES = [
    (
        "none_to_some_aux",
        "none_contradicts_some",
        "No {plural} in the {place} {verb}.",
        "Some {plural} in the {place} {verb}.",
        "No instances contradicts some instances.",
    ),
    (
        "some_to_none_aux",
        "some_contradicts_none",
        "Some {plural} in the {place} {verb}.",
        "No {plural} in the {place} {verb}.",
        "Some instances contradicts no instances.",
    ),
    (
        "no_member_to_member_aux",
        "no_member_contradicts_member",
        "{name} was not a member of the {body}.",
        "{name} was a member of the {body}.",
        "Non-membership contradicts membership.",
    ),
    (
        "never_to_once_aux",
        "never_contradicts_once",
        "{name} never {past_action} during the {event}.",
        "{name} {past_action} once during the {event}.",
        "Never contradicts once.",
    ),
]

RESIDUAL_TEMPLATES = [
    (
        "surface_paraphrase_aux",
        "surface_variation_support",
        "The {artifact} was approved by the {body} on Monday.",
        "On Monday, the {body} approved the {artifact}.",
        "SUPPORT",
        "Surface variation is not primarily coverage-driven.",
    ),
    (
        "alias_granularity_aux",
        "alias_granularity_support",
        "The {body} released the annual summary for {place}.",
        "The {body_alias} released the annual summary for {place}.",
        "SUPPORT",
        "Alias/granularity variation preserves the relation.",
    ),
    (
        "underspecified_residual_aux",
        "underspecified_not_entitled",
        "The bulletin mentioned a delay near {place}.",
        "The bulletin confirmed a two-hour delay near {place}.",
        "NOT_ENTITLED",
        "Underspecified residual case, not a directional coverage template.",
    ),
    (
        "relation_paraphrase_aux",
        "relation_paraphrase_support",
        "{name} signed the {artifact} after the review.",
        "After the review, the {artifact} was signed by {name}.",
        "SUPPORT",
        "Relation-preserving paraphrase not primarily coverage-driven.",
    ),
]

VALUES = [
    {
        "plural": "analysts",
        "place": "Aurora office",
        "verb": "submitted the ledger",
        "name": "Iris Vale",
        "specific_action": "filed a notarized complaint",
        "general_action": "filed a complaint",
        "event": "Harbor review",
        "whole": "Meridian archive",
        "part": "index wing",
        "status": "sealed",
        "role": "delegate",
        "body": "Cedar council",
        "body_alias": "the council",
        "past_action": "entered the chamber",
        "artifact": "zoning memo",
    },
    {
        "plural": "technicians",
        "place": "Lumen depot",
        "verb": "logged the outage",
        "name": "Noah Sato",
        "specific_action": "won the regional design prize",
        "general_action": "won a design prize",
        "event": "Cobalt expo",
        "whole": "Solstice package",
        "part": "routing module",
        "status": "verified",
        "role": "auditor",
        "body": "Marble board",
        "body_alias": "the review board",
        "past_action": "visited the facility",
        "artifact": "safety addendum",
    },
    {
        "plural": "curators",
        "place": "Orchid gallery",
        "verb": "catalogued the exhibit",
        "name": "Mara Keene",
        "specific_action": "completed the alpine rescue course",
        "general_action": "completed a rescue course",
        "event": "winter intake",
        "whole": "Northline report",
        "part": "finance chapter",
        "status": "published",
        "role": "finalist",
        "body": "Raven panel",
        "body_alias": "the panel",
        "past_action": "attended the briefing",
        "artifact": "permit draft",
    },
    {
        "plural": "inspectors",
        "place": "Pine terminal",
        "verb": "approved the manifest",
        "name": "Theo Grant",
        "specific_action": "repaired the primary coolant valve",
        "general_action": "repaired a valve",
        "event": "plant audit",
        "whole": "Atlas convoy",
        "part": "lead vehicle",
        "status": "cleared",
        "role": "witness",
        "body": "Summit tribunal",
        "body_alias": "the tribunal",
        "past_action": "used the access badge",
        "artifact": "procurement note",
    },
    {
        "plural": "fellows",
        "place": "Juniper lab",
        "verb": "replicated the assay",
        "name": "Leah Morgan",
        "specific_action": "translated the emergency protocol into Welsh",
        "general_action": "translated the emergency protocol",
        "event": "readiness drill",
        "whole": "Helio manuscript",
        "part": "methods section",
        "status": "accepted",
        "role": "nominee",
        "body": "Opal committee",
        "body_alias": "the committee",
        "past_action": "signed the register",
        "artifact": "ethics waiver",
    },
]


def _format(template: str, values: dict[str, str], serial: int) -> str:
    extras = dict(values)
    extras["place"] = f"{values['place']} {serial:02d}"
    extras["event"] = f"{values['event']} {serial:02d}"
    extras["artifact"] = f"{values['artifact']} {serial:02d}"
    return template.format(**extras)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    plans = [
        ("ENTAILS_SUPPORT", "SUPPORT", ENTAIL_TEMPLATES),
        ("OVERCLAIM_NOT_ENTITLED", "NOT_ENTITLED", OVERCLAIM_TEMPLATES),
        ("CONTRADICTS_REFUTE", "REFUTE", CONTRADICT_TEMPLATES),
    ]
    for class_label, final_label, templates in plans:
        for idx in range(ROWS_PER_CLASS):
            group, relation, ev_t, cl_t, notes = templates[idx % len(templates)]
            values = VALUES[(idx // len(templates)) % len(VALUES)]
            serial = idx // (len(templates) * len(VALUES))
            rows.append(_row(
                idx=idx,
                class_label=class_label,
                group=group,
                evidence=_format(ev_t, values, serial),
                claim=_format(cl_t, values, serial),
                final_label=final_label,
                coverage_relation=relation,
                notes=notes,
            ))

    for idx in range(ROWS_PER_CLASS):
        group, relation, ev_t, cl_t, final_label, notes = RESIDUAL_TEMPLATES[
            idx % len(RESIDUAL_TEMPLATES)
        ]
        values = VALUES[(idx // len(RESIDUAL_TEMPLATES)) % len(VALUES)]
        serial = idx // (len(RESIDUAL_TEMPLATES) * len(VALUES))
        rows.append(_row(
            idx=idx,
            class_label="OTHER_RESIDUAL",
            group=group,
            evidence=_format(ev_t, values, serial),
            claim=_format(cl_t, values, serial),
            final_label=final_label,
            coverage_relation=relation,
            notes=notes,
        ))
    return rows


REQUIRED_FIELDS = {
    "id", "pair_id", "claim", "evidence", "final_label", "label", "gold",
    "label_id", "coverage_direction_label", "coverage_direction_id",
    "coverage_relation", "expected_owner", "group", "split", "notes",
    "frame_compatible_label", "predicate_covered_label", "sufficiency_label",
    "evidence_sufficient_label", "polarity_label", "intervention_type",
    "probe_type",
}


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != ROWS_PER_CLASS * len(CLASSES):
        errors.append(f"row count {len(rows)} != {ROWS_PER_CLASS * len(CLASSES)}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate id values found")
    for row in rows:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{row.get('id', '?')} missing {sorted(missing)}")
        final_label = row.get("final_label")
        if row.get("label") != final_label:
            errors.append(f"{row['id']} label/final_label mismatch")
        if row.get("gold") != FINAL_LABEL_TO_ID.get(final_label):
            errors.append(f"{row['id']} gold mismatch")
        if row.get("label_id") != FINAL_LABEL_TO_ID.get(final_label):
            errors.append(f"{row['id']} label_id mismatch")
        direction = row.get("coverage_direction_label")
        if row.get("coverage_direction_id") != COVERAGE_LABEL_TO_ID.get(direction):
            errors.append(f"{row['id']} coverage direction mismatch")
        if row.get("pair_id") != row.get("id"):
            errors.append(f"{row['id']} pair_id must equal id")
    class_counts = Counter(r["coverage_direction_label"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    for cls in CLASSES:
        if class_counts[cls] != ROWS_PER_CLASS:
            errors.append(f"{cls} count {class_counts[cls]} != {ROWS_PER_CLASS}")
        train_n = sum(
            r["coverage_direction_label"] == cls and r["split"] == "train"
            for r in rows
        )
        dev_n = sum(
            r["coverage_direction_label"] == cls and r["split"] == "dev"
            for r in rows
        )
        if train_n != ROWS_PER_CLASS - DEV_PER_CLASS or dev_n != DEV_PER_CLASS:
            errors.append(f"{cls} split counts train={train_n} dev={dev_n}")
    if split_counts["train"] != 640 or split_counts["dev"] != 160:
        errors.append(f"split counts {dict(split_counts)} != train=640 dev=160")
    return errors


def build_report(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    return {
        "stage": "Stage31-C",
        "artifact": "coverage_entailment_auxiliary_data",
        "total_rows": len(rows),
        "coverage_direction_mapping": COVERAGE_LABEL_TO_ID,
        "final_label_mapping": FINAL_LABEL_TO_ID,
        "class_counts": dict(Counter(r["coverage_direction_label"] for r in rows)),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "group_counts": dict(Counter(r["group"] for r in rows)),
        "schema_fields": sorted(REQUIRED_FIELDS),
        "diagnostic_only": True,
        "leakage_policy": (
            "Auxiliary diagnostic supervision only. Do not use the Stage31-A/B "
            "evaluation probe for this loss. Do not use this auxiliary set for "
            "final evaluation, calibration, threshold selection, or checkpoint selection."
        ),
        "validation_errors": errors,
        "validation_passed": not errors,
    }


def build_md(report: dict[str, Any]) -> str:
    lines = [
        "# Stage31-C Coverage/Entailment Auxiliary Data Report",
        "",
        "## Purpose",
        "Build separate auxiliary diagnostic supervision for a readout head that",
        "distinguishes entailment-preserving SUPPORT, overclaim NOT_ENTITLED,",
        "contradiction REFUTE, and OTHER_RESIDUAL cases.",
        "",
        "This file is not the Stage31 evaluation probe and does not reuse exact",
        "Stage31-A claim/evidence pairs.",
        "",
        "## Counts",
        f"- Total rows: {report['total_rows']}",
        f"- Split counts: {report['split_counts']}",
        f"- Class counts: {report['class_counts']}",
        "",
        "## Coverage Direction Mapping",
        "| Label | ID |",
        "|---|---|",
    ]
    for label, idx in report["coverage_direction_mapping"].items():
        lines.append(f"| {label} | {idx} |")
    lines.extend([
        "",
        "## Group Counts",
        "| Group | Count |",
        "|---|---|",
    ])
    for group, count in sorted(report["group_counts"].items()):
        lines.append(f"| {group} | {count} |")
    lines.extend([
        "",
        "## Schema",
        "Rows include identity, controlled-style final-label fields,",
        "`coverage_direction_label`, `coverage_direction_id`, split, owner,",
        "group, notes, and controlled-style auxiliary compatibility fields.",
        "",
        "## Leakage Policy",
        report["leakage_policy"],
        "",
        "## Validation",
        "PASSED" if report["validation_passed"] else "FAILED",
        "",
        "*Generated by `scripts/build_stage31c_coverage_entailment_aux.py`.*",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    errors = validate(rows)
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = build_report(rows, errors)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_MD_PATH.write_text(build_md(report), encoding="utf-8")
    print("Stage31-C auxiliary data generation complete.")
    print(f"  rows: {len(rows)}")
    print(f"  splits: {report['split_counts']}")
    print(f"  classes: {report['class_counts']}")
    print(f"  data: {DATA_PATH}")
    print(f"  report: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
