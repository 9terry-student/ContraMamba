"""Stage31-C2: hard-contrast Coverage/Entailment auxiliary data builder.

This auxiliary dataset is diagnostic supervision for a readout head. It is
separate from the Stage31 evaluation probe and must not be used for final
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
}
DIRECTION_TO_FINAL_LABEL = {
    "ENTAILS_SUPPORT": "SUPPORT",
    "OVERCLAIM_NOT_ENTITLED": "NOT_ENTITLED",
    "CONTRADICTS_REFUTE": "REFUTE",
}

CLASSES = list(COVERAGE_LABEL_TO_ID)
CONTRAST_FAMILIES = [
    "all_some_none",
    "specific_general",
    "whole_part",
    "only_also_base",
    "no_member_member",
    "never_once",
    "zero_positive_count",
    "excluded_included",
    "failed_to_did",
    "every_some",
    "participant_scope",
    "org_member_partwhole",
]
SETS_PER_FAMILY = 25
ROWS_PER_CLASS = len(CONTRAST_FAMILIES) * SETS_PER_FAMILY
DEV_SETS_PER_FAMILY = 5
DEV_PER_CLASS = len(CONTRAST_FAMILIES) * DEV_SETS_PER_FAMILY

VALUES = [
    {
        "name": "Iris Vale",
        "plural": "analysts",
        "singular": "analyst",
        "place": "Aurora office",
        "action": "submitted the ledger",
        "specific_action": "filed a notarized complaint",
        "general_action": "filed a complaint",
        "whole": "Meridian archive",
        "part": "index wing",
        "status": "sealed",
        "role": "delegate",
        "body": "Cedar council",
        "event": "Harbor review",
        "count_item": "inspection tickets",
        "member": "North team",
        "org": "Helio institute",
    },
    {
        "name": "Noah Sato",
        "plural": "technicians",
        "singular": "technician",
        "place": "Lumen depot",
        "action": "logged the outage",
        "specific_action": "won the regional design prize",
        "general_action": "won a design prize",
        "whole": "Solstice package",
        "part": "routing module",
        "status": "verified",
        "role": "auditor",
        "body": "Marble board",
        "event": "Cobalt expo",
        "count_item": "repair orders",
        "member": "East unit",
        "org": "Marble board",
    },
    {
        "name": "Mara Keene",
        "plural": "curators",
        "singular": "curator",
        "place": "Orchid gallery",
        "action": "catalogued the exhibit",
        "specific_action": "completed the alpine rescue course",
        "general_action": "completed a rescue course",
        "whole": "Northline report",
        "part": "finance chapter",
        "status": "published",
        "role": "finalist",
        "body": "Raven panel",
        "event": "winter intake",
        "count_item": "gallery notices",
        "member": "Archive desk",
        "org": "Raven panel",
    },
    {
        "name": "Theo Grant",
        "plural": "inspectors",
        "singular": "inspector",
        "place": "Pine terminal",
        "action": "approved the manifest",
        "specific_action": "repaired the primary coolant valve",
        "general_action": "repaired a valve",
        "whole": "Atlas convoy",
        "part": "lead vehicle",
        "status": "cleared",
        "role": "witness",
        "body": "Summit tribunal",
        "event": "plant audit",
        "count_item": "access badges",
        "member": "Logistics cell",
        "org": "Summit tribunal",
    },
    {
        "name": "Leah Morgan",
        "plural": "fellows",
        "singular": "fellow",
        "place": "Juniper lab",
        "action": "replicated the assay",
        "specific_action": "translated the emergency protocol into Welsh",
        "general_action": "translated the emergency protocol",
        "whole": "Helio manuscript",
        "part": "methods section",
        "status": "accepted",
        "role": "nominee",
        "body": "Opal committee",
        "event": "readiness drill",
        "count_item": "ethics waivers",
        "member": "Research office",
        "org": "Opal committee",
    },
]

FAMILY_TEMPLATES: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "all_some_none": {
        "ENTAILS_SUPPORT": ("all_entails_some", "All {plural} in the {place} {action}.", "Some {plural} in the {place} {action}.", "All-to-some weakening preserves support."),
        "OVERCLAIM_NOT_ENTITLED": ("some_does_not_entail_all", "Some {plural} in the {place} {action}.", "All {plural} in the {place} {action}.", "Some-to-all is an overclaim."),
        "CONTRADICTS_REFUTE": ("none_contradicts_some", "No {plural} in the {place} {action}.", "Some {plural} in the {place} {action}.", "None contradicts some."),
    },
    "specific_general": {
        "ENTAILS_SUPPORT": ("specific_entails_general", "{name} {specific_action} during the {event}.", "{name} {general_action} during the {event}.", "Specific action entails the weaker general action."),
        "OVERCLAIM_NOT_ENTITLED": ("general_does_not_entail_specific", "{name} {general_action} during the {event}.", "{name} {specific_action} during the {event}.", "General action does not entail the specific action."),
        "CONTRADICTS_REFUTE": ("specific_contradicted", "{name} did not {specific_action} during the {event}.", "{name} {specific_action} during the {event}.", "Negated specific action contradicts the specific claim."),
    },
    "whole_part": {
        "ENTAILS_SUPPORT": ("whole_entails_part", "The complete {whole} was {status} after inspection.", "The {part} of the {whole} was {status} after inspection.", "Whole-set evidence covers an included part."),
        "OVERCLAIM_NOT_ENTITLED": ("part_does_not_entail_whole", "The {part} of the {whole} was {status} after inspection.", "The complete {whole} was {status} after inspection.", "Part evidence does not cover the whole."),
        "CONTRADICTS_REFUTE": ("part_contradicts_whole", "The {part} of the {whole} was not {status} after inspection.", "The complete {whole} was {status} after inspection.", "A failed included part contradicts the whole being covered."),
    },
    "only_also_base": {
        "ENTAILS_SUPPORT": ("only_entails_base", "{name} was the only {role} recognized by the {body}.", "{name} was a {role} recognized by the {body}.", "Only membership entails base membership."),
        "OVERCLAIM_NOT_ENTITLED": ("base_does_not_entail_only", "{name} was a {role} recognized by the {body}.", "{name} was the only {role} recognized by the {body}.", "Base membership does not entail exclusivity."),
        "CONTRADICTS_REFUTE": ("also_contradicts_only", "{name} and another {role} were recognized by the {body}.", "{name} was the only {role} recognized by the {body}.", "Additional membership contradicts only."),
    },
    "no_member_member": {
        "ENTAILS_SUPPORT": ("member_entails_participant", "{name} was a member of the {body}.", "{name} participated in the {body}.", "Membership entails participation in the body."),
        "OVERCLAIM_NOT_ENTITLED": ("participant_does_not_entail_member", "{name} participated in the {body}.", "{name} was a member of the {body}.", "Participation does not entail membership."),
        "CONTRADICTS_REFUTE": ("nonmember_contradicts_member", "{name} was not a member of the {body}.", "{name} was a member of the {body}.", "Non-membership contradicts membership."),
    },
    "never_once": {
        "ENTAILS_SUPPORT": ("twice_entails_once", "{name} entered the {place} twice during the {event}.", "{name} entered the {place} once during the {event}.", "Twice entails at least once."),
        "OVERCLAIM_NOT_ENTITLED": ("once_does_not_entail_twice", "{name} entered the {place} once during the {event}.", "{name} entered the {place} twice during the {event}.", "Once does not entail twice."),
        "CONTRADICTS_REFUTE": ("never_contradicts_once", "{name} never entered the {place} during the {event}.", "{name} entered the {place} once during the {event}.", "Never contradicts once."),
    },
    "zero_positive_count": {
        "ENTAILS_SUPPORT": ("three_entails_positive", "Three {count_item} were approved at the {place}.", "At least one {count_item} was approved at the {place}.", "A positive count entails existence."),
        "OVERCLAIM_NOT_ENTITLED": ("positive_does_not_entail_three", "At least one {count_item} was approved at the {place}.", "Three {count_item} were approved at the {place}.", "Existence does not entail the exact count."),
        "CONTRADICTS_REFUTE": ("zero_contradicts_positive", "Zero {count_item} were approved at the {place}.", "At least one {count_item} was approved at the {place}.", "Zero contradicts a positive count."),
    },
    "excluded_included": {
        "ENTAILS_SUPPORT": ("included_entails_listed", "{name} was included on the {body} list for the {event}.", "{name} appeared on the {body} list for the {event}.", "Included entails listed."),
        "OVERCLAIM_NOT_ENTITLED": ("listed_does_not_entail_included_final", "{name} appeared in notes about the {body} list for the {event}.", "{name} was included on the {body} list for the {event}.", "Mentioned in notes does not entail inclusion."),
        "CONTRADICTS_REFUTE": ("excluded_contradicts_included", "{name} was excluded from the {body} list for the {event}.", "{name} was included on the {body} list for the {event}.", "Excluded contradicts included."),
    },
    "failed_to_did": {
        "ENTAILS_SUPPORT": ("completed_entails_attempted", "{name} completed the {general_action} during the {event}.", "{name} attempted the {general_action} during the {event}.", "Completion entails an attempt."),
        "OVERCLAIM_NOT_ENTITLED": ("attempted_does_not_entail_completed", "{name} attempted the {general_action} during the {event}.", "{name} completed the {general_action} during the {event}.", "Attempting does not entail completion."),
        "CONTRADICTS_REFUTE": ("failed_contradicts_completed", "{name} failed to complete the {general_action} during the {event}.", "{name} completed the {general_action} during the {event}.", "Failed to complete contradicts completed."),
    },
    "every_some": {
        "ENTAILS_SUPPORT": ("every_entails_some", "Every {singular} assigned to the {place} {action}.", "Some {plural} assigned to the {place} {action}.", "Every entails some for a populated assignment."),
        "OVERCLAIM_NOT_ENTITLED": ("some_does_not_entail_every", "Some {plural} assigned to the {place} {action}.", "Every {singular} assigned to the {place} {action}.", "Some does not entail every."),
        "CONTRADICTS_REFUTE": ("not_every_contradicts_every", "Not every {singular} assigned to the {place} {action}.", "Every {singular} assigned to the {place} {action}.", "Not every contradicts every."),
    },
    "participant_scope": {
        "ENTAILS_SUPPORT": ("named_participant_entails_participant", "{name} was a registered participant in the {event}.", "A participant was registered in the {event}.", "Named participant entails participant existence."),
        "OVERCLAIM_NOT_ENTITLED": ("participant_does_not_entail_named", "A participant was registered in the {event}.", "{name} was a registered participant in the {event}.", "An unnamed participant does not entail this named participant."),
        "CONTRADICTS_REFUTE": ("named_absent_contradicts_named", "{name} was not registered as a participant in the {event}.", "{name} was a registered participant in the {event}.", "Named absence contradicts named participation."),
    },
    "org_member_partwhole": {
        "ENTAILS_SUPPORT": ("org_action_entails_member_unit_action", "The {org} approved the plan through its {member}.", "The {member} of the {org} helped approve the plan.", "Organization action through a member unit entails that unit's participation."),
        "OVERCLAIM_NOT_ENTITLED": ("member_action_does_not_entail_org_action", "The {member} of the {org} reviewed the plan.", "The {org} approved the plan.", "Member review does not entail whole-organization approval."),
        "CONTRADICTS_REFUTE": ("org_rejected_contradicts_org_approved", "The {org} rejected the plan after the {member} review.", "The {org} approved the plan.", "Organization rejection contradicts organization approval."),
    },
}


def _compat(final_label: str) -> dict[str, Any]:
    sufficient = 0 if final_label == "NOT_ENTITLED" else 1
    polarity = {"SUPPORT": "SUPPORT", "NOT_ENTITLED": "NONE", "REFUTE": "REFUTE"}[
        final_label
    ]
    return {
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": sufficient,
        "evidence_sufficient_label": sufficient,
        "polarity_label": polarity,
    }


def _format(template: str, values: dict[str, str], family: str, set_idx: int) -> str:
    extras = dict(values)
    extras["place"] = f"{values['place']} {family.replace('_', '-')} {set_idx:02d}"
    extras["event"] = f"{values['event']} {family.replace('_', '-')} {set_idx:02d}"
    return template.format(**extras)


def _row(
    *,
    family: str,
    set_idx: int,
    class_label: str,
    relation: str,
    evidence: str,
    claim: str,
    notes: str,
) -> dict[str, Any]:
    final_label = DIRECTION_TO_FINAL_LABEL[class_label]
    contrast_set_id = f"stage31c2_{family}_{set_idx:03d}"
    row_id = f"{contrast_set_id}_{class_label.lower()}"
    group = f"{family}_{class_label.lower()}"
    split = "dev" if set_idx % 5 == 4 else "train"
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
        "coverage_relation": relation,
        "expected_owner": "coverage_entailment_direction",
        "group": group,
        "split": split,
        "contrast_set_id": contrast_set_id,
        "contrast_family": family,
        "notes": f"Hard minimal contrast: {notes}",
        "intervention_type": group,
        "probe_type": group,
    }
    row.update(_compat(final_label))
    return row


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in CONTRAST_FAMILIES:
        templates = FAMILY_TEMPLATES[family]
        for set_idx in range(SETS_PER_FAMILY):
            values = VALUES[set_idx % len(VALUES)]
            for class_label in CLASSES:
                relation, ev_t, cl_t, notes = templates[class_label]
                rows.append(
                    _row(
                        family=family,
                        set_idx=set_idx,
                        class_label=class_label,
                        relation=relation,
                        evidence=_format(ev_t, values, family, set_idx),
                        claim=_format(cl_t, values, family, set_idx),
                        notes=notes,
                    )
                )
    return rows


REQUIRED_FIELDS = {
    "id", "pair_id", "claim", "evidence", "final_label", "label", "gold",
    "label_id", "coverage_direction_label", "coverage_direction_id",
    "coverage_relation", "expected_owner", "group", "split",
    "contrast_set_id", "contrast_family", "notes",
    "frame_compatible_label", "predicate_covered_label", "sufficiency_label",
    "evidence_sufficient_label", "polarity_label", "intervention_type",
    "probe_type",
}


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_total = ROWS_PER_CLASS * len(CLASSES)
    if len(rows) != expected_total:
        errors.append(f"row count {len(rows)} != {expected_total}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate id values found")
    for row in rows:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{row.get('id', '?')} missing {sorted(missing)}")
        final_label = row.get("final_label")
        direction = row.get("coverage_direction_label")
        if direction not in COVERAGE_LABEL_TO_ID:
            errors.append(f"{row['id']} invalid direction {direction!r}")
        if final_label != DIRECTION_TO_FINAL_LABEL.get(direction):
            errors.append(f"{row['id']} direction/final_label mismatch")
        if row.get("label") != final_label:
            errors.append(f"{row['id']} label/final_label mismatch")
        if row.get("gold") != FINAL_LABEL_TO_ID.get(final_label):
            errors.append(f"{row['id']} gold mismatch")
        if row.get("label_id") != FINAL_LABEL_TO_ID.get(final_label):
            errors.append(f"{row['id']} label_id mismatch")
        if row.get("coverage_direction_id") != COVERAGE_LABEL_TO_ID.get(direction):
            errors.append(f"{row['id']} coverage direction mismatch")
        if row.get("pair_id") != row.get("id"):
            errors.append(f"{row['id']} pair_id must equal id")
        if row.get("contrast_family") not in CONTRAST_FAMILIES:
            errors.append(f"{row['id']} invalid contrast_family")

    class_counts = Counter(r["coverage_direction_label"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    family_counts = Counter(r["contrast_family"] for r in rows)
    for cls in CLASSES:
        if class_counts[cls] != ROWS_PER_CLASS:
            errors.append(f"{cls} count {class_counts[cls]} != {ROWS_PER_CLASS}")
        train_n = sum(r["coverage_direction_label"] == cls and r["split"] == "train" for r in rows)
        dev_n = sum(r["coverage_direction_label"] == cls and r["split"] == "dev" for r in rows)
        if train_n != ROWS_PER_CLASS - DEV_PER_CLASS or dev_n != DEV_PER_CLASS:
            errors.append(f"{cls} split counts train={train_n} dev={dev_n}")
    for family in CONTRAST_FAMILIES:
        if family_counts[family] != SETS_PER_FAMILY * len(CLASSES):
            errors.append(f"{family} count {family_counts[family]} != {SETS_PER_FAMILY * len(CLASSES)}")
    if split_counts["train"] != 720 or split_counts["dev"] != 180:
        errors.append(f"split counts {dict(split_counts)} != train=720 dev=180")
    return errors


def build_report(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    return {
        "stage": "Stage31-C2",
        "artifact": "coverage_entailment_auxiliary_data",
        "total_rows": len(rows),
        "coverage_direction_mapping": COVERAGE_LABEL_TO_ID,
        "direction_to_final_label": DIRECTION_TO_FINAL_LABEL,
        "final_label_mapping": FINAL_LABEL_TO_ID,
        "class_counts": dict(Counter(r["coverage_direction_label"] for r in rows)),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "split_class_counts": {
            split: dict(Counter(r["coverage_direction_label"] for r in rows if r["split"] == split))
            for split in ("train", "dev")
        },
        "contrast_families": CONTRAST_FAMILIES,
        "contrast_family_counts": dict(Counter(r["contrast_family"] for r in rows)),
        "group_counts": dict(Counter(r["group"] for r in rows)),
        "schema_fields": sorted(REQUIRED_FIELDS),
        "contains_other_residual": any(r["coverage_direction_label"] == "OTHER_RESIDUAL" for r in rows),
        "hard_minimal_contrast": True,
        "diagnostic_only": True,
        "leakage_policy": (
            "Auxiliary diagnostic supervision only. Do not use "
            "data/stage31_coverage_entailment_probe.jsonl for loss, calibration, "
            "threshold selection, checkpoint selection, or final evaluation."
        ),
        "validation_errors": errors,
        "validation_passed": not errors,
    }


def build_md(report: dict[str, Any]) -> str:
    lines = [
        "# Stage31-C2 Coverage/Entailment Auxiliary Data Report",
        "",
        "## Purpose",
        "Build separate hard minimal-contrast auxiliary supervision for a readout",
        "head that distinguishes entailment-preserving SUPPORT, overclaim",
        "NOT_ENTITLED, and contradiction REFUTE.",
        "",
        "This file is not the Stage31 evaluation probe and does not reuse exact",
        "Stage31 claim/evidence pairs.",
        "",
        "## Counts",
        f"- Total rows: {report['total_rows']}",
        f"- Split counts: {report['split_counts']}",
        f"- Class counts: {report['class_counts']}",
        f"- Split class counts: {report['split_class_counts']}",
        f"- Contains OTHER_RESIDUAL: {report['contains_other_residual']}",
        "",
        "## Coverage Direction Mapping",
        "| Label | ID | Final label |",
        "|---|---:|---|",
    ]
    for label, idx in report["coverage_direction_mapping"].items():
        lines.append(f"| {label} | {idx} | {report['direction_to_final_label'][label]} |")
    lines.extend([
        "",
        "## Contrast Families",
        "| Family | Rows |",
        "|---|---:|",
    ])
    for family in report["contrast_families"]:
        lines.append(f"| {family} | {report['contrast_family_counts'].get(family, 0)} |")
    lines.extend([
        "",
        "## Schema",
        "Rows include identity, final-label fields, coverage direction fields,",
        "`contrast_set_id`, `contrast_family`, split, owner, notes, and",
        "controlled-style auxiliary compatibility fields.",
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
    print("Stage31-C2 auxiliary data generation complete.")
    print(f"  rows: {len(rows)}")
    print(f"  splits: {report['split_counts']}")
    print(f"  classes: {report['class_counts']}")
    print(f"  data: {DATA_PATH}")
    print(f"  report: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
