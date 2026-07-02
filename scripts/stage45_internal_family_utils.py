"""Stage45-B internal transformation-family utilities.

These helpers are import-safe and internal-only. They do not read external data,
Stage43-B1 artifacts, or reports; callers provide records already loaded from an
allowed controlled JSONL file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

FAMILY_FIELD_CANDIDATES: tuple[str, ...] = (
    "transformation_family",
    "stage15_probe_type",
    "probe_type",
    "family",
    "source_family",
    "controlled_family",
    "template_family",
    "case_type",
    "metadata.transformation_family",
    "metadata.stage15_probe_type",
    "metadata.probe_type",
)

# Stage45-B1: internal controlled-data fields recovered as usable family proxies
# when no explicit family metadata exists. Internal-only; no external data.
RECOVERED_FAMILY_FIELD_CANDIDATES: tuple[str, ...] = (
    "intervention_type",
    "primary_failure_type",
)

# Stage45-B1: derived composite internal family fields, used only when the
# simple recovered fields above are missing or degenerate.
COMPOSITE_FAMILY_FIELDS: tuple[str, ...] = (
    "intervention_type+primary_failure_type",
    "intervention_type+final_label",
    "primary_failure_type+final_label",
)

# Auto-resolution preference order: explicit family metadata first, then
# recovered internal fields, then fallback. Composites are never chosen
# automatically.
AUTO_FAMILY_FIELD_ORDER: tuple[str, ...] = (
    FAMILY_FIELD_CANDIDATES + RECOVERED_FAMILY_FIELD_CANDIDATES
)

LABELS: tuple[str, ...] = ("SUPPORT", "REFUTE", "NOT_ENTITLED")


def _lookup_dotted(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _clean_family(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _split_composite_field(field: str) -> list[str] | None:
    """Return component field names if `field` is a '+'-joined composite, else None."""
    if "+" not in field:
        return None
    parts = [part.strip() for part in field.split("+") if part.strip()]
    return parts if len(parts) >= 2 else None


def _resolve_field_value(record: dict[str, Any], field: str) -> str | None:
    """Resolve one candidate field (simple or composite) to a clean string value."""
    parts = _split_composite_field(field)
    if parts is None:
        return _clean_family(_lookup_dotted(record, field))
    values = [_clean_family(_lookup_dotted(record, part)) for part in parts]
    if not all(values):
        return None
    return "|".join(values)  # type: ignore[arg-type]


def resolve_family(record: dict[str, Any], family_field: str = "auto") -> tuple[str, str]:
    """Return (family, field_used) for one controlled record."""
    if family_field != "auto":
        value = _resolve_field_value(record, family_field)
        return (value or "unknown_family", family_field if value else "fallback:unknown_family")
    for field in AUTO_FAMILY_FIELD_ORDER:
        value = _resolve_field_value(record, field)
        if value:
            return value, field
    return "unknown_family", "fallback:unknown_family"


def is_time_swap_like(record: dict[str, Any], family: str | None = None) -> bool:
    values: list[Any] = [family]
    for field in (
        "transformation_family",
        "stage15_probe_type",
        "probe_type",
        "family",
        "source_family",
        "controlled_family",
        "template_family",
        "case_type",
        "intervention_type",
        "metadata.transformation_family",
        "metadata.stage15_probe_type",
        "metadata.probe_type",
    ):
        values.append(_lookup_dotted(record, field))
    return any("time_swap" in str(value).lower() for value in values if value is not None)


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        label = str(record.get("final_label", "")).strip()
        if label in counts:
            counts[label] += 1
        elif label:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _available_top_level_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for record in records:
        keys.update(str(key) for key in record.keys())
    return sorted(keys)


def _available_metadata_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for record in records:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            keys.update(str(key) for key in metadata.keys())
    return sorted(keys)


_AUDITED_CANDIDATE_FIELDS: tuple[str, ...] = (
    FAMILY_FIELD_CANDIDATES + RECOVERED_FAMILY_FIELD_CANDIDATES + COMPOSITE_FAMILY_FIELDS
)


def _field_family_counts(records: list[dict[str, Any]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        value = _resolve_field_value(record, field)
        if value is not None:
            counts[value] += 1
    return counts


def _candidate_field_unique_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for field in _AUDITED_CANDIDATE_FIELDS:
        values = _field_family_counts(records, field)
        missing = len(records) - sum(values.values())
        audit[field] = {
            "present_rows": sum(values.values()),
            "missing_rows": missing,
            "unique_count": len(values),
            "unique_values": dict(sorted(values.items())),
        }
    return audit


def _eligible_holdout_families(
    counts: Counter[str], *, total_rows: int, min_family_size: int
) -> list[str]:
    """Families eligible for leave-family-out holdout: not degenerate, meet the
    minimum size, and never consume every row (which would leave zero
    training rows)."""
    if len(counts) < 2:
        return []
    return sorted(
        family
        for family, count in counts.items()
        if count >= min_family_size and count < total_rows
    )


def _stage45b1_recovery_summary(
    records: list[dict[str, Any]], *, min_family_size: int
) -> dict[str, Any]:
    total_rows = len(records)
    field_summary: dict[str, Any] = {}
    for field in RECOVERED_FAMILY_FIELD_CANDIDATES + COMPOSITE_FAMILY_FIELDS:
        counts = _field_family_counts(records, field)
        eligible = _eligible_holdout_families(
            counts, total_rows=total_rows, min_family_size=min_family_size
        )
        field_summary[field] = {
            "family_counts": dict(sorted(counts.items())),
            "unique_count": len(counts),
            "eligible_holdout_families": eligible,
        }

    intervention_type_eligible = field_summary["intervention_type"]["eligible_holdout_families"]
    primary_failure_type_eligible = field_summary["primary_failure_type"][
        "eligible_holdout_families"
    ]
    composite_it_pft_eligible = field_summary["intervention_type+primary_failure_type"][
        "eligible_holdout_families"
    ]

    if intervention_type_eligible:
        recommended_fields = ["intervention_type"]
        recommendation = (
            "intervention_type yields multiple eligible internal families "
            f"({len(intervention_type_eligible)} eligible); recommended as the "
            "primary Stage45-B1 leave-family-out holdout field."
        )
    elif primary_failure_type_eligible:
        recommended_fields = ["primary_failure_type"]
        recommendation = (
            "intervention_type is degenerate or has no eligible holdout families; "
            "primary_failure_type yields eligible internal families "
            f"({len(primary_failure_type_eligible)} eligible) and is recommended instead."
        )
    elif composite_it_pft_eligible:
        recommended_fields = ["intervention_type+primary_failure_type"]
        recommendation = (
            "Both intervention_type and primary_failure_type are weak or degenerate; "
            "recommended composite intervention_type+primary_failure_type as the "
            "Stage45-B1 holdout field "
            f"({len(composite_it_pft_eligible)} eligible)."
        )
    else:
        recommended_fields = []
        recommendation = (
            "No internal recovered or composite field yields an eligible "
            "leave-family-out holdout family. Next valid step is Stage45-C "
            "controlled family annotation/generation plan."
        )

    return {
        "stage45b1_family_recovery_enabled": True,
        "stage45b1_recovered_candidate_fields": list(
            RECOVERED_FAMILY_FIELD_CANDIDATES + COMPOSITE_FAMILY_FIELDS
        ),
        "stage45b1_field_summary": field_summary,
        "stage45b1_recommended_family_fields": recommended_fields,
        "stage45b1_recommendation": recommendation,
        "stage45b1_decision": (
            "STAGE45B1_INTERNAL_FAMILY_RECOVERY_READY"
            if recommended_fields
            else "STAGE45B1_INTERNAL_FAMILY_RECOVERY_INCOMPLETE"
        ),
    }


def build_family_manifest(
    records: list[dict[str, Any]],
    *,
    input_jsonl: str,
    min_family_size: int = 20,
    family_field: str = "auto",
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    label_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    field_counts: Counter[str] = Counter()
    warnings: list[str] = []
    time_swap_families: set[str] = set()

    for record in records:
        family, field_used = resolve_family(record, family_field)
        family_counts[family] += 1
        field_counts[field_used] += 1
        label = str(record.get("final_label", "unknown_label")).strip() or "unknown_label"
        label_counts_by_family[family][label] += 1
        if is_time_swap_like(record, family):
            time_swap_families.add(family)

    all_rows_unknown_family = (
        len(records) > 0
        and len(family_counts) == 1
        and family_counts.get("unknown_family") == len(records)
    )
    eligible = _eligible_holdout_families(
        family_counts, total_rows=len(records), min_family_size=min_family_size
    )
    tiny = sorted(
        family for family, count in family_counts.items()
        if count < min_family_size
    )
    if all_rows_unknown_family:
        warnings.append("all_rows_unknown_family_no_leave_family_out_possible")
    elif len(family_counts) < 2:
        warnings.append("degenerate_single_family_no_leave_family_out_possible")
    if tiny:
        warnings.append(
            "Tiny families below min_family_size: "
            + ", ".join(f"{family}={family_counts[family]}" for family in tiny)
        )
    if time_swap_families:
        warnings.append(
            "time_swap-like family appears in internal data: "
            + ", ".join(sorted(time_swap_families))
        )

    if not records or not eligible or all_rows_unknown_family:
        decision = "STAGE45B_INTERNAL_FAMILY_MANIFEST_INCOMPLETE"
    else:
        decision = "STAGE45B_INTERNAL_FAMILY_MANIFEST_READY"

    stage45b1_summary = _stage45b1_recovery_summary(records, min_family_size=min_family_size)

    if len(field_counts) == 1:
        resolved_field = next(iter(field_counts)) if field_counts else "fallback:unknown_family"
    else:
        resolved_field = "auto:mixed"

    return {
        "decision": decision,
        "input_jsonl": input_jsonl,
        "total_rows": len(records),
        "family_field_used": resolved_field if family_field == "auto" else family_field,
        "family_field_usage": dict(sorted(field_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "label_counts_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(label_counts_by_family.items())
        },
        "eligible_holdout_families": eligible,
        "leave_family_out_possible": bool(eligible) and not all_rows_unknown_family,
        "min_family_size": min_family_size,
        "warnings": warnings,
        "available_top_level_keys": _available_top_level_keys(records),
        "available_metadata_keys": _available_metadata_keys(records),
        "candidate_field_unique_counts": _candidate_field_unique_counts(records),
        "recommendation": (
            "If no internal family metadata exists, Stage45-B cannot perform "
            "leave-family-out validation on this dataset. Next valid step is "
            "Stage45-B1 internal family recovery from existing internal IDs/metadata "
            "if available, or Stage45-C controlled family annotation/generation plan."
        ),
        "stage45b1_family_recovery_enabled": stage45b1_summary["stage45b1_family_recovery_enabled"],
        "stage45b1_recovered_candidate_fields": stage45b1_summary[
            "stage45b1_recovered_candidate_fields"
        ],
        "stage45b1_field_summary": stage45b1_summary["stage45b1_field_summary"],
        "stage45b1_recommended_family_fields": stage45b1_summary[
            "stage45b1_recommended_family_fields"
        ],
        "stage45b1_recommendation": stage45b1_summary["stage45b1_recommendation"],
        "stage45b1_decision": stage45b1_summary["stage45b1_decision"],
        "leakage_policy": {
            "scope": "internal_controlled_jsonl_only",
            "stage43b1_files_read": False,
            "external_examples_used": False,
            "external_labels_or_metrics_used": False,
            "used_for_training": False,
            "used_for_threshold_selection": False,
            "used_for_calibration": False,
            "used_for_checkpoint_selection": False,
        },
    }


def split_leave_family_out(
    records: list[dict[str, Any]],
    *,
    holdout_family: str,
    family_field: str = "auto",
    min_holdout_size: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    warnings: list[str] = []
    time_swap_families: set[str] = set()

    for record in records:
        family, field_used = resolve_family(record, family_field)
        family_counts[family] += 1
        field_counts[field_used] += 1
        if is_time_swap_like(record, family):
            time_swap_families.add(family)
        if family == holdout_family:
            holdout_records.append(record)
        else:
            train_records.append(record)

    if time_swap_families:
        warnings.append(
            "time_swap-like family appears in internal data: "
            + ", ".join(sorted(time_swap_families))
        )
    if holdout_family not in family_counts:
        raise ValueError(
            f"[stage45b] holdout family {holdout_family!r} not found in internal data. "
            f"Available families: {sorted(family_counts)}"
        )
    if not train_records:
        raise ValueError(
            "[stage45b] STAGE45B_INTERNAL_FAMILY_HOLDOUT_INCOMPLETE: "
            "family holdout split would produce zero training rows; "
            "leave-family-out validation is not meaningful. "
            f"total_rows={len(records)} holdout_family={holdout_family!r} "
            f"holdout_rows={len(holdout_records)} train_rows={len(train_records)} "
            f"family_counts={dict(sorted(family_counts.items()))}. "
            "If all rows resolve to unknown_family, Stage45-B cannot perform "
            "leave-family-out validation on this dataset. Next valid step is "
            "Stage45-B1 internal family recovery from existing internal IDs/metadata "
            "if available, or Stage45-C controlled family annotation/generation plan."
        )
    if len(holdout_records) < min_holdout_size:
        raise ValueError(
            f"[stage45b] holdout family {holdout_family!r} has {len(holdout_records)} rows, "
            f"below --stage45-min-holdout-size={min_holdout_size}."
        )

    if len(field_counts) == 1:
        resolved_field = next(iter(field_counts)) if field_counts else "fallback:unknown_family"
    else:
        resolved_field = "auto:mixed"

    meta = {
        "stage45b_enabled": True,
        "stage45b_decision": "STAGE45B_INTERNAL_FAMILY_HOLDOUT_READY",
        "stage45b_family_field_used": resolved_field if family_field == "auto" else family_field,
        "stage45b_family_field_usage": dict(sorted(field_counts.items())),
        "stage45b_holdout_family": holdout_family,
        "stage45b_train_rows": len(train_records),
        "stage45b_holdout_rows": len(holdout_records),
        "stage45b_train_label_counts": label_counts(train_records),
        "stage45b_holdout_label_counts": label_counts(holdout_records),
        "stage45b_family_counts": dict(sorted(family_counts.items())),
        "stage45b_warnings": warnings,
        "stage45b_leakage_policy": (
            "Stage45-B family holdout uses only the internal --data JSONL and internal "
            "controlled metadata. It does not read or use Stage43-B1, VitaminC, "
            "Climate-FEVER, external examples, external labels, external metrics, "
            "external prediction distributions, or external reports for splitting, "
            "selection, thresholding, calibration, training, or design choices."
        ),
        "stage45b_recommendation": (
            "Use this scaffold to diagnose internal controlled-family robustness before "
            "any Stage45-C redesign; do not treat it as external validation."
        ),
    }
    return train_records, holdout_records, meta


def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Stage45-B Internal Family Manifest",
        "",
        "## Decision",
        "",
        f"`{manifest['decision']}`",
        "",
        "## Summary",
        "",
        f"- Input JSONL: `{manifest['input_jsonl']}`",
        f"- Total rows: {manifest['total_rows']}",
        f"- Family field used: `{manifest['family_field_used']}`",
        f"- Minimum family size: {manifest['min_family_size']}",
        f"- Leave-family-out possible: {manifest.get('leave_family_out_possible')}",
        "",
        "## Family Counts",
        "",
        "| Family | Rows |",
        "|---|---:|",
    ]
    for family, count in manifest["family_counts"].items():
        lines.append(f"| {family} | {count} |")
    lines.extend(["", "## Label Counts By Family", ""])
    for family, counts in manifest["label_counts_by_family"].items():
        lines.append(f"### {family}")
        lines.append("")
        for label, count in counts.items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    lines.extend([
        "## Eligible Holdout Families",
        "",
    ])
    if manifest["eligible_holdout_families"]:
        for family in manifest["eligible_holdout_families"]:
            lines.append(f"- {family}")
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if manifest["warnings"]:
        for warning in manifest["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Metadata Audit",
        "",
        "### Available Top-Level Keys",
        "",
    ])
    if manifest.get("available_top_level_keys"):
        for key in manifest["available_top_level_keys"]:
            lines.append(f"- `{key}`")
    else:
        lines.append("- None")
    lines.extend(["", "### Available Metadata Keys", ""])
    if manifest.get("available_metadata_keys"):
        for key in manifest["available_metadata_keys"]:
            lines.append(f"- `{key}`")
    else:
        lines.append("- None")
    lines.extend(["", "### Candidate Field Unique Counts", ""])
    for field, audit in (manifest.get("candidate_field_unique_counts") or {}).items():
        lines.append(
            f"- `{field}`: present_rows={audit.get('present_rows')} "
            f"unique_count={audit.get('unique_count')}"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        manifest.get("recommendation", "None"),
        "",
        "## Stage45-B1 Internal Family Recovery",
        "",
        f"`{manifest.get('stage45b1_decision', 'None')}`",
        "",
        f"- Recovery enabled: {manifest.get('stage45b1_family_recovery_enabled')}",
        "- Recovered/composite candidate fields: "
        + (
            ", ".join(f"`{f}`" for f in manifest.get("stage45b1_recovered_candidate_fields") or [])
            or "None"
        ),
        "- Recommended family field(s): "
        + (
            ", ".join(f"`{f}`" for f in manifest.get("stage45b1_recommended_family_fields") or [])
            or "None"
        ),
        "",
        manifest.get("stage45b1_recommendation", "None"),
        "",
        "## Leakage Policy",
        "",
        "This manifest is internal-only. It does not read Stage43-B1, VitaminC, Climate-FEVER, external examples, external labels, external metrics, external prediction distributions, or external reports.",
    ])
    return "\n".join(lines) + "\n"
