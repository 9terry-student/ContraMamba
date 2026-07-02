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


def resolve_family(record: dict[str, Any], family_field: str = "auto") -> tuple[str, str]:
    """Return (family, field_used) for one controlled record."""
    if family_field != "auto":
        value = _clean_family(_lookup_dotted(record, family_field))
        return (value or "unknown_family", family_field if value else "fallback:unknown_family")
    for field in FAMILY_FIELD_CANDIDATES:
        value = _clean_family(_lookup_dotted(record, field))
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

    eligible = sorted(
        family for family, count in family_counts.items()
        if count >= min_family_size
    )
    tiny = sorted(
        family for family, count in family_counts.items()
        if count < min_family_size
    )
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

    if not records or not eligible:
        decision = "STAGE45B_INTERNAL_FAMILY_MANIFEST_INCOMPLETE"
    else:
        decision = "STAGE45B_INTERNAL_FAMILY_MANIFEST_READY"

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
        "min_family_size": min_family_size,
        "warnings": warnings,
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
    if len(holdout_records) < min_holdout_size:
        raise ValueError(
            f"[stage45b] holdout family {holdout_family!r} has {len(holdout_records)} rows, "
            f"below --stage45-min-holdout-size={min_holdout_size}."
        )
    if not train_records:
        raise ValueError("[stage45b] family holdout split produced zero training rows.")

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
        "## Leakage Policy",
        "",
        "This manifest is internal-only. It does not read Stage43-B1, VitaminC, Climate-FEVER, external examples, external labels, external metrics, external prediction distributions, or external reports.",
    ])
    return "\n".join(lines) + "\n"
