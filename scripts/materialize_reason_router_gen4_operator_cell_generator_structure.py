from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_V1"
AUTHORITY_COMMIT = "91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea"

OUTPUT_FIELDS = (
    "schema_version",
    "authority_commit",
    "row_id",
    "pair_id",
    "intervention_type",
    "intended_changed_axes",
    "generator_source_fields",
    "operator_cells",
)

CANONICAL_AXIS_ORDER = (
    "title",
    "name",
    "role",
    "predicate",
)

_OPERATOR_SPEC: dict[str, tuple[tuple[str, str], ...]] = {
    "entity_swap": (
        ("name", "alternate_name"),
    ),
    "role_swap": (
        ("role", "alternate_role"),
    ),
    "title_name_swap": (
        ("title", "alternate_title"),
        ("name", "alternate_name"),
    ),
    "predicate_swap": (
        ("predicate", "alternate_predicate"),
    ),
}


class MaterializationError(ValueError):
    """Fail-closed Gen4 generator-structure materialization error."""


def validate_authority_commit(authority_commit: str) -> None:
    if authority_commit != AUTHORITY_COMMIT:
        raise MaterializationError(
            "generator semantic authority commit is fixed by scientific authority: "
            f"expected={AUTHORITY_COMMIT!r} observed={authority_commit!r}"
        )


def is_target_operator(intervention_type: object) -> bool:
    return isinstance(intervention_type, str) and intervention_type in _OPERATOR_SPEC


def _require_nonempty_string(row: Mapping[str, Any], field: str) -> str:
    if field not in row:
        raise MaterializationError(f"missing required field: {field}")

    value = row[field]
    if not isinstance(value, str) or not value:
        raise MaterializationError(
            f"required field must be a non-empty string: {field}"
        )
    return value


def operator_spec(intervention_type: str) -> dict[str, object]:
    if not isinstance(intervention_type, str) or not intervention_type:
        raise MaterializationError(
            "intervention_type must be a non-empty string"
        )

    try:
        pairs = _OPERATOR_SPEC[intervention_type]
    except KeyError as exc:
        raise MaterializationError(
            f"unauthorized Gen4 operator: {intervention_type!r}"
        ) from exc

    axes = [axis for axis, _ in pairs]

    ordered_axes = [
        axis for axis in CANONICAL_AXIS_ORDER
        if axis in axes
    ]
    if axes != ordered_axes:
        raise MaterializationError(
            f"operator mapping violates canonical axis order: {intervention_type}"
        )

    generator_source_fields = {
        axis: source_field
        for axis, source_field in pairs
    }

    operator_cells = [
        f"{intervention_type}:{axis}"
        for axis in axes
    ]

    return {
        "intended_changed_axes": axes,
        "generator_source_fields": generator_source_fields,
        "operator_cells": operator_cells,
    }


def materialize_row(row: Mapping[str, Any]) -> dict[str, object] | None:
    if not isinstance(row, Mapping):
        raise MaterializationError("source row must be a mapping")

    if "intervention_type" not in row:
        raise MaterializationError("missing required field: intervention_type")

    intervention_type = row["intervention_type"]
    if not isinstance(intervention_type, str) or not intervention_type:
        raise MaterializationError(
            "required field must be a non-empty string: intervention_type"
        )

    if not is_target_operator(intervention_type):
        return None

    row_id = _require_nonempty_string(row, "id")
    pair_id = _require_nonempty_string(row, "pair_id")

    spec = operator_spec(intervention_type)

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "authority_commit": AUTHORITY_COMMIT,
        "row_id": row_id,
        "pair_id": pair_id,
        "intervention_type": intervention_type,
        "intended_changed_axes": spec["intended_changed_axes"],
        "generator_source_fields": spec["generator_source_fields"],
        "operator_cells": spec["operator_cells"],
    }

    if tuple(result.keys()) != OUTPUT_FIELDS:
        raise MaterializationError(
            "materialized output schema or field order is invalid"
        )

    return result


def materialize_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen_target_row_ids: set[str] = set()

    for row in rows:
        materialized = materialize_row(row)
        if materialized is None:
            continue

        row_id = str(materialized["row_id"])
        if row_id in seen_target_row_ids:
            raise MaterializationError(
                f"duplicate target row id: {row_id}"
            )

        seen_target_row_ids.add(row_id)
        output.append(materialized)

    return output


def serialize_materialized_rows(
    rows: Sequence[Mapping[str, object]],
) -> str:
    lines: list[str] = []

    for row in rows:
        if tuple(row.keys()) != OUTPUT_FIELDS:
            raise MaterializationError(
                "materialized row has unauthorized schema or field order"
            )

        line = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        lines.append(line)

    if not lines:
        return ""

    return "\n".join(lines) + "\n"


def materialize_jsonl(
    rows: Iterable[Mapping[str, Any]],
) -> str:
    return serialize_materialized_rows(materialize_rows(rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(
                    f"invalid JSON at line {line_number}: {path}"
                ) from exc

            if not isinstance(value, dict):
                raise MaterializationError(
                    f"JSON row must be an object at line {line_number}: {path}"
                )

            rows.append(value)

    return rows


def write_sidecar(input_path: Path, output_path: Path) -> None:
    rows = read_jsonl(input_path)
    serialized = materialize_jsonl(rows)
    output_path.write_text(serialized, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize frozen Gen4 operator-cell generator structure. "
            "Canonical scientific execution requires separate authority."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_authority_commit(AUTHORITY_COMMIT)
    write_sidecar(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())