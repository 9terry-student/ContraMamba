from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_controlled_v5 import (
    _statement as _frozen_statement_renderer,
    fact_templates_for_count,
)


GENERATOR_PATH = "scripts/build_controlled_v5.py"

SCHEMA_VERSION = "GEN4_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1"
GENERATOR_AUTHORITY_COMMIT = "91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea"
GENERATOR_SOURCE_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
CONTRAST_SPECIFICATION_COMMIT = "0a0da5354782e542520fb5bba146ab1a599d17ef"
MECHANISM_ID = "masked_slot_substitution_v1"

OUTPUT_FIELDS = (
    "schema_version",
    "generator_authority_commit",
    "generator_source_blob",
    "contrast_specification_commit",
    "mechanism_id",
    "source_pair_id",
    "row_id",
    "contrast_cell_id",
    "axis_mask",
    "intended_changed_axes",
    "generator_source_fields",
    "claim",
    "evidence",
)

CANONICAL_AXIS_ORDER = (
    "title",
    "name",
    "role",
    "predicate",
)

REQUIRED_SEMANTIC_SOURCE_FIELDS = (
    "pair_id",
    "title",
    "name",
    "role",
    "predicate",
    "alternate_title",
    "alternate_name",
    "alternate_role",
    "alternate_predicate",
)

REQUIRED_RENDER_SOURCE_FIELDS = (
    "object",
    "time",
    "location",
)

_REQUIRED_SOURCE_FIELDS = (
    *REQUIRED_SEMANTIC_SOURCE_FIELDS,
    *REQUIRED_RENDER_SOURCE_FIELDS,
)

_AXIS_TO_ALTERNATE = {
    "title": "alternate_title",
    "name": "alternate_name",
    "role": "alternate_role",
    "predicate": "alternate_predicate",
}

_CELL_SPEC: tuple[
    tuple[str, tuple[int, int, int, int], tuple[tuple[str, str], ...]], ...
] = (
    ("C0_SHAM", (0, 0, 0, 0), ()),
    ("C1_TITLE", (1, 0, 0, 0), (("title", "alternate_title"),)),
    ("C2_NAME", (0, 1, 0, 0), (("name", "alternate_name"),)),
    ("C3_ROLE", (0, 0, 1, 0), (("role", "alternate_role"),)),
    ("C4_PREDICATE", (0, 0, 0, 1), (("predicate", "alternate_predicate"),)),
    (
        "C5_TITLE_NAME",
        (1, 1, 0, 0),
        (("title", "alternate_title"), ("name", "alternate_name")),
    ),
)

CELL_IDS = tuple(cell_id for cell_id, _, _ in _CELL_SPEC)
CELL_MASKS = tuple(mask for _, mask, _ in _CELL_SPEC)

FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "final_label",
        "frame_compatible_label",
        "predicate_covered_label",
        "sufficiency_label",
        "polarity_label",
        "primary_failure_type",
        "predictions",
        "logits",
        "probabilities",
        "evaluator_outputs",
        "error_cohort",
        "training_outcomes",
        "evaluation_outcomes",
    }
)


class MaterializationError(ValueError):
    """Fail-closed error for Gen4 six-cell structural materialization."""


def _require_nonempty_string(fact: Mapping[str, Any], field: str) -> str:
    if field not in fact:
        raise MaterializationError(f"missing required source field: {field}")
    value = fact[field]
    if not isinstance(value, str):
        raise MaterializationError(
            f"required source field must be a string: {field}"
        )
    if not value:
        raise MaterializationError(
            f"required source field must be non-empty: {field}"
        )
    return value


def validate_structured_fact(fact: Mapping[str, Any]) -> None:
    if not isinstance(fact, Mapping):
        raise MaterializationError("structured fact must be a mapping")

    for field in _REQUIRED_SOURCE_FIELDS:
        _require_nonempty_string(fact, field)

    for axis in CANONICAL_AXIS_ORDER:
        alternate = _AXIS_TO_ALTERNATE[axis]
        if fact[axis] == fact[alternate]:
            raise MaterializationError(
                f"original and alternate values must differ: {axis}/{alternate}"
            )


def cell_spec(contrast_cell_id: str) -> dict[str, object]:
    for cell_id, mask, pairs in _CELL_SPEC:
        if contrast_cell_id != cell_id:
            continue
        axes = [axis for axis, _ in pairs]
        fields = {axis: source for axis, source in pairs}
        return {
            "contrast_cell_id": cell_id,
            "axis_mask": list(mask),
            "intended_changed_axes": axes,
            "generator_source_fields": fields,
        }
    raise MaterializationError(f"unknown contrast cell: {contrast_cell_id!r}")


def _row_id(source_pair_id: str, contrast_cell_id: str) -> str:
    return (
        f"{source_pair_id}__{MECHANISM_ID}__{contrast_cell_id.lower()}"
    )


def _overrides_for_cell(
    fact: Mapping[str, Any],
    contrast_cell_id: str,
) -> dict[str, str]:
    spec = cell_spec(contrast_cell_id)
    source_fields = spec["generator_source_fields"]
    assert isinstance(source_fields, dict)
    return {
        axis: str(fact[source_field])
        for axis, source_field in source_fields.items()
    }


def materialize_fact(
    fact: Mapping[str, Any],
    *,
    statement_renderer: Callable[..., str] = _frozen_statement_renderer,
) -> list[dict[str, object]]:
    validate_structured_fact(fact)
    source_pair_id = str(fact["pair_id"])
    claim = statement_renderer(dict(fact))

    rows: list[dict[str, object]] = []
    seen_row_ids: set[str] = set()

    for contrast_cell_id in CELL_IDS:
        spec = cell_spec(contrast_cell_id)
        overrides = _overrides_for_cell(fact, contrast_cell_id)
        evidence = statement_renderer(dict(fact), **overrides)
        row_id = _row_id(source_pair_id, contrast_cell_id)

        if row_id in seen_row_ids:
            raise MaterializationError(f"duplicate generated row id: {row_id}")
        seen_row_ids.add(row_id)

        row: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generator_authority_commit": GENERATOR_AUTHORITY_COMMIT,
            "generator_source_blob": GENERATOR_SOURCE_BLOB,
            "contrast_specification_commit": CONTRAST_SPECIFICATION_COMMIT,
            "mechanism_id": MECHANISM_ID,
            "source_pair_id": source_pair_id,
            "row_id": row_id,
            "contrast_cell_id": contrast_cell_id,
            "axis_mask": list(spec["axis_mask"]),
            "intended_changed_axes": list(spec["intended_changed_axes"]),
            "generator_source_fields": dict(spec["generator_source_fields"]),
            "claim": claim,
            "evidence": evidence,
        }

        if tuple(row.keys()) != OUTPUT_FIELDS:
            raise MaterializationError(
                "materialized output schema or field order is invalid"
            )
        if FORBIDDEN_OUTPUT_FIELDS & set(row):
            raise MaterializationError("forbidden label/outcome field emitted")
        rows.append(row)

    validate_materialized_rows(rows)
    return rows


def materialize_facts(
    facts: Iterable[Mapping[str, Any]],
    *,
    statement_renderer: Callable[..., str] = _frozen_statement_renderer,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen_pair_ids: set[str] = set()

    for fact in facts:
        if not isinstance(fact, Mapping):
            raise MaterializationError("structured fact must be a mapping")
        pair_id = _require_nonempty_string(fact, "pair_id")
        if pair_id in seen_pair_ids:
            raise MaterializationError(f"duplicate source pair id: {pair_id}")
        seen_pair_ids.add(pair_id)
        output.extend(
            materialize_fact(fact, statement_renderer=statement_renderer)
        )

    validate_materialized_rows(output)
    return output


def validate_materialized_rows(rows: Sequence[Mapping[str, object]]) -> None:
    seen_row_ids: set[str] = set()
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    pair_order: list[str] = []

    for row in rows:
        if tuple(row.keys()) != OUTPUT_FIELDS:
            raise MaterializationError(
                "materialized row has unauthorized schema or field order"
            )
        if FORBIDDEN_OUTPUT_FIELDS & set(row):
            raise MaterializationError("forbidden label/outcome field emitted")

        if row["schema_version"] != SCHEMA_VERSION:
            raise MaterializationError("schema_version drift")
        if row["generator_authority_commit"] != GENERATOR_AUTHORITY_COMMIT:
            raise MaterializationError("generator authority drift")
        if row["generator_source_blob"] != GENERATOR_SOURCE_BLOB:
            raise MaterializationError("generator source blob drift")
        if row["contrast_specification_commit"] != CONTRAST_SPECIFICATION_COMMIT:
            raise MaterializationError("contrast specification drift")
        if row["mechanism_id"] != MECHANISM_ID:
            raise MaterializationError("mechanism_id drift")

        pair_id = row["source_pair_id"]
        row_id = row["row_id"]
        cell_id = row["contrast_cell_id"]
        if not isinstance(pair_id, str) or not pair_id:
            raise MaterializationError("invalid source_pair_id")
        if not isinstance(row_id, str) or not row_id:
            raise MaterializationError("invalid row_id")
        if not isinstance(cell_id, str):
            raise MaterializationError("invalid contrast_cell_id")

        if row_id in seen_row_ids:
            raise MaterializationError(f"duplicate generated row id: {row_id}")
        seen_row_ids.add(row_id)

        expected = cell_spec(cell_id)
        if row["axis_mask"] != expected["axis_mask"]:
            raise MaterializationError("contrast cell axis mask mismatch")
        if row["intended_changed_axes"] != expected["intended_changed_axes"]:
            raise MaterializationError("contrast cell intended axis mismatch")
        if row["generator_source_fields"] != expected["generator_source_fields"]:
            raise MaterializationError(
                "contrast cell generator source-field mismatch"
            )
        if row_id != _row_id(pair_id, cell_id):
            raise MaterializationError("deterministic row id mismatch")

        if pair_id not in grouped:
            pair_order.append(pair_id)
        grouped[pair_id].append(row)

    for pair_id in pair_order:
        block = grouped[pair_id]
        if len(block) != len(CELL_IDS):
            raise MaterializationError(
                f"incomplete six-cell block: {pair_id}"
            )
        observed_cells = [str(row["contrast_cell_id"]) for row in block]
        if observed_cells != list(CELL_IDS):
            raise MaterializationError(
                f"cell ordering or membership violation: {pair_id}"
            )
        observed_masks = [row["axis_mask"] for row in block]
        if observed_masks != [list(mask) for mask in CELL_MASKS]:
            raise MaterializationError(
                f"canonical mask ordering violation: {pair_id}"
            )
        claims = {row["claim"] for row in block}
        if len(claims) != 1:
            raise MaterializationError(
                f"claim invariance violation: {pair_id}"
            )


def serialize_materialized_rows(
    rows: Sequence[Mapping[str, object]],
) -> str:
    validate_materialized_rows(rows)
    if not rows:
        return ""
    lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def materialize_jsonl(
    facts: Iterable[Mapping[str, Any]],
    *,
    statement_renderer: Callable[..., str] = _frozen_statement_renderer,
) -> str:
    return serialize_materialized_rows(
        materialize_facts(facts, statement_renderer=statement_renderer)
    )


def write_materialized_jsonl(
    facts: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    serialized = materialize_jsonl(facts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8", newline="\n")


def _git_stdout(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise MaterializationError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_frozen_generator_source() -> None:
    authority_blob = _git_stdout(
        "rev-parse",
        f"{GENERATOR_AUTHORITY_COMMIT}:{GENERATOR_PATH}",
    )
    if authority_blob != GENERATOR_SOURCE_BLOB:
        raise MaterializationError("frozen generator authority blob mismatch")

    current_blob = _git_stdout("rev-parse", f"HEAD:{GENERATOR_PATH}")
    if current_blob != GENERATOR_SOURCE_BLOB:
        raise MaterializationError("current HEAD generator blob mismatch")

    result = subprocess.run(
        ["git", "diff", "--quiet", "--", GENERATOR_PATH],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise MaterializationError(
            "working-copy generator differs from current HEAD"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Gen4 six-cell masked-slot-substitution structural rows. "
            "Canonical execution requires separate authority."
        )
    )
    parser.add_argument("--num-pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_pairs < 1:
        raise MaterializationError("num-pairs must be positive")
    verify_frozen_generator_source()
    facts = fact_templates_for_count(args.num_pairs)
    write_materialized_jsonl(facts, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
