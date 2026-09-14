from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

DESIGN_FREEZE_COMMIT = "a31d2bc5ab4b939f52e969c89f3783feb9c3b233"
DESIGN_REL = Path(
    "reports/reason_router_gen4_xg1_cross_generator_prospective_replication_design.md"
)
DESIGN_SHA256 = "21ce2dd133721766a9a18d924774b8f02aa026364e1102809ea839c3985bf403"

SOURCE_SCHEMA = "GEN4_XG1_STRUCTURED_SOURCE_FACT_V1"
ROW_SCHEMA = "GEN4_XG1_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1"
GENERATOR_FAMILY = "xg1_independent_structured_records_v1"
MECHANISM_ID = "masked_slot_substitution_v1"

SOURCE_PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = SOURCE_PAIR_COUNT * ROWS_PER_PAIR

SOURCE_FILE = "structured_source_facts.jsonl"
ROW_FILE = "synthetic_reason_router_six_cell.jsonl"

SOURCE_FIELDS = (
    "schema_version",
    "generator_family",
    "pair_id",
    "title",
    "name",
    "role",
    "predicate",
    "object",
    "time",
    "location",
    "alternate_title",
    "alternate_name",
    "alternate_role",
    "alternate_predicate",
)

ROW_FIELDS = (
    "schema_version",
    "generator_family",
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

CANONICAL_AXIS_ORDER = ("title", "name", "role", "predicate")

CELL_SPEC = (
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
CELL_IDS = tuple(cell_id for cell_id, _, _ in CELL_SPEC)

FORBIDDEN_FIELDS = frozenset(
    {
        "final_label",
        "frame_compatible_label",
        "predicate_covered_label",
        "sufficiency_label",
        "polarity_label",
        "primary_failure_type",
        "predictions",
        "prediction",
        "logits",
        "probabilities",
        "evaluator_outputs",
        "error_cohort",
        "training_outcomes",
        "evaluation_outcomes",
        "target_C",
        "reference_C",
        "alignment_shift_abs",
        "regime",
        "baseline_plus_path_efficiency",
        "baseline_minus_path_efficiency",
        "delta_baseline",
        "alignment_plus_path_efficiency",
        "alignment_minus_path_efficiency",
        "delta_alignment",
        "R_ALIGN",
    }
)


class XG1BuildError(ValueError):
    pass


NAMES = (
    "Talia Voss",
    "Renzo Mirek",
    "Nessa Quill",
    "Borin Vale",
    "Mara Senn",
    "Ivo Calder",
    "Selene Pryce",
    "Damon Rusk",
    "Liora Venn",
    "Cato Merin",
    "Anya Solis",
    "Teo Varin",
    "Kira Dalen",
    "Milo Serrin",
    "Vera Kest",
    "Nico Arden",
)

TITLES = (
    "Envoy",
    "Steward",
    "Prefect",
    "Registrar",
    "Chancellor",
    "Warden",
    "Delegate",
    "Magistrate",
)

ROLES = (
    "archive custodian",
    "field convener",
    "registry steward",
    "district liaison",
    "survey coordinator",
    "program marshal",
    "records delegate",
    "site mediator",
)

PREDICATE_PAIRS = (
    ("certified", "reclassified"),
    ("commissioned", "decommissioned"),
    ("charted", "rerouted"),
    ("consolidated", "partitioned"),
    ("curated", "reindexed"),
    ("deployed", "recalled"),
    ("endorsed", "revoked"),
    ("finalized", "suspended"),
    ("licensed", "quarantined"),
)

OBJECT_STEMS = (
    "the Quasar registry",
    "the Juniper charter",
    "the Kestrel ledger",
    "the Lumen archive",
    "the Mosaic dossier",
    "the Northstar catalog",
    "the Opal register",
    "the Peregrine index",
    "the Quartz folio",
    "the Rowan compendium",
    "the Sable record",
    "the Tern portfolio",
    "the Umber schedule",
    "the Vela inventory",
    "the Willow codex",
    "the Zephyr register",
)

LOCATIONS = (
    "Juniper Reach",
    "Kestrel Port",
    "Lumen Crossing",
    "Mosaic Haven",
    "Northstar Quay",
    "Opal Terrace",
    "Peregrine Ward",
    "Quartz Landing",
    "Rowan Gate",
    "Sable Junction",
    "Tern Basin",
    "Umber Fields",
    "Vela Station",
    "Willow Reach",
    "Zephyr Point",
    "Cobalt Strand",
)

TIMES = (
    "the first winter week",
    "the second winter week",
    "the early spring interval",
    "the late spring interval",
    "the first summer interval",
    "the late summer interval",
    "the early autumn interval",
    "the late autumn interval",
    "the opening quarter",
    "the closing quarter",
    "the first review cycle",
    "the second review cycle",
    "the third review cycle",
    "the fourth review cycle",
    "the initial reporting window",
    "the final reporting window",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise XG1BuildError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def expected_pair_ids() -> list[str]:
    return [f"xg1_fact_{index:03d}" for index in range(1, SOURCE_PAIR_COUNT + 1)]


def render_statement(fact: Mapping[str, Any], **overrides: str) -> str:
    values = {**dict(fact), **overrides}
    return (
        f"During {values['time']}, records from {values['location']} identify "
        f"{values['title']} {values['name']} as {values['role']}; "
        f"this person {values['predicate']} {values['object']}."
    )


def _source_fact(index_zero_based: int) -> dict[str, Any]:
    require(0 <= index_zero_based < SOURCE_PAIR_COUNT, "SOURCE_INDEX_RANGE")
    i = index_zero_based

    name = NAMES[i % len(NAMES)]
    alternate_name = NAMES[(i + 7) % len(NAMES)]
    title = TITLES[(3 * i + 1) % len(TITLES)]
    alternate_title = TITLES[(3 * i + 4) % len(TITLES)]
    role = ROLES[(5 * i + 2) % len(ROLES)]
    alternate_role = ROLES[(5 * i + 5) % len(ROLES)]
    predicate, alternate_predicate = PREDICATE_PAIRS[
        (7 * i + 3) % len(PREDICATE_PAIRS)
    ]
    location = LOCATIONS[(5 * i + 4) % len(LOCATIONS)]
    time = TIMES[(7 * i + 5) % len(TIMES)]
    object_text = f"{OBJECT_STEMS[(11 * i + 6) % len(OBJECT_STEMS)]} unit {i + 1:03d}"

    row = {
        "schema_version": SOURCE_SCHEMA,
        "generator_family": GENERATOR_FAMILY,
        "pair_id": f"xg1_fact_{i + 1:03d}",
        "title": title,
        "name": name,
        "role": role,
        "predicate": predicate,
        "object": object_text,
        "time": time,
        "location": location,
        "alternate_title": alternate_title,
        "alternate_name": alternate_name,
        "alternate_role": alternate_role,
        "alternate_predicate": alternate_predicate,
    }
    validate_source_fact(row)
    return row


def build_source_facts() -> list[dict[str, Any]]:
    rows = [_source_fact(i) for i in range(SOURCE_PAIR_COUNT)]
    validate_source_facts(rows)
    return rows


def validate_source_fact(fact: Mapping[str, Any]) -> None:
    require(set(fact.keys()) == set(SOURCE_FIELDS), "SOURCE_FIELD_SET")
    require(not (FORBIDDEN_FIELDS & set(fact)), "SOURCE_FORBIDDEN_FIELD")
    require(fact["schema_version"] == SOURCE_SCHEMA, "SOURCE_SCHEMA")
    require(fact["generator_family"] == GENERATOR_FAMILY, "SOURCE_FAMILY")
    for field in SOURCE_FIELDS[2:]:
        require(isinstance(fact[field], str) and bool(fact[field]), f"SOURCE_STRING:{field}")
    for axis in CANONICAL_AXIS_ORDER:
        alternate = f"alternate_{axis}"
        require(fact[axis] != fact[alternate], f"SOURCE_AXIS_EQUAL:{axis}")


def validate_source_facts(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == SOURCE_PAIR_COUNT, "SOURCE_PAIR_COUNT")
    for row in rows:
        validate_source_fact(row)
    ids = [str(row["pair_id"]) for row in rows]
    require(ids == expected_pair_ids(), "SOURCE_PAIR_IDS")
    require(len(set(ids)) == SOURCE_PAIR_COUNT, "SOURCE_PAIR_ID_UNIQUENESS")


def cell_spec(
    cell_id: str,
) -> tuple[tuple[int, int, int, int], tuple[tuple[str, str], ...]]:
    for candidate, mask, substitutions in CELL_SPEC:
        if candidate == cell_id:
            return mask, substitutions
    raise XG1BuildError(f"UNKNOWN_CELL:{cell_id}")


def _row_id(pair_id: str, cell_id: str) -> str:
    return f"{pair_id}__{MECHANISM_ID}__{cell_id.lower()}"


def materialize_fact(fact: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_source_fact(fact)
    pair_id = str(fact["pair_id"])
    claim = render_statement(fact)
    rows: list[dict[str, Any]] = []

    for cell_id, mask, substitutions in CELL_SPEC:
        overrides = {axis: str(fact[source]) for axis, source in substitutions}
        evidence = render_statement(fact, **overrides)
        rows.append(
            {
                "schema_version": ROW_SCHEMA,
                "generator_family": GENERATOR_FAMILY,
                "mechanism_id": MECHANISM_ID,
                "source_pair_id": pair_id,
                "row_id": _row_id(pair_id, cell_id),
                "contrast_cell_id": cell_id,
                "axis_mask": list(mask),
                "intended_changed_axes": [axis for axis, _ in substitutions],
                "generator_source_fields": {
                    axis: source for axis, source in substitutions
                },
                "claim": claim,
                "evidence": evidence,
            }
        )

    validate_materialized_rows(rows, expected_pairs=1)
    return rows


def materialize_facts(
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_source_facts(facts)
    output: list[dict[str, Any]] = []
    for fact in facts:
        output.extend(materialize_fact(fact))
    validate_materialized_rows(output, expected_pairs=SOURCE_PAIR_COUNT)
    return output


def validate_materialized_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
) -> None:
    require(len(rows) == expected_pairs * ROWS_PER_PAIR, "MATERIALIZED_ROW_COUNT")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()

    for row in rows:
        require(set(row.keys()) == set(ROW_FIELDS), "ROW_FIELD_SET")
        require(not (FORBIDDEN_FIELDS & set(row)), "ROW_FORBIDDEN_FIELD")
        require(row["schema_version"] == ROW_SCHEMA, "ROW_SCHEMA")
        require(row["generator_family"] == GENERATOR_FAMILY, "ROW_FAMILY")
        require(row["mechanism_id"] == MECHANISM_ID, "ROW_MECHANISM")
        pair_id = str(row["source_pair_id"])
        row_id = str(row["row_id"])
        cell_id = str(row["contrast_cell_id"])
        require(row_id not in seen_ids, f"DUPLICATE_ROW_ID:{row_id}")
        seen_ids.add(row_id)
        mask, substitutions = cell_spec(cell_id)
        require(row["axis_mask"] == list(mask), f"ROW_MASK:{row_id}")
        require(
            row["intended_changed_axes"] == [axis for axis, _ in substitutions],
            f"ROW_AXES:{row_id}",
        )
        require(
            row["generator_source_fields"]
            == {axis: source for axis, source in substitutions},
            f"ROW_SOURCE_FIELDS:{row_id}",
        )
        require(row_id == _row_id(pair_id, cell_id), f"ROW_ID:{row_id}")
        require(isinstance(row["claim"], str) and bool(row["claim"]), f"ROW_CLAIM:{row_id}")
        require(
            isinstance(row["evidence"], str) and bool(row["evidence"]),
            f"ROW_EVIDENCE:{row_id}",
        )
        grouped[pair_id].append(row)

    require(len(grouped) == expected_pairs, "MATERIALIZED_PAIR_COUNT")
    for pair_id, block in grouped.items():
        require(len(block) == ROWS_PER_PAIR, f"CELL_COUNT:{pair_id}")
        require(
            [str(row["contrast_cell_id"]) for row in block] == list(CELL_IDS),
            f"CELL_ORDER:{pair_id}",
        )
        require(len({str(row["claim"]) for row in block}) == 1, f"CLAIM_INVARIANCE:{pair_id}")


def write_cohort(output_dir: Path) -> dict[str, str]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    design = ROOT / DESIGN_REL
    require(design.is_file(), "DESIGN_MISSING")
    require(
        hashlib.sha256(design.read_bytes()).hexdigest() == DESIGN_SHA256,
        "DESIGN_SHA256",
    )

    facts = build_source_facts()
    rows = materialize_facts(facts)
    source_raw = jsonl_bytes(facts)
    row_raw = jsonl_bytes(rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / SOURCE_FILE).write_bytes(source_raw)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    return {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the outcome-blind Gen4-K XG1 independent 300-pair / "
            "1800-row six-cell structural cohort. No tokenizer or model code is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    hashes = write_cohort(args.output_dir)
    print("RESULT=PASS_XG1_COHORT_BUILD")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    for name in (SOURCE_FILE, ROW_FILE):
        print(f"{name.upper().replace('.', '_')}_SHA256={hashes[name]}")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
