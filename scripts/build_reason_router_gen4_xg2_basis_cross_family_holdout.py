from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as calibration


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

SCOPE_FREEZE_COMMIT = "c5679b8a22f103956a46cbc665354b4f9cd7063a"
SCOPE_REL = Path(
    "reports/reason_router_gen4_xg2_basis_cross_family_holdout_scope.md"
)
SCOPE_BLOB = "94baa8cda2c2b2ea5400609f89588ae212507553"

CALIBRATION_BUILDER_REL = Path(
    "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py"
)
CALIBRATION_BUILDER_BLOB = "4505acc99db0627733592694da290a007d281421"

FAMILY_KEYS = ("xg2", "xg4")

HOLDOUT_FIRST_PAIR = 601
HOLDOUT_LAST_PAIR = 900
HOLDOUT_PAIR_COUNT = HOLDOUT_LAST_PAIR - HOLDOUT_FIRST_PAIR + 1

ROWS_PER_PAIR = calibration.ROWS_PER_PAIR
EXPECTED_ROW_COUNT = HOLDOUT_PAIR_COUNT * ROWS_PER_PAIR

SOURCE_FILE = calibration.SOURCE_FILE
ROW_FILE = calibration.ROW_FILE


class XG2BasisCrossFamilyHoldoutBuildError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise XG2BasisCrossFamilyHoldoutBuildError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XG2BasisCrossFamilyHoldoutBuildError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_authority() -> None:
    branch = git("branch", "--show-current")
    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )

    head = git("rev-parse", "HEAD")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", SCOPE_FREEZE_COMMIT, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "SCOPE_FREEZE_NOT_ANCESTOR")

    observed_design_blob = git(
        "rev-parse",
        f"{SCOPE_FREEZE_COMMIT}:{SCOPE_REL.as_posix()}",
    )
    require(
        observed_design_blob == SCOPE_BLOB,
        f"SCOPE_BLOB:{observed_design_blob}",
    )

    current_design_blob = git(
        "rev-parse",
        f"HEAD:{SCOPE_REL.as_posix()}",
    )
    require(
        current_design_blob == SCOPE_BLOB,
        f"SCOPE_DRIFT:{current_design_blob}",
    )

    calibration_builder_blob = git(
        "rev-parse",
        f"HEAD:{CALIBRATION_BUILDER_REL.as_posix()}",
    )
    require(
        calibration_builder_blob == CALIBRATION_BUILDER_BLOB,
        f"CALIBRATION_BUILDER_DRIFT:{calibration_builder_blob}",
    )


def family_spec(key: str) -> calibration.FamilySpec:
    require(key in FAMILY_KEYS, f"UNSUPPORTED_FAMILY:{key}")
    return calibration.family_spec(key)


def expected_pair_ids(key: str) -> list[str]:
    require(key in FAMILY_KEYS, f"UNSUPPORTED_FAMILY:{key}")
    return [
        f"{key}_fact_{pair_number:03d}"
        for pair_number in range(
            HOLDOUT_FIRST_PAIR,
            HOLDOUT_LAST_PAIR + 1,
        )
    ]


def _source_fact_absolute(
    spec: calibration.FamilySpec,
    pair_number: int,
) -> dict[str, Any]:
    require(
        HOLDOUT_FIRST_PAIR <= pair_number <= HOLDOUT_LAST_PAIR,
        f"SOURCE_PAIR_RANGE:{pair_number}",
    )

    # Critical: use the absolute source index, not a restarted 0..299
    # schedule for the fresh holdout.
    i = pair_number - 1
    s = spec.schedule

    name = calibration._select(
        spec.names,
        i,
        s.name_stride,
        s.name_offset,
    )
    alternate_name = calibration._select(
        spec.names,
        i,
        s.name_stride,
        s.name_alt_offset,
    )
    title = calibration._select(
        spec.titles,
        i,
        s.title_stride,
        s.title_offset,
    )
    alternate_title = calibration._select(
        spec.titles,
        i,
        s.title_stride,
        s.title_alt_offset,
    )
    role = calibration._select(
        spec.roles,
        i,
        s.role_stride,
        s.role_offset,
    )
    alternate_role = calibration._select(
        spec.roles,
        i,
        s.role_stride,
        s.role_alt_offset,
    )

    predicate, alternate_predicate = spec.predicate_pairs[
        (s.predicate_stride * i + s.predicate_offset)
        % len(spec.predicate_pairs)
    ]

    location = calibration._select(
        spec.locations,
        i,
        s.location_stride,
        s.location_offset,
    )
    time = calibration._select(
        spec.times,
        i,
        s.time_stride,
        s.time_offset,
    )
    stem = calibration._select(
        spec.object_stems,
        i,
        s.object_stride,
        s.object_offset,
    )

    object_text = (
        f"{stem} packet {spec.object_tag}{pair_number:03d}"
    )

    row = {
        "schema_version": spec.source_schema,
        "generator_family": spec.generator_family,
        "pair_id": f"{spec.key}_fact_{pair_number:03d}",
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

    calibration.validate_source_fact(spec, row)
    return row


def validate_source_facts(
    spec: calibration.FamilySpec,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    require(
        len(rows) == HOLDOUT_PAIR_COUNT,
        f"{spec.key}:SOURCE_COUNT",
    )

    for row in rows:
        calibration.validate_source_fact(spec, row)

    ids = [str(row["pair_id"]) for row in rows]

    require(
        ids == expected_pair_ids(spec.key),
        f"{spec.key}:PAIR_IDS",
    )
    require(
        len(set(ids)) == HOLDOUT_PAIR_COUNT,
        f"{spec.key}:PAIR_ID_UNIQUENESS",
    )


def build_source_facts(
    spec: calibration.FamilySpec,
) -> list[dict[str, Any]]:
    require(
        spec.key in FAMILY_KEYS,
        f"UNSUPPORTED_FAMILY:{spec.key}",
    )

    rows = [
        _source_fact_absolute(spec, pair_number)
        for pair_number in range(
            HOLDOUT_FIRST_PAIR,
            HOLDOUT_LAST_PAIR + 1,
        )
    ]

    validate_source_facts(spec, rows)
    return rows


def validate_materialized_rows(
    spec: calibration.FamilySpec,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    calibration.validate_materialized_rows(
        spec,
        rows,
        expected_pairs=HOLDOUT_PAIR_COUNT,
    )

    pair_order: list[str] = []
    seen: set[str] = set()

    for row in rows:
        pair_id = str(row["source_pair_id"])
        if pair_id not in seen:
            seen.add(pair_id)
            pair_order.append(pair_id)

    require(
        pair_order == expected_pair_ids(spec.key),
        f"{spec.key}:MATERIALIZED_PAIR_IDS",
    )


def materialize_facts(
    spec: calibration.FamilySpec,
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_source_facts(spec, facts)

    output: list[dict[str, Any]] = []

    for fact in facts:
        output.extend(
            calibration.materialize_fact(spec, fact)
        )

    validate_materialized_rows(spec, output)
    return output


def write_holdouts(
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    authenticate_authority()

    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    payloads: dict[str, tuple[bytes, bytes]] = {}
    hashes: dict[str, dict[str, str]] = {}

    for key in FAMILY_KEYS:
        spec = family_spec(key)

        facts = build_source_facts(spec)
        rows = materialize_facts(spec, facts)

        source_raw = calibration.jsonl_bytes(facts)
        row_raw = calibration.jsonl_bytes(rows)

        payloads[key] = (source_raw, row_raw)

        hashes[key] = {
            SOURCE_FILE: calibration.sha256_bytes(source_raw),
            ROW_FILE: calibration.sha256_bytes(row_raw),
        }

    output_dir.mkdir(parents=True, exist_ok=False)

    for key in FAMILY_KEYS:
        family_dir = output_dir / key
        family_dir.mkdir()

        source_raw, row_raw = payloads[key]

        (family_dir / SOURCE_FILE).write_bytes(source_raw)
        (family_dir / ROW_FILE).write_bytes(row_raw)

    return hashes


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-blind XG2/XG4 fresh causal-response "
            "holdouts using absolute source indices 601-900. "
            "No tokenizer, model, CUDA, intervention, response, "
            "training, or backward code is used."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    hashes = write_holdouts(args.output_dir)

    print(
        "RESULT="
        "PASS_XG2_XG4_FRESH_RESPONSE_HOLDOUT_BUILD"
    )

    for key in FAMILY_KEYS:
        print(
            f"{key.upper()}_SOURCE_PAIR_COUNT="
            f"{HOLDOUT_PAIR_COUNT}"
        )
        print(
            f"{key.upper()}_ROW_COUNT="
            f"{EXPECTED_ROW_COUNT}"
        )
        print(
            f"{key.upper()}_FIRST_PAIR="
            f"{HOLDOUT_FIRST_PAIR}"
        )
        print(
            f"{key.upper()}_LAST_PAIR="
            f"{HOLDOUT_LAST_PAIR}"
        )
        print(
            f"{key.upper()}_SOURCE_SHA256="
            f"{hashes[key][SOURCE_FILE]}"
        )
        print(
            f"{key.upper()}_ROW_SHA256="
            f"{hashes[key][ROW_FILE]}"
        )

    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("INTERVENTION_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


if __name__ == "__main__":
    main()