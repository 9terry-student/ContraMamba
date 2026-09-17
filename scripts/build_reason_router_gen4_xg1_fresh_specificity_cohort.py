from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base


ROOT = Path(__file__).resolve().parents[1]

DESIGN_FREEZE_COMMIT = "0bc49ab95cbb2c8735b4bc79422d660fa64e3e01"
DESIGN_REL = Path(
    "reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_design.md"
)
DESIGN_GIT_BLOB = "a2c5652c304ab4efbdeb7c5b4048515501dbfefc"

ORIGINAL_BUILDER_REL = Path(
    "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py"
)
ORIGINAL_BUILDER_GIT_BLOB = "c830026935a6c9f4990c6a3315c75fd5580e7264"

ORIGINAL_SOURCE_REL = Path(
    "data/reason_router_gen4_xg1_cross_generator_v1/structured_source_facts.jsonl"
)
ORIGINAL_ROW_REL = Path(
    "data/reason_router_gen4_xg1_cross_generator_v1/"
    "synthetic_reason_router_six_cell.jsonl"
)
ORIGINAL_SOURCE_SHA256 = (
    "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0"
)
ORIGINAL_ROW_SHA256 = (
    "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f"
)

FRESH_START = 301
FRESH_END = 600
SOURCE_PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = SOURCE_PAIR_COUNT * ROWS_PER_PAIR

SOURCE_FILE = base.SOURCE_FILE
ROW_FILE = base.ROW_FILE


class FreshXG1BuildError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshXG1BuildError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshXG1BuildError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_show_bytes(ref: str, path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshXG1BuildError(
            f"GIT_SHOW_FAILURE:{ref}:{path.as_posix()}"
        ) from exc


def authenticate_static_authority() -> None:
    head = git("rev-parse", "HEAD")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            DESIGN_FREEZE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "DESIGN_FREEZE_NOT_ANCESTOR")

    design_blob = git(
        "rev-parse",
        f"{DESIGN_FREEZE_COMMIT}:{DESIGN_REL.as_posix()}",
    )
    require(
        design_blob == DESIGN_GIT_BLOB,
        f"DESIGN_BLOB_DRIFT:{design_blob}",
    )

    builder_blob = git(
        "rev-parse",
        f"HEAD:{ORIGINAL_BUILDER_REL.as_posix()}",
    )
    require(
        builder_blob == ORIGINAL_BUILDER_GIT_BLOB,
        f"ORIGINAL_BUILDER_BLOB_DRIFT:{builder_blob}",
    )

    for args, label in (
        (
            ["git", "diff", "--quiet", "--", ORIGINAL_BUILDER_REL.as_posix()],
            "ORIGINAL_BUILDER_WORKTREE_DRIFT",
        ),
        (
            [
                "git",
                "diff",
                "--cached",
                "--quiet",
                "--",
                ORIGINAL_BUILDER_REL.as_posix(),
            ],
            "ORIGINAL_BUILDER_INDEX_DRIFT",
        ),
    ):
        rc = subprocess.call(
            args,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, label)


def expected_pair_ids(
    start: int = FRESH_START,
    end: int = FRESH_END,
) -> list[str]:
    require(start >= 1 and end >= start, "PAIR_RANGE")
    return [f"xg1_fact_{index:03d}" for index in range(start, end + 1)]


def source_fact_one_based(index: int) -> dict[str, Any]:
    require(index >= 1, f"SOURCE_INDEX_RANGE:{index}")
    i = index - 1

    name = base.NAMES[i % len(base.NAMES)]
    alternate_name = base.NAMES[(i + 7) % len(base.NAMES)]

    title = base.TITLES[(3 * i + 1) % len(base.TITLES)]
    alternate_title = base.TITLES[(3 * i + 4) % len(base.TITLES)]

    role = base.ROLES[(5 * i + 2) % len(base.ROLES)]
    alternate_role = base.ROLES[(5 * i + 5) % len(base.ROLES)]

    predicate, alternate_predicate = base.PREDICATE_PAIRS[
        (7 * i + 3) % len(base.PREDICATE_PAIRS)
    ]

    location = base.LOCATIONS[(5 * i + 4) % len(base.LOCATIONS)]
    time = base.TIMES[(7 * i + 5) % len(base.TIMES)]

    object_text = (
        f"{base.OBJECT_STEMS[(11 * i + 6) % len(base.OBJECT_STEMS)]} "
        f"unit {i + 1:03d}"
    )

    row = {
        "schema_version": base.SOURCE_SCHEMA,
        "generator_family": base.GENERATOR_FAMILY,
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

    base.validate_source_fact(row)
    return row


def build_source_facts(
    start: int = FRESH_START,
    end: int = FRESH_END,
) -> list[dict[str, Any]]:
    rows = [source_fact_one_based(index) for index in range(start, end + 1)]
    validate_source_facts(rows, start=start, end=end)
    return rows


def validate_source_facts(
    rows: Sequence[Mapping[str, Any]],
    *,
    start: int = FRESH_START,
    end: int = FRESH_END,
) -> None:
    expected = expected_pair_ids(start, end)
    require(len(rows) == len(expected), "SOURCE_PAIR_COUNT")

    for row in rows:
        base.validate_source_fact(row)

    ids = [str(row["pair_id"]) for row in rows]
    require(ids == expected, "SOURCE_PAIR_IDS")
    require(len(set(ids)) == len(expected), "SOURCE_PAIR_ID_UNIQUENESS")


def materialize_facts(
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for fact in facts:
        output.extend(base.materialize_fact(fact))

    base.validate_materialized_rows(
        output,
        expected_pairs=len(facts),
    )
    return output


def prove_original_byte_identity() -> dict[str, str]:
    original_facts = build_source_facts(1, 300)
    original_rows = materialize_facts(original_facts)

    regenerated_source = base.jsonl_bytes(original_facts)
    regenerated_rows = base.jsonl_bytes(original_rows)

    frozen_source = git_show_bytes(
        DESIGN_FREEZE_COMMIT,
        ORIGINAL_SOURCE_REL,
    )
    frozen_rows = git_show_bytes(
        DESIGN_FREEZE_COMMIT,
        ORIGINAL_ROW_REL,
    )

    require(
        sha256_bytes(frozen_source) == ORIGINAL_SOURCE_SHA256,
        "FROZEN_ORIGINAL_SOURCE_SHA",
    )
    require(
        sha256_bytes(frozen_rows) == ORIGINAL_ROW_SHA256,
        "FROZEN_ORIGINAL_ROW_SHA",
    )

    require(
        regenerated_source == frozen_source,
        "ORIGINAL_SOURCE_BYTE_IDENTITY",
    )
    require(
        regenerated_rows == frozen_rows,
        "ORIGINAL_ROW_BYTE_IDENTITY",
    )

    return {
        "source_sha256": sha256_bytes(regenerated_source),
        "row_sha256": sha256_bytes(regenerated_rows),
    }


def write_cohort(output_dir: Path) -> dict[str, str]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")

    authenticate_static_authority()
    prove_original_byte_identity()

    facts = build_source_facts()
    rows = materialize_facts(facts)

    source_raw = base.jsonl_bytes(facts)
    row_raw = base.jsonl_bytes(rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / SOURCE_FILE).write_bytes(source_raw)
    (output_dir / ROW_FILE).write_bytes(row_raw)

    return {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-blind fresh XG1 301..600 structural cohort. "
            "No tokenizer, model, checkpoint, or CUDA execution occurs."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    identity = prove_original_byte_identity()
    hashes = write_cohort(args.output_dir)

    print("RESULT=PASS_XG1_FRESH_SPECIFICITY_COHORT_BUILD")
    print("ORIGINAL_001_300_SOURCE_BYTE_IDENTITY=True")
    print("ORIGINAL_001_300_ROW_BYTE_IDENTITY=True")
    print(
        "ORIGINAL_SOURCE_SHA256=",
        identity["source_sha256"],
        sep="",
    )
    print(
        "ORIGINAL_ROW_SHA256=",
        identity["row_sha256"],
        sep="",
    )
    print("PAIR_ID_FIRST=xg1_fact_301")
    print("PAIR_ID_LAST=xg1_fact_600")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print(
        "SOURCE_FILE_SHA256=",
        hashes[SOURCE_FILE],
        sep="",
    )
    print(
        "ROW_FILE_SHA256=",
        hashes[ROW_FILE],
        sep="",
    )
    print("TOKENIZER_EXECUTED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
