from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.build_controlled_v5 import fact_templates_for_count
from scripts import materialize_reason_router_gen4_six_cell_contrast as materializer


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-large-correction-prospective-validation"
DISCOVERY_FREEZE_COMMIT = "41f4678d09f7c779a8edc2eec6cc08a9effd9f41"

GENERATOR_PATH = "scripts/build_controlled_v5.py"
GENERATOR_AUTHORITY_COMMIT = "91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea"
GENERATOR_SOURCE_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"

MATERIALIZER_PATH = "scripts/materialize_reason_router_gen4_six_cell_contrast.py"
MATERIALIZER_SOURCE_BLOB = "6a0f5bf58614cdff0dad5f78c7d4bd86d507cd3b"

DISCOVERY_PAIR_START = 0
DISCOVERY_PAIR_STOP = 300
VALIDATION_PAIR_START = 300
VALIDATION_PAIR_STOP = 600
VALIDATION_PAIR_COUNT = 300
EXPECTED_ROW_COUNT = VALIDATION_PAIR_COUNT * len(materializer.CELL_IDS)

SCHEMA_VERSION = "GEN4_LARGE_CORRECTION_PROSPECTIVE_COHORT_V1"
ALIGNMENT_SHIFT_ABS_THRESHOLD = 0.11228626366380845


class ProspectiveCohortError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProspectiveCohortError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProspectiveCohortError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def authenticate_repo() -> dict[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            DISCOVERY_FREEZE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "DISCOVERY_FREEZE_NOT_ANCESTOR")

    generator_blob = git("rev-parse", f"HEAD:{GENERATOR_PATH}")
    materializer_blob = git("rev-parse", f"HEAD:{MATERIALIZER_PATH}")

    require(generator_blob == GENERATOR_SOURCE_BLOB, "GENERATOR_BLOB_DRIFT")
    require(
        materializer_blob == MATERIALIZER_SOURCE_BLOB,
        "MATERIALIZER_BLOB_DRIFT",
    )

    authority_blob = git(
        "rev-parse",
        f"{GENERATOR_AUTHORITY_COMMIT}:{GENERATOR_PATH}",
    )
    require(authority_blob == GENERATOR_SOURCE_BLOB, "GENERATOR_AUTHORITY_DRIFT")

    return {
        "branch": branch,
        "head": head,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "generator_authority_commit": GENERATOR_AUTHORITY_COMMIT,
        "generator_source_blob": GENERATOR_SOURCE_BLOB,
        "materializer_source_blob": MATERIALIZER_SOURCE_BLOB,
    }


def expected_validation_pair_ids() -> list[str]:
    return [
        f"generated_fact_{number:03d}"
        for number in range(301, 601)
    ]


def build_source_cohorts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts = fact_templates_for_count(VALIDATION_PAIR_STOP)
    require(len(facts) == VALIDATION_PAIR_STOP, "FACT_COUNT")

    discovery = [
        dict(row)
        for row in facts[DISCOVERY_PAIR_START:DISCOVERY_PAIR_STOP]
    ]
    validation = [
        dict(row)
        for row in facts[VALIDATION_PAIR_START:VALIDATION_PAIR_STOP]
    ]

    require(len(discovery) == 300, "DISCOVERY_COUNT")
    require(len(validation) == VALIDATION_PAIR_COUNT, "VALIDATION_COUNT")

    discovery_ids = [str(row["pair_id"]) for row in discovery]
    validation_ids = [str(row["pair_id"]) for row in validation]

    require(len(set(discovery_ids)) == len(discovery_ids), "DISCOVERY_DUPLICATE_PAIR")
    require(len(set(validation_ids)) == len(validation_ids), "VALIDATION_DUPLICATE_PAIR")
    require(
        set(discovery_ids).isdisjoint(validation_ids),
        "DISCOVERY_VALIDATION_PAIR_OVERLAP",
    )
    require(
        validation_ids == expected_validation_pair_ids(),
        "VALIDATION_PAIR_ID_SEQUENCE",
    )

    discovery_claims = [
        materializer._frozen_statement_renderer(dict(row))
        for row in discovery
    ]
    validation_claims = [
        materializer._frozen_statement_renderer(dict(row))
        for row in validation
    ]

    require(
        len(set(validation_claims)) == VALIDATION_PAIR_COUNT,
        "VALIDATION_CLAIM_DUPLICATE",
    )
    require(
        set(discovery_claims).isdisjoint(validation_claims),
        "DISCOVERY_VALIDATION_CLAIM_OVERLAP",
    )

    return discovery, validation


def materialize_validation_rows(
    validation_facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows = materializer.materialize_facts(validation_facts)
    materializer.validate_materialized_rows(rows)

    require(len(rows) == EXPECTED_ROW_COUNT, "ROW_COUNT")
    pair_ids = [str(row["source_pair_id"]) for row in rows]
    require(len(set(pair_ids)) == VALIDATION_PAIR_COUNT, "ROW_PAIR_COUNT")

    cells = Counter(str(row["contrast_cell_id"]) for row in rows)
    require(
        cells == Counter(
            {
                cell_id: VALIDATION_PAIR_COUNT
                for cell_id in materializer.CELL_IDS
            }
        ),
        "CELL_COUNTS",
    )

    for row in rows:
        require(
            not (materializer.FORBIDDEN_OUTPUT_FIELDS & set(row)),
            "FORBIDDEN_OUTCOME_FIELD",
        )

    return rows


def serialize_rows(rows: Sequence[Mapping[str, object]]) -> bytes:
    text = materializer.serialize_materialized_rows(rows)
    return text.encode("utf-8")


def build_manifest(
    *,
    provenance: Mapping[str, str],
    rows_bytes: bytes,
    validation_facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validation_ids = [str(row["pair_id"]) for row in validation_facts]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "PROSPECTIVE_HOLDOUT_MATERIALIZATION",
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "tokenizer_invoked": False,
        "training_executed": False,
        "evaluation_executed": False,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "discovery_pair_index_range": [0, 300],
        "validation_pair_index_range": [300, 600],
        "validation_pair_count": VALIDATION_PAIR_COUNT,
        "validation_row_count": EXPECTED_ROW_COUNT,
        "first_validation_pair_id": validation_ids[0],
        "last_validation_pair_id": validation_ids[-1],
        "validation_pair_ids_sha256": sha256_bytes(
            ("\n".join(validation_ids) + "\n").encode("utf-8")
        ),
        "rows_sha256": sha256_bytes(rows_bytes),
        "generator_authority_commit": GENERATOR_AUTHORITY_COMMIT,
        "generator_source_blob": GENERATOR_SOURCE_BLOB,
        "materializer_source_blob": MATERIALIZER_SOURCE_BLOB,
        "mechanism_id": materializer.MECHANISM_ID,
        "contrast_cells": list(materializer.CELL_IDS),
        "alignment_shift_abs_threshold": ALIGNMENT_SHIFT_ABS_THRESHOLD,
        "alignment_shift_abs_threshold_source":
            "discovery cohort 1-300 frozen q75",
        "prospective_regime_rule":
            "LARGE iff alignment_shift_abs >= fixed threshold; SMALL otherwise",
        "pair_identity_overlap_with_discovery": 0,
        "claim_string_overlap_with_discovery": 0,
        "claim_boundary": [
            "This artifact materializes an outcome-blind prospective synthetic holdout cohort.",
            "It does not establish tokenizer eligibility, geometry-regime membership, causal response, or scientific support.",
            "No threshold may be re-estimated from validation outcomes.",
        ],
        **dict(provenance),
    }


def write_outputs(
    *,
    output_jsonl: Path,
    manifest_path: Path,
    rows_bytes: bytes,
    manifest: Mapping[str, Any],
) -> None:
    require(not output_jsonl.exists(), f"OUTPUT_COLLISION:{output_jsonl}")
    require(not manifest_path.exists(), f"OUTPUT_COLLISION:{manifest_path}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    output_jsonl.write_bytes(rows_bytes)
    manifest_path.write_text(
        json.dumps(
            dict(manifest),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    provenance = authenticate_repo()
    _discovery, validation = build_source_cohorts()
    rows = materialize_validation_rows(validation)
    rows_bytes = serialize_rows(rows)
    manifest = build_manifest(
        provenance=provenance,
        rows_bytes=rows_bytes,
        validation_facts=validation,
    )
    write_outputs(
        output_jsonl=Path(args.output_jsonl),
        manifest_path=Path(args.manifest),
        rows_bytes=rows_bytes,
        manifest=manifest,
    )

    print("RESULT = PASS_PROSPECTIVE_HOLDOUT_MATERIALIZATION")
    print("VALIDATION_PAIR_COUNT =", VALIDATION_PAIR_COUNT)
    print("VALIDATION_ROW_COUNT =", EXPECTED_ROW_COUNT)
    print("FIRST_PAIR =", expected_validation_pair_ids()[0])
    print("LAST_PAIR =", expected_validation_pair_ids()[-1])
    print("PAIR_OVERLAP_WITH_DISCOVERY = 0")
    print("CLAIM_OVERLAP_WITH_DISCOVERY = 0")
    print("MODEL_FORWARD_COUNT = 0")
    print("TOKENIZER_INVOKED = FALSE")


if __name__ == "__main__":
    main()
