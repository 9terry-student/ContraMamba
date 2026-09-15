from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as calibration
from scripts import build_reason_router_gen4_xg2_xg4_fresh_response_holdouts as holdouts


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"

DESIGN_FREEZE_COMMIT = "2074f52d39bf0fca6d63248016ce47c54bff5e06"
DESIGN_REL = Path(
    "reports/reason_router_gen4_xg2_xg4_cross_generator_response_replication_design.md"
)
DESIGN_BLOB = "809884f817a8ff1314d9237aa28d3ad4627943de"

CALIBRATION_BUILDER_REL = Path(
    "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py"
)
CALIBRATION_BUILDER_BLOB = "4505acc99db0627733592694da290a007d281421"

FRESH_BUILDER_REL = Path(
    "scripts/build_reason_router_gen4_xg2_xg4_fresh_response_holdouts.py"
)
VALIDATOR_REL = Path(
    "scripts/validate_reason_router_gen4_xg2_xg4_fresh_response_holdouts.py"
)

CALIBRATION_ROOT = Path(
    "data/reason_router_gen4_generator_family_prevalence_v1"
)

EXPECTED_CALIBRATION = {
    "xg2": {
        "source": (
            "0d24fe924a9e30ba7341b515e6cd2bf1164eff441462451317ca5ec3beb33c5e"
        ),
        "rows": (
            "c96ce74bb89dbd374386c27f3a7daa10b5c6630c22711074e0b9a44d59ab5cfa"
        ),
    },
    "xg4": {
        "source": (
            "23f81cb93a5deb7a18c98a6a2f4718b34f4bfd1713bd1fb294e037a714f71fb0"
        ),
        "rows": (
            "9146404eefdbcf26e25b81c65496cbcc459d2eb85c90b5da44db35307de9a347"
        ),
    },
}

FAMILY_MANIFEST_SCHEMA = (
    "GEN4_XG2_XG4_FRESH_RESPONSE_HOLDOUT_STRUCTURAL_MANIFEST_V1"
)
CROSS_MANIFEST_SCHEMA = (
    "GEN4_XG2_XG4_FRESH_RESPONSE_HOLDOUT_CROSS_MANIFEST_V1"
)

MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
CROSS_MANIFEST_FILE = "cross_family_structural_manifest.json"
ROOT_CHECKSUM_FILE = "SHA256SUMS.txt"

FORBIDDEN_OUTCOME_FIELDS = set(calibration.FORBIDDEN_FIELDS) | {
    "alignment_shift_abs",
    "reference_C",
    "target_C",
    "regime",
    "regime_label",
    "large",
    "small",
    "R_ALIGN",
    "r_align",
    "endpoint",
    "endpoint_value",
    "baseline_geometry",
    "prediction",
    "predictions",
    "logit",
    "logits",
    "label",
    "labels",
    "loss",
}


class FreshResponseHoldoutValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshResponseHoldoutValidationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshResponseHoldoutValidationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        value = json.loads(line)

        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}",
        )
        rows.append(value)

    return rows


def authenticate_authority() -> None:
    branch = git("branch", "--show-current")

    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )

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

    require(
        rc == 0,
        "DESIGN_FREEZE_NOT_ANCESTOR",
    )

    frozen_design_blob = git(
        "rev-parse",
        f"{DESIGN_FREEZE_COMMIT}:{DESIGN_REL.as_posix()}",
    )

    require(
        frozen_design_blob == DESIGN_BLOB,
        f"DESIGN_FROZEN_BLOB:{frozen_design_blob}",
    )

    head_design_blob = git(
        "rev-parse",
        f"HEAD:{DESIGN_REL.as_posix()}",
    )

    require(
        head_design_blob == DESIGN_BLOB,
        f"DESIGN_HEAD_DRIFT:{head_design_blob}",
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            DESIGN_REL.as_posix(),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "DESIGN_WORKTREE_DRIFT",
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--cached",
            "--quiet",
            "--",
            DESIGN_REL.as_posix(),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "DESIGN_INDEX_DRIFT",
    )

    head_calibration_builder_blob = git(
        "rev-parse",
        f"HEAD:{CALIBRATION_BUILDER_REL.as_posix()}",
    )

    require(
        head_calibration_builder_blob == CALIBRATION_BUILDER_BLOB,
        (
            "CALIBRATION_BUILDER_HEAD_DRIFT:"
            f"{head_calibration_builder_blob}"
        ),
    )

    working_calibration_builder_blob = git(
        "hash-object",
        "--no-filters",
        "--",
        CALIBRATION_BUILDER_REL.as_posix(),
    )

    require(
        working_calibration_builder_blob == CALIBRATION_BUILDER_BLOB,
        (
            "CALIBRATION_BUILDER_WORKTREE_DRIFT:"
            f"{working_calibration_builder_blob}"
        ),
    )


def verify_calibration_regeneration() -> None:
    for key in holdouts.FAMILY_KEYS:
        spec = calibration.family_spec(key)

        facts = calibration.build_source_facts(spec)
        rows = calibration.materialize_facts(
            spec,
            facts,
        )

        source_raw = calibration.jsonl_bytes(facts)
        row_raw = calibration.jsonl_bytes(rows)

        source_path = (
            ROOT
            / CALIBRATION_ROOT
            / key
            / calibration.SOURCE_FILE
        )
        row_path = (
            ROOT
            / CALIBRATION_ROOT
            / key
            / calibration.ROW_FILE
        )

        require(
            source_path.is_file(),
            f"{key}:CALIBRATION_SOURCE_MISSING",
        )
        require(
            row_path.is_file(),
            f"{key}:CALIBRATION_ROWS_MISSING",
        )

        require(
            source_path.read_bytes() == source_raw,
            f"{key}:CALIBRATION_SOURCE_REGEN_BYTE_DRIFT",
        )
        require(
            row_path.read_bytes() == row_raw,
            f"{key}:CALIBRATION_ROW_REGEN_BYTE_DRIFT",
        )

        require(
            sha256_bytes(source_raw)
            == EXPECTED_CALIBRATION[key]["source"],
            f"{key}:CALIBRATION_SOURCE_SHA256",
        )
        require(
            sha256_bytes(row_raw)
            == EXPECTED_CALIBRATION[key]["rows"],
            f"{key}:CALIBRATION_ROW_SHA256",
        )


def _all_keys(
    rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    return {
        str(key)
        for row in rows
        for key in row.keys()
    }


def _validate_no_outcome_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    keys = _all_keys(rows)
    overlap = sorted(keys & FORBIDDEN_OUTCOME_FIELDS)

    require(
        overlap == [],
        f"{label}:FORBIDDEN_OUTCOME_FIELDS:{overlap}",
    )


def _validate_family(
    input_dir: Path,
    key: str,
) -> dict[str, Any]:
    spec = holdouts.family_spec(key)

    family_dir = input_dir / key

    source_path = family_dir / calibration.SOURCE_FILE
    row_path = family_dir / calibration.ROW_FILE
    manifest_path = family_dir / MANIFEST_FILE
    checksum_path = family_dir / CHECKSUM_FILE

    require(
        family_dir.is_dir(),
        f"{key}:FAMILY_DIR_MISSING",
    )

    require(
        source_path.is_file(),
        f"{key}:SOURCE_MISSING",
    )
    require(
        row_path.is_file(),
        f"{key}:ROWS_MISSING",
    )

    require(
        not manifest_path.exists(),
        f"{key}:MANIFEST_COLLISION",
    )
    require(
        not checksum_path.exists(),
        f"{key}:CHECKSUM_COLLISION",
    )

    observed_names = {
        path.name
        for path in family_dir.iterdir()
    }

    require(
        observed_names
        == {
            calibration.SOURCE_FILE,
            calibration.ROW_FILE,
        },
        f"{key}:PREVALIDATION_FILE_SET:{sorted(observed_names)}",
    )

    facts = read_jsonl(source_path)
    rows = read_jsonl(row_path)

    holdouts.validate_source_facts(
        spec,
        facts,
    )
    holdouts.validate_materialized_rows(
        spec,
        rows,
    )

    require(
        len(facts) == 300,
        f"{key}:SOURCE_COUNT",
    )
    require(
        len(rows) == 1800,
        f"{key}:ROW_COUNT",
    )

    expected_ids = holdouts.expected_pair_ids(key)

    observed_ids = [
        str(fact["pair_id"])
        for fact in facts
    ]

    require(
        observed_ids == expected_ids,
        f"{key}:PAIR_IDS",
    )

    require(
        len(set(observed_ids)) == 300,
        f"{key}:PAIR_ID_UNIQUENESS",
    )

    row_ids = [
        str(row["row_id"])
        for row in rows
    ]

    require(
        len(set(row_ids)) == 1800,
        f"{key}:ROW_ID_UNIQUENESS",
    )

    cell_counts = Counter(
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        for row in rows
    )

    for pair_id in expected_ids:
        for cell_id in calibration.CELL_IDS:
            require(
                cell_counts[(pair_id, cell_id)] == 1,
                (
                    f"{key}:CELL_CARDINALITY:"
                    f"{pair_id}:{cell_id}:"
                    f"{cell_counts[(pair_id, cell_id)]}"
                ),
            )

    _validate_no_outcome_fields(
        facts,
        label=f"{key}:SOURCE",
    )
    _validate_no_outcome_fields(
        rows,
        label=f"{key}:ROWS",
    )

    regenerated_facts = holdouts.build_source_facts(spec)
    regenerated_rows = holdouts.materialize_facts(
        spec,
        regenerated_facts,
    )

    regenerated_source_raw = calibration.jsonl_bytes(
        regenerated_facts
    )
    regenerated_row_raw = calibration.jsonl_bytes(
        regenerated_rows
    )

    require(
        source_path.read_bytes()
        == regenerated_source_raw,
        f"{key}:SOURCE_REGEN_BYTE_DRIFT",
    )

    require(
        row_path.read_bytes()
        == regenerated_row_raw,
        f"{key}:ROW_REGEN_BYTE_DRIFT",
    )

    calibration_source_path = (
        ROOT
        / CALIBRATION_ROOT
        / key
        / calibration.SOURCE_FILE
    )
    calibration_row_path = (
        ROOT
        / CALIBRATION_ROOT
        / key
        / calibration.ROW_FILE
    )

    calibration_facts = read_jsonl(
        calibration_source_path
    )
    calibration_rows = read_jsonl(
        calibration_row_path
    )

    calibration_pair_ids = {
        str(fact["pair_id"])
        for fact in calibration_facts
    }
    fresh_pair_ids = {
        str(fact["pair_id"])
        for fact in facts
    }

    pair_overlap = sorted(
        fresh_pair_ids & calibration_pair_ids
    )

    require(
        pair_overlap == [],
        f"{key}:CALIBRATION_PAIR_OVERLAP",
    )

    calibration_claims = {
        str(row["claim"])
        for row in calibration_rows
    }
    fresh_claims = {
        str(row["claim"])
        for row in rows
    }

    claim_overlap = sorted(
        fresh_claims & calibration_claims
    )

    require(
        claim_overlap == [],
        f"{key}:CALIBRATION_CLAIM_OVERLAP",
    )

    calibration_evidence = {
        str(row["evidence"])
        for row in calibration_rows
    }
    fresh_evidence = {
        str(row["evidence"])
        for row in rows
    }

    evidence_overlap = sorted(
        fresh_evidence & calibration_evidence
    )

    require(
        evidence_overlap == [],
        f"{key}:CALIBRATION_EVIDENCE_OVERLAP",
    )

    return {
        "key": key,
        "spec": spec,
        "family_dir": family_dir,
        "source_path": source_path,
        "row_path": row_path,
        "manifest_path": manifest_path,
        "checksum_path": checksum_path,
        "facts": facts,
        "rows": rows,
        "pair_ids": fresh_pair_ids,
        "claims": fresh_claims,
        "evidence": fresh_evidence,
        "source_sha256": sha256_file(source_path),
        "row_sha256": sha256_file(row_path),
        "calibration_pair_overlap_count": 0,
        "calibration_claim_overlap_count": 0,
        "calibration_evidence_overlap_count": 0,
    }


def _validate_cross_family(
    validated: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    xg2 = validated["xg2"]
    xg4 = validated["xg4"]

    pair_overlap = (
        set(xg2["pair_ids"])
        & set(xg4["pair_ids"])
    )
    claim_overlap = (
        set(xg2["claims"])
        & set(xg4["claims"])
    )
    evidence_overlap = (
        set(xg2["evidence"])
        & set(xg4["evidence"])
    )

    require(
        not pair_overlap,
        "CROSS_FAMILY_PAIR_ID_OVERLAP",
    )
    require(
        not claim_overlap,
        "CROSS_FAMILY_CLAIM_OVERLAP",
    )
    require(
        not evidence_overlap,
        "CROSS_FAMILY_EVIDENCE_OVERLAP",
    )

    return {
        "pair_id_overlap_count": 0,
        "claim_overlap_count": 0,
        "evidence_overlap_count": 0,
    }


def _checksum_text(
    entries: Sequence[tuple[str, str]],
) -> bytes:
    return "".join(
        f"{digest}  {name}\n"
        for name, digest in entries
    ).encode("utf-8")


def validate_and_write(
    input_dir: Path,
) -> dict[str, Any]:
    authenticate_authority()
    verify_calibration_regeneration()

    require(
        input_dir.is_dir(),
        "INPUT_DIR_MISSING",
    )

    observed_root_entries = {
        path.name
        for path in input_dir.iterdir()
    }

    require(
        observed_root_entries == {"xg2", "xg4"},
        (
            "PREVALIDATION_ROOT_FILE_SET:"
            f"{sorted(observed_root_entries)}"
        ),
    )

    validated = {
        key: _validate_family(input_dir, key)
        for key in holdouts.FAMILY_KEYS
    }

    cross = _validate_cross_family(validated)

    fresh_builder_path = ROOT / FRESH_BUILDER_REL
    validator_path = ROOT / VALIDATOR_REL
    design_path = ROOT / DESIGN_REL

    require(
        fresh_builder_path.is_file(),
        "FRESH_BUILDER_MISSING",
    )
    require(
        validator_path.is_file(),
        "VALIDATOR_MISSING",
    )

    builder_sha256 = sha256_file(
        fresh_builder_path
    )
    validator_sha256 = sha256_file(
        validator_path
    )
    design_sha256 = sha256_file(
        design_path
    )

    family_artifacts: dict[str, Any] = {}

    for key in holdouts.FAMILY_KEYS:
        item = validated[key]
        spec = item["spec"]

        manifest = {
            "schema_version": FAMILY_MANIFEST_SCHEMA,
            "result": (
                "PASS_FRESH_RESPONSE_HOLDOUT_"
                "STRUCTURAL_GATE"
            ),
            "family_key": key,
            "generator_family": spec.generator_family,
            "design_freeze_commit": DESIGN_FREEZE_COMMIT,
            "design_blob": DESIGN_BLOB,
            "design_sha256": design_sha256,
            "calibration_builder_blob": (
                CALIBRATION_BUILDER_BLOB
            ),
            "fresh_builder_sha256": builder_sha256,
            "validator_sha256": validator_sha256,
            "pair_id_first": (
                f"{key}_fact_301"
            ),
            "pair_id_last": (
                f"{key}_fact_600"
            ),
            "source_pair_count": 300,
            "rows_per_pair": 6,
            "row_count": 1800,
            "source_file_sha256": (
                item["source_sha256"]
            ),
            "row_file_sha256": (
                item["row_sha256"]
            ),
            "calibration_source_sha256": (
                EXPECTED_CALIBRATION[key]["source"]
            ),
            "calibration_row_sha256": (
                EXPECTED_CALIBRATION[key]["rows"]
            ),
            "calibration_pair_overlap_count": 0,
            "calibration_claim_overlap_count": 0,
            "calibration_evidence_overlap_count": 0,
            "deterministic_byte_regeneration": True,
            "calibration_1_300_byte_regeneration": True,
            "absolute_schedule_indexing_301_600": True,
            "labels_present": False,
            "logits_present": False,
            "predictions_present": False,
            "baseline_geometry_present": False,
            "endpoint_values_present": False,
            "response_fields_present": False,
            "tokenizer_executed": False,
            "model_executed": False,
            "cuda_executed": False,
            "intervention_executed": False,
            "training_executed": False,
            "backward_executed": False,
        }

        manifest_raw = canonical_json_bytes(
            manifest
        )

        item["manifest_path"].write_bytes(
            manifest_raw
        )

        manifest_sha256 = sha256_bytes(
            manifest_raw
        )

        checksum_raw = _checksum_text(
            (
                (
                    calibration.SOURCE_FILE,
                    item["source_sha256"],
                ),
                (
                    calibration.ROW_FILE,
                    item["row_sha256"],
                ),
                (
                    MANIFEST_FILE,
                    manifest_sha256,
                ),
            )
        )

        item["checksum_path"].write_bytes(
            checksum_raw
        )

        family_artifacts[key] = {
            "source_sha256": item["source_sha256"],
            "row_sha256": item["row_sha256"],
            "manifest_sha256": manifest_sha256,
            "checksum_sha256": sha256_bytes(
                checksum_raw
            ),
        }

    cross_manifest = {
        "schema_version": CROSS_MANIFEST_SCHEMA,
        "result": (
            "PASS_XG2_XG4_FRESH_RESPONSE_"
            "HOLDOUT_STRUCTURAL_VALIDATION"
        ),
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "design_blob": DESIGN_BLOB,
        "design_sha256": design_sha256,
        "calibration_builder_blob": (
            CALIBRATION_BUILDER_BLOB
        ),
        "fresh_builder_sha256": builder_sha256,
        "validator_sha256": validator_sha256,
        "family_keys": list(
            holdouts.FAMILY_KEYS
        ),
        "source_pair_count_per_family": 300,
        "row_count_per_family": 1800,
        "pair_id_overlap_count": (
            cross["pair_id_overlap_count"]
        ),
        "claim_overlap_count": (
            cross["claim_overlap_count"]
        ),
        "evidence_overlap_count": (
            cross["evidence_overlap_count"]
        ),
        "xg2": family_artifacts["xg2"],
        "xg4": family_artifacts["xg4"],
        "tokenizer_executed": False,
        "model_executed": False,
        "cuda_executed": False,
        "intervention_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "scientific_response_observed": False,
    }

    cross_manifest_path = (
        input_dir / CROSS_MANIFEST_FILE
    )
    root_checksum_path = (
        input_dir / ROOT_CHECKSUM_FILE
    )

    require(
        not cross_manifest_path.exists(),
        "CROSS_MANIFEST_COLLISION",
    )
    require(
        not root_checksum_path.exists(),
        "ROOT_CHECKSUM_COLLISION",
    )

    cross_raw = canonical_json_bytes(
        cross_manifest
    )

    cross_manifest_path.write_bytes(
        cross_raw
    )

    cross_sha256 = sha256_bytes(
        cross_raw
    )

    root_checksum_raw = _checksum_text(
        (
            (
                CROSS_MANIFEST_FILE,
                cross_sha256,
            ),
            (
                "xg2/SHA256SUMS.txt",
                family_artifacts["xg2"][
                    "checksum_sha256"
                ],
            ),
            (
                "xg4/SHA256SUMS.txt",
                family_artifacts["xg4"][
                    "checksum_sha256"
                ],
            ),
        )
    )

    root_checksum_path.write_bytes(
        root_checksum_raw
    )

    return {
        "cross_manifest_sha256": (
            cross_sha256
        ),
        "root_checksum_sha256": (
            sha256_bytes(root_checksum_raw)
        ),
        "builder_sha256": builder_sha256,
        "validator_sha256": validator_sha256,
        "families": family_artifacts,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static structural validation for the frozen "
            "XG2/XG4 fresh causal-response holdouts. "
            "No tokenizer, model, CUDA, intervention, "
            "training, backward, or response execution."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    result = validate_and_write(
        args.input_dir
    )

    print(
        "RESULT="
        "PASS_XG2_XG4_FRESH_RESPONSE_"
        "HOLDOUT_STRUCTURAL_VALIDATION"
    )

    for key in holdouts.FAMILY_KEYS:
        item = result["families"][key]

        print(
            f"{key.upper()}_SOURCE_SHA256="
            f"{item['source_sha256']}"
        )
        print(
            f"{key.upper()}_ROW_SHA256="
            f"{item['row_sha256']}"
        )
        print(
            f"{key.upper()}_MANIFEST_SHA256="
            f"{item['manifest_sha256']}"
        )
        print(
            f"{key.upper()}_CHECKSUM_SHA256="
            f"{item['checksum_sha256']}"
        )

    print(
        "CROSS_MANIFEST_SHA256="
        f"{result['cross_manifest_sha256']}"
    )
    print(
        "ROOT_CHECKSUM_SHA256="
        f"{result['root_checksum_sha256']}"
    )
    print(
        "FRESH_BUILDER_SHA256="
        f"{result['builder_sha256']}"
    )
    print(
        "VALIDATOR_SHA256="
        f"{result['validator_sha256']}"
    )

    print("PAIR_ID_OVERLAP_COUNT=0")
    print("CLAIM_OVERLAP_COUNT=0")
    print("EVIDENCE_OVERLAP_COUNT=0")
    print("CALIBRATION_1_300_BYTE_REGENERATION=True")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("INTERVENTION_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("SCIENTIFIC_RESPONSE_OBSERVED=False")


if __name__ == "__main__":
    main()