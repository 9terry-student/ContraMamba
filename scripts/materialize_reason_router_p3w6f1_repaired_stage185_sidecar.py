#!/usr/bin/env python3
"""Dedicated P3-W6-F1 repaired Stage185 sidecar materializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if not any(Path(entry).resolve() == _REPO_IMPORT_ROOT for entry in sys.path if entry):
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from scripts import analyze_reason_router_p3w6f1_deterministic_polarity_regeneration as p3w6f1
from scripts import build_stage185a_controlled_train_integrity_sidecar as stage185_builder


MATERIALIZER_SOURCE_PATH = "scripts/materialize_reason_router_p3w6f1_repaired_stage185_sidecar.py"
MANIFEST_SCHEMA_VERSION = "p3w6f1_repaired_stage185_materialization_manifest_v1"
MANIFEST_STATUS = "P3W6F1_REPAIRED_STAGE185_MATERIALIZATION_PASS"

F1_EXECUTION_COMMIT = "dc8179e45f7c10416026acdadcbe5cbd8a78d37e"
REPAIRED_JSONL_PATH = (
    "reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_"
    "dc8179e45f7c10416026acdadcbe5cbd8a78d37e/"
    "controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl"
)
REPAIRED_JSONL_SHA256 = "d403c437d982e7d61e524fe21ed24391c84d5103be75890a61be7aa40d942833"
REPAIRED_ROW_COUNT = 3600
AUTHORIZED_F1_ROW_COUNT = 121
AUTHORIZED_F1_ROW_IDS_SHA256 = "386c0c5d5ed80e699f1607f94c2f8ba2861fa0cb1216d5d421f66e62c03d8c64"
BASELINE_ID_SEQUENCE_SHA256 = "898070cd6718f9c677ba68442ee8ed9010200363df01d147528779306917c0eb"
PAIR_COUNT = 300
STRUCTURAL_NEGATIVE_POLARITY_FLIP_ROW_COUNT = 150

GENERATOR_SOURCE_PATH = "scripts/build_controlled_v5.py"
GENERATOR_SOURCE_SHA256 = "37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684"
STAGE184_CONTRACT_MATRIX_PATH = p3w6f1.P3W6F1_STAGE184_CONTRACT_MATRIX_PATH
STAGE184_CONTRACT_MATRIX_SHA256 = "4287bf1ca7f1f2b08e5de53d24ad4019ca5ddff8a16db2dbb65727a5189e96fa"
HISTORICAL_STAGE184_CONTRACT_MATRIX_WINDOWS_CRLF_WORKTREE_SHA256 = "e5f61ac8d0ca3de3dd43767b83bec8c2c171a1635d419466c98d8d32ec2f38e5"
STAGE185_BUILDER_SOURCE_PATH = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
STAGE185_BUILDER_SOURCE_SHA256 = "11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc"

P3W4_SUMMARY_PATH = p3w6f1.P3W6F1_P3W4_SUMMARY_PATH
P3W4_PAIRS_PATH = p3w6f1.P3W6F1_P3W4_PAIRS_PATH
P3W5_MANIFEST_PATH = p3w6f1.P3W6F1_P3W5_MANIFEST_PATH

REGENERATION_MANIFEST_NAME = "p3w6f1_regeneration_execution_manifest.json"
INVOCATION_NAME = "p3w6f1_deterministic_generator_invocation.json"
CONFIGURATION_NAME = "p3w6f1_generator_configuration_identity.json"
REGENERATION_EXECUTION_DIR = (
    "reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_"
    "dc8179e45f7c10416026acdadcbe5cbd8a78d37e"
)
REGENERATION_MANIFEST_PATH = f"{REGENERATION_EXECUTION_DIR}/{REGENERATION_MANIFEST_NAME}"
INVOCATION_PATH = f"{REGENERATION_EXECUTION_DIR}/{INVOCATION_NAME}"
CONFIGURATION_PATH = f"{REGENERATION_EXECUTION_DIR}/{CONFIGURATION_NAME}"
SIDECAR_NAME = "stage185a_controlled_train_integrity_sidecar.jsonl"
MATERIALIZATION_MANIFEST_NAME = "p3w6f1_repaired_stage185_materialization_manifest.json"
OUTPUT_FILE_ORDER = (SIDECAR_NAME, MATERIALIZATION_MANIFEST_NAME)
EXPECTED_OUTPUT_NAMES = {SIDECAR_NAME, MATERIALIZATION_MANIFEST_NAME}

SPLIT_SEED = 174
DEV_RATIO = 0.2
RULE_VERSION = "stage185a_v1"
TRAIN_ROW_COUNT = 2880
DEV_ROW_COUNT = 720

REQUIRED_REPAIRED_RAW_SIGNATURE = {
    "grammar_status": "PASS",
    "intervention_contract_status": "FAIL",
    "integrity_status": "INELIGIBLE",
    "canonical_status": "PASS",
    "polarity_contamination_status": "PASS",
    "dataset_source_status": "PASS",
    "schema_status": "PASS",
    "time_swap_status": "PASS",
    "audit_expected_axes": ["polarity"],
    "audit_changed_axes": ["polarity", "predicate"],
}


class MaterializerError(RuntimeError):
    """Fail-closed materializer rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializerError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def deterministic_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode("utf-8")


def is_full_commit(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value or "") is not None


def git_stdout(repo_root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise MaterializerError(f"GIT_COMMAND_FAILED: {' '.join(args)}") from exc


def git_object_bytes(repo_root: Path, commit: str, source_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{commit}:{source_path}"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise MaterializerError(f"GIT_OBJECT_UNAVAILABLE: {commit}:{source_path}") from exc


def verify_git_repository(repo_root: Path) -> None:
    observed = Path(git_stdout(repo_root, ["rev-parse", "--show-toplevel"])).resolve()
    require(observed == repo_root.resolve(), "REPO_ROOT_MISMATCH")


def current_head(repo_root: Path) -> str:
    return git_stdout(repo_root, ["rev-parse", "HEAD"])


def tracked_worktree_clean(repo_root: Path) -> bool:
    unstaged = subprocess.run(["git", "-C", str(repo_root), "diff", "--quiet", "--"], stderr=subprocess.DEVNULL)
    staged = subprocess.run(["git", "-C", str(repo_root), "diff", "--cached", "--"], stderr=subprocess.DEVNULL)
    return unstaged.returncode == 0 and staged.returncode == 0


def verify_materializer_execution_identity(
    repo_root: Path,
    materializer_execution_commit: str,
    *,
    repo_checker: Any | None = None,
    head_resolver: Any | None = None,
    tracked_clean_checker: Any | None = None,
) -> None:
    require(is_full_commit(materializer_execution_commit), "MATERIALIZER_EXECUTION_COMMIT_NOT_FULL_40_HEX")
    (repo_checker or verify_git_repository)(repo_root)
    observed_head = (head_resolver or current_head)(repo_root)
    require(observed_head == materializer_execution_commit, "MATERIALIZER_EXECUTION_COMMIT_HEAD_MISMATCH")
    require((tracked_clean_checker or tracked_worktree_clean)(repo_root) is True, "TRACKED_WORKTREE_DIRTY")


def verify_f1_execution_commit(value: str) -> None:
    require(value == F1_EXECUTION_COMMIT, "F1_EXECUTION_COMMIT_MISMATCH")


def resolve_under_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise MaterializerError(f"PATH_OUTSIDE_REPO: {value}") from exc
    return resolved


def repo_relative_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def require_canonical_path(repo_root: Path, observed: Path, expected_relative: str, error: str) -> Path:
    expected = (repo_root / expected_relative).resolve()
    require(observed.resolve() == expected, error)
    return expected


def verify_file_identity(path: Path, expected_sha256: str, error: str) -> None:
    require(path.is_file(), f"{error}_MISSING")
    require(file_sha256(path) == expected_sha256, error)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return p3w6f1.load_jsonl(path)


def verify_artifact_identity_json(path: Path, expected: Mapping[str, Any], error: str) -> None:
    observed = load_json(path)
    require(observed == dict(expected), error)


def validate_repaired_jsonl_path_and_sha(repo_root: Path, repaired_jsonl: Path, supplied_sha256: str) -> None:
    require_canonical_path(repo_root, repaired_jsonl, REPAIRED_JSONL_PATH, "REPAIRED_JSONL_PATH_MISMATCH")
    require(supplied_sha256 == REPAIRED_JSONL_SHA256, "REPAIRED_JSONL_SUPPLIED_SHA_MISMATCH")
    verify_file_identity(repaired_jsonl, REPAIRED_JSONL_SHA256, "REPAIRED_JSONL_SHA_MISMATCH")


def validate_fixed_source_argument(repo_root: Path, observed: Path, expected_path: str, expected_sha: str, path_error: str, sha_error: str) -> None:
    canonical = require_canonical_path(repo_root, observed, expected_path, path_error)
    verify_file_identity(canonical, expected_sha, sha_error)


def validate_stage184_contract_matrix_argument(
    repo_root: Path,
    observed: Path,
    materializer_execution_commit: str,
    *,
    git_object_reader: Any | None = None,
) -> None:
    require_canonical_path(repo_root, observed, STAGE184_CONTRACT_MATRIX_PATH, "STAGE184_MATRIX_PATH_MISMATCH")
    blob = (git_object_reader or git_object_bytes)(repo_root, materializer_execution_commit, STAGE184_CONTRACT_MATRIX_PATH)
    require(sha256_bytes(blob) == STAGE184_CONTRACT_MATRIX_SHA256, "STAGE184_MATRIX_SHA_MISMATCH")


def validate_regeneration_artifact_paths(
    repo_root: Path,
    regeneration_manifest: Path,
    invocation_json: Path,
    configuration_json: Path,
) -> None:
    require_canonical_path(repo_root, regeneration_manifest, REGENERATION_MANIFEST_PATH, "REGENERATION_MANIFEST_PATH_MISMATCH")
    require_canonical_path(repo_root, invocation_json, INVOCATION_PATH, "DETERMINISTIC_INVOCATION_PATH_MISMATCH")
    require_canonical_path(repo_root, configuration_json, CONFIGURATION_PATH, "CONFIGURATION_IDENTITY_PATH_MISMATCH")


def derive_authorized_f1_row_ids(
    p3w4_summary: Mapping[str, Any],
    pair_records: Sequence[Mapping[str, Any]],
    p3w5_manifest: Mapping[str, Any],
) -> list[str]:
    try:
        supporting = p3w6f1.extract_decision_supporting_pair_ids(p3w4_summary, p3w5_manifest)
        targets = p3w6f1.extract_authorized_f1_targets(pair_records, supporting)
    except Exception as exc:
        raise MaterializerError(f"AUTHORIZED_F1_DERIVATION_FAILED: {exc}") from exc
    row_ids = targets.get("authorized_F1_row_ids")
    require(isinstance(row_ids, list), "AUTHORIZED_F1_ROW_IDS_MALFORMED")
    require(len(row_ids) == len(set(row_ids)), "AUTHORIZED_F1_ROW_IDS_DUPLICATE")
    require(len(row_ids) == AUTHORIZED_F1_ROW_COUNT, "AUTHORIZED_F1_ROW_COUNT_MISMATCH")
    require(p3w6f1.canonical_sha256(sorted(str(row_id) for row_id in row_ids)) == AUTHORIZED_F1_ROW_IDS_SHA256, "AUTHORIZED_F1_ROW_IDS_SHA_MISMATCH")
    f2_row_ids = p3w6f1.extract_f2_row_ids(pair_records)
    require(not (set(row_ids) & f2_row_ids), "AUTHORIZED_F1_ROW_IDS_INCLUDE_F2")
    return sorted(str(row_id) for row_id in row_ids)


def validate_repaired_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    row_ids = [str(row.get("id", "")) for row in rows]
    require(len(rows) == REPAIRED_ROW_COUNT, "REPAIRED_ROW_COUNT_MISMATCH")
    require(len(row_ids) == len(set(row_ids)), "REPAIRED_DUPLICATE_ROW_ID")
    require(p3w6f1.id_sequence_sha256(row_ids) == BASELINE_ID_SEQUENCE_SHA256, "REPAIRED_ID_SEQUENCE_SHA_MISMATCH")
    require(p3w6f1.baseline_pair_count(rows) == PAIR_COUNT, "PAIR_COUNT_MISMATCH")


def validate_replay(
    repaired_rows: Sequence[Mapping[str, Any]],
    authorized_f1_row_ids: Sequence[str],
    invocation_json: Path,
    configuration_json: Path,
) -> dict[str, Any]:
    try:
        replay = p3w6f1.actual_repaired_generator_replay(repaired_rows, authorized_f1_row_ids)
    except Exception as exc:
        raise MaterializerError(f"REPAIRED_GENERATOR_REPLAY_FAILED: {exc}") from exc
    require(replay.get("pair_count") == PAIR_COUNT, "REPLAY_PAIR_COUNT_MISMATCH")
    require(replay.get("actual_generator_repair_consumed_row_ids") == sorted(authorized_f1_row_ids), "REPAIR_CONSUMPTION_MISMATCH")
    invocation = replay.get("deterministic_generator_invocation")
    configuration = replay.get("generator_configuration_identity")
    require(isinstance(invocation, Mapping), "INVOCATION_IDENTITY_MALFORMED")
    require(isinstance(configuration, Mapping), "CONFIGURATION_IDENTITY_MALFORMED")
    require(invocation.get("baseline_id_sequence_sha256") == BASELINE_ID_SEQUENCE_SHA256, "HISTORICAL_BASELINE_SHA_MISMATCH")
    require(configuration.get("pair_count") == PAIR_COUNT, "CONFIGURATION_PAIR_COUNT_MISMATCH")
    require(configuration.get("authorized_F1_row_count") == AUTHORIZED_F1_ROW_COUNT, "CONFIGURATION_AUTHORIZED_COUNT_MISMATCH")
    require(configuration.get("structural_negative_polarity_flip_row_count") == STRUCTURAL_NEGATIVE_POLARITY_FLIP_ROW_COUNT, "CONFIGURATION_STRUCTURAL_NEGATIVE_COUNT_MISMATCH")
    require(configuration.get("baseline_topology_row_count") == REPAIRED_ROW_COUNT, "CONFIGURATION_TOPOLOGY_COUNT_MISMATCH")
    require(configuration.get("baseline_id_sequence_sha256") == BASELINE_ID_SEQUENCE_SHA256, "CONFIGURATION_BASELINE_SHA_MISMATCH")
    verify_artifact_identity_json(invocation_json, invocation, "DETERMINISTIC_INVOCATION_IDENTITY_MISMATCH")
    verify_artifact_identity_json(configuration_json, configuration, "CONFIGURATION_IDENTITY_MISMATCH")
    return replay


def validate_regeneration_manifest(path: Path, *, repo_root: Path, invocation_json: Path, configuration_json: Path) -> None:
    validate_regeneration_artifact_paths(repo_root, path, invocation_json, configuration_json)
    manifest = load_json(path)
    require(isinstance(manifest, Mapping), "REGENERATION_MANIFEST_MALFORMED")
    expected = {
        "F1_execution_commit": F1_EXECUTION_COMMIT,
        "repaired_output_path": REPAIRED_JSONL_PATH,
        "repaired_output_sha256": REPAIRED_JSONL_SHA256,
        "repaired_generator_source_path": GENERATOR_SOURCE_PATH,
        "repaired_generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "deterministic_generator_invocation_json": repo_relative_path(repo_root, invocation_json),
        "generator_configuration_identity_json": repo_relative_path(repo_root, configuration_json),
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"REGENERATION_MANIFEST_{key.upper()}_MISMATCH")


def runtime_authority_stub() -> dict[str, Any]:
    return {
        "stage185_runtime_authority_pass": True,
        "stage185_runtime_authority_status": "PASS",
        "stage184_contract_matrix_path": STAGE184_CONTRACT_MATRIX_PATH,
        "stage184_contract_matrix_sha256": STAGE184_CONTRACT_MATRIX_SHA256,
        "stage185_integrity_builder_source_path": STAGE185_BUILDER_SOURCE_PATH,
        "stage185_integrity_builder_source_sha256": STAGE185_BUILDER_SOURCE_SHA256,
    }


def build_repaired_sidecar(
    repaired_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    repaired_jsonl_path: Path,
    stage184_contract_matrix: Path,
) -> list[dict[str, Any]]:
    expected_rows, facts = p3w6f1.generator_expected_rows_and_facts_for_source(
        repaired_rows,
        expected_generator_rows=replay_rows,
    )
    contracts = p3w6f1.load_authoritative_stage185_contracts(
        repaired_rows,
        repo_root=repo_root,
        contract_matrix_path=stage184_contract_matrix,
        runtime_authority=runtime_authority_stub(),
    )
    provenance = {
        "source_dataset_path": repo_relative_path(repo_root, repaired_jsonl_path),
        "source_dataset_sha256": REPAIRED_JSONL_SHA256,
        "generator_source_path": GENERATOR_SOURCE_PATH,
        "generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "integrity_builder_sha256": STAGE185_BUILDER_SOURCE_SHA256,
        "stage182a_report_sha256": "",
        "stage184a_report_sha256": "",
    }
    return stage185_builder.build_sidecar(
        [dict(row) for row in repaired_rows],
        contracts,
        expected_rows,
        facts,
        p3w6f1.derive_stage185_dev_ids(repaired_rows),
        provenance,
        RULE_VERSION,
        "",
    )


def validate_sidecar_rows(sidecar_rows: Sequence[Mapping[str, Any]], repaired_rows: Sequence[Mapping[str, Any]], authorized_f1_row_ids: Sequence[str]) -> None:
    require(len(sidecar_rows) == len(repaired_rows), "SIDECAR_ROW_COUNT_MISMATCH")
    missing_schema = [
        str(row.get("row_id", ""))
        for row in sidecar_rows
        if any(field not in row for field in p3w6f1.REQUIRED_STAGE185_SCHEMA_FIELDS)
    ]
    require(not missing_schema, "SIDECAR_SCHEMA_MISMATCH")
    splits = {"train": 0, "dev": 0}
    for row in sidecar_rows:
        split = str(row.get("split", ""))
        if split in splits:
            splits[split] += 1
        require(row.get("source_dataset_path") == REPAIRED_JSONL_PATH, "REPAIRED_SOURCE_PROVENANCE_PATH_SPOOF")
        require(row.get("source_dataset_sha256") == REPAIRED_JSONL_SHA256, "REPAIRED_SOURCE_PROVENANCE_SHA_SPOOF")
        require(row.get("generator_source_path") == GENERATOR_SOURCE_PATH, "GENERATOR_SOURCE_PATH_MISMATCH")
        require(row.get("generator_source_sha256") == GENERATOR_SOURCE_SHA256, "GENERATOR_SOURCE_SHA_MISMATCH")
        require(row.get("integrity_builder_sha256") == STAGE185_BUILDER_SOURCE_SHA256, "STAGE185_BUILDER_PROVENANCE_SHA_MISMATCH")
        require(row.get("created_at") == "", "CREATED_AT_NOT_EMPTY")
        require(row.get("stage182a_report_sha256") == "", "STAGE182_REPORT_SHA_NOT_EMPTY")
        require(row.get("stage184a_report_sha256") == "", "STAGE184_REPORT_SHA_NOT_EMPTY")
    require(splits["train"] == TRAIN_ROW_COUNT and splits["dev"] == DEV_ROW_COUNT, "SPLIT_ACCOUNTING_MISMATCH")
    by_id = {str(row.get("row_id", "")): row for row in sidecar_rows}
    require(len(by_id) == len(sidecar_rows), "SIDECAR_DUPLICATE_ROW_ID")
    for row_id in authorized_f1_row_ids:
        row = by_id.get(row_id)
        require(isinstance(row, Mapping), "AUTHORIZED_F1_SIDECAR_ROW_MISSING")
        for field, expected in REQUIRED_REPAIRED_RAW_SIGNATURE.items():
            require(row.get(field) == expected, f"RAW_REPAIRED_STAGE185_SIGNATURE_MISMATCH:{field}")


def validate_provenance(
    repaired_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    repaired_jsonl_path: Path,
    replay_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validation = p3w6f1.validate_stage185_sidecar_provenance(
        repaired_rows,
        sidecar_rows,
        actual_source_dataset_sha256=REPAIRED_JSONL_SHA256,
        actual_source_dataset_path=repaired_jsonl_path,
        actual_integrity_builder_sha256=STAGE185_BUILDER_SOURCE_SHA256,
        expected_generator_rows=replay_rows,
        repo_root=repo_root,
        runtime_authority=runtime_authority_stub(),
    )
    require(validation.get("stage185_provenance_status") == "PASS", f"STAGE185_PROVENANCE_VALIDATION_FAILED:{validation.get('stage185_provenance_failures')}")
    return validation


def expected_output_dir(repo_root: Path, materializer_execution_commit: str) -> Path:
    return (repo_root / f"reports/stage185a_controlled_train_integrity_sidecar_p3w6f1_{materializer_execution_commit}").resolve()


def verify_output_dir(repo_root: Path, output_dir: Path, materializer_execution_commit: str) -> Path:
    observed = resolve_under_repo(repo_root, output_dir)
    require(observed == expected_output_dir(repo_root, materializer_execution_commit), "OUTPUT_DIR_CONTRACT_MISMATCH")
    return observed


def materializer_source_sha256(repo_root: Path, materializer_execution_commit: str, *, git_object_reader: Any | None = None) -> str:
    blob = (git_object_reader or git_object_bytes)(repo_root, materializer_execution_commit, MATERIALIZER_SOURCE_PATH)
    return sha256_bytes(blob)


def build_manifest(
    *,
    repo_root: Path,
    materializer_execution_commit: str,
    materializer_source_sha: str,
    regeneration_manifest_path: Path,
    invocation_path: Path,
    configuration_path: Path,
    stage184_contract_matrix_path: Path,
    sidecar_path: Path,
    sidecar_bytes: bytes,
    sidecar_rows: Sequence[Mapping[str, Any]],
    provenance_validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": MANIFEST_STATUS,
        "F1_execution_commit": F1_EXECUTION_COMMIT,
        "materializer_execution_commit": materializer_execution_commit,
        "materializer_source_path": MATERIALIZER_SOURCE_PATH,
        "materializer_source_sha256": materializer_source_sha,
        "repaired_source_path": REPAIRED_JSONL_PATH,
        "repaired_source_sha256": REPAIRED_JSONL_SHA256,
        "repaired_source_row_count": REPAIRED_ROW_COUNT,
        "repaired_source_id_sequence_sha256": BASELINE_ID_SEQUENCE_SHA256,
        "authorized_F1_row_count": AUTHORIZED_F1_ROW_COUNT,
        "authorized_F1_row_ids_sha256": AUTHORIZED_F1_ROW_IDS_SHA256,
        "repaired_generator_source_path": GENERATOR_SOURCE_PATH,
        "repaired_generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "regeneration_execution_manifest_path": repo_relative_path(repo_root, regeneration_manifest_path),
        "regeneration_execution_manifest_sha256": file_sha256(regeneration_manifest_path),
        "deterministic_generator_invocation_path": repo_relative_path(repo_root, invocation_path),
        "deterministic_generator_invocation_sha256": file_sha256(invocation_path),
        "generator_configuration_identity_path": repo_relative_path(repo_root, configuration_path),
        "generator_configuration_identity_sha256": file_sha256(configuration_path),
        "stage184_contract_matrix_path": repo_relative_path(repo_root, stage184_contract_matrix_path),
        "stage184_contract_matrix_sha256": STAGE184_CONTRACT_MATRIX_SHA256,
        "historical_stage185_builder_source_path": STAGE185_BUILDER_SOURCE_PATH,
        "historical_stage185_builder_source_sha256": STAGE185_BUILDER_SOURCE_SHA256,
        "historical_stage185_binary_executed": False,
        "split_seed": SPLIT_SEED,
        "dev_ratio": DEV_RATIO,
        "rule_version": RULE_VERSION,
        "train_row_count": TRAIN_ROW_COUNT,
        "dev_row_count": DEV_ROW_COUNT,
        "total_row_count": REPAIRED_ROW_COUNT,
        "sidecar_path": repo_relative_path(repo_root, sidecar_path),
        "sidecar_sha256": sha256_bytes(sidecar_bytes),
        "sidecar_semantic_sha256": p3w6f1.stage185_semantic_sidecar_sha256(sidecar_rows),
        "provenance_validation_status": provenance_validation.get("stage185_provenance_status"),
    }


def publish_artifacts(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    staging_dir_name: str | None = None,
    write_file: Any | None = None,
    before_rename: Any | None = None,
    rename_dir: Any | None = None,
) -> str:
    require(set(payloads) == EXPECTED_OUTPUT_NAMES, "OUTPUT_PAYLOAD_SET_MISMATCH")
    final_paths = {name: output_dir / name for name in EXPECTED_OUTPUT_NAMES}
    if output_dir.exists():
        require(output_dir.is_dir(), "OUTPUT_PATH_EXISTS_NOT_DIRECTORY")
        observed = {entry.name for entry in output_dir.iterdir()}
        require(observed == EXPECTED_OUTPUT_NAMES, "OUTPUT_ARTIFACT_SET_MISMATCH")
        conflicts = [name for name, path in final_paths.items() if not path.is_file() or path.read_bytes() != payloads[name]]
        require(not conflicts, f"OUTPUT_ARTIFACT_CONFLICT:{conflicts}")
        return "IDEMPOTENT_PASS"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_name = staging_dir_name or f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging_dir = output_dir.parent / staging_name
    require(not staging_dir.exists(), "STAGING_DIR_ALREADY_EXISTS")
    writer = write_file or (lambda path, payload: path.write_bytes(payload))
    renamer = rename_dir or (lambda source, target: source.replace(target))
    try:
        staging_dir.mkdir()
        for name in OUTPUT_FILE_ORDER:
            writer(staging_dir / name, payloads[name])
        observed = {entry.name for entry in staging_dir.iterdir()}
        require(observed == EXPECTED_OUTPUT_NAMES, "STAGING_ARTIFACT_SET_MISMATCH")
        mismatches = [
            name
            for name in OUTPUT_FILE_ORDER
            if not (staging_dir / name).is_file() or (staging_dir / name).read_bytes() != payloads[name]
        ]
        require(not mismatches, f"STAGING_ARTIFACT_BYTES_MISMATCH:{mismatches}")
        if before_rename is not None:
            before_rename(staging_dir, output_dir)
        renamer(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    require(output_dir.is_dir(), "OUTPUT_DIR_PUBLICATION_FAILED")
    require({entry.name for entry in output_dir.iterdir()} == EXPECTED_OUTPUT_NAMES, "OUTPUT_ARTIFACT_SET_MISMATCH")
    conflicts = [name for name, path in final_paths.items() if not path.is_file() or path.read_bytes() != payloads[name]]
    require(not conflicts, f"OUTPUT_ARTIFACT_CONFLICT:{conflicts}")
    return "PUBLISHED"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W6-F1 repaired Stage185 sidecar materializer")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--repaired-jsonl", type=Path, required=True)
    parser.add_argument("--repaired-jsonl-sha256", required=True)
    parser.add_argument("--regeneration-execution-manifest-json", type=Path, required=True)
    parser.add_argument("--deterministic-generator-invocation-json", type=Path, required=True)
    parser.add_argument("--generator-configuration-identity-json", type=Path, required=True)
    parser.add_argument("--p3w4-summary-json", type=Path, required=True)
    parser.add_argument("--p3w4-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--p3w5-manifest-json", type=Path, required=True)
    parser.add_argument("--f1-execution-commit", required=True)
    parser.add_argument("--materializer-execution-commit", required=True)
    parser.add_argument("--generator-source", type=Path, required=True)
    parser.add_argument("--stage184-contract-matrix", type=Path, required=True)
    parser.add_argument("--stage185-builder-source", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--dev-ratio", type=float, required=True)
    parser.add_argument("--rule-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    verify_f1_execution_commit(args.f1_execution_commit)
    verify_materializer_execution_identity(repo_root, args.materializer_execution_commit)
    require(args.split_seed == SPLIT_SEED, "SPLIT_SEED_MISMATCH")
    require(args.dev_ratio == DEV_RATIO, "DEV_RATIO_MISMATCH")
    require(args.rule_version == RULE_VERSION, "RULE_VERSION_MISMATCH")

    repaired_jsonl = resolve_under_repo(repo_root, args.repaired_jsonl)
    regeneration_manifest = resolve_under_repo(repo_root, args.regeneration_execution_manifest_json)
    invocation_json = resolve_under_repo(repo_root, args.deterministic_generator_invocation_json)
    configuration_json = resolve_under_repo(repo_root, args.generator_configuration_identity_json)
    p3w4_summary = require_canonical_path(repo_root, resolve_under_repo(repo_root, args.p3w4_summary_json), P3W4_SUMMARY_PATH, "P3W4_SUMMARY_PATH_MISMATCH")
    p3w4_pairs = require_canonical_path(repo_root, resolve_under_repo(repo_root, args.p3w4_pairs_jsonl), P3W4_PAIRS_PATH, "P3W4_PAIRS_PATH_MISMATCH")
    p3w5_manifest = require_canonical_path(repo_root, resolve_under_repo(repo_root, args.p3w5_manifest_json), P3W5_MANIFEST_PATH, "P3W5_MANIFEST_PATH_MISMATCH")
    generator_source = resolve_under_repo(repo_root, args.generator_source)
    stage184_contract_matrix = resolve_under_repo(repo_root, args.stage184_contract_matrix)
    stage185_builder_source = resolve_under_repo(repo_root, args.stage185_builder_source)
    output_dir = verify_output_dir(repo_root, args.output_dir, args.materializer_execution_commit)

    validate_repaired_jsonl_path_and_sha(repo_root, repaired_jsonl, args.repaired_jsonl_sha256)
    validate_fixed_source_argument(repo_root, generator_source, GENERATOR_SOURCE_PATH, GENERATOR_SOURCE_SHA256, "GENERATOR_SOURCE_PATH_MISMATCH", "GENERATOR_SOURCE_SHA_MISMATCH")
    validate_stage184_contract_matrix_argument(repo_root, stage184_contract_matrix, args.materializer_execution_commit)
    validate_fixed_source_argument(repo_root, stage185_builder_source, STAGE185_BUILDER_SOURCE_PATH, STAGE185_BUILDER_SOURCE_SHA256, "STAGE185_BUILDER_PATH_MISMATCH", "STAGE185_BUILDER_SHA_MISMATCH")
    validate_regeneration_manifest(regeneration_manifest, repo_root=repo_root, invocation_json=invocation_json, configuration_json=configuration_json)

    repaired_rows = load_jsonl(repaired_jsonl)
    validate_repaired_rows(repaired_rows)
    authorized_f1_row_ids = derive_authorized_f1_row_ids(load_json(p3w4_summary), load_jsonl(p3w4_pairs), load_json(p3w5_manifest))
    replay = validate_replay(repaired_rows, authorized_f1_row_ids, invocation_json, configuration_json)
    sidecar_rows = build_repaired_sidecar(
        repaired_rows,
        replay["replayed_records"],
        repo_root=repo_root,
        repaired_jsonl_path=repaired_jsonl,
        stage184_contract_matrix=stage184_contract_matrix,
    )
    validate_sidecar_rows(sidecar_rows, repaired_rows, authorized_f1_row_ids)
    provenance_validation = validate_provenance(
        repaired_rows,
        sidecar_rows,
        repo_root=repo_root,
        repaired_jsonl_path=repaired_jsonl,
        replay_rows=replay["replayed_records"],
    )

    sidecar_path = output_dir / SIDECAR_NAME
    sidecar_bytes = deterministic_jsonl_bytes(sidecar_rows)
    source_sha = materializer_source_sha256(repo_root, args.materializer_execution_commit)
    manifest = build_manifest(
        repo_root=repo_root,
        materializer_execution_commit=args.materializer_execution_commit,
        materializer_source_sha=source_sha,
        regeneration_manifest_path=regeneration_manifest,
        invocation_path=invocation_json,
        configuration_path=configuration_json,
        stage184_contract_matrix_path=stage184_contract_matrix,
        sidecar_path=sidecar_path,
        sidecar_bytes=sidecar_bytes,
        sidecar_rows=sidecar_rows,
        provenance_validation=provenance_validation,
    )
    manifest_bytes = deterministic_json_bytes(manifest)
    publish_status = publish_artifacts(
        output_dir,
        {
            SIDECAR_NAME: sidecar_bytes,
            MATERIALIZATION_MANIFEST_NAME: manifest_bytes,
        },
    )
    return {"manifest": manifest, "publish_status": publish_status}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run(args)
    except MaterializerError as exc:
        raise SystemExit(f"P3W6F1_REPAIRED_STAGE185_MATERIALIZER_FAILED_CLOSED: {exc}") from exc
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
