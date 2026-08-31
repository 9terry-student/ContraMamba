#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


RECOVERY_AUTHORITY_COMMIT = "233ed0be080e1d30dd47de2e66136475ec2ede76"
ORIGINAL_EXECUTION_COMMIT = "2737c3c6116ae3766b469801f990e2c45ba9a55e"
ORIGINAL_AUTHORIZED_WRAPPER_SHA256 = "dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e"
SEED = 180
SPLIT_SEED = 174
RECOVERY_SCHEMA = "contramamba-seed180-a0-provenance-recovery-v1"
AUDIT_SCHEMA = "contramamba-seed180-a0-provenance-recovery-audit-v1"
RECOVERY_SCOPE = "P3-W7-A0 seed180 provenance recovery"

SOURCE_RUN_DIR = Path("/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0")
PACKAGE_DIR = Path("/kaggle/working/contramamba_recovery_handoffs")
RUN_REL_DIR = "reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0"
EXPECTED_SELECTED_CHECKPOINT_PATH = str(SOURCE_RUN_DIR / "selected_checkpoint.pt")

TOOLING_AUTHORITY_PATH = "reports/reason_router_p3w7_a0_seed180_provenance_recovery_tooling_implementation_authority_spec_candidate.md"
RECOVERY_AUTHORITY_PATH = "reports/reason_router_p3w7_a0_seed180_provenance_recovery_execution_authority_spec_candidate.md"
ORIGINAL_A0_AUTHORITY_PATH = "reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md"

EXPECTED_DATA_PATH = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
EXPECTED_SIDECAR_PATH = "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
EXPECTED_DATA_PHYSICAL_SHA256 = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
EXPECTED_DATA_SEMANTIC_SHA256 = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
EXPECTED_SIDECAR_PHYSICAL_SHA256 = "2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1"
EXPECTED_SIDECAR_SEMANTIC_SHA256 = "0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08"
EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256 = "9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2"

SOURCE_ARTIFACTS = [
    {"name": "training_report.json", "size": 306114, "sha256": "71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508"},
    {"name": "clean_dev_predictions.json", "size": 4838225, "sha256": "92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2"},
    {"name": "training_report_predictions.jsonl", "size": 3934123, "sha256": "e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef"},
    {"name": "selected_checkpoint.pt", "size": 518269815, "sha256": "dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da"},
    {"name": "run_provenance.json", "size": 68429, "sha256": "4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b"},
]
ARTIFACT_BY_NAME = {item["name"]: item for item in SOURCE_ARTIFACTS}


def zip_artifact_path(artifact_name: str) -> str:
    return f"files/{RUN_REL_DIR}/{artifact_name}"


EXPECTED_ZIP_ENTRIES = [
    "recovery_manifest.json",
    *[zip_artifact_path(item["name"]) for item in SOURCE_ARTIFACTS],
]

EXPECTED_ARGS: dict[str, Any] = {
    "data": EXPECTED_DATA_PATH,
    "architecture": "v6b_minimal",
    "backbone": "mamba",
    "model_name": "state-spaces/mamba-130m-hf",
    "freeze_encoder": True,
    "frame_downstream_gradient_mode": "joint",
    "epochs": 20,
    "max_length": 128,
    "dev_ratio": 0.2,
    "seed": SEED,
    "split_seed": SPLIT_SEED,
    "device": "cuda",
    "flag_source": "controlled_heuristic",
    "select_metric": "final_macro_f1",
    "ranking_weight": 0.0,
    "class_weighting": "none",
    "stage174c_clean_pairwise_mode": "off",
    "stage174c_clean_pairwise_weight": 0.0,
    "stage174c_clean_polarity_preservation_weight": 0.0,
    "stage175b_support_anchor_mode": "off",
    "stage175b_support_anchor_weight": 0.0,
    "stage177c_frame_pairwise_mode": "off",
    "stage177c_frame_pairwise_weight": 0.0,
    "compatible_positive_margin_logit": 0.0,
    "compatible_positive_margin_weight": 0.0,
    "lr": 0.001,
    "reason_router_arm": "A0",
    "reason_router_mode": "explicit_product",
    "gradient_ownership_mode": "joint",
    "controlled_integrity_sidecar_path": EXPECTED_SIDECAR_PATH,
    "expected_integrity_sidecar_semantic_sha256": EXPECTED_SIDECAR_SEMANTIC_SHA256,
    "save_selected_checkpoint": True,
    "selected_checkpoint_filename": "selected_checkpoint.pt",
    "output_json": "/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json",
    "output_predictions_json": "/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json",
}
OMITTED_FLAGS = {"reason_loss_weight", "reason_router_a0_reference_predictions", "batch_size"}
AUTHORIZED_OPTION_NAMES = set(EXPECTED_ARGS) | OMITTED_FLAGS


class Blocker(ValueError):
    pass


def blocker(message: str) -> Blocker:
    return Blocker(f"PROVENANCE_RECOVERY_BLOCKER: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise blocker(message)


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def require_commit(value: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None, "commit must be lowercase 40 hex")
    return value


def run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo_root), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def repo_root() -> Path:
    return Path(run_git(Path.cwd(), ["rev-parse", "--show-toplevel"])).resolve()


def git_head(root: Path) -> str:
    return require_commit(run_git(root, ["rev-parse", "HEAD"]))


def require_clean_repo(root: Path) -> None:
    require(run_git(root, ["status", "--porcelain=v1", "--untracked-files=no"]) == "", "tracked worktree dirty")
    require(run_git(root, ["diff", "--cached", "--name-status"]) == "", "index dirty")


def require_authority_files(root: Path) -> None:
    for rel in (TOOLING_AUTHORITY_PATH, RECOVERY_AUTHORITY_PATH, ORIGINAL_A0_AUTHORITY_PATH):
        path = root / rel
        require(path.exists() and path.is_file(), f"authority file missing: {rel}")


def duplicate_key_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise blocker(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def loads_json_strict(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=duplicate_key_hook)
    except Blocker:
        raise
    except json.JSONDecodeError as exc:
        raise blocker(f"malformed JSON: {exc}") from exc


def load_json_file_strict(path: Path) -> Any:
    return loads_json_strict(path.read_text(encoding="utf-8"))


def deterministic_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": "), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def get_path(obj: Any, dotted: str, default: Any = None) -> Any:
    value = obj
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def exact_int(value: Any, expected: int, field: str) -> None:
    require(type(value) is int and value == expected, f"{field} expected {expected!r} got {value!r}")


def exact_bool(value: Any, expected: bool, field: str) -> None:
    require(type(value) is bool and value is expected, f"{field} expected {expected!r} got {value!r}")


def exact_str(value: Any, expected: str, field: str) -> None:
    require(type(value) is str and value == expected, f"{field} expected {expected!r} got {value!r}")


def exact_number(value: Any, expected: float, field: str) -> None:
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)), f"{field} must be finite numeric")
    require(float(value) == float(expected), f"{field} expected {expected!r} got {value!r}")


def optional_exact(value: Any, expected: Any, field: str) -> None:
    if value is not None:
        if isinstance(expected, bool):
            exact_bool(value, expected, field)
        elif isinstance(expected, int) and not isinstance(expected, bool):
            exact_int(value, expected, field)
        elif isinstance(expected, float):
            exact_number(value, expected, field)
        else:
            exact_str(value, expected, field)


def require_none(value: Any, field: str) -> None:
    require(value is None, f"{field} must be null/None")


_MISSING = object()


def exact_if_present(prov: dict[str, Any], path: str, expected: Any, field: str) -> None:
    value = get_path(prov, path, _MISSING)
    if value is not _MISSING:
        if isinstance(expected, bool):
            exact_bool(value, expected, field)
        elif isinstance(expected, int):
            exact_int(value, expected, field)
        elif isinstance(expected, float):
            exact_number(value, expected, field)
        else:
            exact_str(value, expected, field)


def exact_required_path(prov: dict[str, Any], path: str, expected: Any) -> None:
    value = get_path(prov, path, _MISSING)
    require(value is not _MISSING, f"{path} missing")
    if isinstance(expected, bool):
        exact_bool(value, expected, path)
    elif isinstance(expected, int) and not isinstance(expected, bool):
        exact_int(value, expected, path)
    elif isinstance(expected, float):
        exact_number(value, expected, path)
    else:
        exact_str(value, expected, path)


def validate_present_identity_copies(prov: dict[str, Any]) -> None:
    expected_identities = {
        "dataset_path": (EXPECTED_DATA_PATH, ("parsed_args.data", "data_provenance.main_data.path", "resolved_runtime_config.data_path")),
        "dataset_sha256": (EXPECTED_DATA_PHYSICAL_SHA256, (
            "data_provenance.main_data.sha256",
            "compatible_positive_margin.authoritative_dataset_sha256",
            "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_sha256",
        )),
        "dataset_semantic_sha256": (EXPECTED_DATA_SEMANTIC_SHA256, (
            "compatible_positive_margin.authoritative_dataset_semantic_sha256",
            "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256",
        )),
        "sidecar_path": (EXPECTED_SIDECAR_PATH, ("parsed_args.controlled_integrity_sidecar_path", "resolved_runtime_config.controlled_integrity_sidecar_path")),
        "sidecar_physical_sha256": (EXPECTED_SIDECAR_PHYSICAL_SHA256, (
            "compatible_positive_margin.authoritative_sidecar_physical_sha256",
            "resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_physical_sha256",
        )),
        "sidecar_semantic_sha256": (EXPECTED_SIDECAR_SEMANTIC_SHA256, (
            "compatible_positive_margin.authoritative_sidecar_semantic_sha256",
            "resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_semantic_sha256",
        )),
        "provenance_physical_sha256": (EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256, (
            "compatible_positive_margin.authoritative_provenance_physical_sha256",
            "resolved_runtime_config.compatible_positive_margin.authoritative_provenance_physical_sha256",
        )),
    }
    for label, (expected, paths) in expected_identities.items():
        for path in paths:
            exact_if_present(prov, path, expected, path)

    exact_if_present(prov, "data_provenance.main_data.semantic_sha256", EXPECTED_DATA_SEMANTIC_SHA256, "data_provenance.main_data.semantic_sha256")

    for label, paths in {
        "p4l_provenance_path": ("compatible_positive_margin.authoritative_provenance_path", "resolved_runtime_config.compatible_positive_margin.authoritative_provenance_path"),
        "reason_router_arm": ("reason_router.arm", "resolved_runtime_config.reason_router.arm", "resolved_runtime_config.reason_router_p2_contract.arm"),
        "router_mode": ("reason_router.router_mode", "resolved_runtime_config.reason_router.router_mode", "resolved_runtime_config.reason_router_p2_contract.router_mode"),
        "gradient_ownership": ("reason_router.gradient_ownership_mode", "resolved_runtime_config.reason_router.gradient_ownership_mode", "resolved_runtime_config.reason_router_p2_contract.gradient_ownership_mode"),
    }.items():
        present = [get_path(prov, path, _MISSING) for path in paths]
        present = [value for value in present if value is not _MISSING]
        if present:
            require(all(value == present[0] for value in present), f"contradictory {label} copies")

    require(isinstance(get_path(prov, "finalization.selected_checkpoint", _MISSING), dict), "finalization.selected_checkpoint must be object")
    exact_required_path(prov, "finalization.selected_checkpoint.sha256", ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"])
    exact_required_path(prov, "finalization.selected_checkpoint.filename", "selected_checkpoint.pt")
    exact_required_path(prov, "finalization.selected_checkpoint.path", EXPECTED_SELECTED_CHECKPOINT_PATH)
    exact_required_path(prov, "finalization.selected_checkpoint.size_bytes", ARTIFACT_BY_NAME["selected_checkpoint.pt"]["size"])
    exact_required_path(prov, "finalization.selected_checkpoint.schema_version", "stage176a0_selected_checkpoint_v1")
    exact_required_path(prov, "finalization.selected_checkpoint.saved", True)
    exact_if_present(prov, "finalization.selected_checkpoint.checkpoint_sha256", ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"], "finalization.selected_checkpoint.checkpoint_sha256")
    exact_if_present(prov, "finalization.selected_checkpoint_sha256", ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"], "finalization.selected_checkpoint_sha256")


def validate_artifact_path(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    handle, artifact, _metadata = open_validated_artifact(path, expected)
    handle.close()
    return artifact


def require_meaningful_object_identity(metadata: os.stat_result, path: Path) -> None:
    if os.name == "posix":
        require(getattr(metadata, "st_dev", None) is not None, f"artifact has no device identity: {path}")
        require(bool(getattr(metadata, "st_ino", 0)), f"artifact has no inode identity: {path}")


def open_validated_artifact(path: Path, expected: dict[str, Any]) -> tuple[BinaryIO, dict[str, Any], os.stat_result]:
    require(path.exists(), f"missing artifact: {path}")
    require(not path.is_symlink(), f"symlink artifact rejected: {path}")
    require(path.is_file(), f"artifact is not regular file: {path}")
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise blocker(f"artifact could not be opened: {path}: {exc}") from exc
    try:
        metadata = os.fstat(handle.fileno())
        require_meaningful_object_identity(metadata, path)
        require(stat.S_ISREG(metadata.st_mode), f"artifact is not regular file: {path}")
        require(metadata.st_size == expected["size"], f"artifact size mismatch: {path}")
        digest = sha256_stream(handle)
        require(digest == expected["sha256"], f"artifact sha256 mismatch: {path}")
        handle.seek(0)
        return handle, {"path": f"{RUN_REL_DIR}/{expected['name']}", "size": expected["size"], "sha256": expected["sha256"]}, metadata
    except Exception:
        handle.close()
        raise


def open_validated_source_artifacts(source_dir: Path) -> list[tuple[BinaryIO, dict[str, Any], os.stat_result]]:
    opened: list[tuple[BinaryIO, dict[str, Any], os.stat_result]] = []
    try:
        for item in SOURCE_ARTIFACTS:
            opened.append(open_validated_artifact(source_dir / item["name"], item))
        return opened
    except Exception:
        for handle, _artifact, _metadata in opened:
            handle.close()
        raise


def validate_source_artifacts(source_dir: Path) -> list[dict[str, Any]]:
    return [validate_artifact_path(source_dir / item["name"], item) for item in SOURCE_ARTIFACTS]


def validate_run_provenance(prov: Any) -> None:
    require(isinstance(prov, dict), "run_provenance root must be object")
    exact_str(get_path(prov, "schema_version"), "stage174a_v1", "schema_version")
    exact_str(get_path(prov, "status"), "completed", "status")
    exact_str(get_path(prov, "source_provenance.git_commit"), ORIGINAL_EXECUTION_COMMIT, "source_provenance.git_commit")
    exact_bool(get_path(prov, "source_provenance.git_is_dirty"), False, "source_provenance.git_is_dirty")

    parsed = get_path(prov, "parsed_args")
    resolved = get_path(prov, "resolved_runtime_config")
    split = get_path(prov, "split_seed_contract")
    finalization = get_path(prov, "finalization")
    require(isinstance(parsed, dict), "parsed_args must be object")
    require(isinstance(resolved, dict), "resolved_runtime_config must be object")
    require(isinstance(split, dict), "split_seed_contract must be object")
    require(isinstance(finalization, dict), "finalization must be object")

    exact_int(parsed.get("seed"), SEED, "parsed_args.seed")
    exact_int(resolved.get("seed"), SEED, "resolved_runtime_config.seed")
    exact_int(resolved.get("training_seed"), SEED, "resolved_runtime_config.training_seed")
    exact_int(split.get("training_seed"), SEED, "split_seed_contract.training_seed")
    exact_int(parsed.get("split_seed"), SPLIT_SEED, "parsed_args.split_seed")
    exact_int(resolved.get("resolved_split_seed"), SPLIT_SEED, "resolved_runtime_config.resolved_split_seed")
    exact_int(split.get("resolved_split_seed"), SPLIT_SEED, "split_seed_contract.resolved_split_seed")

    exact_str(parsed.get("architecture"), "v6b_minimal", "parsed_args.architecture")
    exact_str(resolved.get("architecture"), "v6b_minimal", "resolved_runtime_config.architecture")
    exact_str(parsed.get("backbone"), "mamba", "parsed_args.backbone")
    exact_str(resolved.get("backbone"), "mamba", "resolved_runtime_config.backbone")
    exact_str(parsed.get("model_name"), "state-spaces/mamba-130m-hf", "parsed_args.model_name")
    exact_str(resolved.get("model_name"), "state-spaces/mamba-130m-hf", "resolved_runtime_config.model_name")
    exact_str(parsed.get("device"), "cuda", "parsed_args.device")
    exact_str(resolved.get("device_request"), "cuda", "resolved_runtime_config.device_request")
    exact_bool(parsed.get("freeze_encoder"), True, "parsed_args.freeze_encoder")

    p2 = resolved.get("reason_router_p2_contract")
    require(isinstance(p2, dict), "resolved_runtime_config.reason_router_p2_contract must be object")
    exact_str(p2.get("arm"), "A0", "reason_router_p2_contract.arm")
    exact_str(p2.get("router_mode"), "explicit_product", "reason_router_p2_contract.router_mode")
    exact_str(p2.get("gradient_ownership_mode"), "joint", "reason_router_p2_contract.gradient_ownership_mode")
    exact_number(p2.get("reason_loss_weight"), 0.0, "reason_router_p2_contract.reason_loss_weight")
    optional_exact(parsed.get("resolved_reason_router_mode"), "explicit_product", "parsed_args.resolved_reason_router_mode")
    optional_exact(parsed.get("resolved_gradient_ownership_mode"), "joint", "parsed_args.resolved_gradient_ownership_mode")
    optional_exact(parsed.get("resolved_reason_loss_weight"), 0.0, "parsed_args.resolved_reason_loss_weight")

    exact_int(finalization.get("completed_epochs"), 20, "finalization.completed_epochs")
    exact_int(finalization.get("selected_epoch"), 20, "finalization.selected_epoch")
    selected_checkpoint = finalization.get("selected_checkpoint")
    if selected_checkpoint is not None:
        require(isinstance(selected_checkpoint, dict), "finalization.selected_checkpoint must be object")
        optional_exact(selected_checkpoint.get("selected_epoch"), 20, "finalization.selected_checkpoint.selected_epoch")

    count = get_path(prov, "prediction_export_jsonl_audit.prediction_export_row_count", None)
    if count is None:
        count = get_path(prov, "finalization.prediction_export_row_count", None)
    exact_int(count, 720, "prediction_export_row_count")

    exact_str(get_path(prov, "data_provenance.main_data.sha256"), EXPECTED_DATA_PHYSICAL_SHA256, "data_provenance.main_data.sha256")
    exact_str(get_path(prov, "data_provenance.main_data.semantic_sha256"), EXPECTED_DATA_SEMANTIC_SHA256, "data_provenance.main_data.semantic_sha256")
    exact_str(get_path(prov, "compatible_positive_margin.authoritative_dataset_sha256"), EXPECTED_DATA_PHYSICAL_SHA256, "compatible_positive_margin.authoritative_dataset_sha256")
    exact_str(get_path(prov, "compatible_positive_margin.authoritative_dataset_semantic_sha256"), EXPECTED_DATA_SEMANTIC_SHA256, "compatible_positive_margin.authoritative_dataset_semantic_sha256")
    exact_str(get_path(prov, "compatible_positive_margin.authoritative_sidecar_physical_sha256"), EXPECTED_SIDECAR_PHYSICAL_SHA256, "compatible_positive_margin.authoritative_sidecar_physical_sha256")
    exact_str(get_path(prov, "compatible_positive_margin.authoritative_sidecar_semantic_sha256"), EXPECTED_SIDECAR_SEMANTIC_SHA256, "compatible_positive_margin.authoritative_sidecar_semantic_sha256")
    exact_str(get_path(prov, "compatible_positive_margin.authoritative_provenance_physical_sha256"), EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256, "compatible_positive_margin.authoritative_provenance_physical_sha256")
    exact_str(get_path(prov, "resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_physical_sha256"), EXPECTED_SIDECAR_PHYSICAL_SHA256, "resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_physical_sha256")
    exact_str(get_path(prov, "resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_semantic_sha256"), EXPECTED_SIDECAR_SEMANTIC_SHA256, "resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_semantic_sha256")
    exact_str(get_path(prov, "resolved_runtime_config.compatible_positive_margin.authoritative_provenance_physical_sha256"), EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256, "resolved_runtime_config.compatible_positive_margin.authoritative_provenance_physical_sha256")
    exact_str(get_path(prov, "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_sha256"), EXPECTED_DATA_PHYSICAL_SHA256, "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_sha256")
    exact_str(get_path(prov, "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256"), EXPECTED_DATA_SEMANTIC_SHA256, "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256")
    validate_present_identity_copies(prov)


def flag_name(token: str) -> str:
    return token[2:].split("=", 1)[0].replace("-", "_")


def argv_to_options(argv: list[str]) -> dict[str, Any]:
    require(isinstance(argv, list) and all(isinstance(item, str) for item in argv), "argv must be list[str]")
    opts: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        require(token.startswith("--"), f"unexpected positional argument: {token!r}")
        key = flag_name(token)
        require(key in AUTHORIZED_OPTION_NAMES, f"unknown CLI option: --{key.replace('_', '-')}")
        if "=" in token:
            value: Any = token.split("=", 1)[1]
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            value = argv[index + 1]
            index += 1
        else:
            value = True
        if key in opts:
            raise blocker(f"duplicate CLI option: {key}")
        opts[key] = value
        index += 1
    return opts


def command_string_to_argv(command_string: str) -> list[str]:
    require(isinstance(command_string, str) and command_string, "command_string must be nonempty")
    try:
        tokens = shlex.split(command_string)
    except ValueError as exc:
        raise blocker(f"command_string shlex parse failed: {exc}") from exc
    script_index = None
    for index, token in enumerate(tokens):
        normalized = token.replace("\\", "/")
        if normalized.endswith("scripts/train_controlled_v6b_minimal.py") or token == "train_controlled_v6b_minimal.py":
            script_index = index
            break
    require(script_index is not None, "command_string missing trainer script")
    return tokens[script_index + 1 :]


def coerce_expected(value: Any, expected: Any, field: str) -> None:
    if isinstance(expected, bool):
        if isinstance(value, str):
            value = value.lower()
            require(value in {"true", "false"}, f"{field} must be boolean")
            require((value == "true") is expected, f"{field} mismatch")
        else:
            exact_bool(value, expected, field)
    elif isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(value, str):
            require(re.fullmatch(r"-?[0-9]+", value) is not None, f"{field} must be integer")
            value = int(value)
        exact_int(value, expected, field)
    elif isinstance(expected, float):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise blocker(f"{field} must be numeric") from exc
        exact_number(value, expected, field)
    else:
        exact_str(value, expected, field)


def validate_semantic_options(options: dict[str, Any], source: str, *, parsed: bool = False) -> None:
    for key, expected in EXPECTED_ARGS.items():
        require(key in options, f"{source}.{key} missing")
        coerce_expected(options[key], expected, f"{source}.{key}")
    if parsed:
        for field in ("reason_loss_weight", "reason_router_a0_reference_predictions", "train_batch_size", "eval_batch_size"):
            require(field in options, f"{source}.{field} missing")
            require_none(options[field], f"{source}.{field}")
        exact_number(options.get("resolved_reason_loss_weight"), 0.0, f"{source}.resolved_reason_loss_weight")
        optional_exact(options.get("resolved_reason_router_mode"), "explicit_product", f"{source}.resolved_reason_router_mode")
        optional_exact(options.get("resolved_gradient_ownership_mode"), "joint", f"{source}.resolved_gradient_ownership_mode")
    else:
        for omitted in OMITTED_FLAGS:
            require(omitted not in options, f"{source}.{omitted} must be omitted")


def validate_trainer_command(prov: dict[str, Any]) -> None:
    raw_argv = prov.get("raw_sys_argv")
    validate_semantic_options(argv_to_options(raw_argv), "raw_sys_argv")
    parsed_args = prov.get("parsed_args")
    require(isinstance(parsed_args, dict), "parsed_args must be object")
    validate_semantic_options(parsed_args, "parsed_args", parsed=True)
    command_argv = command_string_to_argv(prov.get("command_string"))
    validate_semantic_options(argv_to_options(command_argv), "command_string")


def safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def path_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def create_manifest(implementation_commit: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "artifact_files": artifacts,
        "attempt_disposition": "CONSUMED",
        "execution_status": "completed",
        "implementation_commit": implementation_commit,
        "original_authorized_wrapper_sha256": ORIGINAL_AUTHORIZED_WRAPPER_SHA256,
        "original_execution_commit": ORIGINAL_EXECUTION_COMMIT,
        "recovery_authority_commit": RECOVERY_AUTHORITY_COMMIT,
        "recovery_capture_created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "recovery_capture_time_semantics": "recovery capture time; not historical trainer or cm-wrapper execution time",
        "recovery_scope": RECOVERY_SCOPE,
        "schema": RECOVERY_SCHEMA,
        "scientific_conclusion": "NOT_ESTABLISHED",
        "seed": SEED,
        "source_run_provenance_sha256": ARTIFACT_BY_NAME["run_provenance.json"]["sha256"],
        "source_trainer_git_commit": ORIGINAL_EXECUTION_COMMIT,
        "standard_cm_wrapper_provenance": "missing/incomplete",
    }


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(deterministic_json_bytes(payload))


def collect(args: argparse.Namespace) -> Path:
    expected_commit = require_commit(args.expected_implementation_commit)
    root = repo_root()
    require(git_head(root) == expected_commit, "current HEAD does not match expected implementation commit")
    require_clean_repo(root)
    require_authority_files(root)

    prov_path = SOURCE_RUN_DIR / "run_provenance.json"
    prov = load_json_file_strict(prov_path)
    validate_run_provenance(prov)
    validate_trainer_command(prov)

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    target = PACKAGE_DIR / f"seed180_a0_{expected_commit[:12]}.zip"
    require(not target.exists(), f"target ZIP already exists: {target}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(PACKAGE_DIR))
    os.close(fd)
    temp_path = Path(temp_name)
    opened_artifacts: list[tuple[BinaryIO, dict[str, Any], os.stat_result]] = []
    published = False
    try:
        opened_artifacts = open_validated_source_artifacts(SOURCE_RUN_DIR)
        artifacts = [artifact for _handle, artifact, _metadata in opened_artifacts]
        manifest = create_manifest(expected_commit, artifacts)
        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr("recovery_manifest.json", deterministic_json_bytes(manifest))
                for handle, artifact, _metadata in opened_artifacts:
                    info = zipfile.ZipInfo(zip_artifact_path(artifact["path"].removeprefix(f"{RUN_REL_DIR}/")))
                    info.compress_type = zipfile.ZIP_STORED
                    info.file_size = artifact["size"]
                    with zf.open(info, "w", force_zip64=True) as destination:
                        stream_validated_artifact(handle, artifact, destination)
            validate_completed_temp_package(temp_path, expected_commit)
            revalidate_current_source_paths(SOURCE_RUN_DIR, opened_artifacts)
            with target.open("xb") as output, temp_path.open("rb") as source:
                published = True
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
        except Exception:
            if published and target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            raise
    finally:
        for handle, _artifact, _metadata in opened_artifacts:
            handle.close()
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return target


def validate_zip_structure(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = zf.infolist()
    names = [info.filename for info in infos]
    require(len(names) == len(set(names)), "ZIP duplicate entry")
    logical: set[str] = set()
    by_name: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        require("\x00" not in name, "ZIP NUL path rejected")
        require("\\" not in name, "ZIP backslash path rejected")
        require(not name.startswith("/"), "ZIP absolute path rejected")
        require(not re.match(r"^[A-Za-z]:", name), "ZIP drive path rejected")
        require(not name.endswith("/"), "ZIP directory entry rejected")
        require(not info.is_dir(), "ZIP directory entry rejected")
        pure = PurePosixPath(name)
        require(str(pure) == name, "ZIP path normalization ambiguity")
        require("." not in pure.parts and ".." not in pure.parts, "ZIP traversal or dot path rejected")
        unix_mode = (info.external_attr >> 16) & 0xffff
        unix_type = stat.S_IFMT(unix_mode)
        require(unix_type == 0 or unix_type == stat.S_IFREG, "ZIP special file type rejected")
        require(info.flag_bits & 0x1 == 0, "ZIP encrypted entry rejected")
        require(name not in logical, "ZIP duplicate logical path")
        logical.add(name)
        by_name[name] = info
    require(set(names) == set(EXPECTED_ZIP_ENTRIES), "ZIP allowlist mismatch")
    for item in SOURCE_ARTIFACTS:
        info = by_name[zip_artifact_path(item["name"])]
        require(info.file_size == item["size"], f"packaged artifact size mismatch: {item['name']}")
    return by_name


def sha256_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    with zf.open(info, "r") as handle:
        return sha256_stream(handle)


def stream_validated_artifact(handle: BinaryIO, artifact: dict[str, Any], destination: BinaryIO) -> None:
    digest = hashlib.sha256()
    streamed_size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        streamed_size += len(chunk)
        digest.update(chunk)
        destination.write(chunk)
    require(streamed_size == artifact["size"], f"streamed artifact size mismatch: {artifact['path']}")
    require(digest.hexdigest() == artifact["sha256"], f"streamed artifact sha256 mismatch: {artifact['path']}")


def revalidate_current_source_paths(source_dir: Path, opened_artifacts: list[tuple[BinaryIO, dict[str, Any], os.stat_result]]) -> None:
    source_root = safe_resolve(source_dir)
    for _handle, artifact, original_metadata in opened_artifacts:
        name = PurePosixPath(artifact["path"]).name
        expected = ARTIFACT_BY_NAME.get(name)
        require(expected is not None, f"unknown source artifact during final validation: {name}")
        current_path = source_dir / name
        try:
            current_lstat = os.lstat(current_path)
        except FileNotFoundError as exc:
            raise blocker(f"current source path missing at final validation: {current_path}") from exc
        except OSError as exc:
            raise blocker(f"current source path could not be inspected: {current_path}: {exc}") from exc
        require(not stat.S_ISLNK(current_lstat.st_mode), f"current source path is symlink at final validation: {current_path}")
        require(stat.S_ISREG(current_lstat.st_mode), f"current source path is not regular file at final validation: {current_path}")
        resolved = safe_resolve(current_path)
        require(path_inside(resolved, source_root), f"current source path escaped source directory: {current_path}")
        try:
            with current_path.open("rb") as current_handle:
                current_metadata = os.fstat(current_handle.fileno())
                require(stat.S_ISREG(current_metadata.st_mode), f"current source handle is not regular file at final validation: {current_path}")
                require(os.path.samestat(original_metadata, current_metadata), f"current source path object changed: {current_path}")
                require(current_metadata.st_size == expected["size"], f"current source path size mismatch: {current_path}")
                digest = sha256_stream(current_handle)
                require(digest == expected["sha256"], f"current source path sha256 mismatch: {current_path}")
        except Blocker:
            raise
        except OSError as exc:
            raise blocker(f"current source path could not be opened at final validation: {current_path}: {exc}") from exc


def validate_utc_timestamp(value: Any, field: str) -> None:
    require(isinstance(value, str) and value, f"{field} malformed")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value) is not None, f"{field} malformed")
    try:
        datetime.strptime(value[:-1], "%Y-%m-%dT%H:%M:%S.%f" if "." in value else "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise blocker(f"{field} malformed") from exc


def validate_manifest(manifest: Any, implementation_commit: str) -> None:
    require(isinstance(manifest, dict), "manifest root must be object")
    exact_str(manifest.get("schema"), RECOVERY_SCHEMA, "manifest.schema")
    exact_str(manifest.get("recovery_scope"), RECOVERY_SCOPE, "manifest.recovery_scope")
    exact_str(manifest.get("recovery_authority_commit"), RECOVERY_AUTHORITY_COMMIT, "manifest.recovery_authority_commit")
    exact_str(manifest.get("implementation_commit"), implementation_commit, "manifest.implementation_commit")
    exact_str(manifest.get("original_execution_commit"), ORIGINAL_EXECUTION_COMMIT, "manifest.original_execution_commit")
    exact_int(manifest.get("seed"), SEED, "manifest.seed")
    exact_str(manifest.get("attempt_disposition"), "CONSUMED", "manifest.attempt_disposition")
    exact_str(manifest.get("execution_status"), "completed", "manifest.execution_status")
    exact_str(manifest.get("standard_cm_wrapper_provenance"), "missing/incomplete", "manifest.standard_cm_wrapper_provenance")
    exact_str(manifest.get("original_authorized_wrapper_sha256"), ORIGINAL_AUTHORIZED_WRAPPER_SHA256, "manifest.original_authorized_wrapper_sha256")
    exact_str(manifest.get("source_run_provenance_sha256"), ARTIFACT_BY_NAME["run_provenance.json"]["sha256"], "manifest.source_run_provenance_sha256")
    exact_str(manifest.get("source_trainer_git_commit"), ORIGINAL_EXECUTION_COMMIT, "manifest.source_trainer_git_commit")
    exact_str(manifest.get("scientific_conclusion"), "NOT_ESTABLISHED", "manifest.scientific_conclusion")
    validate_utc_timestamp(manifest.get("recovery_capture_created_at_utc"), "manifest recovery timestamp")
    exact_str(manifest.get("recovery_capture_time_semantics"), "recovery capture time; not historical trainer or cm-wrapper execution time", "manifest.recovery_capture_time_semantics")
    table = manifest.get("artifact_files")
    require(isinstance(table, list), "manifest.artifact_files must be list")
    expected = {f"{RUN_REL_DIR}/{item['name']}": item for item in SOURCE_ARTIFACTS}
    seen: set[str] = set()
    for row in table:
        require(isinstance(row, dict), "manifest artifact row must be object")
        path = row.get("path")
        require(isinstance(path, str) and path in expected and path not in seen, "manifest artifact path mismatch")
        seen.add(path)
        item = expected[path]
        exact_int(row.get("size"), item["size"], f"manifest artifact {path} size")
        exact_str(row.get("sha256"), item["sha256"], f"manifest artifact {path} sha256")
    require(seen == set(expected), "manifest artifact table incomplete")


def validate_completed_temp_package(temp_path: Path, implementation_commit: str) -> None:
    with zipfile.ZipFile(temp_path, "r") as zf:
        infos = validate_zip_structure(zf)
        manifest_text = zf.read(infos["recovery_manifest.json"]).decode("utf-8")
        validate_manifest(loads_json_strict(manifest_text), implementation_commit)
        for item in SOURCE_ARTIFACTS:
            info = infos[zip_artifact_path(item["name"])]
            require(info.file_size == item["size"], f"temporary packaged artifact size mismatch: {item['name']}")
            require(sha256_zip_entry(zf, info) == item["sha256"], f"temporary packaged artifact sha mismatch: {item['name']}")


def audit_import(args: argparse.Namespace) -> Path:
    expected_commit = require_commit(args.expected_implementation_commit)
    root = repo_root()
    require(git_head(root) == expected_commit, "current HEAD does not match expected implementation commit")
    require_clean_repo(root)
    zip_path = safe_resolve(Path(args.zip))
    require(zip_path.exists() and not zip_path.is_symlink() and zip_path.is_file(), "ZIP missing or not regular file")
    audit_output = safe_resolve(Path(args.audit_output))
    require(not path_inside(audit_output, root), "audit-output must be outside repository")
    require(not audit_output.exists(), "audit-output already exists")

    package_hash = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = validate_zip_structure(zf)
        manifest_info = infos["recovery_manifest.json"]
        manifest_text = zf.read(manifest_info).decode("utf-8")
        manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        manifest = loads_json_strict(manifest_text)
        validate_manifest(manifest, expected_commit)
        for item in SOURCE_ARTIFACTS:
            info = infos[zip_artifact_path(item["name"])]
            require(sha256_zip_entry(zf, info) == item["sha256"], f"packaged artifact sha mismatch: {item['name']}")
        prov_text = zf.read(infos[zip_artifact_path("run_provenance.json")]).decode("utf-8")
        prov = loads_json_strict(prov_text)
        validate_run_provenance(prov)
        validate_trainer_command(prov)

    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "artifact_validation": "VALIDATED",
        "execution_success": "OBSERVED",
        "implementation_commit": expected_commit,
        "manifest_sha256": manifest_hash,
        "original_execution_commit": ORIGINAL_EXECUTION_COMMIT,
        "package_path": str(zip_path),
        "package_sha256": package_hash,
        "recovered_artifact_identity": "VALIDATED",
        "recovery_authority_commit": RECOVERY_AUTHORITY_COMMIT,
        "schema": AUDIT_SCHEMA,
        "scientific_conclusion": "NOT_ESTABLISHED",
        "seed": SEED,
        "standard_cm_wrapper_provenance": "INCOMPLETE",
        "trainer_command_validation": "VALIDATED",
        "trainer_provenance_validation": "VALIDATED",
    }
    write_exclusive_json(audit_output, audit)
    return audit_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=RECOVERY_SCOPE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--expected-implementation-commit", required=True)
    collect_parser.set_defaults(func=collect)
    audit_parser = subparsers.add_parser("audit-import")
    audit_parser.add_argument("--zip", required=True)
    audit_parser.add_argument("--expected-implementation-commit", required=True)
    audit_parser.add_argument("--audit-output", required=True)
    audit_parser.set_defaults(func=audit_import)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except Blocker as exc:
        print(str(exc), file=sys.stderr)
        return 64
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
