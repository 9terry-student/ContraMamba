from __future__ import annotations

import ast
import copy
import errno
import hashlib
import importlib.util
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest


def pytest_configure(config):
    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = str(Path(os.environ.get("TEMP", os.getcwd())) / "contramamba_recovery_pytest_tmp")


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reason_router_p3w7_a0_seed180_provenance_recovery.py"
spec = importlib.util.spec_from_file_location("seed180_recovery", MODULE_PATH)
recovery = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recovery)

IMPL = "cdd71ea4f556392eab594ebb5df8258355610e01"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_filesystem_symlink_or_skip(target: Path, link: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("filesystem symlink creation is not available on this host")
    try:
        os.symlink(target, link)
    except PermissionError:
        pytest.skip("filesystem symlink creation is not representable under the current host privileges")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314 or exc.errno in {errno.EPERM, errno.EACCES}:
            pytest.skip("filesystem symlink creation is not representable under the current host privileges")
        raise


def args_list() -> list[str]:
    return [
        "--data", recovery.EXPECTED_DATA_PATH,
        "--architecture", "v6b_minimal",
        "--backbone", "mamba",
        "--model-name", "state-spaces/mamba-130m-hf",
        "--freeze-encoder", "true",
        "--frame-downstream-gradient-mode", "joint",
        "--epochs", "20",
        "--max-length", "128",
        "--dev-ratio", "0.2",
        "--seed", "180",
        "--split-seed", "174",
        "--device", "cuda",
        "--flag-source", "controlled_heuristic",
        "--select-metric", "final_macro_f1",
        "--ranking-weight", "0.0",
        "--class-weighting", "none",
        "--stage174c-clean-pairwise-mode", "off",
        "--stage174c-clean-pairwise-weight", "0.0",
        "--stage174c-clean-polarity-preservation-weight", "0.0",
        "--stage175b-support-anchor-mode", "off",
        "--stage175b-support-anchor-weight", "0.0",
        "--stage177c-frame-pairwise-mode", "off",
        "--stage177c-frame-pairwise-weight", "0.0",
        "--compatible-positive-margin-logit", "0.0",
        "--lr", "0.001",
        "--reason-router-arm", "A0",
        "--reason-router-mode", "explicit_product",
        "--gradient-ownership-mode", "joint",
        "--controlled-integrity-sidecar-path", recovery.EXPECTED_SIDECAR_PATH,
        "--expected-integrity-sidecar-semantic-sha256", recovery.EXPECTED_SIDECAR_SEMANTIC_SHA256,
        "--compatible-positive-margin-weight", "0.0",
        "--save-selected-checkpoint",
        "--selected-checkpoint-filename", "selected_checkpoint.pt",
        "--output-json", "/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json",
        "--output-predictions-json", "/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json",
    ]


def parsed_args() -> dict[str, object]:
    values = copy.deepcopy(recovery.EXPECTED_ARGS)
    values.update(
        {
            "reason_loss_weight": None,
            "reason_router_a0_reference_predictions": None,
            "train_batch_size": None,
            "eval_batch_size": None,
            "resolved_reason_loss_weight": 0.0,
            "resolved_reason_router_mode": "explicit_product",
            "resolved_gradient_ownership_mode": "joint",
        }
    )
    return values


def valid_provenance() -> dict[str, object]:
    return {
        "schema_version": "stage174a_v1",
        "status": "completed",
        "source_provenance": {
            "git_commit": recovery.ORIGINAL_EXECUTION_COMMIT,
            "git_is_dirty": False,
        },
        "raw_sys_argv": args_list(),
        "command_string": "python scripts/train_controlled_v6b_minimal.py " + " ".join(args_list()),
        "parsed_args": parsed_args(),
        "resolved_runtime_config": {
            "architecture": "v6b_minimal",
            "backbone": "mamba",
            "model_name": "state-spaces/mamba-130m-hf",
            "device_request": "cuda",
            "seed": 180,
            "training_seed": 180,
            "resolved_split_seed": 174,
            "reason_router_p2_contract": {
                "arm": "A0",
                "router_mode": "explicit_product",
                "gradient_ownership_mode": "joint",
                "reason_loss_weight": 0.0,
            },
            "compatible_positive_margin": {
                "authoritative_dataset_sha256": recovery.EXPECTED_DATA_PHYSICAL_SHA256,
                "authoritative_dataset_semantic_sha256": recovery.EXPECTED_DATA_SEMANTIC_SHA256,
                "authoritative_sidecar_physical_sha256": recovery.EXPECTED_SIDECAR_PHYSICAL_SHA256,
                "authoritative_sidecar_semantic_sha256": recovery.EXPECTED_SIDECAR_SEMANTIC_SHA256,
                "authoritative_provenance_physical_sha256": recovery.EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256,
            },
            "active_bridge_auxiliary_modes_and_row_counts": {
                "row_counts": {"dev_rows": 720},
            },
        },
        "split_seed_contract": {
            "training_seed": 180,
            "resolved_split_seed": 174,
            "clean_main_dev_rows": 720,
        },
        "data_provenance": {
            "main_data": {
                "sha256": recovery.EXPECTED_DATA_PHYSICAL_SHA256,
                "semantic_sha256": recovery.EXPECTED_DATA_SEMANTIC_SHA256,
            },
            "auxiliary_activity": {
                "row_counts": {"dev_rows": 720},
            },
        },
        "compatible_positive_margin": {
            "authoritative_dataset_sha256": recovery.EXPECTED_DATA_PHYSICAL_SHA256,
            "authoritative_dataset_semantic_sha256": recovery.EXPECTED_DATA_SEMANTIC_SHA256,
            "authoritative_sidecar_physical_sha256": recovery.EXPECTED_SIDECAR_PHYSICAL_SHA256,
            "authoritative_sidecar_semantic_sha256": recovery.EXPECTED_SIDECAR_SEMANTIC_SHA256,
            "authoritative_provenance_physical_sha256": recovery.EXPECTED_P4L_PROVENANCE_PHYSICAL_SHA256,
        },
        "finalization": {
            "completed_epochs": 20,
            "selected_epoch": 20,
            "selected_checkpoint": {
                "checkpoint_is_selected_clean_dev_state": True,
                "external_data_used": False,
                "external_labels_used": False,
                "filename": "selected_checkpoint.pt",
                "path": recovery.EXPECTED_SELECTED_CHECKPOINT_PATH,
                "saved": True,
                "schema_version": "stage176a0_selected_checkpoint_v1",
                "selected_epoch": 20,
                "sha256": recovery.ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"],
                "size_bytes": recovery.ARTIFACT_BY_NAME["selected_checkpoint.pt"]["size"],
                "teacher_checkpoint_used": False,
                "time_swap_used": False,
            },
        },
    }


def clean_predictions_bytes(count: int = 720) -> bytes:
    return json.dumps({"predictions": [{"row_id": index} for index in range(count)]}, sort_keys=True).encode("utf-8")


def jsonl_predictions_bytes(count: int = 720) -> bytes:
    return "".join(json.dumps({"row_id": index}) + "\n" for index in range(count)).encode("utf-8")


@pytest.fixture()
def synthetic_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    package_dir = tmp_path / "packages"
    source.mkdir()
    package_dir.mkdir()
    contents = {
        "training_report.json": b"report",
        "clean_dev_predictions.json": clean_predictions_bytes(),
        "training_report_predictions.jsonl": jsonl_predictions_bytes(),
        "selected_checkpoint.pt": b"checkpoint",
    }
    artifacts = []
    for name, data in contents.items():
        (source / name).write_bytes(data)
        artifacts.append({"name": name, "size": len(data), "sha256": sha(data)})
    monkeypatch.setattr(recovery, "SOURCE_RUN_DIR", source)
    monkeypatch.setattr(recovery, "PACKAGE_DIR", package_dir)
    monkeypatch.setattr(recovery, "SOURCE_ARTIFACTS", artifacts)
    monkeypatch.setattr(recovery, "ARTIFACT_BY_NAME", {item["name"]: item for item in artifacts})
    monkeypatch.setattr(recovery, "EXPECTED_ZIP_ENTRIES", ["recovery_manifest.json", *[recovery.zip_artifact_path(item["name"]) for item in artifacts]])
    run_provenance_data = json.dumps(valid_provenance(), sort_keys=True).encode("utf-8")
    (source / "run_provenance.json").write_bytes(run_provenance_data)
    artifacts.append({"name": "run_provenance.json", "size": len(run_provenance_data), "sha256": sha(run_provenance_data)})
    monkeypatch.setattr(recovery, "SOURCE_ARTIFACTS", artifacts)
    monkeypatch.setattr(recovery, "ARTIFACT_BY_NAME", {item["name"]: item for item in artifacts})
    monkeypatch.setattr(recovery, "EXPECTED_ZIP_ENTRIES", ["recovery_manifest.json", *[recovery.zip_artifact_path(item["name"]) for item in artifacts]])
    root = tmp_path / "repo"
    for rel in (recovery.TOOLING_AUTHORITY_PATH, recovery.RECOVERY_AUTHORITY_PATH, recovery.ORIGINAL_A0_AUTHORITY_PATH):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("authority\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "repo_root", lambda: root)
    monkeypatch.setattr(recovery, "git_head", lambda root_arg: IMPL)
    monkeypatch.setattr(recovery, "require_clean_repo", lambda root_arg: None)
    return source, package_dir, root


def assert_blocked(fn, *args, **kwargs):
    with pytest.raises(recovery.Blocker, match="PROVENANCE_RECOVERY_BLOCKER"):
        fn(*args, **kwargs)


def set_nested(payload: dict[str, object], path: str, value: object) -> None:
    cursor = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def delete_nested(payload: dict[str, object], path: str) -> None:
    cursor = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    del cursor[parts[-1]]


def rewrite_source_artifact(source: Path, name: str, data: bytes) -> None:
    (source / name).write_bytes(data)
    item = next(item for item in recovery.SOURCE_ARTIFACTS if item["name"] == name)
    item["size"] = len(data)
    item["sha256"] = sha(data)
    recovery.ARTIFACT_BY_NAME[name] = item


def rewrite_run_provenance(source: Path, prov: dict[str, object]) -> None:
    rewrite_source_artifact(source, "run_provenance.json", json.dumps(prov, sort_keys=True).encode("utf-8"))


def assert_collect_blocks_without_zip(package_dir: Path) -> None:
    assert recovery.main(["collect", "--expected-implementation-commit", IMPL]) == 64
    assert not list(package_dir.glob("*.zip"))


def test_duplicate_key_rejecting_json_happy_path():
    assert recovery.loads_json_strict('{"a": 1}') == {"a": 1}
    assert_blocked(recovery.loads_json_strict, '{"a": 1, "a": 2}')


def test_valid_synthetic_collect_exact_six_entry_zip_and_audit_import(synthetic_env, tmp_path: Path):
    _source, package_dir, _root = synthetic_env
    assert "prediction_export_jsonl_audit" not in valid_provenance()
    target = recovery.main(["collect", "--expected-implementation-commit", IMPL])
    assert target == 0
    zips = list(package_dir.glob("*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as zf:
        assert zf.namelist() == recovery.EXPECTED_ZIP_ENTRIES
        assert all(name == "recovery_manifest.json" or name.startswith("files/") for name in zf.namelist())
        assert not any(name.startswith("reports/") for name in zf.namelist())
        manifest = recovery.loads_json_strict(zf.read("recovery_manifest.json").decode("utf-8"))
    recovery.validate_manifest(manifest, IMPL)

    audit_path = tmp_path / "outside" / "audit.json"
    assert recovery.main(["audit-import", "--zip", str(zips[0]), "--expected-implementation-commit", IMPL, "--audit-output", str(audit_path)]) == 0
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["artifact_validation"] == "VALIDATED"
    assert audit["standard_cm_wrapper_provenance"] == "INCOMPLETE"


def test_deterministic_artifact_validation(synthetic_env):
    source, _package_dir, _root = synthetic_env
    first = recovery.validate_source_artifacts(source)
    second = recovery.validate_source_artifacts(source)
    assert first == second


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("schema_version", "bad"),
        ("source_provenance.git_commit", "0" * 40),
        ("source_provenance.git_is_dirty", True),
        ("parsed_args.seed", 181),
        ("resolved_runtime_config.seed", 181),
        ("parsed_args.split_seed", 175),
        ("resolved_runtime_config.resolved_split_seed", 175),
        ("parsed_args.architecture", "bad"),
        ("parsed_args.backbone", "bert"),
        ("parsed_args.model_name", "bad"),
        ("parsed_args.device", "cpu"),
        ("parsed_args.freeze_encoder", False),
        ("resolved_runtime_config.reason_router_p2_contract.arm", "A1"),
        ("resolved_runtime_config.reason_router_p2_contract.router_mode", "conditional_first_blocker"),
        ("resolved_runtime_config.reason_router_p2_contract.gradient_ownership_mode", "explicit_local"),
        ("resolved_runtime_config.reason_router_p2_contract.reason_loss_weight", 1.0),
        ("data_provenance.main_data.sha256", "bad"),
        ("data_provenance.main_data.semantic_sha256", "bad"),
        ("resolved_runtime_config.compatible_positive_margin.authoritative_dataset_sha256", "bad"),
        ("resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256", "bad"),
        ("compatible_positive_margin.authoritative_dataset_semantic_sha256", "bad"),
        ("compatible_positive_margin.authoritative_sidecar_physical_sha256", "bad"),
        ("compatible_positive_margin.authoritative_sidecar_semantic_sha256", "bad"),
        ("compatible_positive_margin.authoritative_provenance_physical_sha256", "bad"),
        ("resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_physical_sha256", "bad"),
        ("resolved_runtime_config.compatible_positive_margin.authoritative_sidecar_semantic_sha256", "bad"),
        ("resolved_runtime_config.compatible_positive_margin.authoritative_provenance_physical_sha256", "bad"),
        ("split_seed_contract.resolved_split_seed", 175),
        ("finalization.selected_checkpoint.sha256", "bad"),
        ("finalization.completed_epochs", 19),
        ("finalization.selected_epoch", 19),
        ("data_provenance.auxiliary_activity.row_counts.dev_rows", 719),
        ("resolved_runtime_config.active_bridge_auxiliary_modes_and_row_counts.row_counts.dev_rows", 719),
        ("split_seed_contract.clean_main_dev_rows", 719),
    ],
)
def test_run_provenance_validation_failures(path: str, value: object):
    prov = valid_provenance()
    set_nested(prov, path, value)
    assert_blocked(recovery.validate_run_provenance, prov)


@pytest.mark.parametrize(
    "path",
    [
        "data_provenance.auxiliary_activity.row_counts.dev_rows",
        "resolved_runtime_config.active_bridge_auxiliary_modes_and_row_counts.row_counts.dev_rows",
        "split_seed_contract.clean_main_dev_rows",
    ],
)
def test_required_provenance_dev_row_paths_missing(path: str):
    prov = valid_provenance()
    delete_nested(prov, path)
    assert_blocked(recovery.validate_run_provenance, prov)


@pytest.mark.parametrize(
    "path",
    [
        "prediction_export_jsonl_audit.prediction_export_row_count",
        "finalization.prediction_export_row_count",
    ],
)
def test_legacy_prediction_export_row_count_absent_allowed_but_contradiction_blocks(path: str):
    recovery.validate_run_provenance(valid_provenance())
    prov = valid_provenance()
    parts = path.split(".")
    cursor = prov
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = 719
    assert_blocked(recovery.validate_run_provenance, prov)


@pytest.mark.parametrize("count", [719, 721])
def test_clean_dev_predictions_wrong_cardinality_blocks_collect(synthetic_env, count: int):
    source, package_dir, _root = synthetic_env
    rewrite_source_artifact(source, "clean_dev_predictions.json", clean_predictions_bytes(count))
    assert_collect_blocks_without_zip(package_dir)


@pytest.mark.parametrize(
    "data",
    [
        b"{bad",
        b'{"predictions": [], "predictions": []}',
        b"{}",
        b'{"predictions": {}}',
        b"\xff",
    ],
)
def test_clean_dev_predictions_malformed_duplicate_missing_non_list_or_utf8_blocks_collect(synthetic_env, data: bytes):
    source, package_dir, _root = synthetic_env
    rewrite_source_artifact(source, "clean_dev_predictions.json", data)
    assert_collect_blocks_without_zip(package_dir)


@pytest.mark.parametrize("count", [719, 721])
def test_training_report_predictions_jsonl_wrong_cardinality_blocks_collect(synthetic_env, count: int):
    source, package_dir, _root = synthetic_env
    rewrite_source_artifact(source, "training_report_predictions.jsonl", jsonl_predictions_bytes(count))
    assert_collect_blocks_without_zip(package_dir)


@pytest.mark.parametrize(
    "data",
    [
        jsonl_predictions_bytes(719) + b"\n\n",
        jsonl_predictions_bytes(719) + b"{bad\n",
        jsonl_predictions_bytes(719) + b'{"row_id": 719, "row_id": 720}\n',
        b"\xff",
    ],
)
def test_training_report_predictions_jsonl_malformed_duplicate_or_utf8_blocks_collect(synthetic_env, data: bytes):
    source, package_dir, _root = synthetic_env
    rewrite_source_artifact(source, "training_report_predictions.jsonl", data)
    assert_collect_blocks_without_zip(package_dir)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data_provenance.auxiliary_activity.row_counts.dev_rows", 719),
        ("resolved_runtime_config.active_bridge_auxiliary_modes_and_row_counts.row_counts.dev_rows", 719),
        ("split_seed_contract.clean_main_dev_rows", 719),
    ],
)
def test_provenance_cardinality_disagreement_blocks_collect(synthetic_env, path: str, value: object):
    source, package_dir, _root = synthetic_env
    prov = valid_provenance()
    set_nested(prov, path, value)
    rewrite_run_provenance(source, prov)
    assert_collect_blocks_without_zip(package_dir)


@pytest.mark.parametrize(
    "path",
    [
        "compatible_positive_margin.authoritative_dataset_semantic_sha256",
        "resolved_runtime_config.compatible_positive_margin.authoritative_dataset_semantic_sha256",
    ],
)
def test_dataset_semantic_sha_required_missing_and_null(path: str):
    missing = valid_provenance()
    delete_nested(missing, path)
    assert_blocked(recovery.validate_run_provenance, missing)

    null_value = valid_provenance()
    cursor = null_value
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = None
    assert_blocked(recovery.validate_run_provenance, null_value)


def test_dataset_semantic_sha_contradictory_duplicate_rejected():
    prov = valid_provenance()
    prov["resolved_runtime_config"]["compatible_positive_margin"]["authoritative_dataset_semantic_sha256"] = "0" * 64
    assert_blocked(recovery.validate_run_provenance, prov)


def test_dataset_semantic_sha_required_value_accepted():
    recovery.validate_run_provenance(valid_provenance())


@pytest.mark.parametrize(
    "path",
    [
        "finalization.selected_checkpoint",
        "finalization.selected_checkpoint.sha256",
        "finalization.selected_checkpoint.filename",
        "finalization.selected_checkpoint.path",
    ],
)
def test_selected_checkpoint_identity_required_missing_and_null(path: str):
    missing = valid_provenance()
    delete_nested(missing, path)
    assert_blocked(recovery.validate_run_provenance, missing)

    null_value = valid_provenance()
    cursor = null_value
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = None
    assert_blocked(recovery.validate_run_provenance, null_value)


def test_selected_checkpoint_wrong_sha_and_contradictory_duplicate_rejected():
    wrong = valid_provenance()
    wrong["finalization"]["selected_checkpoint"]["sha256"] = "0" * 64
    assert_blocked(recovery.validate_run_provenance, wrong)

    contradictory = valid_provenance()
    contradictory["finalization"]["selected_checkpoint"]["checkpoint_sha256"] = "0" * 64
    assert_blocked(recovery.validate_run_provenance, contradictory)


def test_selected_checkpoint_identity_required_value_accepted():
    recovery.validate_run_provenance(valid_provenance())


@pytest.mark.parametrize(
    ("mutator"),
    [
        lambda p: p["raw_sys_argv"].extend(["--reason-loss-weight", "0.0"]),
        lambda p: p["raw_sys_argv"].extend(["--reason-router-a0-reference-predictions", "x.jsonl"]),
        lambda p: p["raw_sys_argv"].extend(["--batch-size", "8"]),
        lambda p: p["raw_sys_argv"].__setitem__(p["raw_sys_argv"].index("180"), "181"),
        lambda p: p["parsed_args"].__setitem__("lr", 0.002),
        lambda p: p["parsed_args"].__setitem__("reason_loss_weight", 0.0),
        lambda p: p["parsed_args"].__setitem__("train_batch_size", 8),
        lambda p: p.__setitem__("command_string", "python scripts/train_controlled_v6b_minimal.py --seed 180"),
    ],
)
def test_trainer_command_semantic_failures(mutator):
    prov = valid_provenance()
    mutator(prov)
    assert_blocked(recovery.validate_trainer_command, prov)


@pytest.mark.parametrize(
    ("mutator"),
    [
        lambda p: p["raw_sys_argv"].extend(["--unknown-option"]),
        lambda p: p["raw_sys_argv"].extend(["--unknown-option", "value"]),
        lambda p: p.__setitem__("command_string", p["command_string"] + " --unknown-option value"),
        lambda p: p["raw_sys_argv"].append("unexpected-positional"),
        lambda p: p["raw_sys_argv"].extend(["--seed", "180"]),
    ],
)
def test_trainer_command_rejects_unknown_or_unexpected_options(mutator):
    prov = valid_provenance()
    mutator(prov)
    assert_blocked(recovery.validate_trainer_command, prov)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-99-01T00:00:00Z",
        "2026-02-30T00:00:00Z",
        "2026-01-01T24:00:00Z",
        "2026-01-01T00:60:00Z",
        "2026-01-01T00:00:60Z",
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00Ztrailing",
        "",
    ],
)
def test_manifest_timestamp_rejects_noncanonical_or_impossible_values(timestamp):
    artifacts = [{"path": f"{recovery.RUN_REL_DIR}/{item['name']}", "size": item["size"], "sha256": item["sha256"]} for item in recovery.SOURCE_ARTIFACTS]
    manifest = recovery.create_manifest(IMPL, artifacts)
    manifest["recovery_capture_created_at_utc"] = timestamp
    assert_blocked(recovery.validate_manifest, manifest, IMPL)


def test_manifest_timestamp_accepts_canonical_utc_value():
    artifacts = [{"path": f"{recovery.RUN_REL_DIR}/{item['name']}", "size": item["size"], "sha256": item["sha256"]} for item in recovery.SOURCE_ARTIFACTS]
    manifest = recovery.create_manifest(IMPL, artifacts)
    manifest["recovery_capture_created_at_utc"] = "2026-08-31T12:34:56.123456Z"
    recovery.validate_manifest(manifest, IMPL)


def test_malformed_expected_implementation_commit(synthetic_env):
    assert recovery.main(["collect", "--expected-implementation-commit", "BAD"]) == 64


def test_wrong_current_head(synthetic_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(recovery, "git_head", lambda root_arg: "0" * 40)
    assert recovery.main(["collect", "--expected-implementation-commit", IMPL]) == 64


def test_tracked_dirty_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(recovery, "run_git", lambda root, args: " M x.py" if args[:2] == ["status", "--porcelain=v1"] else "")
    assert_blocked(recovery.require_clean_repo, tmp_path)


def test_staged_dirty_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(recovery, "run_git", lambda root, args: "A\tx.py" if args == ["diff", "--cached", "--name-status"] else "")
    assert_blocked(recovery.require_clean_repo, tmp_path)


def test_missing_authority_file(synthetic_env, root=None):
    _source, _package_dir, repo = synthetic_env
    (repo / recovery.RECOVERY_AUTHORITY_PATH).unlink()
    assert recovery.main(["collect", "--expected-implementation-commit", IMPL]) == 64


def test_source_artifact_missing_wrong_size_wrong_sha_and_symlink(synthetic_env, tmp_path: Path):
    source, _package_dir, _root = synthetic_env
    (source / "training_report.json").unlink()
    assert_blocked(recovery.validate_source_artifacts, source)
    (source / "training_report.json").write_bytes(b"bad-size")
    assert_blocked(recovery.validate_source_artifacts, source)
    item = next(item for item in recovery.SOURCE_ARTIFACTS if item["name"] == "training_report.json")
    (source / "training_report.json").write_bytes(b"report")
    item["sha256"] = "0" * 64
    assert_blocked(recovery.validate_source_artifacts, source)
    item["sha256"] = sha(b"report")
    (source / "training_report.json").unlink()
    create_filesystem_symlink_or_skip(tmp_path / "target", source / "training_report.json")
    assert_blocked(recovery.validate_source_artifacts, source)


def test_malformed_and_duplicate_run_provenance_json(synthetic_env):
    source, _package_dir, _root = synthetic_env
    (source / "run_provenance.json").write_text("{bad", encoding="utf-8")
    assert recovery.main(["collect", "--expected-implementation-commit", IMPL]) == 64
    (source / "run_provenance.json").write_text('{"schema_version":"stage174a_v1","schema_version":"x"}', encoding="utf-8")
    assert recovery.main(["collect", "--expected-implementation-commit", IMPL]) == 64


def test_output_zip_collision(synthetic_env):
    _source, package_dir, _root = synthetic_env
    (package_dir / f"seed180_a0_{IMPL[:12]}.zip").write_bytes(b"exists")
    assert recovery.main(["collect", "--expected-implementation-commit", IMPL]) == 64


def test_package_creation_failure_leaves_no_completed_target_zip(synthetic_env, monkeypatch: pytest.MonkeyPatch):
    _source, package_dir, _root = synthetic_env
    original = zipfile.ZipFile

    class FailingZip(original):
        def writestr(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(recovery.zipfile, "ZipFile", FailingZip)
    with pytest.raises(RuntimeError):
        recovery.collect(type("Args", (), {"expected_implementation_commit": IMPL})())
    assert not (package_dir / f"seed180_a0_{IMPL[:12]}.zip").exists()


def make_zip(path: Path, entries: dict[str, bytes], attrs: dict[str, int] | None = None, flags: dict[str, int] | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (attrs or {}).get(name, 0)
            info.flag_bits = (flags or {}).get(name, 0)
            zf.writestr(info, data)


def valid_zip_from_env(synthetic_env, path: Path) -> None:
    source, _package_dir, _root = synthetic_env
    recovery.collect(type("Args", (), {"expected_implementation_commit": IMPL})())
    produced = next((_package_dir).glob("*.zip"))
    path.write_bytes(produced.read_bytes())


@pytest.mark.parametrize(
    ("bad_name"),
    [
        "/absolute",
        "../traversal",
        "files\\bad",
        "files/./bad",
    ],
)
def test_zip_path_security_rejections(synthetic_env, tmp_path: Path, bad_name: str):
    entries = {name: b"x" for name in recovery.EXPECTED_ZIP_ENTRIES}
    entries.pop(recovery.EXPECTED_ZIP_ENTRIES[-1])
    entries[bad_name] = b"x"
    zip_path = tmp_path / "bad.zip"
    make_zip(zip_path, entries)
    with zipfile.ZipFile(zip_path) as zf:
        assert_blocked(recovery.validate_zip_structure, zf)


def test_zip_nul_rejection_if_constructible(synthetic_env, tmp_path: Path):
    info = zipfile.ZipInfo("bad\x00name")
    zip_path = tmp_path / "nul.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, b"x")
    with zipfile.ZipFile(zip_path) as zf:
        assert_blocked(recovery.validate_zip_structure, zf)


def test_zip_symlink_encrypted_duplicate_missing_extra_directory(synthetic_env, tmp_path: Path):
    entries = {name: b"x" for name in recovery.EXPECTED_ZIP_ENTRIES}
    zip_path = tmp_path / "symlink.zip"
    make_zip(zip_path, entries, attrs={recovery.EXPECTED_ZIP_ENTRIES[-1]: (stat.S_IFLNK | 0o777) << 16})
    with zipfile.ZipFile(zip_path) as zf:
        assert_blocked(recovery.validate_zip_structure, zf)

    zip_path = tmp_path / "encrypted.zip"
    make_zip(zip_path, entries, flags={recovery.EXPECTED_ZIP_ENTRIES[-1]: 1})
    with zipfile.ZipFile(zip_path) as zf:
        zf.infolist()[-1].flag_bits = 1
        assert_blocked(recovery.validate_zip_structure, zf)

    duplicate = tmp_path / "dup.zip"
    with zipfile.ZipFile(duplicate, "w") as zf:
        for name in recovery.EXPECTED_ZIP_ENTRIES:
            zf.writestr(name, b"x")
        zf.writestr(recovery.EXPECTED_ZIP_ENTRIES[-1], b"y")
    with zipfile.ZipFile(duplicate) as zf:
        assert_blocked(recovery.validate_zip_structure, zf)

    for label, mutated in [
        ("missing", {name: b"x" for name in recovery.EXPECTED_ZIP_ENTRIES[:-1]}),
        ("extra", {**{name: b"x" for name in recovery.EXPECTED_ZIP_ENTRIES}, "extra": b"x"}),
        ("directory", {**{name: b"x" for name in recovery.EXPECTED_ZIP_ENTRIES}, "dir/": b""}),
    ]:
        path = tmp_path / f"{label}.zip"
        make_zip(path, mutated)
        with zipfile.ZipFile(path) as zf:
            assert_blocked(recovery.validate_zip_structure, zf)


@pytest.mark.parametrize(
    ("mode", "accepted"),
    [
        (0, True),
        (stat.S_IFREG | 0o644, True),
        (stat.S_IFDIR | 0o755, False),
        (stat.S_IFLNK | 0o777, False),
        (stat.S_IFIFO | 0o644, False),
        (stat.S_IFSOCK | 0o644, False),
        (stat.S_IFCHR | 0o644, False),
        (stat.S_IFBLK | 0o644, False),
    ],
)
def test_zip_allowed_entry_requires_regular_file_type(synthetic_env, tmp_path: Path, mode: int, accepted: bool):
    entries = {"recovery_manifest.json": b"manifest"}
    for item in recovery.SOURCE_ARTIFACTS:
        entries[f"files/{recovery.RUN_REL_DIR}/{item['name']}"] = b"x" * item["size"]
    path = tmp_path / f"mode-{mode:o}.zip"
    make_zip(path, entries, attrs={recovery.EXPECTED_ZIP_ENTRIES[-1]: mode << 16})
    with zipfile.ZipFile(path) as zf:
        if accepted:
            recovery.validate_zip_structure(zf)
        else:
            assert_blocked(recovery.validate_zip_structure, zf)


def test_collect_packages_same_validated_handles_after_source_replacement(synthetic_env, monkeypatch: pytest.MonkeyPatch):
    original = recovery.open_validated_source_artifacts

    def open_then_replace(source_dir):
        opened = original(source_dir)
        (source_dir / "training_report.json").write_bytes(b"tamper")
        return opened

    monkeypatch.setattr(recovery, "open_validated_source_artifacts", open_then_replace)
    with pytest.raises(recovery.Blocker, match="PROVENANCE_RECOVERY_BLOCKER"):
        recovery.collect(type("Args", (), {"expected_implementation_commit": IMPL})())
    assert not list(synthetic_env[1].glob("*.zip"))


def assert_collect_blocks_after_atomic_replace(synthetic_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_bytes: bytes) -> None:
    source, package_dir, _root = synthetic_env
    original = recovery.open_validated_artifact

    def open_then_replace(path: Path, expected: dict[str, object]):
        if expected["name"] != "training_report.json":
            return original(path, expected)
        handle, artifact, metadata = original(path, expected)
        old_object_link = tmp_path / "training_report.old-object"
        os.link(path, old_object_link)
        handle.close()
        old_handle = old_object_link.open("rb")
        assert os.path.samestat(metadata, os.fstat(old_handle.fileno()))
        replacement = tmp_path / "training_report.replacement"
        replacement.write_bytes(replacement_bytes)
        os.replace(replacement, source / "training_report.json")
        return old_handle, artifact, metadata

    monkeypatch.setattr(recovery, "open_validated_artifact", open_then_replace)
    with pytest.raises(recovery.Blocker, match="PROVENANCE_RECOVERY_BLOCKER"):
        recovery.collect(type("Args", (), {"expected_implementation_commit": IMPL})())
    assert not list(package_dir.glob("*.zip"))
    assert not list(package_dir.glob("*.tmp"))


def test_collect_blocks_byte_identical_atomic_os_replace(synthetic_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert_collect_blocks_after_atomic_replace(synthetic_env, tmp_path, monkeypatch, b"report")


def test_collect_blocks_same_size_different_byte_atomic_os_replace(synthetic_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert_collect_blocks_after_atomic_replace(synthetic_env, tmp_path, monkeypatch, b"tamper")


def test_collect_does_not_use_zip_path_reopen_write(synthetic_env, monkeypatch: pytest.MonkeyPatch):
    def forbidden_write(*args, **kwargs):
        raise AssertionError("ZipFile.write must not be used for source packaging")

    monkeypatch.setattr(recovery.zipfile.ZipFile, "write", forbidden_write)
    target = recovery.collect(type("Args", (), {"expected_implementation_commit": IMPL})())
    assert target.exists()


def test_packaged_artifact_size_sha_and_run_provenance_tampering(synthetic_env, tmp_path: Path):
    entries = {"recovery_manifest.json": recovery.deterministic_json_bytes(recovery.create_manifest(IMPL, recovery.validate_source_artifacts(synthetic_env[0])))}
    for item in recovery.SOURCE_ARTIFACTS:
        entries[f"files/{recovery.RUN_REL_DIR}/{item['name']}"] = (synthetic_env[0] / item["name"]).read_bytes()
    entries[f"files/{recovery.RUN_REL_DIR}/training_report.json"] = b"wrong"
    zip_path = tmp_path / "bad.zip"
    make_zip(zip_path, entries)
    assert recovery.main(["audit-import", "--zip", str(zip_path), "--expected-implementation-commit", IMPL, "--audit-output", str(tmp_path / "audit.json")]) == 64

    item = next(item for item in recovery.SOURCE_ARTIFACTS if item["name"] == "training_report.json")
    entries[f"files/{recovery.RUN_REL_DIR}/training_report.json"] = b"x" * item["size"]
    make_zip(zip_path, entries)
    assert recovery.main(["audit-import", "--zip", str(zip_path), "--expected-implementation-commit", IMPL, "--audit-output", str(tmp_path / "audit2.json")]) == 64

    entries[f"files/{recovery.RUN_REL_DIR}/training_report.json"] = (synthetic_env[0] / "training_report.json").read_bytes()
    prov = valid_provenance()
    prov["source_provenance"]["git_commit"] = "0" * 40
    data = json.dumps(prov, sort_keys=True).encode("utf-8")
    item = next(item for item in recovery.SOURCE_ARTIFACTS if item["name"] == "run_provenance.json")
    item["size"] = len(data)
    item["sha256"] = sha(data)
    recovery.ARTIFACT_BY_NAME["run_provenance.json"] = item
    entries[f"files/{recovery.RUN_REL_DIR}/run_provenance.json"] = data
    manifest = recovery.create_manifest(IMPL, [{"path": f"{recovery.RUN_REL_DIR}/{i['name']}", "size": i["size"], "sha256": i["sha256"]} for i in recovery.SOURCE_ARTIFACTS])
    entries["recovery_manifest.json"] = recovery.deterministic_json_bytes(manifest)
    make_zip(zip_path, entries)
    assert recovery.main(["audit-import", "--zip", str(zip_path), "--expected-implementation-commit", IMPL, "--audit-output", str(tmp_path / "audit3.json")]) == 64


def test_manifest_schema_duplicate_type_and_artifact_table_tampering(synthetic_env):
    manifest = recovery.create_manifest(IMPL, recovery.validate_source_artifacts(synthetic_env[0]))
    for key, value in [("schema", "bad"), ("seed", "180")]:
        bad = dict(manifest)
        bad[key] = value
        assert_blocked(recovery.validate_manifest, bad, IMPL)
    table_bad = copy.deepcopy(manifest)
    table_bad["artifact_files"][0]["sha256"] = "bad"
    assert_blocked(recovery.validate_manifest, table_bad, IMPL)
    assert_blocked(recovery.loads_json_strict, '{"schema":"x","schema":"y"}')


def test_audit_zip_missing_symlink_output_inside_repo_collision_and_failure_no_output(synthetic_env, tmp_path: Path):
    missing = tmp_path / "missing.zip"
    assert recovery.main(["audit-import", "--zip", str(missing), "--expected-implementation-commit", IMPL, "--audit-output", str(tmp_path / "audit.json")]) == 64
    real = tmp_path / "real.zip"
    real.write_bytes(b"bad")
    link = tmp_path / "link.zip"
    create_filesystem_symlink_or_skip(real, link)
    assert recovery.main(["audit-import", "--zip", str(link), "--expected-implementation-commit", IMPL, "--audit-output", str(tmp_path / "audit.json")]) == 64
    repo_audit = synthetic_env[2] / "audit.json"
    zip_path = tmp_path / "valid.zip"
    valid_zip_from_env(synthetic_env, zip_path)
    assert recovery.main(["audit-import", "--zip", str(zip_path), "--expected-implementation-commit", IMPL, "--audit-output", str(repo_audit)]) == 64
    audit = tmp_path / "audit.json"
    audit.write_text("exists", encoding="utf-8")
    assert recovery.main(["audit-import", "--zip", str(zip_path), "--expected-implementation-commit", IMPL, "--audit-output", str(audit)]) == 64
    fail_out = tmp_path / "fail.json"
    assert recovery.main(["audit-import", "--zip", str(missing), "--expected-implementation-commit", IMPL, "--audit-output", str(fail_out)]) == 64
    assert not fail_out.exists()


def test_no_forbidden_or_third_party_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "torch",
        "transformers",
        "mamba_ssm",
        "scripts.train_controlled_v6b_minimal",
        "train_controlled_v6b_minimal",
    }
    assert not (imported & forbidden)
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "os",
        "re",
        "shlex",
        "stat",
        "subprocess",
        "sys",
        "tempfile",
        "zipfile",
        "datetime",
        "pathlib",
        "typing",
    }
    assert imported <= allowed
