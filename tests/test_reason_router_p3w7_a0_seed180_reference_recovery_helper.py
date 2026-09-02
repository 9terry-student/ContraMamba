from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reason_router_p3w7_a0_seed180_reference_recovery_helper.py"
)
spec = importlib.util.spec_from_file_location("seed180_reference_helper", MODULE_PATH)
helper = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helper)

HEAD = "f" * 40


@pytest.fixture()
def work_tmp(request):
    base = Path.cwd() / ".rr_helper_test_tmp"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)[:80]
    path = base / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(json_bytes(row) + b"\n" for row in rows)


def dataset_rows() -> list[dict[str, object]]:
    labels = ["SUPPORT", "NOT_ENTITLED", "REFUTE", "SUPPORT", "NOT_ENTITLED"]
    return [
        {"id": f"r{index}", "pair_id": f"p{index}", "final_label": labels[index % len(labels)]}
        for index in range(10)
    ]


def dev_prediction_rows() -> list[dict[str, object]]:
    dev = helper.split_dev_records(dataset_rows())
    preds = []
    for row in dev:
        pred = "SUPPORT" if row["final_label"] == "SUPPORT" else "REFUTE"
        preds.append(
            {
                "row_id": row["id"],
                "pair_id": row["pair_id"],
                "gold_label": row["final_label"],
                "pred_label": pred,
            }
        )
    return preds


def report_bytes(best_epoch: int = 7) -> bytes:
    return json_bytes({"runs": {"single": {"best_epoch": best_epoch}}})


def run_provenance_bytes() -> bytes:
    return json_bytes(
        {
            "source_provenance": {
                "git_commit": helper.SOURCE_EXECUTION_COMMIT,
                "git_is_dirty": False,
            },
            "parsed_args": {"seed": 180, "split_seed": 174, "reason_router_arm": "A0"},
            "finalization": {
                "selected_checkpoint": {
                    "sha256": helper.ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"]
                }
            },
        }
    )


@pytest.fixture()
def synthetic(work_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    root = work_tmp / "repo"
    dest = root / helper.DESTINATION_REL
    dataset = root / helper.DATASET_REL
    sidecar = root / helper.SIDECAR_REL
    dataset.parent.mkdir(parents=True)
    sidecar.parent.mkdir(parents=True)
    dataset.write_bytes(jsonl_bytes(dataset_rows()))
    sidecar.write_bytes(jsonl_bytes([{"row_id": "r1", "value": 1}]))
    artifacts = [
        {"name": "training_report.json", "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json", "data": report_bytes()},
        {"name": "clean_dev_predictions.json", "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json", "data": json_bytes({"predictions": dev_prediction_rows()})},
        {"name": "training_report_predictions.jsonl", "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl", "data": jsonl_bytes(dev_prediction_rows())},
        {"name": "selected_checkpoint.pt", "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/selected_checkpoint.pt", "data": b"raw-checkpoint-bytes"},
        {"name": "run_provenance.json", "zip": "files/reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json", "data": b""},
    ]
    patched = []
    for artifact in artifacts:
        data = artifact["data"]
        patched.append(
            {
                "name": artifact["name"],
                "zip": artifact["zip"],
                "size": len(data),
                "sha256": sha(data),
                "data": data,
            }
        )
    monkeypatch.setattr(helper, "SOURCE_ARTIFACTS", patched)
    monkeypatch.setattr(helper, "ARTIFACT_BY_NAME", {item["name"]: item for item in patched})
    run_provenance_data = run_provenance_bytes()
    run_provenance_item = next(item for item in patched if item["name"] == "run_provenance.json")
    run_provenance_item["data"] = run_provenance_data
    run_provenance_item["size"] = len(run_provenance_data)
    run_provenance_item["sha256"] = sha(run_provenance_data)
    monkeypatch.setattr(helper, "SOURCE_ARTIFACTS", patched)
    monkeypatch.setattr(helper, "ARTIFACT_BY_NAME", {item["name"]: item for item in patched})
    monkeypatch.setattr(helper, "EXPECTED_ZIP_ENTRIES", ("recovery_manifest.json", *(item["zip"] for item in patched)))
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", "zip-sha-placeholder")
    manifest = valid_manifest()
    manifest_data = canonical_manifest_bytes(manifest)
    monkeypatch.setattr(helper, "EXPECTED_RECOVERY_MANIFEST_SHA256", sha(manifest_data))
    zip_path = work_tmp / "synthetic.zip"
    write_zip(zip_path, {"recovery_manifest.json": manifest_data, **{item["zip"]: item["data"] for item in patched}})
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", sha(zip_path.read_bytes()))
    monkeypatch.setattr(helper, "EXPECTED_DATASET_SHA256", sha(dataset.read_bytes()))
    monkeypatch.setattr(helper, "EXPECTED_SIDECAR_SEMANTIC_SHA256", helper.semantic_sidecar_sha256(sidecar))
    monkeypatch.setattr(helper, "repo_root", lambda: root)
    monkeypatch.setattr(helper, "run_git", lambda root_arg, args: git_stub(args))
    return root, dest, dataset, sidecar, zip_path


def git_stub(args: list[str]) -> str:
    if args[:2] == ["cat-file", "-t"]:
        if args[2] == "e" * 40:
            return "blob"
        if args[2] == "0" * 40:
            helper.block("git command failed: nonexistent")
        return "commit"
    if args == ["rev-parse", "HEAD"]:
        return HEAD
    if args == ["rev-parse", "--show-toplevel"]:
        return "unused"
    return ""


def canonical_manifest_bytes(payload: dict[str, object]) -> bytes:
    return helper.canonical_json_bytes(payload)


def valid_manifest() -> dict[str, object]:
    return {
        "schema": helper.RECOVERY_SCHEMA,
        "seed": helper.SEED,
        "attempt_disposition": "CONSUMED",
        "execution_status": "completed",
        "original_authorized_wrapper_sha256": helper.ORIGINAL_AUTHORIZED_WRAPPER_SHA256,
        "original_execution_commit": helper.SOURCE_EXECUTION_COMMIT,
        "recovery_authority_commit": helper.RECOVERY_AUTHORITY_COMMIT,
        "source_run_provenance_sha256": helper.ARTIFACT_BY_NAME["run_provenance.json"]["sha256"],
        "source_trainer_git_commit": helper.SOURCE_EXECUTION_COMMIT,
        "standard_cm_wrapper_provenance": "missing/incomplete",
        "scientific_conclusion": "NOT_ESTABLISHED",
        "artifact_files": [
            {
                "path": item["zip"].removeprefix("files/"),
                "size": item["size"],
                "sha256": item["sha256"],
            }
            for item in helper.SOURCE_ARTIFACTS
        ],
    }


def write_zip(path: Path, entries: dict[str, bytes], attrs: dict[str, int] | None = None, flags: dict[str, int] | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (attrs or {}).get(name, 0)
            info.flag_bits = (flags or {}).get(name, 0)
            zf.writestr(info, data)


def assert_blocked(fn, *args, **kwargs):
    with pytest.raises(helper.Blocker, match="REFERENCE_RECOVERY_BLOCKED"):
        fn(*args, **kwargs)


def test_happy_path_materializes_and_audit_is_pass(synthetic):
    _root, dest, _dataset, _sidecar, zip_path = synthetic
    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 0
    audit = helper.strict_json_file(dest / "A0_REFERENCE_AUDIT.json")
    assert audit["status"] == "PASS"
    assert audit["standard_cm_wrapper_provenance"] == "INCOMPLETE"
    assert audit["recovery_reference_status"] == "RECOVERY_REFERENCE_AUDIT_PASS"
    assert "source_execution_commit" in audit


@pytest.mark.parametrize("value", ["BAD", "A" * 40, "0" * 39, "g" * 40])
def test_authority_commit_malformed_or_nonlowercase_rejected(synthetic, value: str):
    root, *_ = synthetic
    assert_blocked(helper.require_runtime_authority, root, value)


def test_authority_commit_nonexistent_noncommit_and_head_mismatch_rejected(synthetic):
    root, *_ = synthetic
    assert_blocked(helper.require_runtime_authority, root, "0" * 40)
    assert_blocked(helper.require_runtime_authority, root, "e" * 40)
    assert_blocked(helper.require_runtime_authority, root, "1" * 40)
    assert helper.require_runtime_authority(root, HEAD) == HEAD


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "files\\x", ".", "files/./x"])
def test_zip_bad_paths_rejected(synthetic, work_tmp: Path, name: str):
    entries = {n: b"x" for n in helper.EXPECTED_ZIP_ENTRIES}
    entries.pop(helper.EXPECTED_ZIP_ENTRIES[-1])
    entries[name] = b"x"
    path = work_tmp / "bad.zip"
    write_zip(path, entries)
    with zipfile.ZipFile(path) as zf:
        assert_blocked(helper.validate_zip_structure, zf)


def test_zip_nul_path_rejected_by_production_member_validator():
    # ZipInfo truncates names at NUL construction time, so exercise the same
    # member-path validator used by production ZIP inspection directly.
    assert_blocked(helper.validate_zip_member_path, "files/reports/bad\x00name", False, 0, 0)


def test_zip_missing_unexpected_duplicate_directory_encrypted_special_rejected(synthetic, work_tmp: Path):
    base = {n: b"x" for n in helper.EXPECTED_ZIP_ENTRIES}
    for label, entries in {
        "missing": dict(list(base.items())[:-1]),
        "unexpected": {**base, "extra": b"x"},
        "directory": {**base, "dir/": b""},
    }.items():
        path = work_tmp / f"{label}.zip"
        write_zip(path, entries)
        with zipfile.ZipFile(path) as zf:
            assert_blocked(helper.validate_zip_structure, zf)
    dup = work_tmp / "dup.zip"
    with zipfile.ZipFile(dup, "w") as zf:
        for name in helper.EXPECTED_ZIP_ENTRIES:
            zf.writestr(name, b"x")
        zf.writestr(helper.EXPECTED_ZIP_ENTRIES[-1], b"y")
    with zipfile.ZipFile(dup) as zf:
        assert_blocked(helper.validate_zip_structure, zf)
    for mode in [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644]:
        path = work_tmp / f"mode-{mode}.zip"
        write_zip(path, base, attrs={helper.EXPECTED_ZIP_ENTRIES[-1]: mode << 16})
        with zipfile.ZipFile(path) as zf:
            assert_blocked(helper.validate_zip_structure, zf)
    path = work_tmp / "encrypted.zip"
    write_zip(path, base)
    with zipfile.ZipFile(path) as zf:
        zf.infolist()[-1].flag_bits = 1
        assert_blocked(helper.validate_zip_structure, zf)


def test_duplicate_json_keys_rejected():
    assert helper.strict_json_loads('{"a": 1}') == {"a": 1}
    assert_blocked(helper.strict_json_loads, '{"a": 1, "a": 2}')


def test_packaged_recovery_manifest_duplicate_key_rejected_before_public_write(
    synthetic, work_tmp: Path, monkeypatch: pytest.MonkeyPatch
):
    root, dest, _dataset, _sidecar, _zip_path = synthetic
    duplicate_manifest = (
        b'{"schema":"'
        + helper.RECOVERY_SCHEMA.encode("utf-8")
        + b'","schema":"'
        + helper.RECOVERY_SCHEMA.encode("utf-8")
        + b'"}'
    )
    entries = {"recovery_manifest.json": duplicate_manifest}
    for item in helper.SOURCE_ARTIFACTS:
        entries[item["zip"]] = item["data"]
    bad_zip = work_tmp / "duplicate-manifest.zip"
    write_zip(bad_zip, entries)
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", sha(bad_zip.read_bytes()))
    monkeypatch.setattr(helper, "EXPECTED_RECOVERY_MANIFEST_SHA256", sha(duplicate_manifest))

    assert_blocked(helper.validate_zip_source, bad_zip)
    assert helper.main(["materialize-reference", "--zip", str(bad_zip), "--expected-recovery-execution-authority-commit", HEAD]) == 64
    assert not dest.exists()
    assert not list(root.rglob(".seed180_reference_recovery_staging.*"))


def test_packaged_run_provenance_duplicate_key_rejected_before_public_write(
    synthetic, work_tmp: Path, monkeypatch: pytest.MonkeyPatch
):
    root, dest, _dataset, _sidecar, _zip_path = synthetic
    duplicate_run_provenance = (
        b'{"source_provenance":{"git_commit":"'
        + helper.SOURCE_EXECUTION_COMMIT.encode("utf-8")
        + b'","git_commit":"'
        + helper.SOURCE_EXECUTION_COMMIT.encode("utf-8")
        + b'","git_is_dirty":false},'
        b'"parsed_args":{"seed":180,"split_seed":174,"reason_router_arm":"A0"},'
        b'"finalization":{"selected_checkpoint":{"sha256":"'
        + helper.ARTIFACT_BY_NAME["selected_checkpoint.pt"]["sha256"].encode("utf-8")
        + b'"}}}'
    )
    patched = copy.deepcopy(helper.SOURCE_ARTIFACTS)
    run_item = next(item for item in patched if item["name"] == "run_provenance.json")
    run_item["data"] = duplicate_run_provenance
    run_item["size"] = len(duplicate_run_provenance)
    run_item["sha256"] = sha(duplicate_run_provenance)
    monkeypatch.setattr(helper, "SOURCE_ARTIFACTS", patched)
    monkeypatch.setattr(helper, "ARTIFACT_BY_NAME", {item["name"]: item for item in patched})
    monkeypatch.setattr(helper, "EXPECTED_ZIP_ENTRIES", ("recovery_manifest.json", *(item["zip"] for item in patched)))
    manifest_data = canonical_manifest_bytes(valid_manifest())
    entries = {"recovery_manifest.json": manifest_data}
    for item in patched:
        entries[item["zip"]] = item["data"]
    bad_zip = work_tmp / "duplicate-run-provenance.zip"
    write_zip(bad_zip, entries)
    monkeypatch.setattr(helper, "EXPECTED_RECOVERY_MANIFEST_SHA256", sha(manifest_data))
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", sha(bad_zip.read_bytes()))

    assert_blocked(helper.validate_zip_source, bad_zip)
    assert helper.main(["materialize-reference", "--zip", str(bad_zip), "--expected-recovery-execution-authority-commit", HEAD]) == 64
    assert not dest.exists()
    assert not list(root.rglob(".seed180_reference_recovery_staging.*"))


def test_manifest_and_run_provenance_binding_mismatches_rejected(synthetic):
    manifest = valid_manifest()
    manifest["schema"] = "bad"
    assert_blocked(helper.validate_manifest, manifest)
    prov = helper.strict_json_loads(run_provenance_bytes().decode())
    prov["source_provenance"]["git_commit"] = "0" * 40
    assert_blocked(helper.validate_run_provenance, prov)


def test_zip_sha_manifest_sha_and_artifact_hash_rejected(synthetic, work_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    *_rest, zip_path = synthetic
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", "0" * 64)
    assert_blocked(helper.validate_zip_source, zip_path)
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", sha(zip_path.read_bytes()))
    monkeypatch.setattr(helper, "EXPECTED_RECOVERY_MANIFEST_SHA256", "0" * 64)
    assert_blocked(helper.validate_zip_source, zip_path)
    entries = {"recovery_manifest.json": canonical_manifest_bytes(valid_manifest())}
    for item in helper.SOURCE_ARTIFACTS:
        entries[item["zip"]] = b"x" * item["size"]
    bad = work_tmp / "bad-artifact.zip"
    write_zip(bad, entries)
    monkeypatch.setattr(helper, "EXPECTED_ZIP_SHA256", sha(bad.read_bytes()))
    monkeypatch.setattr(helper, "EXPECTED_RECOVERY_MANIFEST_SHA256", sha(entries["recovery_manifest.json"]))
    assert_blocked(helper.validate_zip_source, bad)


def test_dataset_and_sidecar_identity_rejected(synthetic, monkeypatch: pytest.MonkeyPatch):
    root, dest, dataset, sidecar, _zip = synthetic
    monkeypatch.setattr(helper, "EXPECTED_DATASET_SHA256", "0" * 64)
    assert_blocked(helper.prevalidate_environment, root, dest, dataset, sidecar)
    monkeypatch.setattr(helper, "EXPECTED_DATASET_SHA256", sha(dataset.read_bytes()))
    monkeypatch.setattr(helper, "EXPECTED_SIDECAR_SEMANTIC_SHA256", "0" * 64)
    assert_blocked(helper.prevalidate_environment, root, dest, dataset, sidecar)


def test_nonidentical_collision_audit_preexistence_and_identical_acceptance(synthetic):
    root, dest, dataset, sidecar, _zip = synthetic
    dest.mkdir(parents=True)
    item = helper.ARTIFACT_BY_NAME["training_report.json"]
    (dest / "training_report.json").write_bytes(b"wrong")
    assert_blocked(helper.prevalidate_environment, root, dest, dataset, sidecar)
    (dest / "training_report.json").write_bytes(item["data"])
    helper.prevalidate_environment(root, dest, dataset, sidecar)
    (dest / "A0_REFERENCE_AUDIT.json").write_text("exists", encoding="utf-8")
    assert_blocked(helper.prevalidate_environment, root, dest, dataset, sidecar)


def mutate_prediction_file(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(jsonl_bytes(rows))


def test_dev_identity_mismatches_and_invalid_prediction_rejected(synthetic, work_tmp: Path):
    _root, _dest, dataset, _sidecar, _zip = synthetic
    pred = work_tmp / "pred.jsonl"
    rows = dev_prediction_rows()
    mutate_prediction_file(pred, rows[:-1])
    assert_blocked(helper.validate_dev_identity, dataset, pred)
    bad = copy.deepcopy(rows)
    bad[0]["pair_id"] = "wrong"
    mutate_prediction_file(pred, bad)
    assert_blocked(helper.validate_dev_identity, dataset, pred)
    bad = copy.deepcopy(rows)
    bad[0]["gold_label"] = "REFUTE" if bad[0]["gold_label"] != "REFUTE" else "SUPPORT"
    mutate_prediction_file(pred, bad)
    assert_blocked(helper.validate_dev_identity, dataset, pred)
    bad = copy.deepcopy(rows)
    bad[0]["pred_label"] = "BAD"
    mutate_prediction_file(pred, bad)
    assert_blocked(helper.validate_dev_identity, dataset, pred)
    bad = copy.deepcopy(rows)
    bad[1]["row_id"] = bad[0]["row_id"]
    mutate_prediction_file(pred, bad)
    assert_blocked(helper.validate_dev_identity, dataset, pred)


def test_duplicate_source_row_and_persisted_audit_checks_rejected(synthetic, work_tmp: Path):
    _root, _dest, dataset, _sidecar, _zip = synthetic
    rows = dataset_rows()
    rows[1]["id"] = rows[0]["id"]
    dataset.write_bytes(jsonl_bytes(rows))
    pred = work_tmp / "pred.jsonl"
    mutate_prediction_file(pred, dev_prediction_rows())
    assert_blocked(helper.validate_dev_identity, dataset, pred)
    expected = {"status": "PASS", "standard_cm_wrapper_provenance": "INCOMPLETE"}
    assert_blocked(helper.validate_persisted_audit, {"status": "FAIL"}, expected)
    assert_blocked(
        helper.validate_persisted_audit,
        {"status": "PASS", "standard_cm_wrapper_provenance": "PASS"},
        expected,
    )


def test_persisted_audit_reread_normal_and_recovery_corruption_rejected(synthetic):
    root, dest, dataset, sidecar, zip_path = synthetic
    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 0
    audit_path = dest / "A0_REFERENCE_AUDIT.json"
    expected = helper.strict_json_file(audit_path)

    normal_corrupt = copy.deepcopy(expected)
    normal_corrupt["prediction_sha256"] = "0" * 64
    audit_path.write_bytes(helper.canonical_json_bytes(normal_corrupt))
    assert_blocked(helper.validate_persisted_audit, helper.strict_json_file(audit_path), expected)

    recovery_corrupt = copy.deepcopy(expected)
    recovery_corrupt["retained_zip_sha256"] = "1" * 64
    audit_path.write_bytes(helper.canonical_json_bytes(recovery_corrupt))
    assert_blocked(helper.validate_persisted_audit, helper.strict_json_file(audit_path), expected)

    audit_path.write_bytes(helper.canonical_json_bytes(expected))
    helper.validate_published_reference(root, dest, dataset, sidecar, expected)


def test_persisted_audit_reread_prediction_artifact_mismatch_rejected(synthetic):
    root, dest, dataset, sidecar, zip_path = synthetic
    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 0
    expected = helper.strict_json_file(dest / "A0_REFERENCE_AUDIT.json")
    (dest / "training_report_predictions.jsonl").write_bytes(b"tampered\n")
    assert_blocked(helper.validate_published_reference, root, dest, dataset, sidecar, expected)


def test_checkpoint_report_reread_and_audit_nonpass_not_published(synthetic, monkeypatch: pytest.MonkeyPatch):
    root, dest, _dataset, _sidecar, zip_path = synthetic
    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 0
    (dest / "selected_checkpoint.pt").write_bytes(b"tamper")
    assert_blocked(helper.build_audit, root, dest, zip_path, root / helper.DATASET_REL, root / helper.SIDECAR_REL)
    (dest / "selected_checkpoint.pt").write_bytes(helper.ARTIFACT_BY_NAME["selected_checkpoint.pt"]["data"])
    (dest / "training_report.json").write_bytes(json_bytes({}))
    assert_blocked(helper.build_audit, root, dest, zip_path, root / helper.DATASET_REL, root / helper.SIDECAR_REL)


def test_prevalidation_before_public_write_and_zip_unchanged(synthetic, monkeypatch: pytest.MonkeyPatch):
    _root, dest, _dataset, _sidecar, zip_path = synthetic
    before = zip_path.read_bytes()
    monkeypatch.setattr(helper, "EXPECTED_DATASET_SHA256", "0" * 64)
    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 64
    assert not dest.exists()
    assert zip_path.read_bytes() == before


def test_late_prevalidation_failure_leaves_absent_destination_and_staging_absent(
    synthetic, monkeypatch: pytest.MonkeyPatch
):
    root, dest, _dataset, _sidecar, zip_path = synthetic
    assert not dest.exists()
    monkeypatch.setattr(helper, "EXPECTED_SIDECAR_SEMANTIC_SHA256", "0" * 64)

    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 64
    assert not dest.exists()
    assert not list(root.rglob(".seed180_reference_recovery_staging.*"))


def test_no_overwrite_delete_or_rename_over_existing_file(synthetic, monkeypatch: pytest.MonkeyPatch):
    root, dest, _dataset, _sidecar, zip_path = synthetic
    dest.mkdir(parents=True)
    sentinel = dest / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_link = os.link

    def fail_after_first(source, target):
        if Path(target).name == "clean_dev_predictions.json":
            helper.block("synthetic publish failure")
        return original_link(source, target)

    monkeypatch.setattr(helper.os, "link", fail_after_first)
    assert helper.main(["materialize-reference", "--zip", str(zip_path), "--expected-recovery-execution-authority-commit", HEAD]) == 64
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (dest / "training_report.json").exists()
    assert not (dest / "clean_dev_predictions.json").exists()


def test_static_import_and_no_checkpoint_deserialization_symbols():
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
    assert not imported & forbidden
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "torch.load" not in text
    assert "pickle.load" not in text
    assert "--expected-recovery-execution-authority-commit" in text
    assert "--expected-authority-freeze-commit" not in text
