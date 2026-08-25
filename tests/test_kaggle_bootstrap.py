from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import kaggle_bootstrap as kb


def write_wheelhouse(root: Path, *, marker=kb.RESTORE_MARKER, boundary=kb.EVIDENCE_BOUNDARY, missing_wheels=None):
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    requirements = wheelhouse / kb.RESTORE_REQUIREMENTS
    requirements.write_text("\n".join(sorted(kb.ALLOWED_REQUIREMENT_LINES)) + "\n", encoding="utf-8")
    restore_packages = []
    first_wheel = None
    for line in sorted(kb.ALLOWED_REQUIREMENT_LINES):
        name, version = line.split("==", 1)
        filename = f"{name.replace('_', '-')}-{version}-py3-none-any.whl"
        wheel = wheelhouse / filename
        wheel.write_bytes(f"NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST_WHEEL:{name}".encode("utf-8"))
        if first_wheel is None:
            first_wheel = wheel
        restore_packages.append(
            {
                "name": name,
                "version": version,
                "wheels": [
                    {
                        "filename": wheel.name,
                        "size": wheel.stat().st_size,
                        "sha256": kb.file_sha256(wheel),
                    }
                ],
            }
        )
    manifest = {
        "marker": marker,
        "evidence_boundary": boundary,
        "environment_contract": {"kind": "NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST"},
        "restore_packages": restore_packages,
        "inherit_from_kaggle_base": ["python", "torch", "cuda"],
        "explicitly_not_restored": {"torch": True, "cuda": True},
        "missing_wheels": [] if missing_wheels is None else missing_wheels,
    }
    (wheelhouse / kb.RESTORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return wheelhouse, requirements, first_wheel, manifest


def completed(args, code, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def verifier_stdout(status):
    return f"KAGGLE_ENVIRONMENT\nFINAL_STATUS: {status}\nNOT_SCIENTIFIC_EVIDENCE\n"


def make_runner(sequence, calls):
    outcomes = iter(sequence)

    def runner(args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        outcome = next(outcomes)
        if callable(outcome):
            return outcome(args, kwargs)
        return outcome

    return runner


def statuses(report):
    return {check.name: check.status for check in [*report.checks, *report.wheel_integrity]}


def test_already_compatible_skips_restore_and_passes(tmp_path):
    calls = []
    runner = make_runner([completed([], 0, verifier_stdout("PASS"))], calls)

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "PASS"
    assert report.restore_result == "RESTORE_SKIPPED_ALREADY_COMPATIBLE"
    assert report.restore_attempted is False
    assert len(calls) == 1


def test_install_required_uses_local_wheelhouse_only(tmp_path):
    write_wheelhouse(tmp_path)
    calls = []
    runner = make_runner(
        [
            completed([], 2, verifier_stdout("INSTALL_REQUIRED")),
            completed([], 0, "pip ok"),
            completed([], 0, verifier_stdout("PASS")),
        ],
        calls,
    )

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "PASS"
    assert report.restore_result == "RESTORE_SUCCEEDED"
    pip_args = calls[1]["args"]
    assert pip_args[:4] == [kb.sys.executable, "-m", "pip", "install"]
    assert "--no-index" in pip_args
    assert f"--find-links={tmp_path / 'wheelhouse'}" in pip_args
    assert "--no-deps" in pip_args
    assert "-r" in pip_args
    assert "torch" not in pip_args
    assert "cuda" not in " ".join(pip_args).lower()


def test_missing_requirements_fails_closed(tmp_path):
    wheelhouse, requirements, _wheel, _manifest = write_wheelhouse(tmp_path)
    requirements.unlink()
    calls = []
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], calls)

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["restore_requirements"] == "FAIL"
    assert report.restore_result == "RESTORE_INPUT_VALIDATION_FAILED"
    assert len(calls) == 1


def test_missing_restore_manifest_fails_closed(tmp_path):
    wheelhouse, _requirements, _wheel, _manifest = write_wheelhouse(tmp_path)
    (wheelhouse / kb.RESTORE_MANIFEST).unlink()
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["restore_manifest"] == "FAIL"


def test_unexpected_restore_requirement_fails_before_pip(tmp_path):
    wheelhouse, requirements, _wheel, _manifest = write_wheelhouse(tmp_path)
    requirements.write_text("torch==999.0.0\n", encoding="utf-8")
    calls = []
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], calls)

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["restore_requirements_allowlist"] == "FAIL"
    assert len(calls) == 1


def test_malformed_restore_manifest_fails_closed(tmp_path):
    wheelhouse, _requirements, _wheel, _manifest = write_wheelhouse(tmp_path)
    (wheelhouse / kb.RESTORE_MANIFEST).write_text("{", encoding="utf-8")
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["restore_manifest"] == "FAIL"


def test_missing_restore_packages_fails_closed(tmp_path):
    wheelhouse, _requirements, _wheel, manifest = write_wheelhouse(tmp_path)
    manifest.pop("restore_packages")
    (wheelhouse / kb.RESTORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["restore_manifest_restore_packages"] == "FAIL"


def test_package_missing_wheels_fails_closed(tmp_path):
    wheelhouse, _requirements, _wheel, manifest = write_wheelhouse(tmp_path)
    manifest["restore_packages"][0]["wheels"] = []
    (wheelhouse / kb.RESTORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert any(check.name.endswith("_wheels") and check.status == "FAIL" for check in report.wheel_integrity)


def test_package_requirements_manifest_mismatch_fails_closed(tmp_path):
    wheelhouse, requirements, _wheel, _manifest = write_wheelhouse(tmp_path)
    requirements.write_text("transformers==4.45.0\n", encoding="utf-8")
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["restore_requirements_manifest_match"] == "FAIL"


def test_duplicate_package_definition_fails_closed(tmp_path):
    wheelhouse, _requirements, _wheel, manifest = write_wheelhouse(tmp_path)
    manifest["restore_packages"].append(dict(manifest["restore_packages"][0]))
    (wheelhouse / kb.RESTORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert any(check.name.endswith("_duplicate") and check.status == "FAIL" for check in report.wheel_integrity)


@pytest.mark.parametrize(
    ("marker", "boundary", "missing_wheels", "failing_check"),
    [
        ("WRONG", kb.EVIDENCE_BOUNDARY, [], "restore_manifest_marker"),
        (kb.RESTORE_MARKER, "SCIENTIFIC_EVIDENCE", [], "restore_manifest_boundary"),
        (kb.RESTORE_MARKER, kb.EVIDENCE_BOUNDARY, ["x.whl"], "restore_manifest_missing_wheels"),
    ],
)
def test_manifest_marker_boundary_and_missing_wheels_fail(tmp_path, marker, boundary, missing_wheels, failing_check):
    write_wheelhouse(tmp_path, marker=marker, boundary=boundary, missing_wheels=missing_wheels)
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)[failing_check] == "FAIL"


def test_missing_listed_wheel_fails_closed(tmp_path):
    _wheelhouse, _requirements, wheel, _manifest = write_wheelhouse(tmp_path)
    wheel.unlink()
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert "missing wheel" in [check.detail for check in report.wheel_integrity]


def test_wheel_sha_and_size_mismatch_fail_closed(tmp_path):
    wheelhouse, _requirements, _wheel, manifest = write_wheelhouse(tmp_path)
    manifest["restore_packages"][0]["wheels"][0]["size"] += 1
    manifest["restore_packages"][0]["wheels"][0]["sha256"] = "0" * 64
    (wheelhouse / kb.RESTORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    runner = make_runner([completed([], 2, verifier_stdout("INSTALL_REQUIRED"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert any("size expected" in check.detail for check in report.wheel_integrity)
    assert any(check.name.endswith("_sha256") and check.status == "FAIL" for check in report.wheel_integrity)


def test_expected_head_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "run_git", lambda args, cwd: "a" * 40 if args == ("rev-parse", "HEAD") else "")
    calls = []
    runner = make_runner([completed([], 0, verifier_stdout("PASS"))], calls)

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head="b" * 40,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["expected_head"] == "FAIL"
    assert calls == []


def test_local_restore_nonzero_fails(tmp_path):
    write_wheelhouse(tmp_path)
    runner = make_runner(
        [
            completed([], 2, verifier_stdout("INSTALL_REQUIRED")),
            completed([], 1, "", "pip failed"),
        ],
        [],
    )

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert report.restore_result == "RESTORE_FAILED"


def test_environment_verifier_nonzero_fail_without_restore(tmp_path):
    runner = make_runner([completed([], 1, verifier_stdout("FAIL"))], [])

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["environment_verifier_initial"] == "FAIL"


def test_post_restore_verifier_nonzero_fails(tmp_path):
    write_wheelhouse(tmp_path)
    runner = make_runner(
        [
            completed([], 2, verifier_stdout("INSTALL_REQUIRED")),
            completed([], 0, "pip ok"),
            completed([], 1, verifier_stdout("FAIL")),
        ],
        [],
    )

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["environment_verifier_post_restore"] == "FAIL"


def test_offline_model_env_propagates_to_subprocesses(tmp_path):
    write_wheelhouse(tmp_path)
    calls = []
    runner = make_runner(
        [
            completed([], 2, verifier_stdout("INSTALL_REQUIRED")),
            completed([], 0, "pip ok"),
            completed([], 0, verifier_stdout("PASS")),
        ],
        calls,
    )

    report = kb.build_report(
        cache_root=tmp_path,
        expected_head=None,
        cuda_smoke=True,
        offline_model=True,
        manifest_path=None,
        repo_root=tmp_path,
        runner=runner,
    )

    assert report.final_status == "PASS"
    for call in calls:
        env = call["kwargs"]["env"]
        assert env["HF_HOME"] == str((tmp_path / "huggingface").resolve())
        assert env["HF_HUB_CACHE"] == str((tmp_path / "huggingface" / "hub").resolve())
        assert env["PIP_CACHE_DIR"] == str((tmp_path / "pip").resolve())
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert "--cuda-smoke" in calls[0]["args"]
    assert "--cuda-smoke" in calls[2]["args"]


def test_bootstrap_manifest_schema_and_marker(tmp_path):
    manifest_path = tmp_path / "bootstrap_manifest.json"
    runner = make_runner([completed([], 0, verifier_stdout("PASS"))], [])

    report = kb.build_report(
        cache_root=tmp_path / "cache",
        expected_head=None,
        cuda_smoke=False,
        offline_model=False,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        runner=runner,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report.final_status == "PASS"
    assert payload["marker"] == kb.BOOTSTRAP_MARKER
    assert payload["evidence_boundary"] == kb.EVIDENCE_BOUNDARY
    assert payload["metadata"]["evidence_boundary"] == kb.EVIDENCE_BOUNDARY
    assert payload["cache"]["HF_HOME"].endswith("huggingface")
    assert payload["restore_attempted"] is False
    assert payload["environment_verifier"]["initial"]["returncode"] == 0
    assert payload["metadata"]["cuda_smoke_requested"] is False
    assert payload["metadata"]["offline_model_requested"] is False


def test_atomic_manifest_preserves_existing_on_replace_failure(tmp_path, monkeypatch):
    manifest = tmp_path / "bootstrap_manifest.json"
    manifest.write_text('{"existing": true}', encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(kb.os, "replace", failing_replace)
    with pytest.raises(OSError):
        kb.atomic_write_json(manifest, {"new": True})

    assert manifest.read_text(encoding="utf-8") == '{"existing": true}'
    assert list(tmp_path.glob(".bootstrap_manifest.json.*.tmp")) == []


def test_main_exit_and_output_for_pass_and_fail(monkeypatch, tmp_path, capsys):
    reports = iter(
        [
            kb.BootstrapReport(
                cache=kb.cache_paths(tmp_path),
                repository={},
                restore_attempted=False,
                restore_result="RESTORE_SKIPPED_ALREADY_COMPATIBLE",
                restore_requirements=None,
                checks=[kb.CheckResult("synthetic", "PASS", "ok")],
            ),
            kb.BootstrapReport(
                cache=kb.cache_paths(tmp_path),
                repository={},
                restore_attempted=False,
                restore_result="NOT_ATTEMPTED",
                restore_requirements=None,
                checks=[kb.CheckResult("synthetic", "FAIL", "bad")],
            ),
        ]
    )
    monkeypatch.setattr(kb, "build_report", lambda **kwargs: next(reports))

    assert kb.main(["--cache-root", str(tmp_path)]) == 0
    assert "BOOTSTRAP_STATUS: PASS" in capsys.readouterr().out
    assert kb.main(["--cache-root", str(tmp_path)]) == 1
    assert "BOOTSTRAP_STATUS: FAIL" in capsys.readouterr().out
