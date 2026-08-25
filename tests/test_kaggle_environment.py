from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import kaggle_environment as ke


def observed_runtime(**overrides):
    observed = {
        "python": "3.12.13",
        "python_executable": "python",
        "platform": "NON_SCIENTIFIC_PRE_URP_INFRASTRUCTURE_TEST",
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "cuda_available": True,
        "gpu_count": 2,
        "gpu_names": ["Tesla T4", "Tesla T4"],
        "gpu_capabilities": [[7, 5], [7, 5]],
        "transformers": "4.45.0",
        "mamba_ssm": "2.3.2.post1",
        "causal_conv1d": "1.7.0",
        "datasets": None,
        "scikit_learn": None,
        "environment": {
            "CUDA_VISIBLE_DEVICES": None,
            "HF_HOME": None,
            "TRANSFORMERS_CACHE": None,
            "HF_HUB_CACHE": None,
        },
    }
    observed.update(overrides)
    return observed


def passing_imports():
    return {
        "causal_conv1d": "PASS",
        "mamba_ssm": "PASS",
        "causal_conv1d_fn": "PASS",
        "selective_scan_fn": "PASS",
        "transformers_mamba_fast_path": True,
    }, [
        ke.CheckResult("causal_conv1d", "PASS", "import ok"),
        ke.CheckResult("mamba_ssm", "PASS", "import ok"),
        ke.CheckResult("causal_conv1d_fn", "PASS", "import ok"),
        ke.CheckResult("selective_scan_fn", "PASS", "import ok"),
        ke.CheckResult("transformers_mamba_fast_path", "PASS", "config.use_mamba_kernels=True"),
    ]


def build_with(
    tmp_path: Path,
    *,
    observed_sequence,
    install=False,
    cuda_smoke=False,
    manifest_path=None,
    expected_head=None,
    trainer_path=None,
    import_verifier=passing_imports,
    pip_runner=None,
    cuda_smoke_runner=None,
):
    observations = iter(observed_sequence)

    def runtime_inspector():
        return next(observations)

    if pip_runner is None:
        pip_runner = lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", "")
    if cuda_smoke_runner is None:
        def cuda_smoke_runner(required):
            raise AssertionError("cuda smoke must not run unless explicitly requested")

    return ke.build_report(
        install=install,
        cuda_smoke=cuda_smoke,
        manifest_path=manifest_path,
        expected_head=expected_head,
        trainer_path=trainer_path,
        repo_root=tmp_path,
        runtime_inspector=runtime_inspector,
        import_verifier=import_verifier,
        pip_runner=pip_runner,
        cuda_smoke_runner=cuda_smoke_runner,
    )


def statuses(report):
    return {check.name: check.status for check in report.checks}


def make_report(status):
    status_check = {
        "PASS": ke.CheckResult("synthetic", "PASS", "ok"),
        "FAIL": ke.CheckResult("synthetic", "FAIL", "failed"),
        "INSTALL_REQUIRED": ke.CheckResult("synthetic", "INSTALL_REQUIRED", "install needed"),
    }[status]
    return ke.EnvironmentReport(
        expected_profile=ke.EXPECTED_PROFILE,
        observed_runtime=observed_runtime(),
        verification={},
        repository={},
        metadata={
            "kind": ke.METADATA_KIND,
            "evidence_boundary": ke.EVIDENCE_BOUNDARY,
        },
        checks=[status_check],
    )


def test_exact_environment_profile_pass(tmp_path):
    report = build_with(tmp_path, observed_sequence=[observed_runtime()])

    assert report.final_status == "PASS"
    assert statuses(report)["cuda_smoke"] == "NOT_CHECKED"
    assert report.install_actions == []


def test_wrong_inherited_torch_fails_and_no_pip_call(tmp_path):
    pip_calls = []

    def pip_runner(*args, **kwargs):
        pip_calls.append(args)
        return subprocess.CompletedProcess(args[0], 0, "", "")

    report = build_with(
        tmp_path,
        observed_sequence=[observed_runtime(torch="2.9.0+cu128", mamba_ssm=None)],
        install=True,
        pip_runner=pip_runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["torch"] == "FAIL"
    assert pip_calls == []


def test_missing_exact_package_verify_only_reports_install_required(tmp_path):
    import_called = False

    def import_verifier():
        nonlocal import_called
        import_called = True
        return passing_imports()

    report = build_with(
        tmp_path,
        observed_sequence=[observed_runtime(mamba_ssm=None)],
        import_verifier=import_verifier,
    )

    assert report.final_status == "INSTALL_REQUIRED"
    assert statuses(report)["mamba-ssm"] == "INSTALL_REQUIRED"
    assert statuses(report)["selective_scan_fn"] == "INSTALL_REQUIRED"
    assert import_called is False


def test_install_invokes_only_allowed_targets(tmp_path):
    pip_calls = []

    def pip_runner(args, **kwargs):
        pip_calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    report = build_with(
        tmp_path,
        observed_sequence=[
            observed_runtime(transformers=None, mamba_ssm=None, causal_conv1d=None),
            observed_runtime(),
        ],
        install=True,
        pip_runner=pip_runner,
    )

    assert report.final_status == "PASS"
    targets = [args[4] for args in pip_calls]
    assert targets == [
        "transformers==4.45.0",
        "causal-conv1d==1.7.0",
        "mamba-ssm==2.3.2.post1",
    ]
    flat = [item for args in pip_calls for item in args]
    assert "torch" not in flat
    assert "datasets" not in flat
    assert "scikit-learn" not in flat
    assert pip_calls[1][5] == "--no-build-isolation"
    assert pip_calls[2][5] == "--no-build-isolation"


def test_already_valid_install_mode_skips_pip(tmp_path):
    pip_calls = []
    report = build_with(
        tmp_path,
        observed_sequence=[observed_runtime()],
        install=True,
        pip_runner=lambda *args, **kwargs: pip_calls.append(args),
    )

    assert report.final_status == "PASS"
    assert report.install_actions == [{"status": "SKIPPED", "reason": "already compatible"}]
    assert pip_calls == []


def test_optional_package_absent_is_nonfatal(tmp_path):
    report = build_with(tmp_path, observed_sequence=[observed_runtime(datasets=None, scikit_learn=None)])

    check_statuses = statuses(report)
    assert report.final_status == "PASS"
    assert check_statuses["datasets"] == "NOT_CHECKED"
    assert check_statuses["scikit-learn"] == "NOT_CHECKED"


def test_mamba_extension_import_failure_fails_when_packages_present(tmp_path):
    def failing_imports():
        verification, checks = passing_imports()
        verification["selective_scan_fn"] = "FAIL: ImportError"
        checks = [check for check in checks if check.name != "selective_scan_fn"]
        checks.append(ke.CheckResult("selective_scan_fn", "FAIL", "FAIL: ImportError"))
        return verification, checks

    report = build_with(tmp_path, observed_sequence=[observed_runtime()], import_verifier=failing_imports)

    assert report.final_status == "FAIL"
    assert statuses(report)["selective_scan_fn"] == "FAIL"


def test_transformers_mamba_fast_path_failure_fails_when_packages_present(tmp_path):
    def failing_fast_path_imports():
        verification, checks = passing_imports()
        verification["transformers_mamba_fast_path"] = False
        checks = [check for check in checks if check.name != "transformers_mamba_fast_path"]
        checks.append(ke.CheckResult("transformers_mamba_fast_path", "FAIL", "use_mamba_kernels unavailable"))
        return verification, checks

    report = build_with(tmp_path, observed_sequence=[observed_runtime()], import_verifier=failing_fast_path_imports)

    assert report.final_status == "FAIL"
    assert statuses(report)["transformers_mamba_fast_path"] == "FAIL"


def test_cuda_smoke_not_checked_unless_requested(tmp_path):
    smoke_calls = []

    def cuda_smoke_runner(required):
        smoke_calls.append(required)
        return {"status": "FAIL"}, ke.CheckResult("cuda_smoke", "FAIL", "should not matter")

    report = build_with(
        tmp_path,
        observed_sequence=[observed_runtime(cuda_available=False, gpu_count=0, gpu_names=[], gpu_capabilities=[])],
        cuda_smoke=False,
        cuda_smoke_runner=cuda_smoke_runner,
    )

    assert report.final_status == "PASS"
    assert statuses(report)["cuda_smoke"] == "NOT_CHECKED"
    assert smoke_calls == []


def test_requested_cuda_smoke_failure_is_fail(tmp_path):
    def cuda_smoke_runner(required):
        assert required is True
        return {"status": "FAIL", "reason": "CUDA unavailable"}, ke.CheckResult("cuda_smoke", "FAIL", "CUDA unavailable")

    report = build_with(
        tmp_path,
        observed_sequence=[observed_runtime(cuda_available=False, gpu_count=0, gpu_names=[], gpu_capabilities=[])],
        cuda_smoke=True,
        cuda_smoke_runner=cuda_smoke_runner,
    )

    assert report.final_status == "FAIL"
    assert statuses(report)["cuda_smoke"] == "FAIL"


def test_main_exit_codes_for_pass_install_required_and_fail(monkeypatch):
    reports = iter([make_report("PASS"), make_report("INSTALL_REQUIRED"), make_report("FAIL")])

    def fake_build_report(**kwargs):
        return next(reports)

    monkeypatch.setattr(ke, "build_report", fake_build_report)

    assert ke.main([]) == 0
    assert ke.main([]) == 2
    assert ke.main([]) == 1


def test_verify_only_missing_required_package_returns_install_required_exit(monkeypatch):
    def fake_build_report(**kwargs):
        assert kwargs["install"] is False
        return make_report("INSTALL_REQUIRED")

    monkeypatch.setattr(ke, "build_report", fake_build_report)

    assert ke.main([]) == 2


def test_successful_mocked_install_reverification_returns_pass_exit(monkeypatch):
    def fake_build_report(**kwargs):
        assert kwargs["install"] is True
        return make_report("PASS")

    monkeypatch.setattr(ke, "build_report", fake_build_report)

    assert ke.main(["--install"]) == 0


def test_fast_path_failure_returns_nonzero_exit(monkeypatch):
    def fake_build_report(**kwargs):
        return make_report("FAIL")

    monkeypatch.setattr(ke, "build_report", fake_build_report)

    assert ke.main([]) == 1


def test_fast_path_failure_through_main_returns_exit_one(monkeypatch, tmp_path, capsys):
    original_build_report = ke.build_report

    def failing_fast_path_imports():
        verification, checks = passing_imports()
        verification["transformers_mamba_fast_path"] = False
        checks = [check for check in checks if check.name != "transformers_mamba_fast_path"]
        checks.append(ke.CheckResult("transformers_mamba_fast_path", "FAIL", "use_mamba_kernels unavailable"))
        return verification, checks

    def wrapped_build_report(**kwargs):
        assert kwargs["install"] is False
        assert kwargs["cuda_smoke"] is False
        return original_build_report(
            install=kwargs["install"],
            cuda_smoke=kwargs["cuda_smoke"],
            manifest_path=kwargs["manifest_path"],
            expected_head=kwargs["expected_head"],
            trainer_path=None,
            repo_root=tmp_path,
            runtime_inspector=lambda: observed_runtime(),
            import_verifier=failing_fast_path_imports,
            pip_runner=lambda *args, **kwargs: pytest.fail("pip must not run"),
            cuda_smoke_runner=lambda required: pytest.fail("cuda smoke must not run"),
        )

    monkeypatch.setattr(ke, "build_report", wrapped_build_report)

    assert ke.main([]) == 1
    output = capsys.readouterr().out
    assert "transformers_mamba_fast_path: FAIL" in output
    assert "FINAL_STATUS: FAIL" in output


def test_expected_git_head_match_and_mismatch(tmp_path, monkeypatch):
    def fake_run_git(args, cwd):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("rev-parse", "origin/main"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(ke, "run_git", fake_run_git)
    match = build_with(tmp_path, observed_sequence=[observed_runtime()], expected_head="a" * 40)
    mismatch = build_with(tmp_path, observed_sequence=[observed_runtime()], expected_head="b" * 40)

    assert statuses(match)["expected_head"] == "PASS"
    assert statuses(mismatch)["expected_head"] == "FAIL"
    assert mismatch.final_status == "FAIL"


def test_manifest_schema_and_marker(tmp_path):
    manifest = tmp_path / "manifest.json"
    report = build_with(tmp_path, observed_sequence=[observed_runtime()], manifest_path=manifest)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert report.final_status == "PASS"
    assert payload["expected_profile"]["torch"] == "2.10.0+cu128"
    assert payload["observed_runtime"]["torch"] == "2.10.0+cu128"
    assert payload["metadata"]["kind"] == "PRE_URP_INFRASTRUCTURE_ENVIRONMENT_METADATA"
    assert payload["metadata"]["evidence_boundary"] == "NOT_SCIENTIFIC_EVIDENCE"


def test_atomic_manifest_replacement_preserves_existing_on_replace_failure(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"existing": true}', encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(ke.os, "replace", failing_replace)

    with pytest.raises(OSError):
        ke.atomic_write_json(manifest, {"new": True})

    assert manifest.read_text(encoding="utf-8") == '{"existing": true}'
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
