from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_CACHE_ROOT = Path("/kaggle/working/contramamba_cache")
RESTORE_REQUIREMENTS = "minimal_restore_requirements.txt"
RESTORE_MANIFEST = "minimal_restore_manifest.json"
RESTORE_MARKER = "PRE_URP_KAGGLE_MINIMAL_RESTORE_LOCK"
BOOTSTRAP_MARKER = "PRE_URP_KAGGLE_BOOTSTRAP_METADATA"
EVIDENCE_BOUNDARY = "NOT_SCIENTIFIC_EVIDENCE"
ENVIRONMENT_SCRIPT = Path("scripts/kaggle_environment.py")
ALLOWED_REQUIREMENT_LINES = {
    "transformers==4.45.0",
    "huggingface_hub==0.36.2",
    "tokenizers==0.20.3",
    "causal-conv1d==1.7.0",
    "mamba-ssm==2.3.2.post1",
    "tilelang==0.1.8",
    "apache-tvm-ffi==0.1.9",
    "quack-kernels==0.6.4",
    "torch-c-dlpack-ext==0.1.5",
    "z3-solver==4.15.4.0",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class BootstrapReport:
    cache: dict[str, str]
    repository: dict[str, Any]
    restore_attempted: bool
    restore_result: str
    restore_requirements: str | None
    wheel_integrity: list[CheckResult] = field(default_factory=list)
    verifier: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def final_status(self) -> str:
        statuses = {check.status for check in self.checks}
        statuses.update(check.status for check in self.wheel_integrity)
        if "FAIL" in statuses:
            return "FAIL"
        return "PASS"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_paths(cache_root: Path) -> dict[str, str]:
    root = cache_root.resolve()
    hf_home = root / "huggingface"
    return {
        "cache_root": str(root),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "PIP_CACHE_DIR": str(root / "pip"),
        "wheelhouse": str(root / "wheelhouse"),
    }


def subprocess_env(paths: dict[str, str], *, offline_model: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": paths["HF_HOME"],
            "HF_HUB_CACHE": paths["HF_HUB_CACHE"],
            "PIP_CACHE_DIR": paths["PIP_CACHE_DIR"],
        }
    )
    if offline_model:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def run_git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repository_info(repo_root: Path, expected_head: str | None) -> tuple[dict[str, Any], list[CheckResult]]:
    head = run_git(("rev-parse", "HEAD"), repo_root)
    origin_main = run_git(("rev-parse", "origin/main"), repo_root)
    tracked = run_git(("diff", "--name-only"), repo_root)
    staged = run_git(("diff", "--cached", "--name-only"), repo_root)
    info = {
        "git_head": head,
        "origin_main": origin_main,
        "tracked_dirty": None if tracked is None else bool(tracked),
        "index_dirty": None if staged is None else bool(staged),
        "expected_head": expected_head,
    }
    checks = [CheckResult("git_identity", "PASS" if head else "WARN", "git HEAD recorded" if head else "not a Git repository or unavailable")]
    if expected_head is None:
        checks.append(CheckResult("expected_head", "NOT_CHECKED", "no expected head supplied"))
    elif head == expected_head:
        checks.append(CheckResult("expected_head", "PASS", "matches"))
    else:
        checks.append(CheckResult("expected_head", "FAIL", f"expected={expected_head} observed={head}"))
    return info, checks


def load_restore_manifest(path: Path) -> tuple[dict[str, Any] | None, list[CheckResult]]:
    checks: list[CheckResult] = []
    if not path.is_file():
        return None, [CheckResult("restore_manifest", "FAIL", f"missing={path}")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [CheckResult("restore_manifest", "FAIL", f"malformed JSON: {type(exc).__name__}: {exc}")]
    if not isinstance(payload, dict):
        checks.append(CheckResult("restore_manifest", "FAIL", "top-level JSON is not an object"))
        return None, checks
    marker = payload.get("marker")
    boundary = payload.get("evidence_boundary")
    missing_wheels = payload.get("missing_wheels")
    wheels = payload.get("wheels")
    checks.append(CheckResult("restore_manifest_marker", "PASS" if marker == RESTORE_MARKER else "FAIL", f"observed={marker!r}"))
    checks.append(CheckResult("restore_manifest_boundary", "PASS" if boundary == EVIDENCE_BOUNDARY else "FAIL", f"observed={boundary!r}"))
    checks.append(CheckResult("restore_manifest_missing_wheels", "PASS" if missing_wheels == [] else "FAIL", f"observed={missing_wheels!r}"))
    checks.append(CheckResult("restore_manifest_wheels", "PASS" if isinstance(wheels, list) else "FAIL", "wheels list"))
    return payload, checks


def wheel_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("wheels", [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def wheel_name(entry: dict[str, Any]) -> str | None:
    for key in ("filename", "file", "name", "path"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return Path(value).name
    return None


def expected_size(entry: dict[str, Any]) -> int | None:
    for key in ("size", "size_bytes", "byte_size"):
        value = entry.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def expected_sha(entry: dict[str, Any]) -> str | None:
    for key in ("sha256", "physical_sha256"):
        value = entry.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    return None


def validate_restore_inputs(wheelhouse: Path) -> tuple[Path | None, list[CheckResult]]:
    checks: list[CheckResult] = []
    requirements = wheelhouse / RESTORE_REQUIREMENTS
    manifest_path = wheelhouse / RESTORE_MANIFEST
    if requirements.is_file():
        checks.append(CheckResult("restore_requirements", "PASS", str(requirements)))
        requirement_lines = [
            line.strip()
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        unexpected = [line for line in requirement_lines if line not in ALLOWED_REQUIREMENT_LINES]
        if unexpected:
            checks.append(CheckResult("restore_requirements_allowlist", "FAIL", f"unexpected={unexpected!r}"))
        else:
            checks.append(CheckResult("restore_requirements_allowlist", "PASS", f"count={len(requirement_lines)}"))
    else:
        checks.append(CheckResult("restore_requirements", "FAIL", f"missing={requirements}"))
    manifest, manifest_checks = load_restore_manifest(manifest_path)
    checks.extend(manifest_checks)
    if manifest is None:
        return None, checks
    for index, entry in enumerate(wheel_entries(manifest)):
        name = wheel_name(entry)
        if name is None:
            checks.append(CheckResult(f"wheel_{index}", "FAIL", "missing wheel filename"))
            continue
        wheel_path = wheelhouse / name
        if not wheel_path.is_file():
            checks.append(CheckResult(name, "FAIL", "missing wheel"))
            continue
        actual_size = wheel_path.stat().st_size
        declared_size = expected_size(entry)
        declared_sha = expected_sha(entry)
        if declared_size is None:
            checks.append(CheckResult(name, "FAIL", "missing declared size"))
        elif actual_size != declared_size:
            checks.append(CheckResult(name, "FAIL", f"size expected={declared_size} observed={actual_size}"))
        else:
            checks.append(CheckResult(f"{name}_size", "PASS", str(actual_size)))
        if declared_sha is None:
            checks.append(CheckResult(name, "FAIL", "missing declared sha256"))
        else:
            actual_sha = file_sha256(wheel_path)
            checks.append(CheckResult(f"{name}_sha256", "PASS" if actual_sha == declared_sha else "FAIL", f"expected={declared_sha} observed={actual_sha}"))
    return requirements if requirements.is_file() else None, checks


def run_verifier(
    *,
    repo_root: Path,
    expected_head: str | None,
    cuda_smoke: bool,
    env: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any]:
    args = [sys.executable, str(repo_root / ENVIRONMENT_SCRIPT)]
    if expected_head:
        args.extend(["--expected-head", expected_head])
    if cuda_smoke:
        args.append("--cuda-smoke")
    result = runner(args, cwd=str(repo_root), env=env, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout or ""
    status = None
    for line in stdout.splitlines():
        if line.startswith("FINAL_STATUS:"):
            status = line.split(":", 1)[1].strip()
    return {
        "args": args,
        "returncode": result.returncode,
        "final_status": status,
        "stdout_tail": stdout[-1000:],
        "stderr_tail": (result.stderr or "")[-1000:],
    }


def run_local_restore(
    *,
    requirements: Path,
    wheelhouse: Path,
    env: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any]:
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        f"--find-links={wheelhouse}",
        "--no-deps",
        "-r",
        str(requirements),
        "-q",
    ]
    result = runner(args, env=env, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "args": args,
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "")[-1000:],
        "stderr_tail": (result.stderr or "")[-1000:],
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False, prefix=f".{path.name}.", suffix=".tmp") as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.stat().st_size <= 0:
            raise RuntimeError("temporary bootstrap manifest is empty")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def report_to_dict(report: BootstrapReport) -> dict[str, Any]:
    return {
        "marker": BOOTSTRAP_MARKER,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "cache": report.cache,
        "repository": report.repository,
        "restore_attempted": report.restore_attempted,
        "restore_result": report.restore_result,
        "restore_requirements": report.restore_requirements,
        "wheel_integrity": [check.__dict__ for check in report.wheel_integrity],
        "environment_verifier": report.verifier,
        "metadata": report.metadata,
        "checks": [check.__dict__ for check in report.checks],
        "final_status": report.final_status,
    }


def build_report(
    *,
    cache_root: Path,
    expected_head: str | None,
    cuda_smoke: bool,
    offline_model: bool,
    manifest_path: Path | None,
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> BootstrapReport:
    paths = cache_paths(cache_root)
    env = subprocess_env(paths, offline_model=offline_model)
    repo, repo_checks = repository_info(repo_root, expected_head)
    checks = list(repo_checks)
    if any(check.status == "FAIL" for check in repo_checks):
        report = BootstrapReport(
            cache=paths,
            repository=repo,
            restore_attempted=False,
            restore_result="NOT_ATTEMPTED_REPOSITORY_IDENTITY_FAILED",
            restore_requirements=None,
            verifier={},
            metadata={
                "marker": BOOTSTRAP_MARKER,
                "evidence_boundary": EVIDENCE_BOUNDARY,
                "cuda_smoke_requested": bool(cuda_smoke),
                "offline_model_requested": bool(offline_model),
            },
            checks=checks,
        )
        if manifest_path is not None:
            atomic_write_json(manifest_path, report_to_dict(report))
        return report
    first = run_verifier(repo_root=repo_root, expected_head=expected_head, cuda_smoke=cuda_smoke, env=env, runner=runner)
    verifier = {"initial": first}
    restore_attempted = False
    restore_result = "RESTORE_SKIPPED_ALREADY_COMPATIBLE" if first["returncode"] == 0 else "NOT_ATTEMPTED"
    wheel_checks: list[CheckResult] = []
    restore_requirements: Path | None = None

    if first["returncode"] == 0:
        checks.append(CheckResult("environment_verifier_initial", "PASS", "already compatible"))
    elif first["returncode"] == 2:
        checks.append(CheckResult("environment_verifier_initial", "INSTALL_REQUIRED", "local restore required"))
        restore_attempted = True
        wheelhouse = Path(paths["wheelhouse"])
        restore_requirements, wheel_checks = validate_restore_inputs(wheelhouse)
        if any(check.status == "FAIL" for check in wheel_checks) or restore_requirements is None:
            restore_result = "RESTORE_INPUT_VALIDATION_FAILED"
            checks.append(CheckResult("local_restore", "FAIL", "restore inputs invalid"))
        else:
            restore = run_local_restore(requirements=restore_requirements, wheelhouse=wheelhouse, env=env, runner=runner)
            verifier["restore"] = restore
            if restore["returncode"] != 0:
                restore_result = "RESTORE_FAILED"
                checks.append(CheckResult("local_restore", "FAIL", f"pip exit={restore['returncode']}"))
            else:
                restore_result = "RESTORE_SUCCEEDED"
                checks.append(CheckResult("local_restore", "PASS", "local wheelhouse restore completed"))
                second = run_verifier(repo_root=repo_root, expected_head=expected_head, cuda_smoke=cuda_smoke, env=env, runner=runner)
                verifier["post_restore"] = second
                if second["returncode"] == 0:
                    checks.append(CheckResult("environment_verifier_post_restore", "PASS", "compatible"))
                else:
                    checks.append(CheckResult("environment_verifier_post_restore", "FAIL", f"exit={second['returncode']}"))
    else:
        checks.append(CheckResult("environment_verifier_initial", "FAIL", f"exit={first['returncode']}"))

    report = BootstrapReport(
        cache=paths,
        repository=repo,
        restore_attempted=restore_attempted,
        restore_result=restore_result,
        restore_requirements=str(restore_requirements) if restore_requirements else None,
        wheel_integrity=wheel_checks,
        verifier=verifier,
        metadata={
            "marker": BOOTSTRAP_MARKER,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "cuda_smoke_requested": bool(cuda_smoke),
            "offline_model_requested": bool(offline_model),
        },
        checks=checks,
    )
    if manifest_path is not None:
        atomic_write_json(manifest_path, report_to_dict(report))
    return report


def print_report(report: BootstrapReport) -> None:
    print("KAGGLE_BOOTSTRAP")
    print(f"cache_root: {report.cache['cache_root']}")
    print(f"wheelhouse: {report.cache['wheelhouse']}")
    print(f"restore_result: {report.restore_result}")
    for check in [*report.checks, *report.wheel_integrity]:
        detail = f" {check.detail}" if check.detail else ""
        print(f"{check.name}: {check.status}{detail}")
    print(f"BOOTSTRAP_STATUS: {report.final_status}")
    print(EVIDENCE_BOUNDARY)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command pre-URP ContraMamba Kaggle environment bootstrap.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--offline-model", action="store_true")
    parser.add_argument("--expected-head", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        cache_root=args.cache_root,
        expected_head=args.expected_head,
        cuda_smoke=args.cuda_smoke,
        offline_model=args.offline_model,
        manifest_path=args.manifest,
        repo_root=Path.cwd(),
    )
    print_report(report)
    return 0 if report.final_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
