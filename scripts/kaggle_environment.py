from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


EXPECTED_PROFILE = {
    "python": "3.12.13",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "packages": {
        "transformers": "4.45.0",
        "mamba-ssm": "2.3.2.post1",
        "causal-conv1d": "1.7.0",
    },
}
OBSERVED_ONLY_PACKAGES = ("datasets", "scikit-learn")
ALLOWED_INSTALLS = (
    ("transformers", "transformers==4.45.0", ()),
    ("causal-conv1d", "causal-conv1d==1.7.0", ("--no-build-isolation",)),
    ("mamba-ssm", "mamba-ssm==2.3.2.post1", ("--no-build-isolation",)),
)
METADATA_KIND = "PRE_URP_INFRASTRUCTURE_ENVIRONMENT_METADATA"
EVIDENCE_BOUNDARY = "NOT_SCIENTIFIC_EVIDENCE"
NON_SCIENTIFIC_SEED = 9017
EXIT_CODES = {
    "PASS": 0,
    "FAIL": 1,
    "INSTALL_REQUIRED": 2,
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class EnvironmentReport:
    expected_profile: dict[str, Any]
    observed_runtime: dict[str, Any]
    verification: dict[str, Any]
    repository: dict[str, Any]
    metadata: dict[str, Any]
    checks: list[CheckResult] = field(default_factory=list)
    install_actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final_status(self) -> str:
        statuses = {check.status for check in self.checks}
        if "FAIL" in statuses:
            return "FAIL"
        if "INSTALL_REQUIRED" in statuses:
            return "INSTALL_REQUIRED"
        return "PASS"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def repository_info(repo_root: Path, trainer_path: Path | None, expected_head: str | None) -> tuple[dict[str, Any], list[CheckResult]]:
    checks: list[CheckResult] = []
    head = run_git(("rev-parse", "HEAD"), repo_root)
    origin_main = run_git(("rev-parse", "origin/main"), repo_root)
    status = run_git(("status", "--porcelain"), repo_root)
    dirty = None if status is None else bool(status)
    info: dict[str, Any] = {
        "git_head": head,
        "origin_main": origin_main,
        "dirty": dirty,
        "expected_head": expected_head,
    }
    if expected_head:
        if head == expected_head:
            checks.append(CheckResult("expected_head", "PASS", "matches"))
        else:
            checks.append(CheckResult("expected_head", "FAIL", f"expected={expected_head} observed={head}"))
    else:
        checks.append(CheckResult("expected_head", "NOT_CHECKED", "no expected head supplied"))
    if trainer_path is not None:
        resolved = (repo_root / trainer_path).resolve() if not trainer_path.is_absolute() else trainer_path.resolve()
        if not resolved.is_file():
            checks.append(CheckResult("trainer_path", "FAIL", f"missing={resolved}"))
            info["trainer"] = {"path": str(resolved), "sha256": None}
        else:
            info["trainer"] = {"path": str(resolved), "sha256": file_sha256(resolved)}
            checks.append(CheckResult("trainer_path", "PASS", str(resolved)))
    return info, checks


def inspect_runtime() -> dict[str, Any]:
    observed: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        },
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised through injection in tests
        observed["torch_import_error"] = f"{type(exc).__name__}: {exc}"
        observed["torch"] = None
        observed["torch_cuda"] = None
        observed["cuda_available"] = False
        observed["gpu_count"] = 0
        observed["gpu_names"] = []
        observed["gpu_capabilities"] = []
        return observed

    observed["torch"] = getattr(torch, "__version__", None)
    observed["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    observed["cuda_available"] = bool(torch.cuda.is_available())
    observed["gpu_count"] = int(torch.cuda.device_count()) if observed["cuda_available"] else 0
    observed["gpu_names"] = [
        torch.cuda.get_device_name(index) for index in range(observed["gpu_count"])
    ]
    observed["gpu_capabilities"] = [
        list(torch.cuda.get_device_capability(index)) for index in range(observed["gpu_count"])
    ]
    for package_name in (*EXPECTED_PROFILE["packages"].keys(), *OBSERVED_ONLY_PACKAGES):
        observed[package_name.replace("-", "_")] = package_version(package_name)
    return observed


def evaluate_profile(observed: dict[str, Any]) -> list[CheckResult]:
    checks = [
        CheckResult(
            "python",
            "PASS" if observed.get("python") == EXPECTED_PROFILE["python"] else "FAIL",
            f"expected={EXPECTED_PROFILE['python']} observed={observed.get('python')}",
        ),
        CheckResult(
            "torch",
            "PASS" if observed.get("torch") == EXPECTED_PROFILE["torch"] else "FAIL",
            f"expected={EXPECTED_PROFILE['torch']} observed={observed.get('torch')}",
        ),
        CheckResult(
            "torch_cuda",
            "PASS" if observed.get("torch_cuda") == EXPECTED_PROFILE["torch_cuda"] else "FAIL",
            f"expected={EXPECTED_PROFILE['torch_cuda']} observed={observed.get('torch_cuda')}",
        ),
    ]
    for package_name, expected in EXPECTED_PROFILE["packages"].items():
        key = package_name.replace("-", "_")
        observed_version = observed.get(key)
        status = "PASS" if observed_version == expected else "INSTALL_REQUIRED"
        checks.append(CheckResult(package_name, status, f"expected={expected} observed={observed_version}"))
    for package_name in OBSERVED_ONLY_PACKAGES:
        key = package_name.replace("-", "_")
        observed_version = observed.get(key)
        checks.append(
            CheckResult(
                package_name,
                "PASS" if observed_version is not None else "NOT_CHECKED",
                f"observed={observed_version}",
            )
        )
    return checks


def verify_imports(import_module: Callable[[str], Any] = importlib.import_module) -> tuple[dict[str, Any], list[CheckResult]]:
    verification: dict[str, Any] = {}
    checks: list[CheckResult] = []
    for module_name in ("causal_conv1d", "mamba_ssm"):
        try:
            import_module(module_name)
        except Exception as exc:
            verification[module_name] = f"FAIL: {type(exc).__name__}: {exc}"
            checks.append(CheckResult(module_name, "FAIL", verification[module_name]))
        else:
            verification[module_name] = "PASS"
            checks.append(CheckResult(module_name, "PASS", "import ok"))
    try:
        module = import_module("causal_conv1d.causal_conv1d_interface")
        getattr(module, "causal_conv1d_fn")
    except Exception as exc:
        verification["causal_conv1d_fn"] = f"FAIL: {type(exc).__name__}: {exc}"
        checks.append(CheckResult("causal_conv1d_fn", "FAIL", verification["causal_conv1d_fn"]))
    else:
        verification["causal_conv1d_fn"] = "PASS"
        checks.append(CheckResult("causal_conv1d_fn", "PASS", "import ok"))
    try:
        module = import_module("mamba_ssm.ops.selective_scan_interface")
        getattr(module, "selective_scan_fn")
    except Exception as exc:
        verification["selective_scan_fn"] = f"FAIL: {type(exc).__name__}: {exc}"
        checks.append(CheckResult("selective_scan_fn", "FAIL", verification["selective_scan_fn"]))
    else:
        verification["selective_scan_fn"] = "PASS"
        checks.append(CheckResult("selective_scan_fn", "PASS", "import ok"))
    try:
        modeling_mamba = import_module("transformers.models.mamba.modeling_mamba")
        available = getattr(modeling_mamba, "is_fast_path_available")
    except Exception as exc:
        verification["transformers_mamba_fast_path"] = f"FAIL: {type(exc).__name__}: {exc}"
        checks.append(CheckResult("transformers_mamba_fast_path", "FAIL", verification["transformers_mamba_fast_path"]))
    else:
        verification["transformers_mamba_fast_path"] = bool(available)
        if available is True:
            checks.append(CheckResult("transformers_mamba_fast_path", "PASS", "modeling_mamba.is_fast_path_available=True"))
        else:
            checks.append(CheckResult("transformers_mamba_fast_path", "FAIL", f"modeling_mamba.is_fast_path_available={available!r}"))
    return verification, checks


def mamba_package_checks(checks: Sequence[CheckResult]) -> list[CheckResult]:
    package_names = set(EXPECTED_PROFILE["packages"].keys())
    return [check for check in checks if check.name in package_names]


def all_mamba_packages_present(checks: Sequence[CheckResult]) -> bool:
    return all(check.status == "PASS" for check in mamba_package_checks(checks))


def install_required_verification(checks: Sequence[CheckResult]) -> tuple[dict[str, Any], list[CheckResult]]:
    verification: dict[str, Any] = {}
    deferred: list[CheckResult] = []
    for check in mamba_package_checks(checks):
        if check.status == "PASS":
            continue
        verification[check.name.replace("-", "_")] = "INSTALL_REQUIRED"
        deferred.append(CheckResult(f"{check.name}_import", "INSTALL_REQUIRED", "package pin not satisfied"))
    verification["causal_conv1d_fn"] = "INSTALL_REQUIRED"
    verification["selective_scan_fn"] = "INSTALL_REQUIRED"
    verification["transformers_mamba_fast_path"] = "INSTALL_REQUIRED"
    deferred.extend(
        [
            CheckResult("causal_conv1d_fn", "INSTALL_REQUIRED", "package pins not satisfied"),
            CheckResult("selective_scan_fn", "INSTALL_REQUIRED", "package pins not satisfied"),
            CheckResult("transformers_mamba_fast_path", "INSTALL_REQUIRED", "package pins not satisfied"),
        ]
    )
    return verification, deferred


def run_cuda_smoke(required: bool) -> tuple[dict[str, Any], CheckResult]:
    try:
        import torch
        from transformers import MambaConfig, MambaModel
    except Exception as exc:
        status = "FAIL" if required else "NOT_CHECKED"
        return {"status": status, "error": f"{type(exc).__name__}: {exc}"}, CheckResult("cuda_smoke", status, "imports unavailable")
    if not torch.cuda.is_available():
        status = "FAIL" if required else "NOT_CHECKED"
        return {"status": status, "reason": "CUDA unavailable"}, CheckResult("cuda_smoke", status, "CUDA unavailable")
    torch.manual_seed(NON_SCIENTIFIC_SEED)
    device = torch.device("cuda")
    try:
        config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        config.use_mamba_kernels = True
        model = MambaModel.from_pretrained("state-spaces/mamba-130m-hf", config=config, torch_dtype=torch.float16).to(device)
        model.train()
        input_ids = torch.randint(0, int(config.vocab_size), (1, 8), device=device)
        output = model(input_ids=input_ids).last_hidden_state
        loss = output.float().square().mean()
        loss.backward()
        finite_grad = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, CheckResult("cuda_smoke", "FAIL", "synthetic smoke failed")
    return {
        "status": "PASS" if finite_grad else "FAIL",
        "non_scientific_seed": NON_SCIENTIFIC_SEED,
        "finite_gradients": finite_grad,
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
    }, CheckResult("cuda_smoke", "PASS" if finite_grad else "FAIL", "synthetic forward/backward")


def run_pip_install(target: str, extra_args: Sequence[str], runner: Callable[..., subprocess.CompletedProcess]) -> dict[str, Any]:
    args = [sys.executable, "-m", "pip", "install", target, *extra_args, "-q"]
    result = runner(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "target": target,
        "extra_args": list(extra_args),
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "")[-500:],
        "stderr_tail": (result.stderr or "")[-500:],
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
            raise RuntimeError("temporary manifest is empty")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def build_report(
    *,
    install: bool,
    cuda_smoke: bool,
    manifest_path: Path | None,
    expected_head: str | None,
    trainer_path: Path | None,
    repo_root: Path,
    runtime_inspector: Callable[[], dict[str, Any]] = inspect_runtime,
    import_verifier: Callable[[], tuple[dict[str, Any], list[CheckResult]]] = verify_imports,
    pip_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    cuda_smoke_runner: Callable[[bool], tuple[dict[str, Any], CheckResult]] = run_cuda_smoke,
) -> EnvironmentReport:
    observed = runtime_inspector()
    checks = evaluate_profile(observed)
    if all_mamba_packages_present(checks):
        verification, import_checks = import_verifier()
    else:
        verification, import_checks = install_required_verification(checks)
    checks.extend(import_checks)
    repo, repo_checks = repository_info(repo_root, trainer_path, expected_head)
    checks.extend(repo_checks)
    report = EnvironmentReport(
        expected_profile=EXPECTED_PROFILE,
        observed_runtime=observed,
        verification=verification,
        repository=repo,
        metadata={
            "kind": METADATA_KIND,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "install_mode": bool(install),
            "cuda_smoke_requested": bool(cuda_smoke),
        },
        checks=checks,
    )
    inherited_fail = any(check.status == "FAIL" and check.name in {"python", "torch", "torch_cuda"} for check in report.checks)
    install_required = [check for check in report.checks if check.status == "INSTALL_REQUIRED"]
    if install and install_required and not inherited_fail:
        install_names = {check.name for check in install_required}
        for package_name, target, extra_args in ALLOWED_INSTALLS:
            if package_name not in install_names:
                continue
            action = run_pip_install(target, extra_args, pip_runner)
            report.install_actions.append(action)
            if action["returncode"] != 0:
                report.checks.append(CheckResult(f"install_{package_name}", "FAIL", f"target={target}"))
        if not any(check.status == "FAIL" for check in report.checks):
            observed = runtime_inspector()
            next_checks = evaluate_profile(observed)
            if all_mamba_packages_present(next_checks):
                verification, import_checks = import_verifier()
            else:
                verification, import_checks = install_required_verification(next_checks)
            report.observed_runtime = observed
            report.verification = verification
            report.checks = []
            for check in next_checks:
                if check.status == "INSTALL_REQUIRED":
                    report.checks.append(CheckResult(check.name, "FAIL", f"post-install mismatch: {check.detail}"))
                else:
                    report.checks.append(check)
            report.checks.extend(import_checks)
            report.checks.extend(repo_checks)
    elif install and not install_required:
        report.install_actions.append({"status": "SKIPPED", "reason": "already compatible"})
    if cuda_smoke:
        smoke, smoke_check = cuda_smoke_runner(True)
    else:
        smoke_check = CheckResult("cuda_smoke", "NOT_CHECKED", "not requested")
        smoke = {"status": "NOT_CHECKED", "reason": "not requested"}
    report.verification["cuda_smoke"] = smoke
    report.checks.append(smoke_check)
    if manifest_path is not None:
        atomic_write_json(manifest_path, report_to_dict(report))
    return report


def report_to_dict(report: EnvironmentReport) -> dict[str, Any]:
    return {
        "expected_profile": report.expected_profile,
        "observed_runtime": report.observed_runtime,
        "verification": report.verification,
        "repository": report.repository,
        "metadata": report.metadata,
        "checks": [check.__dict__ for check in report.checks],
        "install_actions": report.install_actions,
        "final_status": report.final_status,
    }


def print_report(report: EnvironmentReport) -> None:
    print("KAGGLE_ENVIRONMENT")
    for check in report.checks:
        detail = f" {check.detail}" if check.detail else ""
        print(f"{check.name}: {check.status}{detail}")
    print(f"FINAL_STATUS: {report.final_status}")
    print(EVIDENCE_BOUNDARY)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the pre-URP ContraMamba Kaggle environment profile.")
    parser.add_argument("--install", action="store_true", help="Explicitly install only allowed Mamba-facing package pins when required.")
    parser.add_argument("--cuda-smoke", action="store_true", help="Run a non-scientific synthetic CUDA Mamba forward/backward smoke test.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional path for an atomic environment manifest JSON.")
    parser.add_argument("--expected-head", default=None, help="Optional expected Git HEAD SHA.")
    parser.add_argument("--trainer-path", type=Path, default=Path("scripts/train_controlled_v6b_minimal.py"), help="Optional trainer path whose exact bytes are recorded.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        install=args.install,
        cuda_smoke=args.cuda_smoke,
        manifest_path=args.manifest,
        expected_head=args.expected_head,
        trainer_path=args.trainer_path,
        repo_root=Path.cwd(),
    )
    print_report(report)
    return EXIT_CODES[report.final_status]


if __name__ == "__main__":
    raise SystemExit(main())
