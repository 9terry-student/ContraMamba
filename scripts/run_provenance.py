from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "stage174a_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return json_safe(value.value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if type(value).__module__ == "torch" and type(value).__name__ == "device":
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            tensor = value.detach().cpu()
            return json_safe(tensor.item() if tensor.numel() == 1 else tensor.tolist())
        except Exception:
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = json_safe(payload)
    text = json.dumps(
        safe_payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_row_count(path: Path) -> int | None:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def dataset_record(path_value: Any, *, mode: str | None, expected: bool = False) -> dict[str, Any]:
    configured = path_value is not None
    if not configured:
        return {
            "byte_size": None,
            "configured": False,
            "error": None,
            "exists": False,
            "expected": expected,
            "mode": mode,
            "path": None,
            "resolved_path": None,
            "row_count": None,
            "sha256": None,
        }
    path = Path(path_value)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    record = {
        "byte_size": None,
        "configured": True,
        "error": None,
        "exists": resolved.exists(),
        "expected": expected,
        "mode": mode,
        "path": str(path),
        "resolved_path": str(resolved),
        "row_count": None,
        "sha256": None,
    }
    if not resolved.exists():
        record["error"] = "configured path does not exist"
        return record
    try:
        record["byte_size"] = resolved.stat().st_size
        record["sha256"] = file_sha256(resolved)
        record["row_count"] = jsonl_row_count(resolved)
    except (OSError, UnicodeError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_info(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    info: dict[str, Any] = {
        "git_branch": None,
        "git_commit": None,
        "git_diff_names": None,
        "git_error": None,
        "git_is_dirty": None,
    }
    try:
        info["git_commit"] = run_git(["rev-parse", "HEAD"]) or None
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or None
        info["git_branch"] = None if branch == "HEAD" else branch
        changed = run_git(["status", "--porcelain"])
        info["git_is_dirty"] = bool(changed)
        status_paths: list[str] = []
        for line in changed.splitlines():
            raw_path = line[3:] if len(line) > 3 else ""
            if " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[1]
            if raw_path:
                status_paths.append(raw_path.strip().strip(chr(34)))
        info["git_diff_names"] = (
            run_git(["diff", "--name-only"]).splitlines()
            + run_git(["diff", "--name-only", "--cached"]).splitlines()
            + status_paths
        )
        info["git_diff_names"] = sorted(set(info["git_diff_names"]))
    except Exception as exc:
        info["git_error"] = f"{type(exc).__name__}: {exc}"
    return info


def command_string(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def initial_record(
    *,
    run_dir: Path,
    training_script: Path,
    raw_sys_argv: list[str],
    working_directory: Path,
    parsed_args: dict[str, Any],
    resolved_runtime_config: dict[str, Any],
    data_provenance: dict[str, Any],
    model_runtime_versions: dict[str, Any],
    training_selection_policy: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "finalized_at_utc": None,
        "status": "running",
        "run_dir": str(run_dir),
        "training_script": str(training_script),
        "process_id": os.getpid(),
        "hostname": socket.gethostname(),
        "raw_sys_argv": raw_sys_argv,
        "command_string": command_string([sys.executable, str(training_script), *raw_sys_argv]),
        "working_directory": str(working_directory),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "source_provenance": git_info(repo_root),
        "data_provenance": data_provenance,
        "parsed_args": parsed_args,
        "resolved_runtime_config": resolved_runtime_config,
        "model_runtime_versions": model_runtime_versions,
        "training_selection_policy": training_selection_policy,
        "finalization": {
            "completed_epochs": None,
            "final_checkpoint_path": None,
            "prediction_artifact_paths": [],
            "report_artifact_paths": [],
            "selected_checkpoint_path": None,
            "selected_checkpoint": None,
            "selected_clean_dev_metric_values": None,
            "selected_epoch": None,
            "total_runtime_seconds": None,
        },
    }
    return json_safe(record)


def runtime_versions(torch_module: Any, model: Any, architecture: str) -> dict[str, Any]:
    return {
        "causal_conv1d_version": package_version("causal-conv1d"),
        "cuda_runtime_reported_by_pytorch": getattr(getattr(torch_module, "version", None), "cuda", None),
        "imported_model_class": type(model).__name__ if model is not None else None,
        "mamba_ssm_version": package_version("mamba-ssm"),
        "numpy_version": np.__version__,
        "pandas_version": package_version("pandas"),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": getattr(torch_module, "__version__", None),
        "transformers_version": package_version("transformers"),
        "architecture_identifier": architecture,
    }
