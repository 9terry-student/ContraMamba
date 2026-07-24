"""Execute Stage196-B2-B6P9-P3 separate observational runs sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


RUN_IDS = [
    "control_off_none",
    "previous_step_direction",
    "previous_step_candidate_order",
    "previous_epoch_direction",
    "previous_epoch_candidate_order",
    "ema_direction",
    "ema_candidate_order",
]
SIDECARS = [
    "teacher_observer_manifest.json",
    "teacher_observer_batch_metrics.jsonl",
    "teacher_observer_epoch_metrics.csv",
    "teacher_observer_run_summary.json",
    "teacher_observer_state_audit.json",
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != "STAGE196B2B6P9P3P0_MANIFEST_READY":
        raise SystemExit("manifest decision is not STAGE196B2B6P9P3P0_MANIFEST_READY")
    rows = payload.get("run_table")
    if not isinstance(rows, list) or [row.get("run_id") for row in rows] != RUN_IDS:
        raise SystemExit("manifest does not contain the exact seven rows in order")
    return payload


def args_to_cli(args: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key in sorted(args):
        value = args[key]
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])
    return argv


def validate_row(row: dict[str, Any]) -> None:
    run_id = row.get("run_id")
    mode = row.get("observer_mode")
    family = row.get("target_family")
    if run_id not in RUN_IDS:
        raise ValueError(f"unknown run_id {run_id!r}")
    if mode == "off":
        if family != "none" or row.get("ema_decay") is not None:
            raise ValueError("control run must use mode off, target none, and no EMA decay")
        if row.get("expected_runtime_sidecars") != []:
            raise ValueError("control run must expect zero observer sidecars")
    else:
        if family not in {"direction", "candidate_order"}:
            raise ValueError("enabled runs require exactly one target family")
        if row.get("expected_runtime_sidecars") != SIDECARS:
            raise ValueError("enabled runs must expect exactly five P9-P2 sidecars")
        if mode == "ema" and row.get("ema_decay") != 0.99:
            raise ValueError("EMA rows require frozen decay 0.99")
        if mode != "ema" and row.get("ema_decay") is not None:
            raise ValueError("non-EMA rows must not carry EMA decay")
    if row.get("seed") != 183:
        raise ValueError("all rows require seed 183")


def output_hashes(row: dict[str, Any]) -> dict[str, str]:
    paths = [Path(row["checkpoint_path"])]
    paths.extend(Path(row["observer_output_dir"]) / name for name in row.get("expected_runtime_sidecars", []))
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            hashes[str(path)] = file_sha256(path)
    return hashes


def completed_compatible(row: dict[str, Any], command_fp: str) -> bool:
    status_path = Path(row["output_dir"]) / "execution_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("success") or status.get("returncode") != 0:
        return False
    if status.get("resolved_command_fingerprint") != command_fp:
        return False
    required = [Path(row["checkpoint_path"])]
    required.extend(Path(row["observer_output_dir"]) / name for name in row.get("expected_runtime_sidecars", []))
    if any(not path.is_file() for path in required):
        return False
    return output_hashes(row) == status.get("completion_output_hashes")


def validate_outputs(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint.is_file():
        reasons.append("checkpoint_missing")
    observer_dir = Path(row["observer_output_dir"]) if row.get("observer_output_dir") else None
    expected = row.get("expected_runtime_sidecars", [])
    if row["observer_mode"] == "off":
        if observer_dir is not None and observer_dir.exists() and any(observer_dir.iterdir()):
            reasons.append("control_observer_sidecars_present")
    else:
        observed = sorted(p.name for p in observer_dir.iterdir() if p.is_file()) if observer_dir and observer_dir.exists() else []
        if observed != sorted(expected):
            reasons.append(f"enabled_sidecar_set_mismatch={observed}")
    return not reasons, reasons


def run_one(row: dict[str, Any], repo_root: Path, python_executable: str) -> int:
    validate_row(row)
    run_dir = Path(row["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str((repo_root / row["trainer_script"]).resolve()),
        *args_to_cli(row["trainer_args"]),
    ]
    command_record = {
        "run_id": row["run_id"],
        "command": command,
        "config_fingerprint": row["config_fingerprint"],
        "manifest_command_fingerprint": row["config_fingerprint"],
    }
    command_fp = sha256_json(command_record)
    command_record["resolved_command_fingerprint"] = command_fp
    resolved_path = run_dir / "resolved_command.json"
    if resolved_path.is_file():
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous.get("resolved_command_fingerprint") != command_fp:
            raise SystemExit(f"refusing to overwrite incompatible completed or partial run: {row['run_id']}")
    if completed_compatible(row, command_fp):
        return 0
    resolved_path.write_text(json.dumps(command_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stdout_path = Path(row["stdout_log"])
    stderr_path = Path(row["stderr_log"])
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(command, cwd=str(repo_root), stdout=stdout, stderr=stderr, check=False)
    outputs_ok, output_reasons = validate_outputs(row)
    success = proc.returncode == 0 and outputs_ok
    status = {
        "run_id": row["run_id"],
        "success": success,
        "returncode": proc.returncode,
        "resolved_command_fingerprint": command_fp,
        "config_fingerprint": row["config_fingerprint"],
        "output_validation_passed": outputs_ok,
        "output_validation_reasons": output_reasons,
        "completion_output_hashes": output_hashes(row) if success else {},
    }
    (run_dir / "execution_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return proc.returncode if proc.returncode != 0 else (0 if outputs_ok else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest_json)
    rows = manifest["run_table"]
    if args.run_id:
        rows = [row for row in rows if row["run_id"] == args.run_id]
        if len(rows) != 1:
            raise SystemExit(f"--run-id did not match exactly one row: {args.run_id}")
    rc = 0
    for row in rows:
        rc = run_one(row, args.repo_root.resolve(), args.python_executable)
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
