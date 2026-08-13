#!/usr/bin/env python3
"""Dedicated P3-W6-F1 deterministic polarity-regeneration wrapper.

This script is an execution wrapper for the already-approved Python-level
repair API in scripts.build_controlled_v5. It intentionally does not expose a
dataset-version argument and does not implement an independent repair algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if not any(
    Path(entry).resolve() == _REPO_IMPORT_ROOT
    for entry in sys.path
    if entry
):
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from scripts import build_controlled_v5 as generator
from scripts import analyze_reason_router_p3w6f1_deterministic_polarity_regeneration as p3w6f1


WRAPPER_SOURCE_PATH = "scripts/regenerate_reason_router_p3w6f1_deterministic_polarity.py"
REPAIRED_GENERATOR_SOURCE_PATH = "scripts/build_controlled_v5.py"

BASE_FORM_AUTHORITY_COMMIT = "11102ea05b28f6638fdead205b4a9ee0f35ca0de"
BASE_FORM_AUTHORITY_PATH = "scripts/build_controlled_v5.py"
BASE_FORM_AUTHORITY_SYMBOL = "_BASE_PREDICATE_BY_INFLECTED"
BASE_FORM_AUTHORITY_SHA256 = "37e47a3ef60b26c7186d37367d59db158c28c6b9c9eb9e25a13927fc85810684"

EXPECTED_F1_TARGET_COUNT = 121
EXPECTED_OUTPUT_DIR_PREFIX = (
    "reports/reason_router_p2_p3w6f1_deterministic_polarity_regeneration_execution_"
)

REPAIRED_JSONL_NAME = "controlled_v5_v3_without_time_swap_p3w6f1_repaired.jsonl"
INVOCATION_JSON_NAME = "p3w6f1_deterministic_generator_invocation.json"
CONFIGURATION_JSON_NAME = "p3w6f1_generator_configuration_identity.json"
EXECUTION_MANIFEST_NAME = "p3w6f1_regeneration_execution_manifest.json"

EXPECTED_ARTIFACT_NAMES = {
    REPAIRED_JSONL_NAME,
    INVOCATION_JSON_NAME,
    CONFIGURATION_JSON_NAME,
    EXECUTION_MANIFEST_NAME,
}


class RegenerationWrapperError(RuntimeError):
    """Fail-closed wrapper rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegenerationWrapperError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def deterministic_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


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
        raise RegenerationWrapperError(f"git command failed: {' '.join(args)}") from exc


def git_object_bytes(repo_root: Path, commit: str, source_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{commit}:{source_path}"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RegenerationWrapperError(
            f"git object unavailable: {commit}:{source_path}"
        ) from exc


def current_head(repo_root: Path) -> str:
    return git_stdout(repo_root, ["rev-parse", "HEAD"])


def verify_git_repository(repo_root: Path) -> None:
    observed = Path(git_stdout(repo_root, ["rev-parse", "--show-toplevel"])).resolve()
    require(
        observed == repo_root.resolve(),
        f"repo root mismatch: expected {repo_root.resolve()}, observed {observed}",
    )


def tracked_worktree_clean(repo_root: Path) -> bool:
    unstaged = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", "--"],
        stderr=subprocess.DEVNULL,
    )
    staged = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--"],
        stderr=subprocess.DEVNULL,
    )
    return unstaged.returncode == 0 and staged.returncode == 0


def verify_execution_identity(
    repo_root: Path,
    f1_execution_commit: str,
    *,
    head_resolver: Any | None = None,
    tracked_clean_checker: Any | None = None,
    repo_checker: Any | None = None,
) -> None:
    require(is_full_commit(f1_execution_commit), "F1_EXECUTION_COMMIT_NOT_FULL_40_HEX")
    (repo_checker or verify_git_repository)(repo_root)
    observed_head = (head_resolver or current_head)(repo_root)
    require(observed_head == f1_execution_commit, "F1_EXECUTION_COMMIT_HEAD_MISMATCH")
    clean = (tracked_clean_checker or tracked_worktree_clean)(repo_root)
    require(clean is True, "TRACKED_WORKTREE_DIRTY")


def repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RegenerationWrapperError(f"path outside repo root: {path}") from exc
    return relative.as_posix()


def resolve_under_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def verify_output_dir(repo_root: Path, output_dir: Path, f1_execution_commit: str) -> Path:
    expected = (
        repo_root
        / f"{EXPECTED_OUTPUT_DIR_PREFIX}{f1_execution_commit}"
    ).resolve()
    observed = resolve_under_repo(repo_root, output_dir)
    require(observed == expected, "OUTPUT_DIR_CONTRACT_MISMATCH")
    return observed


def verify_base_form_authority(
    repo_root: Path,
    *,
    commit: str,
    source_path: str,
    symbol: str,
    source_sha256: str,
    git_object_reader: Any | None = None,
) -> dict[str, str]:
    require(commit == BASE_FORM_AUTHORITY_COMMIT, "BASE_FORM_AUTHORITY_COMMIT_MISMATCH")
    require(source_path == BASE_FORM_AUTHORITY_PATH, "BASE_FORM_AUTHORITY_PATH_MISMATCH")
    require(symbol == BASE_FORM_AUTHORITY_SYMBOL, "BASE_FORM_AUTHORITY_SYMBOL_MISMATCH")
    require(source_sha256 == BASE_FORM_AUTHORITY_SHA256, "BASE_FORM_AUTHORITY_SHA256_MISMATCH")
    blob = (git_object_reader or git_object_bytes)(repo_root, commit, source_path)
    observed_sha = sha256_bytes(blob)
    require(observed_sha == source_sha256, "BASE_FORM_AUTHORITY_GIT_OBJECT_SHA_MISMATCH")
    require(symbol.encode("utf-8") in blob, "BASE_FORM_AUTHORITY_SYMBOL_UNRESOLVED")
    return {
        "commit": commit,
        "path": source_path,
        "symbol": symbol,
        "raw_source_sha256": observed_sha,
    }


def verify_baseline_sha(path: Path, expected_sha256: str) -> None:
    require(file_sha256(path) == expected_sha256, "BASELINE_JSONL_SHA256_MISMATCH")


def verify_input_authorities(
    repo_root: Path,
    *,
    baseline_jsonl_path: Path,
    baseline_sidecar_jsonl_path: Path,
    p3w4_summary_json_path: Path,
    p3w4_pairs_jsonl_path: Path,
    p3w5_manifest_json_path: Path,
    f1_input_sha256: str,
) -> dict[str, Any]:
    try:
        return p3w6f1.validate_p3w6f1_input_authority_identity(
            repo_root=repo_root,
            baseline_jsonl_path=baseline_jsonl_path,
            baseline_sidecar_jsonl_path=baseline_sidecar_jsonl_path,
            p3w4_summary_json_path=p3w4_summary_json_path,
            p3w4_pairs_jsonl_path=p3w4_pairs_jsonl_path,
            p3w5_manifest_json_path=p3w5_manifest_json_path,
            f1_input_sha256=f1_input_sha256,
        )
    except Exception as exc:
        raise RegenerationWrapperError(f"INPUT_AUTHORITY_IDENTITY_FAILED: {exc}") from exc


def derive_authorized_f1_row_ids_from_authority(
    p3w4_summary: Mapping[str, Any],
    p3w4_pair_records: Sequence[Mapping[str, Any]],
    p3w5_manifest: Mapping[str, Any],
    *,
    expected_count: int = EXPECTED_F1_TARGET_COUNT,
) -> list[str]:
    try:
        supporting = p3w6f1.extract_decision_supporting_pair_ids(p3w4_summary, p3w5_manifest)
        targets = p3w6f1.extract_authorized_f1_targets(p3w4_pair_records, supporting)
    except Exception as exc:
        raise RegenerationWrapperError(f"AUTHORIZED_F1_DERIVATION_FAILED: {exc}") from exc
    row_ids = targets.get("authorized_F1_row_ids")
    require(isinstance(row_ids, list), "AUTHORIZED_F1_ROW_IDS_MALFORMED")
    require(len(row_ids) == len(set(row_ids)), "AUTHORIZED_F1_ROW_IDS_DUPLICATE")
    require(len(row_ids) == expected_count, "AUTHORIZED_F1_ROW_IDS_CARDINALITY_MISMATCH")
    f2_row_ids = p3w6f1.extract_f2_row_ids(p3w4_pair_records)
    require(not (set(row_ids) & f2_row_ids), "AUTHORIZED_F1_ROW_IDS_INCLUDE_F2")
    return sorted(str(row_id) for row_id in row_ids)


def derive_authorized_f1_row_ids(
    p3w4_summary_path: Path,
    p3w4_pairs_path: Path,
    p3w5_manifest_path: Path,
) -> list[str]:
    p3w4_summary = p3w6f1.load_json(p3w4_summary_path)
    p3w4_pair_records = p3w6f1.load_jsonl(p3w4_pairs_path)
    p3w5_manifest = p3w6f1.load_json(p3w5_manifest_path)
    return derive_authorized_f1_row_ids_from_authority(
        p3w4_summary,
        p3w4_pair_records,
        p3w5_manifest,
    )


def extract_f2_row_ids_from_authority(p3w4_pairs_path: Path) -> set[str]:
    return p3w6f1.extract_f2_row_ids(p3w6f1.load_jsonl(p3w4_pairs_path))


def build_repaired_payload(
    baseline_rows: Sequence[Mapping[str, Any]],
    authorized_f1_row_ids: Sequence[str],
) -> dict[str, Any]:
    pair_count = p3w6f1.baseline_pair_count(baseline_rows)
    replay_rows, audit = generator.build_controlled_records_with_f1_polarity_repair_audit(
        pair_count,
        set(authorized_f1_row_ids),
    )
    projected_rows = p3w6f1.project_replay_to_baseline_topology(replay_rows, baseline_rows)
    consumed = sorted(str(row_id) for row_id in audit.get("repair_consumed_row_ids", []))
    baseline_ids = [str(row.get("id", "")) for row in baseline_rows]
    invocation_identity = {
        "pair_count": pair_count,
        "authorized_F1_row_ids_sha256": p3w6f1.canonical_sha256(sorted(authorized_f1_row_ids)),
        "repair_api": "build_controlled_records_with_f1_polarity_repair_audit",
        "baseline_id_sequence_sha256": p3w6f1.id_sequence_sha256(baseline_ids),
        "projection_policy": "baseline_id_sequence",
        "repair_mode": "f1_authorized_polarity_negative_only",
    }
    configuration_identity = {
        "generator_source_path": REPAIRED_GENERATOR_SOURCE_PATH,
        "pair_count": pair_count,
        "authorized_F1_row_count": len(set(authorized_f1_row_ids)),
        "structural_negative_polarity_flip_row_count": len(
            p3w6f1.structural_negative_polarity_flip_row_ids_for_pair_count(pair_count)
        ),
        "baseline_topology_row_count": len(baseline_rows),
        "baseline_id_sequence_sha256": invocation_identity["baseline_id_sequence_sha256"],
    }
    return {
        "pair_count": pair_count,
        "repaired_rows": projected_rows,
        "repair_consumed_row_ids": consumed,
        "deterministic_generator_invocation": invocation_identity,
        "generator_configuration_identity": configuration_identity,
    }


def validate_wrapper_isolation(
    baseline_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    *,
    authorized_f1_row_ids: Iterable[str],
    f2_row_ids: Iterable[str],
    repair_consumed_row_ids: Iterable[str],
    deterministic_generator_invocation: Mapping[str, Any],
    generator_configuration_identity: Mapping[str, Any],
    f1_execution_commit: str,
    repaired_generator_source_sha256: str,
) -> dict[str, Any]:
    pair_count = p3w6f1.baseline_pair_count(baseline_rows)
    isolation = p3w6f1.full_output_isolation(
        baseline_rows,
        repaired_rows,
        authorized_f1_row_ids=authorized_f1_row_ids,
        structural_negative_polarity_flip_row_ids=(
            p3w6f1.structural_negative_polarity_flip_row_ids_for_pair_count(pair_count)
        ),
        repair_consumed_row_ids=repair_consumed_row_ids,
        f2_row_ids=f2_row_ids,
        repaired_generator_commit=f1_execution_commit,
        repaired_generator_source_path=REPAIRED_GENERATOR_SOURCE_PATH,
        repaired_generator_source_sha256=repaired_generator_source_sha256,
        deterministic_generator_invocation=deterministic_generator_invocation,
        generator_configuration_identity=generator_configuration_identity,
    )
    validation = p3w6f1.validate_full_output_isolation(isolation)
    require(
        validation.get("full_output_isolation_pass") is True,
        f"FULL_OUTPUT_ISOLATION_FAILED: {validation.get('full_output_isolation_failures')}",
    )
    baseline_by_id = p3w6f1.row_map(baseline_rows)
    repaired_by_id = p3w6f1.row_map(repaired_rows)
    for row_id in isolation["changed_ids"]:
        changed = p3w6f1.changed_fields(baseline_by_id[row_id], repaired_by_id[row_id])
        require(changed == ["evidence"], f"AUTHORIZED_ROW_FIELD_SCOPE_MISMATCH: {row_id}")
    return {"isolation": isolation, "validation": validation}


def build_execution_manifest(
    *,
    repo_root: Path,
    f1_execution_commit: str,
    baseline_jsonl_path: Path,
    baseline_jsonl_sha256: str,
    output_dir: Path,
    repaired_output_sha256: str,
    repaired_generator_source_sha256: str,
    base_form_authority: Mapping[str, str],
) -> dict[str, Any]:
    repaired_path = output_dir / REPAIRED_JSONL_NAME
    invocation_path = output_dir / INVOCATION_JSON_NAME
    configuration_path = output_dir / CONFIGURATION_JSON_NAME
    return {
        "F1_execution_commit": f1_execution_commit,
        "baseline_input_path": repo_relative_path(repo_root, baseline_jsonl_path),
        "baseline_input_sha256": baseline_jsonl_sha256,
        "repaired_output_path": repo_relative_path(repo_root, repaired_path),
        "repaired_output_sha256": repaired_output_sha256,
        "repaired_generator_source_path": REPAIRED_GENERATOR_SOURCE_PATH,
        "repaired_generator_source_sha256": repaired_generator_source_sha256,
        "base_form_authority_commit": base_form_authority["commit"],
        "base_form_authority_path": base_form_authority["path"],
        "base_form_authority_symbol": base_form_authority["symbol"],
        "base_form_authority_sha256": base_form_authority["raw_source_sha256"],
        "deterministic_generator_invocation_json": repo_relative_path(repo_root, invocation_path),
        "generator_configuration_identity_json": repo_relative_path(repo_root, configuration_path),
        "wrapper_source_path": WRAPPER_SOURCE_PATH,
    }


def ensure_no_conflicting_outputs(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    if not output_dir.exists():
        return
    require(output_dir.is_dir(), "OUTPUT_PATH_EXISTS_NOT_DIRECTORY")
    observed = {entry.name for entry in output_dir.iterdir()}
    unexpected = sorted(observed - EXPECTED_ARTIFACT_NAMES)
    require(not unexpected, f"OUTPUT_DIR_CONTAINS_UNEXPECTED_ARTIFACTS: {unexpected}")
    for name, payload in payloads.items():
        path = output_dir / name
        if path.exists():
            require(path.is_file(), f"OUTPUT_ARTIFACT_PATH_NOT_FILE: {name}")
            require(path.read_bytes() == payload, f"OUTPUT_ARTIFACT_CONFLICT: {name}")


def write_output_artifacts(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    ensure_no_conflicting_outputs(output_dir, payloads)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        final_path = output_dir / name
        tmp_path = output_dir / f".{name}.tmp"
        tmp_path.write_bytes(payload)
        tmp_path.replace(final_path)
    final_names = {entry.name for entry in output_dir.iterdir()}
    require(final_names == EXPECTED_ARTIFACT_NAMES, "OUTPUT_ARTIFACT_SET_MISMATCH")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3-W6-F1 deterministic polarity regeneration wrapper"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-jsonl-sha256", required=True)
    parser.add_argument("--p3w4-summary-json", type=Path, required=True)
    parser.add_argument("--p3w4-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--p3w5-manifest-json", type=Path, required=True)
    parser.add_argument("--baseline-sidecar-jsonl", type=Path, required=True)
    parser.add_argument("--base-form-authority-commit", required=True)
    parser.add_argument("--base-form-authority-path", required=True)
    parser.add_argument("--base-form-authority-symbol", required=True)
    parser.add_argument("--base-form-authority-sha256", required=True)
    parser.add_argument("--f1-execution-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    baseline_jsonl_path = resolve_under_repo(repo_root, args.baseline_jsonl)
    p3w4_summary_path = resolve_under_repo(repo_root, args.p3w4_summary_json)
    p3w4_pairs_path = resolve_under_repo(repo_root, args.p3w4_pairs_jsonl)
    p3w5_manifest_path = resolve_under_repo(repo_root, args.p3w5_manifest_json)
    baseline_sidecar_path = resolve_under_repo(repo_root, args.baseline_sidecar_jsonl)

    verify_execution_identity(repo_root, args.f1_execution_commit)
    output_dir = verify_output_dir(repo_root, args.output_dir, args.f1_execution_commit)
    base_form_authority = verify_base_form_authority(
        repo_root,
        commit=args.base_form_authority_commit,
        source_path=args.base_form_authority_path,
        symbol=args.base_form_authority_symbol,
        source_sha256=args.base_form_authority_sha256,
    )
    verify_baseline_sha(baseline_jsonl_path, args.baseline_jsonl_sha256)
    verify_input_authorities(
        repo_root,
        baseline_jsonl_path=baseline_jsonl_path,
        baseline_sidecar_jsonl_path=baseline_sidecar_path,
        p3w4_summary_json_path=p3w4_summary_path,
        p3w4_pairs_jsonl_path=p3w4_pairs_path,
        p3w5_manifest_json_path=p3w5_manifest_path,
        f1_input_sha256=args.baseline_jsonl_sha256,
    )

    baseline_rows = p3w6f1.load_jsonl(baseline_jsonl_path)
    baseline_ids = [str(row.get("id", "")) for row in baseline_rows]
    require(len(baseline_ids) == len(set(baseline_ids)), "BASELINE_DUPLICATE_IDS")

    authorized_f1_row_ids = derive_authorized_f1_row_ids(
        p3w4_summary_path,
        p3w4_pairs_path,
        p3w5_manifest_path,
    )
    f2_row_ids = extract_f2_row_ids_from_authority(p3w4_pairs_path)
    require(not (set(authorized_f1_row_ids) & f2_row_ids), "AUTHORIZED_F1_ROW_IDS_INCLUDE_F2")

    payload = build_repaired_payload(baseline_rows, authorized_f1_row_ids)
    repaired_rows = payload["repaired_rows"]
    repaired_generator_source_sha256 = sha256_bytes(
        git_object_bytes(repo_root, args.f1_execution_commit, REPAIRED_GENERATOR_SOURCE_PATH)
    )

    validate_wrapper_isolation(
        baseline_rows,
        repaired_rows,
        authorized_f1_row_ids=authorized_f1_row_ids,
        f2_row_ids=f2_row_ids,
        repair_consumed_row_ids=payload["repair_consumed_row_ids"],
        deterministic_generator_invocation=payload["deterministic_generator_invocation"],
        generator_configuration_identity=payload["generator_configuration_identity"],
        f1_execution_commit=args.f1_execution_commit,
        repaired_generator_source_sha256=repaired_generator_source_sha256,
    )

    repaired_jsonl_bytes = deterministic_jsonl_bytes(repaired_rows)
    repaired_output_sha256 = sha256_bytes(repaired_jsonl_bytes)
    manifest = build_execution_manifest(
        repo_root=repo_root,
        f1_execution_commit=args.f1_execution_commit,
        baseline_jsonl_path=baseline_jsonl_path,
        baseline_jsonl_sha256=args.baseline_jsonl_sha256,
        output_dir=output_dir,
        repaired_output_sha256=repaired_output_sha256,
        repaired_generator_source_sha256=repaired_generator_source_sha256,
        base_form_authority=base_form_authority,
    )
    output_payloads = {
        REPAIRED_JSONL_NAME: repaired_jsonl_bytes,
        INVOCATION_JSON_NAME: deterministic_json_bytes(payload["deterministic_generator_invocation"]),
        CONFIGURATION_JSON_NAME: deterministic_json_bytes(payload["generator_configuration_identity"]),
        EXECUTION_MANIFEST_NAME: deterministic_json_bytes(manifest),
    }
    write_output_artifacts(output_dir, output_payloads)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run(args)
    except RegenerationWrapperError as exc:
        raise SystemExit(f"P3W6F1_REGENERATION_WRAPPER_FAILED_CLOSED: {exc}") from exc
    print(json.dumps({"status": "PASS", "execution_manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
