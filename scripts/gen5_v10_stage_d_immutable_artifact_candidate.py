#!/usr/bin/env python3
"""Stage D CPU-only, untrusted V3/failure artifact writer and read-back validator.

Never runs a model or GPU. SHA256 here is a corruption check, NOT provenance
attestation. Existing frozen V3 verifier owns all numeric verdicts and remains
unchanged. No experiment CLI or production execution entrypoint is exposed.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
from scripts import gen5_v10_bounded_worker_export_candidate as stage_b

SCHEMA = "GEN5_V10_BOUNDED_ARTIFACT_DIRECTORY_D1"
FAILURE_SCHEMA = "GEN5_V10_BOUNDED_FAILURE_ENVELOPE_D1"
BASE = "869b92b1f3b17f97f4d844a20a7c1388f24cb10f"
V3_BASE = v3.BASE
MAX_FILE_BYTES = 64 * 1024 * 1024
MIN_FREE_BYTES = 64 * 1024 * 1024
KINDS = ("V3_UNAUTHENTICATED", "FAILURE_ONLY")
FAIL_STAGES = ("PRESELECT", "INPUT_CAPTURE", "FROZEN", "V10", "ARTIFACT")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}\Z")
SAFE_REASON = re.compile(r"[A-Z][A-Z0-9_]{0,95}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactBlocked(ValueError):
    pass


def require(cond: bool, reason: str) -> None:
    if not cond:
        raise ArtifactBlocked("GEN5_V10_D1_ARTIFACT_BLOCKED:" + reason)


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactBlocked("NONCANONICAL_JSON") from exc


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, "DUPLICATE_JSON_KEY:" + key)
        output[key] = value
    return output


def _read_json(path: Path) -> tuple[Any, bytes]:
    require(path.is_file() and not path.is_symlink(), "MISSING_OR_LINKED_FILE:" + path.name)
    require(0 < path.stat().st_size <= MAX_FILE_BYTES, "FILE_SIZE:" + path.name)
    data = path.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs_unique,
                         parse_constant=lambda x: (_ for _ in ()).throw(ArtifactBlocked("JSON_NONFINITE:" + x)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactBlocked("INVALID_JSON:" + path.name) from exc
    require(data == _json_bytes(obj), "NONCANONICAL_ON_DISK:" + path.name)
    return obj, data


def _safe_destination(dest: Path, *, repo: Path | None) -> tuple[Path, Path]:
    require(isinstance(dest, Path) and SAFE_NAME.fullmatch(dest.name) is not None,
            "ARTIFACT_ROOT_NAME")
    require(dest.name not in (".", "..") and not dest.exists() and not dest.is_symlink(),
            "ARTIFACT_ROOT_COLLISION")
    parent = dest.parent
    require(parent.is_dir() and not parent.is_symlink(), "PARENT_NOT_DIRECTORY_OR_SYMLINK")
    canonical_parent = parent.resolve(strict=True)
    require(parent.absolute() == canonical_parent, "PARENT_SYMLINK_PATH")
    target = canonical_parent / dest.name
    require(not target.exists() and not target.is_symlink(), "CANONICAL_TARGET_COLLISION")
    for root in (Path(__file__).resolve().parents[1], repo):
        if root is None:
            continue
        base = root.resolve(strict=True)
        require(target != base and base not in target.parents,
                "ARTIFACT_INSIDE_CODE_WORKTREE")
    require(shutil.disk_usage(canonical_parent).free >= MIN_FREE_BYTES,
            "FREE_DISK_BUDGET")
    return canonical_parent, target


def _validate_failure(obj: Any) -> dict[str, Any]:
    require(type(obj) is dict and set(obj) == {
        "schema_version", "candidate_base", "phase", "reason_code", "message",
        "selector", "source_kind", "evidence_status", "V10_E0_PROVENANCE",
        "SCIENTIFIC_EXECUTION", "PRODUCTION",
    }, "FAILURE_FIELDS")
    require(obj["schema_version"] == FAILURE_SCHEMA and obj["candidate_base"] == V3_BASE,
            "FAILURE_SCHEMA_BASE")
    require(type(obj["phase"]) is str and obj["phase"] in FAIL_STAGES,
            "FAILURE_PHASE")
    require(type(obj["reason_code"]) is str and SAFE_REASON.fullmatch(obj["reason_code"]) is not None,
            "FAILURE_REASON")
    require(type(obj["message"]) is str and len(obj["message"]) <= 1024 and
            all((ord(ch) >= 32 and ord(ch) != 127) for ch in obj["message"]),
            "FAILURE_MESSAGE")
    require(obj["source_kind"] in ("CPU_MOCK_ONLY", "RUNTIME_CLAIM_UNAUTHENTICATED"),
            "FAILURE_SOURCE_KIND")
    require(obj["evidence_status"] == "FAILURE_ONLY_NO_COMPLETE_BUNDLE" and
            obj["V10_E0_PROVENANCE"] == "NOT_EVALUATED" and
            obj["SCIENTIFIC_EXECUTION"] == "NOT_AUTHORIZED" and
            obj["PRODUCTION"] == "BLOCKED", "FAILURE_FALSE_SUCCESS_CLAIM")
    selector = obj["selector"]
    if selector is None:
        require(obj["phase"] == "PRESELECT", "MISSING_SELECTOR_NONPRESELECT")
    else:
        require(type(selector) is dict and type(selector.get("row_ids")) is list,
                "FAILURE_SELECTOR_STRUCTURE")
        try:
            actual = stage_b.choose_bounded_locus(
                worker_id=selector.get("worker_id"),
                batch_index=selector.get("batch_index_one_based"),
                row_start=selector.get("row_start"), row_stop=selector.get("row_stop"),
                row_ids=selector["row_ids"], source_cells=stage_b.SOURCE_CELLS,
                orientations=stage_b.FROZEN_ORIENTATIONS,
                selected_source=stage_b.SOURCE_CELLS[0],
                selected_targets=(stage_b.SOURCE_CELLS[1], stage_b.SOURCE_CELLS[2]),
                target_group_index=selector.get("target_group_index"))
        except stage_b.StageBBlocked as exc:
            raise ArtifactBlocked("INVALID_FAILURE_SELECTOR") from exc
        require(selector == actual, "FAILURE_SELECTOR_IDENTITY")
    return {"V10_E0_PROVENANCE": "NOT_EVALUATED", "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
            "V10_E2_PAYLOAD": "NOT_EVALUATED", "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
            "PRODUCTION": "BLOCKED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
            "failure_reason": obj["reason_code"]}


def failure_envelope(*, stage: str, reason_code: str, message: str,
                     selector: dict[str, Any] | None = None,
                     source_kind: str = "CPU_MOCK_ONLY") -> dict[str, Any]:
    value = {"schema_version": FAILURE_SCHEMA, "candidate_base": V3_BASE,
             "phase": stage, "reason_code": reason_code, "message": message,
             "selector": selector, "source_kind": source_kind,
             "evidence_status": "FAILURE_ONLY_NO_COMPLETE_BUNDLE",
             "V10_E0_PROVENANCE": "NOT_EVALUATED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
             "PRODUCTION": "BLOCKED"}
    _validate_failure(value)
    return value


def _write_new_file(path: Path, data: bytes) -> None:
    # A staging directory cannot be promoted until all bytes are durable and
    # independently read back. No file inside a completed artifact is edited.
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _sidecar(name: str, data: bytes) -> bytes:
    return f"{_sha(data)}  {name}\n".encode("ascii")


def _publish_no_replace(staging: Path, target: Path) -> None:
    """Atomically publish a directory without ever replacing an existing one.

    Windows os.rename has no-replace semantics. Linux needs renameat2 with
    RENAME_NOREPLACE; ordinary POSIX os.rename can replace an empty directory
    created after the preceding existence check, so it is forbidden here.
    Other platforms fail closed until an equivalent primitive is audited.
    """
    if os.name == "nt":
        try:
            os.rename(staging, target)
        except FileExistsError as exc:
            raise ArtifactBlocked("TARGET_COLLISION_AT_PUBLISH") from exc
        return
    require(sys.platform.startswith("linux"), "ATOMIC_NOREPLACE_UNSUPPORTED_PLATFORM")
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "ATOMIC_NOREPLACE_UNAVAILABLE")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    # AT_FDCWD=-100 and RENAME_NOREPLACE=1 are Linux syscall constants.
    result = renameat2(-100, os.fsencode(staging), -100,
                       os.fsencode(target), 1)
    if result != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise ArtifactBlocked("TARGET_COLLISION_AT_PUBLISH")
        if code in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise ArtifactBlocked("ATOMIC_NOREPLACE_UNAVAILABLE")
        raise OSError(code, os.strerror(code), str(target))


def _write_untrusted(kind: str, evidence: Any, dest: Path,
                     *, repo: Path | None = None) -> dict[str, Any]:
    require(kind in KINDS, "PACKAGE_KIND")
    if kind == "V3_UNAUTHENTICATED":
        # V3 itself replays K4 per-gap and diagnostics. Never manufacture a
        # sealed bundle, upgrade a verdict, or change the submitted object.
        from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as verify
        verdict = verify.validate(evidence)
        require(verdict["V10_E0_PROVENANCE"] == "NOT_EVALUATED" and
                verdict["V10_E3_DIAGNOSTICS"] == "NOT_EVALUATED" and
                verdict["PRODUCTION"] == "BLOCKED", "UNEXPECTED_V3_PROMOTION")
        basefiles = {"bundle.v3.json": _json_bytes(evidence),
                     "verification.v3.json": _json_bytes(verdict)}
    else:
        verdict = _validate_failure(evidence)
        basefiles = {"failure.d1.json": _json_bytes(evidence)}
    require(all(len(raw) <= MAX_FILE_BYTES for raw in basefiles.values()),
            "PAYLOAD_FILE_TOO_LARGE")
    manifest = {"schema_version": SCHEMA, "writer_base": BASE,
                "kind": kind, "evidence_status": "UNAUTHENTICATED_NO_PROMOTION",
                "file_sha256": {name: _sha(raw) for name, raw in sorted(basefiles.items())},
                "V10_E0_PROVENANCE": "NOT_EVALUATED", "PRODUCTION": "BLOCKED",
                "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED"}
    manifest_bytes = _json_bytes(manifest)
    parent, target = _safe_destination(dest, repo=repo)
    claim = parent / ("." + target.name + ".reserved")
    require(not claim.exists() and not claim.is_symlink(), "RESERVATION_COLLISION")
    tmp = None
    # Exclusive sibling reservation; no preexisting target is overwritten.
    try:
        with claim.open("x", encoding="ascii") as handle:
            handle.write("STAGE_D_IMMUTABLE_ARTIFACT_RESERVATION\n")
        tmp = Path(tempfile.mkdtemp(prefix="." + target.name + ".writing-", dir=parent))
        for name, raw in basefiles.items():
            _write_new_file(tmp / name, raw)
            _write_new_file(tmp / (name + ".sha256"), _sidecar(name, raw))
        _write_new_file(tmp / "artifact_manifest.d1.json", manifest_bytes)
        _write_new_file(tmp / "artifact_manifest.d1.json.sha256",
                        _sidecar("artifact_manifest.d1.json", manifest_bytes))
        # A failed or truncated staging directory is never renamed into place.
        verify_artifact(tmp)
        require(not target.exists() and not target.is_symlink(), "TARGET_COLLISION_BEFORE_PUBLISH")
        _publish_no_replace(tmp, target)
        tmp = None
    finally:
        if tmp is not None:
            shutil.rmtree(tmp)  # Only own unpromoted temporary directory.
        if claim.exists() and claim.read_text(encoding="ascii") == "STAGE_D_IMMUTABLE_ARTIFACT_RESERVATION\n":
            claim.unlink()  # Only own reservation file.
    return {"status": "UNAUTHENTICATED_ARTIFACT_WRITTEN_NOT_PROMOTED",
            "root": str(target), "kind": kind, "manifest_file_sha256": _sha(manifest_bytes),
            "verdict": verdict}


def write_v3_bundle(bundle: Any, dest: Path, *, repo: Path | None = None) -> dict[str, Any]:
    return _write_untrusted("V3_UNAUTHENTICATED", bundle, dest, repo=repo)


def write_failure(envelope: Any, dest: Path, *, repo: Path | None = None) -> dict[str, Any]:
    return _write_untrusted("FAILURE_ONLY", envelope, dest, repo=repo)


def verify_artifact(directory: Path) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "ROOT_MISSING_OR_SYMLINK")
    manifest, manifest_raw = _read_json(directory / "artifact_manifest.d1.json")
    require(type(manifest) is dict and set(manifest) == {
        "schema_version", "writer_base", "kind", "evidence_status", "file_sha256",
        "V10_E0_PROVENANCE", "PRODUCTION", "SCIENTIFIC_EXECUTION"}, "MANIFEST_FIELDS")
    require(manifest["schema_version"] == SCHEMA and manifest["writer_base"] == BASE and
            manifest["kind"] in KINDS and manifest["evidence_status"] == "UNAUTHENTICATED_NO_PROMOTION" and
            manifest["V10_E0_PROVENANCE"] == "NOT_EVALUATED" and
            manifest["PRODUCTION"] == "BLOCKED" and manifest["SCIENTIFIC_EXECUTION"] == "NOT_AUTHORIZED",
            "MANIFEST_FALSE_STATUS")
    files = ("bundle.v3.json", "verification.v3.json") if manifest["kind"] == KINDS[0] else ("failure.d1.json",)
    require(type(manifest["file_sha256"]) is dict and set(manifest["file_sha256"]) == set(files) and
            all(type(v) is str and SHA.fullmatch(v) for v in manifest["file_sha256"].values()),
            "MANIFEST_FILE_HASH_SET")
    names = set(files) | {name + ".sha256" for name in files} | {
        "artifact_manifest.d1.json", "artifact_manifest.d1.json.sha256"}
    require({p.name for p in directory.iterdir()} == names, "EXTRA_OR_MISSING_ARTIFACT_FILE")
    def check_sha(name: str, raw: bytes) -> None:
        sidecar = directory / (name + ".sha256")
        require(sidecar.is_file() and not sidecar.is_symlink() and
                sidecar.read_bytes() == _sidecar(name, raw), "PHYSICAL_SHA256:" + name)
    check_sha("artifact_manifest.d1.json", manifest_raw)
    records: dict[str, Any] = {}
    for name in files:
        obj, raw = _read_json(directory / name)
        require(_sha(raw) == manifest["file_sha256"][name], "MANIFEST_FILE_TAMPER:" + name)
        check_sha(name, raw)
        records[name] = obj
    if manifest["kind"] == KINDS[0]:
        report = v3.validate(records["bundle.v3.json"])
        require(report == records["verification.v3.json"], "VERIFICATION_REPORT_MISMATCH")
    else:
        report = _validate_failure(records["failure.d1.json"])
    return {"status": "OFFLINE_INTEGRITY_ONLY_NO_PROVENANCE_ATTESTATION",
            "kind": manifest["kind"], "manifest_file_sha256": _sha(manifest_raw),
            "verdict": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check", action="store_true", required=True)
    args = parser.parse_args()
    require(args.static_contract_check, "STATIC_ONLY")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "CUDA_MUST_BE_DISABLED")
    print("GEN5_V10_STAGE_D_CPU_ARTIFACT_STATIC_CONTRACT_PASS")
    print("WRITER_BASE=" + BASE + ";V3_INPUT_SCHEMA=" + v3.SCHEMA)
    print("GPU=OFF;MODEL_LOAD=NO;V10_E0_PROVENANCE=NOT_EVALUATED;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
