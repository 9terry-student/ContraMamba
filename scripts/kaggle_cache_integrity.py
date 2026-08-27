"""Read-only integrity manifests for a local Hugging Face model cache.

This module validates bytes and cache layout only.  It does not prove model
completeness or usability; those checks remain the responsibility of the
environment verifier.  No network access or cache mutation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
MARKER = "PRE_URP_KAGGLE_HF_CACHE_INTEGRITY_MANIFEST"
EVIDENCE_BOUNDARY = "NOT_SCIENTIFIC_EVIDENCE"
DEFAULT_MODEL_ID = "state-spaces/mamba-130m-hf"


class CacheIntegrityError(RuntimeError):
    """Raised when cache or manifest integrity cannot be established."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CacheIntegrityError(f"malformed {label}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PurePosixPath(value).parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CacheIntegrityError(f"unsafe {label}: {value!r}")
    return value


def _inside(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CacheIntegrityError(f"{label} escapes or is unavailable") from exc
    return resolved


def _model_family(hub_cache: Path, model_id: str) -> tuple[Path, str]:
    if not isinstance(model_id, str) or model_id.count("/") != 1:
        raise CacheIntegrityError("model_id must be an organization/model pair")
    family_name = "models--" + model_id.replace("/", "--")
    family = (hub_cache.resolve() / family_name)
    if not family.is_dir():
        raise CacheIntegrityError(f"model cache family missing: {family_name}")
    return family, family_name


def _revision_name(value: str) -> str:
    return _safe_relative(value, "revision")


def _available_revisions(family: Path) -> list[str]:
    snapshots = family / "snapshots"
    if not snapshots.is_dir():
        raise CacheIntegrityError("snapshots directory missing")
    revisions = sorted(entry.name for entry in snapshots.iterdir() if entry.is_dir() and not entry.is_symlink())
    if not revisions:
        raise CacheIntegrityError("no usable snapshots found")
    return revisions


def select_revision(family: Path, revision: str | None) -> str:
    revisions = _available_revisions(family)
    if revision is not None:
        selected = _revision_name(revision)
        if selected not in revisions:
            raise CacheIntegrityError(f"requested revision is not a usable snapshot: {selected}")
        return selected
    refs_main = family / "refs" / "main"
    if refs_main.is_file() and not refs_main.is_symlink():
        target = refs_main.read_text(encoding="utf-8").strip()
        if target in revisions and target and "\n" not in target:
            return target
        raise CacheIntegrityError("refs/main does not resolve to one usable snapshot")
    if len(revisions) == 1:
        return revisions[0]
    raise CacheIntegrityError("snapshot revision is ambiguous; supply --revision")


def _snapshot_files(snapshot: Path, family: Path) -> list[dict[str, Any]]:
    snapshot = _inside(snapshot, family, "snapshot")
    if not snapshot.is_dir():
        raise CacheIntegrityError("snapshot is not a directory")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root, dirs, names in os.walk(snapshot, topdown=True, followlinks=False):
        root_path = Path(root)
        for directory in list(dirs):
            candidate = root_path / directory
            if candidate.is_symlink():
                raise CacheIntegrityError(f"symlinked snapshot directory rejected: {directory}")
        for name in names:
            logical = (root_path / name).relative_to(snapshot).as_posix()
            _safe_relative(logical, "logical_path")
            if logical in seen:
                raise CacheIntegrityError(f"duplicate logical path: {logical}")
            seen.add(logical)
            candidate = root_path / name
            try:
                resolved = _inside(candidate, family, f"content target for {logical}")
                size = resolved.stat().st_size
                digest = file_sha256(resolved)
            except OSError as exc:
                raise CacheIntegrityError(f"unreadable snapshot file: {logical}") from exc
            records.append(
                {
                    "logical_path": logical,
                    "size_bytes": size,
                    "sha256": digest,
                    "resolved_content_relative_path": resolved.relative_to(family).as_posix(),
                }
            )
    return sorted(records, key=lambda record: record["logical_path"])


def _manifest_from_snapshot(hub_cache: Path, model_id: str, revision: str | None) -> dict[str, Any]:
    family, family_name = _model_family(hub_cache, model_id)
    selected = select_revision(family, revision)
    snapshot = family / "snapshots" / selected
    records = _snapshot_files(snapshot, family)
    unique_targets = {record["resolved_content_relative_path"]: record for record in records}
    return {
        "schema_version": SCHEMA_VERSION,
        "marker": MARKER,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "model_id": model_id,
        "revision": selected,
        "model_cache_family_relative_path": family_name,
        "snapshot_relative_path": f"snapshots/{selected}",
        "files": records,
        "summary": {
            "logical_file_count": len(records),
            "unique_content_object_count": len(unique_targets),
            "logical_total_bytes": sum(record["size_bytes"] for record in records),
            "unique_content_total_bytes": sum(record["size_bytes"] for record in unique_targets.values()),
        },
    }


def _validate_manifest_shape(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CacheIntegrityError("manifest root must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CacheIntegrityError("unsupported schema_version")
    if manifest.get("marker") != MARKER or manifest.get("evidence_boundary") != EVIDENCE_BOUNDARY:
        raise CacheIntegrityError("manifest marker or evidence boundary mismatch")
    for key in ("model_id", "revision", "model_cache_family_relative_path", "snapshot_relative_path", "files", "summary"):
        if key not in manifest:
            raise CacheIntegrityError(f"manifest field missing: {key}")
    if not isinstance(manifest["model_id"], str) or not isinstance(manifest["files"], list) or not isinstance(manifest["summary"], dict):
        raise CacheIntegrityError("malformed manifest structure")
    _safe_relative(manifest["revision"], "revision")
    _safe_relative(manifest["model_cache_family_relative_path"], "model_cache_family_relative_path")
    _safe_relative(manifest["snapshot_relative_path"], "snapshot_relative_path")
    seen: set[str] = set()
    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise CacheIntegrityError("malformed file record")
        for key in ("logical_path", "resolved_content_relative_path", "size_bytes", "sha256"):
            if key not in record:
                raise CacheIntegrityError(f"file record field missing: {key}")
        logical = _safe_relative(record["logical_path"], "logical_path")
        resolved = _safe_relative(record["resolved_content_relative_path"], "resolved_content_relative_path")
        if logical in seen:
            raise CacheIntegrityError(f"duplicate manifest logical path: {logical}")
        seen.add(logical)
        if not isinstance(record["size_bytes"], int) or record["size_bytes"] < 0:
            raise CacheIntegrityError("malformed file size")
        if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in record["sha256"]):
            raise CacheIntegrityError("malformed file SHA256")
    expected_summary = {"logical_file_count", "unique_content_object_count", "logical_total_bytes", "unique_content_total_bytes"}
    if set(manifest["summary"]) != expected_summary or any(not isinstance(manifest["summary"].get(key), int) or manifest["summary"][key] < 0 for key in expected_summary):
        raise CacheIntegrityError("malformed summary")
    return manifest


def verify_manifest(hub_cache: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = _validate_manifest_shape(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, CacheIntegrityError) as exc:
        if isinstance(exc, CacheIntegrityError):
            raise
        raise CacheIntegrityError(f"cannot read manifest: {exc}") from exc
    family, family_name = _model_family(hub_cache, manifest["model_id"])
    if manifest["model_cache_family_relative_path"] != family_name:
        raise CacheIntegrityError("model cache family path mismatch")
    revision = manifest["revision"]
    if manifest["snapshot_relative_path"] != f"snapshots/{revision}":
        raise CacheIntegrityError("snapshot path mismatch")
    actual = _snapshot_files(family / "snapshots" / revision, family)
    if actual != sorted(manifest["files"], key=lambda record: record["logical_path"]):
        raise CacheIntegrityError("snapshot file identity mismatch")
    unique = {record["resolved_content_relative_path"]: record for record in actual}
    summary = {
        "logical_file_count": len(actual),
        "unique_content_object_count": len(unique),
        "logical_total_bytes": sum(record["size_bytes"] for record in actual),
        "unique_content_total_bytes": sum(record["size_bytes"] for record in unique.values()),
    }
    if manifest["summary"] != summary:
        raise CacheIntegrityError("summary mismatch")
    return manifest


def atomic_write_manifest(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise CacheIntegrityError("manifest exists; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size == 0:
            raise CacheIntegrityError("temporary manifest is empty")
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            try:
                fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def create_manifest(hub_cache: Path, model_id: str, manifest_path: Path, revision: str | None = None, *, overwrite: bool = False) -> dict[str, Any]:
    payload = _manifest_from_snapshot(hub_cache.resolve(), model_id, revision)
    atomic_write_manifest(manifest_path, payload, overwrite=overwrite)
    try:
        written = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_shape(written)
    except (OSError, json.JSONDecodeError, CacheIntegrityError) as exc:
        raise CacheIntegrityError(f"written manifest validation failed: {exc}") from exc
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or verify a read-only Hugging Face cache integrity manifest.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--hub-cache", type=Path, required=True)
    create.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--revision", default=None)
    create.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--hub-cache", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.operation == "create":
            manifest = create_manifest(args.hub_cache, args.model_id, args.manifest, args.revision, overwrite=args.overwrite)
            print(f"CACHE_INTEGRITY: PASS files={manifest['summary']['logical_file_count']} revision={manifest['revision']}")
        else:
            manifest = verify_manifest(args.hub_cache, args.manifest)
            print(f"CACHE_INTEGRITY: PASS files={manifest['summary']['logical_file_count']} revision={manifest['revision']}")
        print(EVIDENCE_BOUNDARY)
        return 0
    except (CacheIntegrityError, OSError, ValueError) as exc:
        print(f"CACHE_INTEGRITY: FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
